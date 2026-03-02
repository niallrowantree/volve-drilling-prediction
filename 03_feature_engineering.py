# Databricks notebook source
# MAGIC %md
# MAGIC # Volve ROP — Notebook 03: Feature Engineering
# MAGIC
# MAGIC Reads `workspace.volve_ml.train_raw` (907 K rows, 12 wells) and
# MAGIC `workspace.volve_ml.test_raw` (543 K rows, 2 wells) at 30-second resolution,
# MAGIC then produces clean feature matrices ready for model training.
# MAGIC
# MAGIC **Pipeline**
# MAGIC
# MAGIC | Step | Operation |
# MAGIC |------|-----------|
# MAGIC | 1 | Derive `spm_total` (sum of pump stroke channels) |
# MAGIC | 1b | Join formation labels from `formation_tops` (built in NB05) |
# MAGIC | 2 | Forward-fill nulls within each well (sensor dropout) |
# MAGIC | 3 | Fill remaining nulls with train-set medians |
# MAGIC | 3b | Clip outliers to [1st, 99th] percentile from train |
# MAGIC | 4 | Compute rolling features (5-min and 30-min windows) |
# MAGIC | 5 | Compute lag features (t−1, t−5) |
# MAGIC | 6 | Drop warm-up rows (insufficient preceding data for 30-min window) |
# MAGIC | 7 | Write `train_features` and `test_features` Delta tables |
# MAGIC
# MAGIC **All statistics (medians) are computed on train only and applied to test
# MAGIC to avoid data leakage.**
# MAGIC
# MAGIC **Rolling/lag windows are computed within each well** (partitioned by
# MAGIC `well_name`) so no signal bleeds across wells.
# MAGIC
# MAGIC **`dmea` and `bpos` are excluded** — raw depth is a well-identity proxy that
# MAGIC prevents generalisation to unseen wells. Geological context is instead encoded
# MAGIC as `formation_id` (ordinal integer from NB05 formation tops join).

# COMMAND ----------

TRAIN_RAW      = "workspace.volve_ml.train_raw"
TEST_RAW       = "workspace.volve_ml.test_raw"
TRAIN_FEAT     = "workspace.volve_ml.train_features"
TEST_FEAT      = "workspace.volve_ml.test_features"
FORMATION_TOPS = "workspace.volve_ml.formation_tops"

TARGET = "rop"

# Channels to impute and include as base features.
# spm1/spm2/spm3 are collapsed into spm_total; raw pump channels dropped after.
# dmea and bpos are EXCLUDED — raw depth is a well-identity proxy that causes
# the model to memorise well-specific geology rather than learn transferable
# drilling physics. Geological context is encoded via formation_id instead.
BASE_CHANNELS = ["wob", "rpm", "spp", "flow_in", "torque", "hkld",
                 "spm1", "spm2", "spm3", "mwti"]

# Channels to generate rolling and lag features for.
ROLL_CHANNELS = ["wob", "rpm", "spp", "flow_in", "torque"]
LAG_CHANNELS  = ["wob", "rpm"]

# Rolling window sizes (rows at 30-second intervals).
ROLL_SHORT = 10   # 5 minutes
ROLL_LONG  = 60   # 30 minutes

# Rows to drop per well after feature computation (= ROLL_LONG warm-up rows).
WARMUP_ROWS = ROLL_LONG

# COMMAND ----------
# MAGIC %md ## Step 1: Load tables and derive spm_total

# COMMAND ----------

from pyspark.sql import functions as F, Window

df_train = spark.read.table(TRAIN_RAW)
df_test  = spark.read.table(TEST_RAW)

print(f"Train raw: {df_train.count():,} rows  |  "
      f"Test raw: {df_test.count():,} rows")

def add_spm_total(df):
    """Sum pump stroke channels; treat nulls as zero so partial readings count."""
    return df.withColumn(
        "spm_total",
        F.coalesce(F.col("spm1"), F.lit(0.0)) +
        F.coalesce(F.col("spm2"), F.lit(0.0)) +
        F.coalesce(F.col("spm3"), F.lit(0.0))
    ).drop("spm1", "spm2", "spm3")

df_train = add_spm_total(df_train)
df_test  = add_spm_total(df_test)

# Updated base channel list after collapsing pump channels
FEAT_CHANNELS = [c for c in BASE_CHANNELS if c not in ("spm1", "spm2", "spm3")] + ["spm_total"]
print(f"\nFeature channels ({len(FEAT_CHANNELS)}): {FEAT_CHANNELS}")

# COMMAND ----------
# MAGIC %md ## Step 1b: Join formation labels
# MAGIC
# MAGIC Formation tops (built in Notebook 05) give a `formation_id` ordinal integer
# MAGIC that encodes which geological formation the bit is in at each row. This
# MAGIC captures the "Brent Group vs Chalk vs Hordaland" context that drives
# MAGIC fundamentally different drilling behaviour, without leaking well identity.
# MAGIC
# MAGIC **Depth unit correction:** F-15S has `dmea` recorded in feet in the WITSML
# MAGIC source data (max ~24,939 vs ~3,500 m for all other wells). We detect this
# MAGIC per-well and convert before the interval join. The corrected depth is only
# MAGIC used for the join and is not kept as a feature.

# COMMAND ----------

ft_df = spark.read.table(FORMATION_TOPS)

# Detect per-well depth unit: if max(dmea) > 10000 the channel is in feet
well_units = (
    df_train.union(df_test)
    .groupBy("well_name")
    .agg(F.max("dmea").alias("max_dmea"))
    .withColumn("dmea_scale", F.when(F.col("max_dmea") > 10000, 1.0 / 3.28084).otherwise(1.0))
    .select("well_name", "dmea_scale")
)

def add_formation(df, ft_df, well_units):
    """Join formation_id from formation_tops using depth-corrected dmea."""
    # Add depth scale factor per well
    df = df.join(well_units, on="well_name", how="left")
    df = df.withColumn("dmea_m", F.col("dmea") * F.col("dmea_scale")).drop("dmea_scale")

    # Left join: find the interval [md_from_m, md_to_m) that contains dmea_m
    ft_df2 = ft_df.select(
        F.col("well_name").alias("_ft_well"),
        "md_from_m", "md_to_m", "formation", "formation_id"
    )
    df = (df.join(ft_df2,
                  (df.well_name   == ft_df2._ft_well) &
                  (df.dmea_m      >= ft_df2.md_from_m) &
                  (df.dmea_m       < ft_df2.md_to_m),
                  how="left")
            .drop("_ft_well", "md_from_m", "md_to_m", "formation", "dmea_m"))

    # Rows above all formation tops (conductor/surface section) get id 0
    df = df.fillna({"formation_id": 0})
    return df

df_train = add_formation(df_train, ft_df, well_units)
df_test  = add_formation(df_test,  ft_df, well_units)

print("Formation label join complete.")
print("\nTrain formation distribution:")
df_train.groupBy("formation_id").count().orderBy("formation_id").show()
print("Test formation distribution:")
df_test.groupBy("formation_id").count().orderBy("formation_id").show()

# COMMAND ----------
# MAGIC %md ## Step 2: Forward-fill nulls within each well
# MAGIC
# MAGIC Sensor dropouts produce short null runs mid-well. Forward-filling within the
# MAGIC well boundary propagates the last valid reading forward — appropriate for
# MAGIC slowly-varying drilling parameters. Partitioned by well to avoid cross-well leakage.

# COMMAND ----------

w_ffill = (Window.partitionBy("well_name")
                 .orderBy("ts")
                 .rowsBetween(Window.unboundedPreceding, 0))

def forward_fill(df, channels):
    for c in channels:
        df = df.withColumn(c, F.last(F.col(c), ignorenulls=True).over(w_ffill))
    return df

df_train = forward_fill(df_train, FEAT_CHANNELS)
df_test  = forward_fill(df_test,  FEAT_CHANNELS)

print("Forward fill complete.")

# COMMAND ----------
# MAGIC %md ## Step 3: Fill remaining nulls with train-set medians
# MAGIC
# MAGIC After forward-fill, nulls only remain at the very start of a well where no
# MAGIC prior value exists. These are filled with the global median computed on the
# MAGIC **train set only**. The same values are applied to the test set.

# COMMAND ----------

# Compute medians on train only
median_exprs = [F.percentile_approx(c, 0.5).alias(c) for c in FEAT_CHANNELS]
train_medians_row = df_train.agg(*median_exprs).collect()[0]
train_medians = {c: float(train_medians_row[c]) for c in FEAT_CHANNELS
                 if train_medians_row[c] is not None}

print("Train medians (used for null fill on both splits):")
for c, v in sorted(train_medians.items()):
    print(f"  {c:<12} {v:.4f}")

df_train = df_train.fillna(train_medians)
df_test  = df_test.fillna(train_medians)

# Verify no nulls remain in feature channels
print("\nRemaining nulls after imputation:")
null_exprs = [F.sum(F.col(c).isNull().cast("int")).alias(c) for c in FEAT_CHANNELS]
df_train.agg(*null_exprs).show(truncate=False)

# COMMAND ----------
# MAGIC %md ## Step 3b: Clip outliers
# MAGIC
# MAGIC Some channels contain physically impossible sensor error values (e.g. SPP > 1e27).
# MAGIC Values are clipped to the [1st, 99th] percentile range computed from the
# MAGIC **train set only** and applied to both splits. This preserves the true tails of
# MAGIC the distribution while removing clear instrument faults.
# MAGIC
# MAGIC The target (`rop`) is also clipped — extreme ROP values are equally unreliable.

# COMMAND ----------

CLIP_CHANNELS = FEAT_CHANNELS + [TARGET]

# Compute percentile bounds on train only.
# Using 1st/99th percentile: aggressive enough to catch ~1% sensor-fault readings
# (spp, torque, flow_in have contamination at roughly that level) while preserving
# 98% of real values for channels that are already clean.
pct_exprs = []
for c in CLIP_CHANNELS:
    pct_exprs += [
        F.percentile_approx(c, 0.01).alias(f"{c}_lo"),
        F.percentile_approx(c, 0.99).alias(f"{c}_hi"),
    ]
bounds_row = df_train.agg(*pct_exprs).collect()[0]
clip_bounds = {
    c: (float(bounds_row[f"{c}_lo"]), float(bounds_row[f"{c}_hi"]))
    for c in CLIP_CHANNELS
}

print("Clip bounds (0.1th – 99.9th percentile from train):")
print(f"  {'Channel':<12}  {'Low':>12}  {'High':>12}")
print("  " + "-" * 40)
for c, (lo, hi) in sorted(clip_bounds.items()):
    print(f"  {c:<12}  {lo:>12.4f}  {hi:>12.4f}")

def apply_clip(df, bounds):
    for c, (lo, hi) in bounds.items():
        df = df.withColumn(c, F.greatest(F.lit(lo), F.least(F.lit(hi), F.col(c))))
    return df

df_train = apply_clip(df_train, clip_bounds)
df_test  = apply_clip(df_test,  clip_bounds)

print("\nClipping complete. Re-checking distributions:")
stats_exprs = []
for c in ["wob", "rpm", "spp", "flow_in", "torque", "rop"]:
    stats_exprs += [
        F.round(F.mean(c), 3).alias(f"{c}_mean"),
        F.round(F.stddev(c), 3).alias(f"{c}_std"),
    ]
print("Train:"); df_train.agg(*stats_exprs).show(truncate=False)
print("Test: "); df_test.agg(*stats_exprs).show(truncate=False)

# COMMAND ----------
# MAGIC %md ## Step 4: Rolling window features
# MAGIC
# MAGIC Windows are computed on the **preceding** rows only (not including the current
# MAGIC row) to avoid leaking the current observation into its own smoothed feature.
# MAGIC Partitioned by well so no cross-well signal.
# MAGIC
# MAGIC | Feature suffix | Window | Duration |
# MAGIC |---------------|--------|----------|
# MAGIC | `_r10_mean` | preceding 10 rows | 5 min |
# MAGIC | `_r60_mean` | preceding 60 rows | 30 min |
# MAGIC | `_r60_std` | preceding 60 rows | 30 min |
# MAGIC
# MAGIC ROP is excluded from rolling/lag features — including past ROP values causes
# MAGIC the model to learn autocorrelation rather than the physical relationship between
# MAGIC drilling parameters and ROP.

# COMMAND ----------

w_base  = Window.partitionBy("well_name").orderBy("ts")
w_short = w_base.rowsBetween(-ROLL_SHORT, -1)
w_long  = w_base.rowsBetween(-ROLL_LONG,  -1)

def add_rolling_features(df, channels):
    exprs = []
    for c in channels:
        exprs += [
            F.avg(c).over(w_short).alias(f"{c}_r10_mean"),
            F.avg(c).over(w_long).alias(f"{c}_r60_mean"),
            F.stddev(c).over(w_long).alias(f"{c}_r60_std"),
        ]
    return df.select("*", *exprs)

df_train = add_rolling_features(df_train, ROLL_CHANNELS)
df_test  = add_rolling_features(df_test,  ROLL_CHANNELS)

print(f"Added {len(ROLL_CHANNELS) * 3} rolling features.")

# COMMAND ----------
# MAGIC %md ## Step 5: Lag features
# MAGIC
# MAGIC Lag 1 (30 s ago) and lag 5 (2.5 min ago), again within-well only.
# MAGIC Including lagged ROP is valid because at prediction time we know the ROP
# MAGIC that was measured one or more 30-second intervals ago.

# COMMAND ----------

w_lag = Window.partitionBy("well_name").orderBy("ts")

def add_lag_features(df, channels):
    exprs = []
    for c in channels:
        exprs += [
            F.lag(c, 1).over(w_lag).alias(f"{c}_lag1"),
            F.lag(c, 5).over(w_lag).alias(f"{c}_lag5"),
        ]
    return df.select("*", *exprs)

df_train = add_lag_features(df_train, LAG_CHANNELS)
df_test  = add_lag_features(df_test,  LAG_CHANNELS)

print(f"Added {len(LAG_CHANNELS) * 2} lag features.")

# COMMAND ----------
# MAGIC %md ## Step 6: Drop warm-up rows and verify feature completeness
# MAGIC
# MAGIC The first `ROLL_LONG` (60) rows of each well have insufficient preceding data
# MAGIC for the 30-minute rolling window and will contain nulls. These rows are dropped.
# MAGIC Any remaining nulls (e.g. std with < 2 values in window) are zero-filled.

# COMMAND ----------

# Identify all feature columns (everything except ts, well_name, rop)
all_feature_cols = [c for c in df_train.columns
                    if c not in ("ts", "well_name", TARGET)]

def drop_warmup_and_clean(df, warmup=WARMUP_ROWS):
    # Row number within each well
    w_rn = Window.partitionBy("well_name").orderBy("ts")
    df = df.withColumn("_rn", F.row_number().over(w_rn))
    df = df.filter(F.col("_rn") > warmup).drop("_rn")
    # Zero-fill any residual nulls (e.g. stddev with <2 rows in window)
    df = df.fillna(0.0, subset=all_feature_cols)
    return df

df_train = drop_warmup_and_clean(df_train)
df_test  = drop_warmup_and_clean(df_test)

n_train = df_train.count()
n_test  = df_test.count()

print(f"Train after warmup drop: {n_train:,} rows")
print(f"Test  after warmup drop: {n_test:,} rows")

# Confirm zero nulls in final feature set
print("\nNull counts in final train feature matrix:")
null_check = [F.sum(F.col(c).isNull().cast("int")).alias(c) for c in all_feature_cols]
df_train.agg(*null_check).show(truncate=False)

# COMMAND ----------
# MAGIC %md ## Step 7: Summary and write feature tables

# COMMAND ----------

print(f"Feature columns ({len(all_feature_cols)}):")
for c in sorted(all_feature_cols):
    print(f"  {c}")

print(f"\nTarget : {TARGET}")
print(f"\nTrain  : {n_train:,} rows  |  {df_train.select('well_name').distinct().count()} wells")
print(f"Test   : {n_test:,} rows  |  {df_test.select('well_name').distinct().count()} wells")

for tbl in [TRAIN_FEAT, TEST_FEAT]:
    spark.sql(f"DROP TABLE IF EXISTS {tbl}")

df_train.write.format("delta").mode("overwrite").saveAsTable(TRAIN_FEAT)
print(f"\nWritten: {TRAIN_FEAT}  ({n_train:,} rows)")

df_test.write.format("delta").mode("overwrite").saveAsTable(TEST_FEAT)
print(f"Written: {TEST_FEAT}  ({n_test:,} rows)")

# COMMAND ----------
# MAGIC %md ## Step 8: Feature distribution check (train vs test)
# MAGIC
# MAGIC Compare mean and std of key features across splits — large divergence would
# MAGIC indicate distribution shift between training wells and test wells.

# COMMAND ----------

from pyspark.sql.functions import col

check_cols = ["wob", "rpm", "spp", "flow_in", "torque", "hkld", "spm_total", "mwti"]

stats_exprs = []
for c in check_cols:
    stats_exprs += [
        F.round(F.mean(c), 3).alias(f"{c}_mean"),
        F.round(F.stddev(c), 3).alias(f"{c}_std"),
    ]

print("── TRAIN ──")
df_train.agg(*stats_exprs).show(truncate=False)

print("── TEST ──")
df_test.agg(*stats_exprs).show(truncate=False)

print("── TRAIN formation_id distribution ──")
df_train.groupBy("formation_id").count().orderBy("formation_id").show()

print("── TEST formation_id distribution ──")
df_test.groupBy("formation_id").count().orderBy("formation_id").show()
