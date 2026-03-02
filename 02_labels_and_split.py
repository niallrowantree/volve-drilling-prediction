# Databricks notebook source
# MAGIC %md
# MAGIC # Volve ROP — Notebook 02: Labels, Resampling & Train/Test Split
# MAGIC
# MAGIC Reads `workspace.volve_ml.drilling_raw` (15.09 M rows, 14 wells, 2007–2009),
# MAGIC downsamples to a 30-second grid, drops rows without an ROP label, and splits
# MAGIC into train / test by well to avoid temporal data leakage.
# MAGIC
# MAGIC **Test well selection rationale**
# MAGIC
# MAGIC Wells are held out *in their entirety* so the model is evaluated on wells it
# MAGIC has never seen — the realistic production use-case. Two wells were chosen for
# MAGIC their good ROP coverage (< 30 % null) and coverage of different time windows:
# MAGIC
# MAGIC | Well | Rows | ROP null % | Drilling window |
# MAGIC |------|------|-----------|-----------------|
# MAGIC | F-12 (test) | 2.09 M | 26 % | Jun 2007 – Feb 2008 |
# MAGIC | F-15S (test) | 1.88 M | 22 % | Sep – Dec 2008 |
# MAGIC
# MAGIC **Outputs**
# MAGIC
# MAGIC | Table | Contents |
# MAGIC |-------|----------|
# MAGIC | `workspace.volve_ml.train_raw` | 12 training wells, 30 s grid, rop non-null |
# MAGIC | `workspace.volve_ml.test_raw` | 2 held-out wells, 30 s grid, rop non-null |

# COMMAND ----------

INPUT_TABLE  = "workspace.volve_ml.drilling_raw"
TRAIN_TABLE  = "workspace.volve_ml.train_raw"
TEST_TABLE   = "workspace.volve_ml.test_raw"

TEST_WELLS = [
    "Norway-Statoil-15_$47$_9-F-12",
    "Norway-StatoilHydro-15_$47$_9-F-15S",
]

RESAMPLE_SECONDS = 30   # downsample from ~4 s raw to 30 s grid
TARGET           = "rop"
CHANNELS         = ["rop", "wob", "rpm", "spp", "flow_in", "torque", "hkld", "dmea",
                    "bpos", "spm1", "spm2", "spm3", "mwti"]
# ecd dropped: 91.6% null across all wells — genuinely absent from this dataset

# COMMAND ----------
# MAGIC %md ## Step 1: Load raw table and audit channel coverage

# COMMAND ----------

from pyspark.sql import functions as F

df = spark.read.table(INPUT_TABLE)
n_raw = df.count()
print(f"Raw rows: {n_raw:,}  across {df.select('well_name').distinct().count()} wells\n")

print("Null % per channel per well:")
null_exprs = [
    F.round(100.0 * F.sum(F.col(c).isNull().cast("int")) / F.count("*"), 1).alias(c)
    for c in CHANNELS
]
(df.groupBy("well_name")
   .agg(F.count("*").alias("rows"), *null_exprs)
   .orderBy("well_name")
   .show(20, truncate=False))

# COMMAND ----------
# MAGIC %md ## Step 2: Resample to 30-second grid
# MAGIC
# MAGIC Raw data is logged at ~4-second intervals. Bucketing to 30 seconds:
# MAGIC - Reduces volume ~7× (fewer redundant rows, less autocorrelation)
# MAGIC - Averages out sensor noise within each window
# MAGIC - Produces a uniform time grid per well

# COMMAND ----------

df_resampled = (
    df
    .withColumn(
        "ts",
        F.to_timestamp(
            (F.unix_timestamp("ts") / RESAMPLE_SECONDS).cast("long") * RESAMPLE_SECONDS
        )
    )
    .groupBy("well_name", "ts")
    .agg(*[F.mean(c).alias(c) for c in CHANNELS])
    .orderBy("well_name", "ts")
)

n_resampled = df_resampled.count()
print(f"After 30 s resampling: {n_resampled:,} rows  "
      f"({n_resampled / n_raw * 100:.1f}% of raw)")

# COMMAND ----------
# MAGIC %md ## Step 3: Drop rows without ROP label
# MAGIC
# MAGIC ROP is the prediction target. Rows where ROP is null carry no training signal
# MAGIC and are removed. Other channels may still be null — that is handled in
# MAGIC Notebook 03 during feature engineering.

# COMMAND ----------

df_labeled = df_resampled.filter(F.col(TARGET).isNotNull())

n_labeled = df_labeled.count()
print(f"After dropping null-ROP rows: {n_labeled:,}  "
      f"({n_labeled / n_resampled * 100:.1f}% of resampled)\n")

print("Rows and ROP stats per well:")
(df_labeled.groupBy("well_name")
    .agg(
        F.count("*").alias("rows"),
        F.round(F.mean(TARGET), 2).alias("rop_mean"),
        F.round(F.stddev(TARGET), 2).alias("rop_std"),
        F.round(F.min(TARGET), 2).alias("rop_min"),
        F.round(F.max(TARGET), 2).alias("rop_max"),
    )
    .orderBy("well_name")
    .show(20, truncate=False))

# COMMAND ----------
# MAGIC %md ## Step 4: Train / test split by well
# MAGIC
# MAGIC Two wells are held out entirely as the test set. All other wells form the
# MAGIC training set. Splitting by well (rather than by time within a well) prevents
# MAGIC the model from learning well-specific temporal patterns and gives a more
# MAGIC honest estimate of generalisation to new wells.

# COMMAND ----------

df_test  = df_labeled.filter( F.col("well_name").isin(TEST_WELLS))
df_train = df_labeled.filter(~F.col("well_name").isin(TEST_WELLS))

n_train = df_train.count()
n_test  = df_test.count()

print(f"Train: {n_train:,} rows  ({n_train / n_labeled * 100:.1f}%)")
print(f"Test : {n_test:,} rows  ({n_test  / n_labeled * 100:.1f}%)\n")

print("Training wells:")
df_train.groupBy("well_name").count().orderBy("well_name").show(truncate=False)

print("Test wells:")
df_test.groupBy("well_name").count().orderBy("well_name").show(truncate=False)

# COMMAND ----------
# MAGIC %md ## Step 5: Write train and test tables

# COMMAND ----------

for tbl in [TRAIN_TABLE, TEST_TABLE]:
    spark.sql(f"DROP TABLE IF EXISTS {tbl}")

df_train.write.format("delta").mode("overwrite").saveAsTable(TRAIN_TABLE)
print(f"Written: {TRAIN_TABLE}  ({n_train:,} rows)")

df_test.write.format("delta").mode("overwrite").saveAsTable(TEST_TABLE)
print(f"Written: {TEST_TABLE}  ({n_test:,} rows)")

# COMMAND ----------
# MAGIC %md ## Step 6: Verify — channel coverage in train and test

# COMMAND ----------

for label, tbl in [("TRAIN", TRAIN_TABLE), ("TEST", TEST_TABLE)]:
    df_v = spark.read.table(tbl)
    print(f"\n── {label} ({tbl}) ──")
    null_exprs = [
        F.round(100.0 * F.sum(F.col(c).isNull().cast("int")) / F.count("*"), 1).alias(c)
        for c in CHANNELS
    ]
    df_v.agg(F.count("*").alias("rows"), *null_exprs).show(truncate=False)
