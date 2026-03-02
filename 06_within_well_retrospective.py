# Databricks notebook source
# MAGIC %pip install xgboost

# COMMAND ----------
# MAGIC %md
# MAGIC # Volve ROP — Notebook 06: Within-Well Retrospective
# MAGIC
# MAGIC Trains an XGBoost model with an **80/20 time-based split within each well**:
# MAGIC the model learns from the first 80 % of each well's drilling timeline and is
# MAGIC evaluated on the final 20 %. Because the split is within-well, `dmea` is a
# MAGIC legitimate feature (depth is meaningful within a single well campaign).
# MAGIC
# MAGIC **On-bottom filter:** only rows where WOB > 2 kN, RPM > 20, and ROP > 1 m/hr
# MAGIC are used for training and evaluation, to exclude non-drilling states
# MAGIC (tripping, reaming, washing, surface operations). Rolling/lag features are
# MAGIC computed on the full time series before filtering so temporal context is preserved.
# MAGIC
# MAGIC **Outputs**
# MAGIC
# MAGIC | Table | Contents |
# MAGIC |-------|----------|
# MAGIC | `workspace.volve_ml.retrospective` | Per-row predictions + efficiency score (on-bottom only) |
# MAGIC
# MAGIC **Efficiency score** = `actual_rop / predicted_rop`. Values below 1.0 indicate
# MAGIC the well was drilling slower than the model expected given those parameters and
# MAGIC depth. Aggregated to 1-hour windows this highlights sustained underperformance.

# COMMAND ----------

ALL_WELLS_TABLES = [
    "workspace.volve_ml.train_raw",
    "workspace.volve_ml.test_raw",
]
RETRO_TABLE = "workspace.volve_ml.retrospective"

TARGET        = "rop"
TRAIN_FRAC    = 0.80   # first 80 % of each well's timeline → train
ROLL_SHORT    = 10     # 5-min rolling window (rows at 30 s)
ROLL_LONG     = 60     # 30-min rolling window
WARMUP_ROWS   = ROLL_LONG

# On-bottom filter — exclude non-drilling states
MIN_WOB = 2.0    # kN
MIN_RPM = 20.0   # rpm
MIN_ROP = 1.0    # m/hr

XGB_PARAMS = {
    "n_estimators":     400,
    "max_depth":        5,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 30,
    "objective":        "reg:squarederror",
    "tree_method":      "hist",
    "random_state":     42,
}
EARLY_STOPPING_ROUNDS = 20

# COMMAND ----------
# MAGIC %md ## Step 1: Load and combine all wells

# COMMAND ----------

from pyspark.sql import functions as F, Window
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

dfs = [spark.read.table(t) for t in ALL_WELLS_TABLES]
df  = dfs[0]
for d in dfs[1:]:
    df = df.union(d)

n_total = df.count()
n_wells = df.select("well_name").distinct().count()
print(f"Combined dataset: {n_total:,} rows  |  {n_wells} wells")

# COMMAND ----------
# MAGIC %md ## Step 2: Derive spm_total

# COMMAND ----------

df = (df
      .withColumn("spm_total",
                  F.coalesce(F.col("spm1"), F.lit(0.0)) +
                  F.coalesce(F.col("spm2"), F.lit(0.0)) +
                  F.coalesce(F.col("spm3"), F.lit(0.0)))
      .drop("spm1", "spm2", "spm3"))

BASE_CHANNELS = ["wob", "rpm", "spp", "flow_in", "torque",
                 "hkld", "mwti", "spm_total"]
# dmea excluded: in the combined cross-well model, depth encodes "which well"
# rather than physics, and clipping causes F-10/F-15S test rows to be stuck
# at the training 99th-percentile ceiling (3793 m).

ROLL_CHANNELS = ["wob", "rpm", "spp", "flow_in", "torque"]
LAG_CHANNELS  = ["wob", "rpm"]

print(f"Base channels: {BASE_CHANNELS}")

# COMMAND ----------
# MAGIC %md ## Step 3: Impute, clip, compute features — within-well
# MAGIC
# MAGIC All statistics are computed **on the training portion only** and applied
# MAGIC to the test portion to avoid leakage.

# COMMAND ----------

# ── 3a: Forward-fill within each well ──────────────────────────────────
w_ffill = (Window.partitionBy("well_name")
                 .orderBy("ts")
                 .rowsBetween(Window.unboundedPreceding, 0))

for c in BASE_CHANNELS:
    df = df.withColumn(c, F.last(F.col(c), ignorenulls=True).over(w_ffill))

# ── 3b: Time-based split flag ─────────────────────────────────────────
# For each well find the 80th-percentile timestamp; rows ≤ that are "train"
w_well = Window.partitionBy("well_name").orderBy("ts")
df = df.withColumn("_rn",    F.row_number().over(w_well))
df = df.withColumn("_total", F.count("*").over(Window.partitionBy("well_name")))
df = df.withColumn("split",
                   F.when(F.col("_rn") <= (F.col("_total") * TRAIN_FRAC).cast("int"),
                          "train")
                    .otherwise("test")
                   ).drop("_rn", "_total")

n_train = df.filter(F.col("split") == "train").count()
n_test  = df.filter(F.col("split") == "test").count()
print(f"Train rows: {n_train:,}  |  Test rows: {n_test:,}")

# ── 3c: Median-fill from train portion only ───────────────────────────
df_tr = df.filter(F.col("split") == "train")
median_exprs = [F.percentile_approx(c, 0.5).alias(c) for c in BASE_CHANNELS]
medians_row  = df_tr.agg(*median_exprs).collect()[0]
medians = {c: float(medians_row[c]) for c in BASE_CHANNELS
           if medians_row[c] is not None}

df = df.fillna(medians)

# ── 3d: Clip to [1st, 99th] percentile from train ────────────────────
CLIP_CHANNELS = BASE_CHANNELS + [TARGET]
pct_exprs = []
for c in CLIP_CHANNELS:
    pct_exprs += [
        F.percentile_approx(c, 0.01).alias(f"{c}_lo"),
        F.percentile_approx(c, 0.99).alias(f"{c}_hi"),
    ]
bounds_row = df_tr.agg(*pct_exprs).collect()[0]
clip_bounds = {
    c: (float(bounds_row[f"{c}_lo"]), float(bounds_row[f"{c}_hi"]))
    for c in CLIP_CHANNELS
}

for c, (lo, hi) in clip_bounds.items():
    df = df.withColumn(c, F.greatest(F.lit(lo), F.least(F.lit(hi), F.col(c))))

print("Imputation and clipping complete.")

# COMMAND ----------
# MAGIC %md ## Step 4: Rolling and lag features

# COMMAND ----------

w_base  = Window.partitionBy("well_name").orderBy("ts")
w_short = w_base.rowsBetween(-ROLL_SHORT, -1)
w_long  = w_base.rowsBetween(-ROLL_LONG,  -1)
w_lag   = w_base

roll_exprs = []
for c in ROLL_CHANNELS:
    roll_exprs += [
        F.avg(c).over(w_short).alias(f"{c}_r10_mean"),
        F.avg(c).over(w_long).alias(f"{c}_r60_mean"),
        F.stddev(c).over(w_long).alias(f"{c}_r60_std"),
    ]
df = df.select("*", *roll_exprs)

lag_exprs = []
for c in LAG_CHANNELS:
    lag_exprs += [
        F.lag(c, 1).over(w_lag).alias(f"{c}_lag1"),
        F.lag(c, 5).over(w_lag).alias(f"{c}_lag5"),
    ]
df = df.select("*", *lag_exprs)

print(f"Added {len(ROLL_CHANNELS)*3} rolling + {len(LAG_CHANNELS)*2} lag features.")

# COMMAND ----------
# MAGIC %md ## Step 5: Drop warm-up rows and zero-fill residual nulls

# COMMAND ----------

feat_cols = [c for c in df.columns
             if c not in ("ts", "well_name", TARGET, "split")]

df = df.withColumn("_rn2", F.row_number().over(
    Window.partitionBy("well_name").orderBy("ts")))
df = df.filter(F.col("_rn2") > WARMUP_ROWS).drop("_rn2")
df = df.fillna(0.0, subset=feat_cols)

n_tr = df.filter(F.col("split") == "train").count()
n_te = df.filter(F.col("split") == "test").count()
print(f"After warm-up drop — Train: {n_tr:,}  Test: {n_te:,}")
print(f"Feature columns ({len(feat_cols)}): {sorted(feat_cols)}")

# COMMAND ----------
# MAGIC %md ## Step 6: Collect to pandas and apply on-bottom filter
# MAGIC
# MAGIC Rolling/lag features were computed on the full time series to preserve
# MAGIC temporal context. Now filter to active drilling rows only before training.

# COMMAND ----------

pdf = df.toPandas()

# Apply on-bottom filter — exclude tripping, reaming, surface operations
on_bottom_mask = (
    (pdf["wob"]  > MIN_WOB) &
    (pdf["rpm"]  > MIN_RPM) &
    (pdf["rop"]  > MIN_ROP)
)
pdf_ob = pdf[on_bottom_mask].copy()

print(f"Total rows:      {len(pdf):,}")
print(f"On-bottom rows:  {len(pdf_ob):,}  ({len(pdf_ob)/len(pdf):.1%} of total)")
print(f"\nOn-bottom rows per well:")
print(pdf_ob.groupby("well_name")["rop"].count().rename("n_ob").to_string())

train_pd = pdf_ob[pdf_ob["split"] == "train"]
test_pd  = pdf_ob[pdf_ob["split"] == "test"]

print(f"\nOn-bottom train: {len(train_pd):,}  |  On-bottom test: {len(test_pd):,}")

# COMMAND ----------
# MAGIC %md ## Step 7: Train XGBoost

# COMMAND ----------

X_tr = train_pd[feat_cols].values
y_tr = train_pd[TARGET].values
X_te = test_pd[feat_cols].values
y_te = test_pd[TARGET].values

# 10 % of train rows held out for early stopping
rng     = np.random.default_rng(42)
val_idx = rng.choice(len(X_tr), size=int(len(X_tr) * 0.10), replace=False)
tr_idx  = np.setdiff1d(np.arange(len(X_tr)), val_idx)

model = XGBRegressor(**XGB_PARAMS, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
model.fit(
    X_tr[tr_idx], y_tr[tr_idx],
    eval_set=[(X_tr[val_idx], y_tr[val_idx])],
    verbose=50,
)
print(f"\nBest iteration: {model.best_iteration}")

# COMMAND ----------
# MAGIC %md ## Step 8: Evaluate

# COMMAND ----------

def metrics(y_true, y_pred, label):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"\n{label}")
    print(f"  RMSE : {rmse:.3f}")
    print(f"  MAE  : {mae:.3f}")
    print(f"  R²   : {r2:.4f}")
    return {"rmse": rmse, "mae": mae, "r2": r2}

train_pred = model.predict(X_tr)
test_pred  = model.predict(X_te)

metrics(y_tr, train_pred, "TRAIN (all wells, first 80 %, on-bottom only)")
metrics(y_te, test_pred,  "TEST  (all wells, last 20 %, on-bottom only)")

print("\nPer-well test performance (last 20 % of each well, on-bottom only):")
print(f"  {'Well':<55} {'N':>7}  {'RMSE':>7}  {'MAE':>7}  {'R²':>7}")
print("  " + "-" * 82)
for well, grp in test_pd.groupby("well_name"):
    if len(grp) < 10:
        print(f"  {well[:55]:<55} {len(grp):>7,}  (too few on-bottom rows)")
        continue
    pred_w = model.predict(grp[feat_cols].values)
    rmse_w = np.sqrt(mean_squared_error(grp[TARGET], pred_w))
    mae_w  = mean_absolute_error(grp[TARGET], pred_w)
    r2_w   = r2_score(grp[TARGET], pred_w)
    print(f"  {well[:55]:<55} {len(grp):>7,}  {rmse_w:>7.3f}  {mae_w:>7.3f}  {r2_w:>7.4f}")

# COMMAND ----------
# MAGIC %md ## Step 9: Feature importance

# COMMAND ----------

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

importances = (pd.Series(model.feature_importances_, index=feat_cols)
               .sort_values(ascending=True).tail(20))

fig, ax = plt.subplots(figsize=(8, 7))
importances.plot(kind="barh", ax=ax, color="#00EDED")
ax.set_title("Within-Well XGBoost — Feature Importance (top 20)", fontsize=13)
ax.set_xlabel("Importance score")
ax.set_facecolor("#0a1628"); fig.patch.set_facecolor("#0a1628")
ax.tick_params(colors="white"); ax.xaxis.label.set_color("white")
ax.title.set_color("white")
plt.tight_layout()
plt.savefig("/tmp/within_well_feature_importance.png", dpi=120, bbox_inches="tight")
plt.show()

print("Top 10 features:")
for feat, score in importances.tail(10)[::-1].items():
    print(f"  {feat:<30} {score:.5f}")

# COMMAND ----------
# MAGIC %md ## Step 10: Compute efficiency score
# MAGIC
# MAGIC Efficiency = actual ROP ÷ predicted ROP (on-bottom rows only).
# MAGIC - **> 1.0** : drilling faster than the model expected — favourable conditions or
# MAGIC   aggressive (but effective) parameters
# MAGIC - **< 0.7** : sustained underperformance — possible bit wear, poor parameters,
# MAGIC   wellbore issues, or unexpected hard formation

# COMMAND ----------

test_pd = test_pd.copy()
test_pd["pred_rop"]   = model.predict(X_te)
test_pd["efficiency"] = test_pd[TARGET] / test_pd["pred_rop"].clip(lower=0.5)

# Aggregate to 1-hour windows per well
test_pd["ts_hour"] = pd.to_datetime(test_pd["ts"]).dt.floor("1h")

hourly = (test_pd
          .groupby(["well_name", "ts_hour"])
          .agg(
              actual_rop   = (TARGET,        "mean"),
              pred_rop     = ("pred_rop",    "mean"),
              efficiency   = ("efficiency",  "mean"),
              wob_mean     = ("wob",         "mean"),
              rpm_mean     = ("rpm",         "mean"),
              dmea_mean    = ("dmea",        "mean"),
              n_rows       = ("rop",         "count"),
          )
          .reset_index())

print(f"Hourly windows (on-bottom): {len(hourly):,}")
print(f"\nEfficiency stats across all test windows:")
print(hourly["efficiency"].describe().round(3).to_string())

# COMMAND ----------
# MAGIC %md ## Step 11: Identify underperforming periods

# COMMAND ----------

UNDERPERF_THRESHOLD = 0.70   # efficiency below this = significant underperformance

underperf = (hourly[hourly["efficiency"] < UNDERPERF_THRESHOLD]
             .sort_values("efficiency")
             .head(30))

print(f"Hours with efficiency < {UNDERPERF_THRESHOLD:.0%}: "
      f"{(hourly['efficiency'] < UNDERPERF_THRESHOLD).sum()} of {len(hourly)} "
      f"({(hourly['efficiency'] < UNDERPERF_THRESHOLD).mean():.1%})")

print(f"\nTop 20 worst-performing 1-hour windows:")
print(f"  {'Well':<45} {'Time':>20}  {'Actual':>8}  {'Pred':>8}  "
      f"{'Eff%':>6}  {'WOB':>6}  {'RPM':>6}  {'Depth':>7}")
print("  " + "-" * 110)
for _, r in underperf.head(20).iterrows():
    print(f"  {str(r['well_name'])[:45]:<45} {str(r['ts_hour']):>20}  "
          f"{r['actual_rop']:>8.2f}  {r['pred_rop']:>8.2f}  "
          f"{r['efficiency']:>6.1%}  {r['wob_mean']:>6.1f}  "
          f"{r['rpm_mean']:>6.1f}  {r['dmea_mean']:>7.0f}")

# COMMAND ----------
# MAGIC %md ## Step 12: Per-well efficiency summary

# COMMAND ----------

well_summary = (hourly
                .groupby("well_name")
                .agg(
                    hours_analysed   = ("ts_hour",    "count"),
                    mean_efficiency  = ("efficiency", "mean"),
                    pct_underperf    = ("efficiency",
                                       lambda x: (x < UNDERPERF_THRESHOLD).mean()),
                    median_actual_rop = ("actual_rop", "median"),
                    median_pred_rop   = ("pred_rop",   "median"),
                )
                .reset_index()
                .sort_values("mean_efficiency"))

print("Per-well efficiency summary (test period, on-bottom hours only):")
print(f"  {'Well':<55} {'Hours':>6}  {'Mean Eff':>9}  {'% Under':>8}  "
      f"{'Actual ROP':>11}  {'Pred ROP':>9}")
print("  " + "-" * 105)
for _, r in well_summary.iterrows():
    print(f"  {str(r['well_name'])[:55]:<55} {r['hours_analysed']:>6}  "
          f"{r['mean_efficiency']:>9.1%}  {r['pct_underperf']:>8.1%}  "
          f"{r['median_actual_rop']:>11.2f}  {r['median_pred_rop']:>9.2f}")

# COMMAND ----------
# MAGIC %md ## Step 13: Write results table

# COMMAND ----------

retro_sp = spark.createDataFrame(
    test_pd[["well_name", "ts", TARGET, "pred_rop", "efficiency",
             "dmea", "wob", "rpm", "spp", "flow_in", "torque", "split"]]
)

spark.sql(f"DROP TABLE IF EXISTS {RETRO_TABLE}")
retro_sp.write.format("delta").mode("overwrite").saveAsTable(RETRO_TABLE)
print(f"Written: {RETRO_TABLE}  ({len(test_pd):,} on-bottom rows)")

print("\nDone.")
print(f"  On-bottom filter: WOB > {MIN_WOB} kN, RPM > {MIN_RPM}, ROP > {MIN_ROP} m/hr")
print(f"  All efficiency scores are computed on active drilling rows only.")
