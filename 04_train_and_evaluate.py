# Databricks notebook source
# MAGIC %pip install xgboost

# COMMAND ----------
# MAGIC %md
# MAGIC # Volve ROP — Notebook 04: Train & Evaluate
# MAGIC
# MAGIC Trains an XGBoost regression model on `workspace.volve_ml.train_features`
# MAGIC (12 wells) and evaluates on `workspace.volve_ml.test_features` (F-12, F-15S).
# MAGIC
# MAGIC **Evaluation design**
# MAGIC
# MAGIC The test wells are held out entirely — the model has never seen them.
# MAGIC This tests generalisation to *unseen wells*, which is the realistic production
# MAGIC scenario (predict ROP on a new well before drilling it).
# MAGIC
# MAGIC **Metrics**
# MAGIC
# MAGIC | Metric | Description |
# MAGIC |--------|-------------|
# MAGIC | RMSE | Root mean squared error (same units as ROP) |
# MAGIC | MAE | Mean absolute error — more robust to tail errors |
# MAGIC | R² | Fraction of variance explained |
# MAGIC
# MAGIC Results are logged to MLflow for experiment tracking.

# COMMAND ----------

TRAIN_FEAT  = "workspace.volve_ml.train_features"
TEST_FEAT   = "workspace.volve_ml.test_features"

TARGET = "rop"

# Columns that are metadata — excluded from the feature matrix
META_COLS = ["ts", "well_name", TARGET]

# XGBoost hyperparameters (sensible defaults for ~850K rows, ~34 features)
XGB_PARAMS = {
    "n_estimators":     500,
    "max_depth":        6,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 20,
    "objective":        "reg:squarederror",
    "tree_method":      "hist",       # fast histogram-based split finder
    "random_state":     42,
}
EARLY_STOPPING_ROUNDS = 30    # stop if val RMSE doesn't improve for 30 rounds
VAL_FRACTION          = 0.15  # fraction of train rows held out for early stopping

# COMMAND ----------
# MAGIC %md ## Step 1: Load feature tables

# COMMAND ----------

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df_train_sp = spark.read.table(TRAIN_FEAT)
df_test_sp  = spark.read.table(TEST_FEAT)

feature_cols = [c for c in df_train_sp.columns if c not in META_COLS]
print(f"Features ({len(feature_cols)}):")
for c in sorted(feature_cols):
    print(f"  {c}")

print(f"\nTrain rows: {df_train_sp.count():,}")
print(f"Test  rows: {df_test_sp.count():,}")

# COMMAND ----------
# MAGIC %md ## Step 2: Collect to pandas
# MAGIC
# MAGIC At ~850K × 34 features the dataset fits comfortably in driver memory.

# COMMAND ----------

train_pd = df_train_sp.toPandas()
test_pd  = df_test_sp.toPandas()

X_train_full = train_pd[feature_cols].values
y_train_full = train_pd[TARGET].values

X_test = test_pd[feature_cols].values
y_test = test_pd[TARGET].values

print(f"X_train: {X_train_full.shape}  y_train: {y_train_full.shape}")
print(f"X_test : {X_test.shape}   y_test : {y_test.shape}")
print(f"\ny_train stats:  mean={y_train_full.mean():.2f}  std={y_train_full.std():.2f}  "
      f"min={y_train_full.min():.2f}  max={y_train_full.max():.2f}")
print(f"y_test  stats:  mean={y_test.mean():.2f}  std={y_test.std():.2f}  "
      f"min={y_test.min():.2f}  max={y_test.max():.2f}")

# COMMAND ----------
# MAGIC %md ## Step 3: Train / validation split for early stopping
# MAGIC
# MAGIC A random 15% of training rows are held back as a validation set for
# MAGIC XGBoost early stopping. This prevents overfitting without touching the
# MAGIC held-out test wells.

# COMMAND ----------

rng = np.random.default_rng(42)
n_val = int(len(X_train_full) * VAL_FRACTION)
val_idx = rng.choice(len(X_train_full), size=n_val, replace=False)
train_idx = np.setdiff1d(np.arange(len(X_train_full)), val_idx)

X_train, y_train = X_train_full[train_idx], y_train_full[train_idx]
X_val,   y_val   = X_train_full[val_idx],   y_train_full[val_idx]

print(f"Train split : {X_train.shape[0]:,} rows")
print(f"Val split   : {X_val.shape[0]:,} rows  (early stopping only)")
print(f"Test (held) : {X_test.shape[0]:,} rows")

# COMMAND ----------
# MAGIC %md ## Step 4: Train XGBoost

# COMMAND ----------

model = XGBRegressor(
    **XGB_PARAMS,
    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=50,
)

best_iter = model.best_iteration
print(f"\nBest iteration: {best_iter}")

# ── Evaluate ──────────────────────────────────────────────────────────
def metrics(y_true, y_pred, label):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"\n{label}")
    print(f"  RMSE : {rmse:.3f}")
    print(f"  MAE  : {mae:.3f}")
    print(f"  R²   : {r2:.4f}")
    return {"rmse": rmse, "mae": mae, "r2": r2}

train_pred = model.predict(X_train_full)
test_pred  = model.predict(X_test)

train_m = metrics(y_train_full, train_pred, "TRAIN (all 12 wells)")
test_m  = metrics(y_test,       test_pred,  "TEST  (F-12, F-15S)")

# COMMAND ----------
# MAGIC %md ## Step 5: Feature importance

# COMMAND ----------

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

importances = pd.Series(
    model.feature_importances_,
    index=feature_cols
).sort_values(ascending=True).tail(25)

fig, ax = plt.subplots(figsize=(8, 8))
importances.plot(kind="barh", ax=ax, color="#00EDED")
ax.set_title("XGBoost Feature Importance (top 25)", fontsize=13)
ax.set_xlabel("Importance score")
ax.set_facecolor("#0a1628")
fig.patch.set_facecolor("#0a1628")
ax.tick_params(colors="white")
ax.xaxis.label.set_color("white")
ax.title.set_color("white")
plt.tight_layout()
plt.savefig("/tmp/feature_importance.png", dpi=120, bbox_inches="tight")
plt.show()
print("Feature importance chart saved to /tmp/feature_importance.png")

# Top 10 by importance
print("\nTop 10 features:")
for feat, score in importances.tail(10)[::-1].items():
    print(f"  {feat:<30} {score:.5f}")

# COMMAND ----------
# MAGIC %md ## Step 6: Residual analysis by test well

# COMMAND ----------

test_pd["pred"] = test_pred
test_pd["residual"] = test_pd[TARGET] - test_pd["pred"]

print("Per-well test performance:")
print(f"  {'Well':<45} {'N':>7}  {'RMSE':>7}  {'MAE':>7}  {'R²':>7}")
print("  " + "-" * 75)
for well, grp in test_pd.groupby("well_name"):
    rmse = np.sqrt(mean_squared_error(grp[TARGET], grp["pred"]))
    mae  = mean_absolute_error(grp[TARGET], grp["pred"])
    r2   = r2_score(grp[TARGET], grp["pred"])
    print(f"  {well[:45]:<45} {len(grp):>7,}  {rmse:>7.3f}  {mae:>7.3f}  {r2:>7.4f}")

print("\nResidual stats (test):")
print(f"  mean : {test_pd['residual'].mean():.3f}  (bias)")
print(f"  std  : {test_pd['residual'].std():.3f}")
print(f"  p5   : {test_pd['residual'].quantile(0.05):.3f}")
print(f"  p95  : {test_pd['residual'].quantile(0.95):.3f}")
