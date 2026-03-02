# Databricks notebook source
# MAGIC %md
# MAGIC # Volve ROP — Notebook 07: Parameter Benchmarking
# MAGIC
# MAGIC Answers the question: **for each formation, what drilling parameters were
# MAGIC associated with the highest ROP across the Volve campaign?**
# MAGIC
# MAGIC Uses the formation tops table built in Notebook 05. For each formation the
# MAGIC dataset is split into ROP quartiles; the top-quartile rows (fastest drilling)
# MAGIC are compared against the bottom-quartile rows to identify the parameter
# MAGIC combinations that characterise efficient vs inefficient drilling.
# MAGIC
# MAGIC **Outputs**
# MAGIC
# MAGIC | Table | Contents |
# MAGIC |-------|----------|
# MAGIC | `workspace.volve_ml.param_benchmarks` | Median parameters by formation × ROP quartile |
# MAGIC | `workspace.volve_ml.well_formation_perf` | Per-well efficiency by formation |
# MAGIC
# MAGIC **On-bottom filter:** only rows where WOB > 2 kN and ROP > 1 m/hr are used,
# MAGIC to exclude non-drilling states (tripping, reaming, washing).

# COMMAND ----------

from pyspark.sql import functions as F, Window
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ALL_WELLS_TABLES  = [
    "workspace.volve_ml.train_raw",
    "workspace.volve_ml.test_raw",
]
FORMATION_TOPS    = "workspace.volve_ml.formation_tops"
BENCHMARK_TABLE   = "workspace.volve_ml.param_benchmarks"
WELL_PERF_TABLE   = "workspace.volve_ml.well_formation_perf"

# Active-drilling filters (exclude trips, reaming, surface operations)
MIN_WOB = 2.0    # kN
MIN_ROP = 1.0    # m/hr

# Formation ordinal → name (matches NB05)
FORMATION_NAMES = {
    0:  "Unknown",
    1:  "Utsira",
    2:  "Skade",
    3:  "Grid",
    4:  "Balder",
    5:  "Sele",
    6:  "Lista",
    7:  "Ty",
    8:  "Chalk",
    9:  "Aasgard",
    10: "Draupne",
    11: "Heather",
    12: "Hugin",
    13: "Skagerrak",
}

# COMMAND ----------
# MAGIC %md ## Step 1: Load drilling data and formation tops

# COMMAND ----------

dfs = [spark.read.table(t) for t in ALL_WELLS_TABLES]
df  = dfs[0]
for d in dfs[1:]:
    df = df.union(d)

# Derive spm_total
df = (df
      .withColumn("spm_total",
                  F.coalesce(F.col("spm1"), F.lit(0.0)) +
                  F.coalesce(F.col("spm2"), F.lit(0.0)) +
                  F.coalesce(F.col("spm3"), F.lit(0.0)))
      .drop("spm1", "spm2", "spm3"))

print(f"Loaded {df.count():,} rows from {df.select('well_name').distinct().count()} wells")

# COMMAND ----------
# MAGIC %md ## Step 2: Join formation labels
# MAGIC
# MAGIC Detect per-well depth units (F-15S stores dmea in feet) and join the
# MAGIC formation_tops interval table on corrected measured depth.

# COMMAND ----------

ft_df = spark.read.table(FORMATION_TOPS)

# Detect depth units per well
well_units = (
    df.groupBy("well_name")
      .agg(F.max("dmea").alias("max_dmea"))
      .withColumn("scale", F.when(F.col("max_dmea") > 10000, 1.0/3.28084)
                            .otherwise(1.0))
      .select("well_name", "scale")
)

df = df.join(well_units, on="well_name", how="left")
df = df.withColumn("dmea_m", F.col("dmea") * F.col("scale")).drop("scale")

ft2 = ft_df.select(
    F.col("well_name").alias("_wn"),
    "md_from_m", "md_to_m", "formation", "formation_id"
)

df = (df.join(ft2,
              (df.well_name == ft2._wn) &
              (df.dmea_m   >= ft2.md_from_m) &
              (df.dmea_m    < ft2.md_to_m),
              how="left")
        .drop("_wn", "md_from_m", "md_to_m"))

df = df.fillna({"formation_id": 0, "formation": "Unknown"})

print("Formation join complete.")
df.groupBy("formation_id", "formation").count().orderBy("formation_id").show(20, truncate=False)

# COMMAND ----------
# MAGIC %md ## Step 3: Apply on-bottom filter

# COMMAND ----------

df_ob = df.filter(
    (F.col("wob") > MIN_WOB) &
    (F.col("rop") > MIN_ROP) &
    F.col("wob").isNotNull() &
    F.col("rpm").isNotNull() &
    F.col("rop").isNotNull()
)

n_ob = df_ob.count()
n_total = df.count()
print(f"On-bottom rows: {n_ob:,}  ({n_ob/n_total:.1%} of total)")
print("\nOn-bottom rows per formation:")
df_ob.groupBy("formation_id", "formation").count().orderBy("formation_id").show(20, truncate=False)

# COMMAND ----------
# MAGIC %md ## Step 4: ROP quartile benchmarking by formation
# MAGIC
# MAGIC For each formation, label each row Q1 (slowest) through Q4 (fastest) based
# MAGIC on ROP. Compare median drilling parameters across quartiles.

# COMMAND ----------

w_fm = Window.partitionBy("formation_id")

df_ob = df_ob.withColumn(
    "rop_quartile",
    F.ntile(4).over(w_fm.orderBy("rop"))
)

# Collect benchmark stats to pandas
PARAM_COLS = ["wob", "rpm", "spp", "flow_in", "torque", "spm_total", "mwti"]

agg_exprs = (
    [F.count("*").alias("n_rows"),
     F.round(F.percentile_approx("rop", 0.5), 2).alias("rop_median"),
     F.round(F.percentile_approx("rop", 0.25), 2).alias("rop_p25"),
     F.round(F.percentile_approx("rop", 0.75), 2).alias("rop_p75")] +
    [F.round(F.percentile_approx(c, 0.5), 3).alias(c) for c in PARAM_COLS]
)

benchmarks_sp = (df_ob
                 .groupBy("formation_id", "formation", "rop_quartile")
                 .agg(*agg_exprs)
                 .orderBy("formation_id", "rop_quartile"))

benchmarks_pd = benchmarks_sp.toPandas()

# COMMAND ----------
# MAGIC %md ## Step 5: Display benchmark comparison (Q1 vs Q4)

# COMMAND ----------

print("=" * 100)
print("DRILLING PARAMETER BENCHMARKS: Bottom Quartile (Q1) vs Top Quartile (Q4) by Formation")
print("=" * 100)

q1 = benchmarks_pd[benchmarks_pd["rop_quartile"] == 1].set_index("formation_id")
q4 = benchmarks_pd[benchmarks_pd["rop_quartile"] == 4].set_index("formation_id")

for fm_id in sorted(benchmarks_pd["formation_id"].unique()):
    if fm_id == 0:
        continue
    fm_name = FORMATION_NAMES.get(fm_id, f"ID_{fm_id}")
    if fm_id not in q1.index or fm_id not in q4.index:
        continue

    r1 = q1.loc[fm_id]
    r4 = q4.loc[fm_id]

    print(f"\n── {fm_name} (formation_id={fm_id}) ─────────────────")
    print(f"  {'Parameter':<14} {'Q1 (slow)':>12}  {'Q4 (fast)':>12}  {'Δ (fast-slow)':>14}")
    print(f"  {'-'*54}")
    print(f"  {'ROP (m/hr)':<14} {r1['rop_median']:>12.2f}  {r4['rop_median']:>12.2f}  "
          f"{r4['rop_median'] - r1['rop_median']:>+14.2f}")
    for p in PARAM_COLS:
        if p in r1 and p in r4 and pd.notna(r1[p]) and pd.notna(r4[p]):
            delta = r4[p] - r1[p]
            print(f"  {p:<14} {r1[p]:>12.3f}  {r4[p]:>12.3f}  {delta:>+14.3f}")
    print(f"  {'Rows':<14} {int(r1['n_rows']):>12,}  {int(r4['n_rows']):>12,}")

# COMMAND ----------
# MAGIC %md ## Step 6: Per-well, per-formation efficiency
# MAGIC
# MAGIC For each (well, formation) combination, compare median ROP against the
# MAGIC field-wide median for that formation. Ratio > 1.0 means this well drilled
# MAGIC that formation above the field average.

# COMMAND ----------

# Field-wide median ROP per formation
field_median = (df_ob
                .groupBy("formation_id", "formation")
                .agg(F.percentile_approx("rop", 0.5).alias("field_median_rop"),
                     F.count("*").alias("field_rows"))
                .withColumnRenamed("formation_id", "_fm_id")
                .drop("formation"))   # avoid duplicate when joining with well_fm

# Per-well per-formation median ROP
well_fm = (df_ob
           .groupBy("well_name", "formation_id", "formation")
           .agg(F.percentile_approx("rop", 0.5).alias("well_median_rop"),
                F.count("*").alias("well_rows"))
           .filter(F.col("well_rows") >= 50))   # minimum 50 rows for stability

well_fm = (well_fm
           .join(field_median,
                 well_fm.formation_id == field_median._fm_id,
                 how="left")
           .drop("_fm_id")
           .withColumn("rop_ratio",
                       F.round(F.col("well_median_rop") / F.col("field_median_rop"), 3))
           .orderBy("formation_id", F.desc("rop_ratio")))

print("Per-well ROP performance vs field median by formation")
print("(ratio > 1.0 = above field average for that formation)\n")
well_fm.show(60, truncate=False)

# COMMAND ----------
# MAGIC %md ## Step 7: WOB × RPM heat map for key formations
# MAGIC
# MAGIC For the most data-rich formations, bin WOB and RPM into deciles and show
# MAGIC the median ROP in each cell — a quick visual guide to the optimal drilling
# MAGIC window.

# COMMAND ----------

KEY_FORMATIONS = [7, 8, 12]   # Ty, Chalk, Hugin — adjust if sparse

df_ob_pd = df_ob.select(
    "formation_id", "formation", "wob", "rpm", "rop", "well_name"
).toPandas()

for fm_id in KEY_FORMATIONS:
    fm_name = FORMATION_NAMES.get(fm_id, f"ID_{fm_id}")
    subset  = df_ob_pd[df_ob_pd["formation_id"] == fm_id].copy()

    if len(subset) < 200:
        print(f"Skipping {fm_name} — only {len(subset)} rows")
        continue

    # Bin into 5 × 5 grid
    subset["wob_bin"] = pd.qcut(subset["wob"], q=5, duplicates="drop")
    subset["rpm_bin"] = pd.qcut(subset["rpm"], q=5, duplicates="drop")

    pivot = (subset
             .groupby(["wob_bin", "rpm_bin"], observed=True)["rop"]
             .median()
             .unstack("rpm_bin"))

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", origin="lower")
    plt.colorbar(im, ax=ax, label="Median ROP (m/hr)")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns], rotation=30, ha="right", fontsize=7)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(i) for i in pivot.index], fontsize=7)
    ax.set_xlabel("RPM bin")
    ax.set_ylabel("WOB bin")
    ax.set_title(f"{fm_name} — Median ROP by WOB × RPM", fontsize=12)

    ax.set_facecolor("#0a1628"); fig.patch.set_facecolor("#0a1628")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")
    ax.title.set_color("white")

    plt.tight_layout()
    plt.savefig(f"/tmp/heatmap_{fm_name.lower()}.png", dpi=120, bbox_inches="tight")
    plt.show()
    print(f"Saved heatmap for {fm_name} ({len(subset):,} rows)")

# COMMAND ----------
# MAGIC %md ## Step 8: Write output tables

# COMMAND ----------

spark.sql(f"DROP TABLE IF EXISTS {BENCHMARK_TABLE}")
benchmarks_sp.write.format("delta").mode("overwrite").saveAsTable(BENCHMARK_TABLE)
print(f"Written: {BENCHMARK_TABLE}  ({benchmarks_pd.shape[0]} rows)")

spark.sql(f"DROP TABLE IF EXISTS {WELL_PERF_TABLE}")
well_fm.write.format("delta").mode("overwrite").saveAsTable(WELL_PERF_TABLE)
print(f"Written: {WELL_PERF_TABLE}")

print("\nDone. Key tables:")
print(f"  {BENCHMARK_TABLE} — Q1/Q4 parameter comparison by formation")
print(f"  {WELL_PERF_TABLE} — per-well ROP efficiency vs field median")
