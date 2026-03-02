# Databricks notebook source
# MAGIC %md
# MAGIC # Volve ROP — Notebook 05: Formation Tops
# MAGIC
# MAGIC Parses the Volve EDM XML to extract formation top depths for each well,
# MAGIC normalises the inconsistent naming conventions used across wells, and writes
# MAGIC a `workspace.volve_ml.formation_tops` interval table of the form:
# MAGIC
# MAGIC | Column | Description |
# MAGIC |--------|-------------|
# MAGIC | `well_name` | WITSML well name (matches drilling tables) |
# MAGIC | `md_from_m` | Top of formation interval (metres MD) |
# MAGIC | `md_to_m` | Base of formation interval (metres MD) |
# MAGIC | `formation` | Canonical formation name |
# MAGIC | `formation_id` | Ordinal integer (1 = Utsira … 13 = Skagerrak) |
# MAGIC
# MAGIC **Phase preference:** ACTUAL > PLAN > PROTOTYPE (best available per well).
# MAGIC **Depth units:** EDM stores MD in feet → converted to metres (÷ 3.28084).
# MAGIC **F-15S note:** WITSML drilling data for F-15S has `dmea` recorded in feet
# MAGIC (max ~24,939 vs ~3,500 m for other wells). The join in Notebook 03 corrects
# MAGIC for this before matching formation intervals.

# COMMAND ----------

import re
import pandas as pd
from collections import defaultdict
from pyspark.sql import functions as F

EDM_PATH   = ("/Volumes/equinor_asa_volve_data_village/public/volve"
              "/Well_technical_data/EDM.XML/Volve F.edm.xml")
TOPS_TABLE = "workspace.volve_ml.formation_tops"

# ── Canonical formation rules ────────────────────────────────────────
# Ordered list of (lowercase-substring, canonical_name).
# First match wins. Norwegian XML entities (&#xF8; = ø etc.) are
# decoded before matching.
FORMATION_RULES = [
    ("hugin",     "Hugin"),
    ("sleipner",  "Skagerrak"),
    ("skagerrak", "Skagerrak"),
    ("heather",   "Heather"),
    ("draupne",   "Draupne"),
    ("aasgard",   "Aasgard"),
    ("asgard",    "Aasgard"),
    ("roedby",    "Chalk"),   # Rødbyvej Fm
    ("rodby",     "Chalk"),
    ("hidra",     "Chalk"),
    ("blod",      "Chalk"),   # Blødøks Fm
    ("ekofisk",   "Chalk"),
    ("hod",       "Chalk"),
    ("tor ",      "Chalk"),   # Tor Fm — trailing space avoids partial hits
    ("tor_",      "Chalk"),
    ("tor fm",    "Chalk"),
    ("top tor",   "Chalk"),
    ("topp tor",  "Chalk"),
    (" tor",      "Chalk"),
    ("ty fm",     "Ty"),
    ("ty_",       "Ty"),
    ("top ty",    "Ty"),
    ("topp ty",   "Ty"),
    (" ty",       "Ty"),
    ("ty ",       "Ty"),
    ("lista",     "Lista"),
    ("sele",      "Sele"),
    ("balder",    "Balder"),
    ("grid",      "Grid"),
    ("skade",     "Skade"),
    ("utsira",    "Utsira"),
]

# Ordinal encoding: stratigraphic order shallow → deep
FORMATION_ORDINAL = {
    "Utsira":    1,
    "Skade":     2,
    "Grid":      3,
    "Balder":    4,
    "Sele":      5,
    "Lista":     6,
    "Ty":        7,
    "Chalk":     8,
    "Aasgard":   9,
    "Draupne":   10,
    "Heather":   11,
    "Hugin":     12,
    "Skagerrak": 13,
}

# Best phase gets highest rank; lower phases used as fallback
PHASE_RANK = {"ACTUAL": 3, "PLAN": 2, "PROTOTYPE": 1}


# COMMAND ----------
# MAGIC %md ## Step 1: Extract CD_WELL from EDM XML

# COMMAND ----------

wells_df = spark.sql(f"""
    SELECT
        regexp_extract(value, 'well_id="([^"]+)"', 1)          AS well_id,
        regexp_extract(value, 'well_common_name="([^"]+)"', 1) AS well_common_name
    FROM read_files('{EDM_PATH}', format => 'text')
    WHERE value LIKE '%<CD_WELL %'
""").toPandas()

well_id_to_common = dict(zip(wells_df["well_id"], wells_df["well_common_name"]))
print(f"EDM wells found: {len(wells_df)}")
print(wells_df[["well_id", "well_common_name"]].to_string(index=False))


# COMMAND ----------
# MAGIC %md ## Step 2: Extract CD_WELLBORE_FORMATION

# COMMAND ----------

tops_raw = spark.sql(f"""
    SELECT
        regexp_extract(value, 'well_id="([^"]+)"', 1)                         AS well_id,
        regexp_extract(value, 'formation_name="([^"]+)"', 1)                  AS formation_name_raw,
        TRY_CAST(regexp_extract(value, 'prognosed_md="([^"]+)"', 1) AS DOUBLE) AS md_ft,
        regexp_extract(value, 'phase="([^"]+)"', 1)                           AS phase
    FROM read_files('{EDM_PATH}', format => 'text')
    WHERE value LIKE '%CD_WELLBORE_FORMATION%'
      AND regexp_extract(value, 'prognosed_md="([^"]+)"', 1) != ''
""").toPandas()

tops_raw["phase_rank"] = tops_raw["phase"].map(PHASE_RANK).fillna(0).astype(int)
tops_raw = tops_raw[tops_raw["md_ft"].notna() & (tops_raw["md_ft"] > 0)]
print(f"Raw formation top records (with valid MD): {len(tops_raw)}")
print(tops_raw["phase"].value_counts().to_string())


# COMMAND ----------
# MAGIC %md ## Step 3: Canonicalise formation names

# COMMAND ----------

def canonical_formation(name_raw):
    if not name_raw:
        return None
    n = name_raw.lower()
    # Decode common XML/HTML entities for Norwegian characters
    n = (n.replace("&#xf8;", "o").replace("&#xc5;", "a")
          .replace("&#xe6;", "ae").replace("&#xf6;", "o")
          .replace("&oslash;", "o").replace("&aring;", "a"))
    for pattern, canonical in FORMATION_RULES:
        if pattern in n:
            return canonical
    return None  # not a recognised formation — skip


tops_raw["formation"] = tops_raw["formation_name_raw"].apply(canonical_formation)
tops_valid = tops_raw[tops_raw["formation"].notna()].copy()
print(f"Rows with recognised formation: {len(tops_valid)}")
print(tops_valid["formation"].value_counts().to_string())


# COMMAND ----------
# MAGIC %md ## Step 4: Deduplicate — best phase, then shallowest MD per (well, formation)

# COMMAND ----------

# Sort: highest phase_rank first, then shallowest md_ft first.
# groupby.first() then keeps the best-phase shallowest record.
tops_dedup = (
    tops_valid
    .sort_values(["well_id", "formation", "phase_rank", "md_ft"],
                 ascending=[True, True, False, True])
    .groupby(["well_id", "formation"], as_index=False)
    .first()
)

# Convert EDM feet → metres
tops_dedup["md_m"] = tops_dedup["md_ft"] / 3.28084
tops_dedup["well_common_name"] = tops_dedup["well_id"].map(well_id_to_common)
tops_dedup = tops_dedup[["well_common_name", "formation", "md_m"]].sort_values(
    ["well_common_name", "md_m"]
)
print(f"\nDeduplicated tops: {len(tops_dedup)}")
print("\nWells covered:")
print(tops_dedup["well_common_name"].value_counts().to_string())


# COMMAND ----------
# MAGIC %md ## Step 5: Map EDM common names to WITSML well names

# COMMAND ----------

witsml_wells = (spark.sql("SELECT DISTINCT well_name FROM workspace.volve_ml.drilling_raw")
                     .toPandas()["well_name"].tolist())

def edm_common_name(witsml_name):
    """Extract F-<n> from WITSML name; strip suffixes like A/B/S."""
    m = re.search(r"F-(\d+)", witsml_name)
    return f"F-{m.group(1)}" if m else None

witsml_to_common = {w: edm_common_name(w) for w in witsml_wells}
print("WITSML → EDM common name mapping:")
for wn, cn in sorted(witsml_to_common.items()):
    print(f"  {wn:<55}  →  {cn}")


# COMMAND ----------
# MAGIC %md ## Step 6: Build formation interval table
# MAGIC
# MAGIC For each WITSML well, sort tops by MD and create half-open intervals
# MAGIC [md_from_m, md_to_m). The last formation in the well extends to 99 999 m.
# MAGIC Wells with no EDM tops receive no rows (handled as `Unknown` in NB03).

# COMMAND ----------

# Compute field-wide median top per formation as fallback for missing wells
field_medians = (tops_dedup.groupby("formation")["md_m"]
                           .median()
                           .to_dict())
print("Field-wide median formation tops (m MD):")
for fm, md in sorted(field_medians.items(), key=lambda x: x[1]):
    print(f"  {fm:<12}  {md:>8.1f} m")


# COMMAND ----------

interval_records = []
no_tops_wells = []

for witsml_name in sorted(witsml_wells):
    common = witsml_to_common.get(witsml_name)
    if not common:
        no_tops_wells.append(witsml_name)
        continue

    w_tops = tops_dedup[tops_dedup["well_common_name"] == common].copy()

    if len(w_tops) == 0:
        no_tops_wells.append(witsml_name)
        continue

    # If multiple EDM wellbores for same common_name, take shallowest per formation
    w_tops = (w_tops.sort_values("md_m")
                    .groupby("formation", as_index=False)
                    .first()
                    .sort_values("md_m")
                    .reset_index(drop=True))

    for i, row in w_tops.iterrows():
        md_from = row["md_m"]
        md_to   = w_tops.iloc[i + 1]["md_m"] if i + 1 < len(w_tops) else 99_999.0
        fm      = row["formation"]
        interval_records.append({
            "well_name":    witsml_name,
            "md_from_m":   round(md_from, 3),
            "md_to_m":     round(md_to,   3),
            "formation":   fm,
            "formation_id": FORMATION_ORDINAL.get(fm, 0),
        })

intervals_pd = pd.DataFrame(interval_records)

print(f"\nInterval rows written: {len(intervals_pd)}")
print(f"Wells with no EDM tops (will use formation_id=0): {no_tops_wells}")
print("\nFormation coverage across wells:")
print(intervals_pd.groupby("formation")["well_name"].nunique()
                  .sort_values(ascending=False).to_string())


# COMMAND ----------
# MAGIC %md ## Step 7: Write formation_tops table

# COMMAND ----------

spark.sql(f"DROP TABLE IF EXISTS {TOPS_TABLE}")
(spark.createDataFrame(intervals_pd)
      .write.format("delta").mode("overwrite").saveAsTable(TOPS_TABLE))

print(f"Written: {TOPS_TABLE}  ({len(intervals_pd)} rows)")
spark.read.table(TOPS_TABLE).orderBy("well_name", "md_from_m").show(50, truncate=False)


# COMMAND ----------
# MAGIC %md ## Step 8: Verify coverage against drilling depths

# COMMAND ----------

# Check that formation intervals cover the actual drilling depth range per well
print("Formation coverage check (drilling data dmea vs formation intervals):")
print(f"  {'Well':<55} {'dmea_max':>10}  {'top_fm_tops':>12}  {'covered?':>10}")
print("  " + "-" * 92)

drill_depths = (spark.sql("""
    SELECT well_name,
           MAX(CASE WHEN dmea < 10000 THEN dmea ELSE dmea/3.28084 END) AS max_dmea_m
    FROM workspace.volve_ml.drilling_raw
    GROUP BY well_name
""").toPandas())

tops_per_well = intervals_pd.groupby("well_name")["md_from_m"].min()

for _, row in drill_depths.sort_values("well_name").iterrows():
    wn = row["well_name"]
    max_d = row["max_dmea_m"]
    min_top = tops_per_well.get(wn, float("nan"))
    covered = "YES" if wn in tops_per_well else "NO TOPS"
    print(f"  {wn:<55} {max_d:>10.0f}  {min_top:>12.0f}  {covered:>10}")
