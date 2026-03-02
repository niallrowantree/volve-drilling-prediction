# Databricks notebook source
# MAGIC %md
# MAGIC # Volve WITSML — Notebook 01: Parse to Delta
# MAGIC
# MAGIC Walks all 23+ Volve wells, identifies time-indexed drilling log objects,
# MAGIC normalises channel names to a canonical set, and writes a unified
# MAGIC `workspace.volve_ml.drilling_raw` Delta table.
# MAGIC
# MAGIC **Canonical channels extracted:**
# MAGIC | Name | Source mnemonics (priority order) |
# MAGIC |------|----------------------------------|
# MAGIC | rop | ROP, ROP5, ROP30s, ROP2M, GS_ROP, QROP |
# MAGIC | wob | SWOB, CWOB, GS_SWOB, DWOB_RT, DWOB_RAW_RT |
# MAGIC | rpm | RPM, DRPM, TRPM_RT, CRPM_RT, DRPM30s, Bit_RPM |
# MAGIC | spp | SPPA, SPP, GS_SPPA, SIG_SPP5s |
# MAGIC | flow_in | TFLO, FLOWIN, FLOW-TRPM |
# MAGIC | torque | TQA, ACTC, TORQUE_AVG, TRQ |
# MAGIC | ecd | ECD_CT_FPWD, ACTECDM, ECD_ARC_RT, ECD_ECO_RT |
# MAGIC | hkld | HKLD |
# MAGIC | dmea | DMEA, DBTM, DEPTH, DEPT |
# MAGIC | bpos | BPOS |
# MAGIC | spm1 | SPM1 |
# MAGIC | spm2 | SPM2 |
# MAGIC | spm3 | SPM3 |
# MAGIC | mwti | MWTI |

# COMMAND ----------

WELLS_ROOT   = "/Volumes/workspace/volve_ml/witsml_raw/sitecom14.statoil.no"
OUTPUT_TABLE = "workspace.volve_ml.drilling_raw"
NS           = "{http://www.witsml.org/schemas/1series}"

# Canonical name → source mnemonics in priority order.
# The first matching mnemonic found in a file's logCurveInfo sequence is used.
CHANNEL_MAP = {
    "rop":     ["ROP", "ROP5", "ROP30s", "ROP2M", "GS_ROP", "QROP"],
    "wob":     ["SWOB", "CWOB", "GS_SWOB", "DWOB_RT", "DWOB_RAW_RT"],
    "rpm":     ["RPM", "DRPM", "TRPM_RT", "CRPM_RT", "DRPM30s", "Bit_RPM"],
    "spp":     ["SPPA", "SPP", "GS_SPPA", "SIG_SPP5s"],
    "flow_in": ["TFLO", "FLOWIN", "FLOW-TRPM"],
    "torque":  ["TQA", "ACTC", "TORQUE_AVG", "TRQ"],
    "ecd":     ["ECD_CT_FPWD", "ACTECDM", "ECD_ARC_RT", "ECD_ECO_RT"],
    "hkld":    ["HKLD"],
    "dmea":    ["DMEA", "DBTM", "DEPTH", "DEPT"],
    "bpos":    ["BPOS"],
    "spm1":    ["SPM1"],
    "spm2":    ["SPM2"],
    "spm3":    ["SPM3"],
    "mwti":    ["MWTI"],
}

MIN_CANONICAL = 3    # log objects with fewer canonical channels are skipped
MIN_ROWS      = 100  # log objects with fewer data rows are skipped

# COMMAND ----------
# MAGIC %md ## Step 1: Discover all log objects and identify index type

# COMMAND ----------

import os
import xml.etree.ElementTree as ET
from collections import defaultdict


def read_log_header(xml_path):
    """
    Parse the first XML in a log chunk to get:
    - indexType string (e.g. "date time", "measured depth")
    - list of mnemonics from logCurveInfo
    Returns (idx_type, mnemonics) or ("error", []) on failure.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        idx_el = root.find(f".//{NS}indexType") or root.find(".//indexType")
        idx_type = idx_el.text.strip().lower() if idx_el is not None else "unknown"

        mnemonics = []
        for lci in root.iter(f"{NS}logCurveInfo"):
            m = lci.find(f"{NS}mnemonic")
            if m is not None:
                mnemonics.append(m.text.strip())
        if not mnemonics:
            for lci in root.iter("logCurveInfo"):
                m = lci.find("mnemonic")
                if m is not None:
                    mnemonics.append(m.text.strip())

        return idx_type, mnemonics
    except Exception as e:
        return f"error: {e}", []


def build_canonical_map(mnemonics):
    """Map canonical names to the highest-priority matching source mnemonic."""
    mnem_set = set(mnemonics)
    result = {}
    for canon, candidates in CHANNEL_MAP.items():
        for c in candidates:
            if c in mnem_set:
                result[canon] = c
                break
    return result


log_objects = []

for well_folder in sorted(os.listdir(WELLS_ROOT)):
    well_path = os.path.join(WELLS_ROOT, well_folder)
    if not os.path.isdir(well_path):
        continue

    for wb in sorted(d for d in os.listdir(well_path)
                     if os.path.isdir(os.path.join(well_path, d)) and not d.startswith('_')):
        log_base = os.path.join(well_path, wb, "log")
        if not os.path.isdir(log_base):
            continue

        for idx_folder in sorted(os.listdir(log_base)):
            idx_path = os.path.join(log_base, idx_folder)
            if not os.path.isdir(idx_path):
                continue

            for log_obj in sorted(os.listdir(idx_path)):
                log_obj_path = os.path.join(idx_path, log_obj)
                if not os.path.isdir(log_obj_path):
                    continue

                # Read header from first XML of first chunk
                first_xml = None
                total_size = 0
                for chunk in sorted(os.listdir(log_obj_path)):
                    chunk_path = os.path.join(log_obj_path, chunk)
                    if not os.path.isdir(chunk_path):
                        continue
                    xmls = sorted(f for f in os.listdir(chunk_path) if f.endswith('.xml'))
                    for x in xmls:
                        xp = os.path.join(chunk_path, x)
                        total_size += os.path.getsize(xp)
                        if first_xml is None:
                            first_xml = xp

                if first_xml is None:
                    continue

                idx_type, mnemonics = read_log_header(first_xml)
                canon_map = build_canonical_map(mnemonics)

                log_objects.append({
                    "well":         well_folder,
                    "wellbore":     wb,
                    "log_obj_path": log_obj_path,
                    "idx_type":     idx_type,
                    "mnemonics":    mnemonics,
                    "canon_map":    canon_map,
                    "n_canonical":  len(canon_map),
                    "total_mb":     total_size / 1e6,
                })

print(f"Found {len(log_objects)} log objects across all wells\n")
print(f"{'Well':<50} {'IndexType':<20} {'Canon':>6} {'MB':>8}")
print("-" * 90)
for lo in log_objects:
    print(f"{lo['well'][:50]:<50} {lo['idx_type']:<20} {lo['n_canonical']:>6} {lo['total_mb']:>8.1f}")

# COMMAND ----------
# MAGIC %md ## Step 2: Show which canonical channel each well will use

# COMMAND ----------

# SiteCom WITSML exports omit <indexType>, so we cannot filter by it.
# Instead include all log objects with enough canonical channels.
# Depth-indexed objects will produce 0 valid timestamps in parse_log_object
# and be dropped by the MIN_ROWS check.
candidate_objects = [lo for lo in log_objects if lo["n_canonical"] >= MIN_CANONICAL]

print(f"Log objects with {MIN_CANONICAL}+ canonical channels: {len(candidate_objects)}\n")

print(f"{'Well':<50} {'Canon':>6}  Channel mapping")
print("-" * 110)
for lo in candidate_objects:
    mapping = "  ".join(f"{k}={v}" for k, v in sorted(lo["canon_map"].items()))
    print(f"{lo['well'][:50]:<50} {lo['n_canonical']:>6}  {mapping}")

# COMMAND ----------
# MAGIC %md ## Step 2b: Diagnostic — inspect XML structure of first candidate

# COMMAND ----------

# Run this cell once to confirm what's inside the log object XMLs.
# It tells us whether mnemonicList is present and what data rows look like.
_lo = candidate_objects[0]
print(f"Well        : {_lo['well']}")
print(f"log_obj_path: {_lo['log_obj_path']}")
print(f"canon_map   : {_lo['canon_map']}")

_chunks = sorted(c for c in os.listdir(_lo['log_obj_path'])
                 if os.path.isdir(os.path.join(_lo['log_obj_path'], c)))
print(f"\nChunk dirs  : {_chunks[:8]}")

for _chunk in _chunks[:2]:
    _chunk_path = os.path.join(_lo['log_obj_path'], _chunk)
    _xmls = sorted(f for f in os.listdir(_chunk_path) if f.endswith('.xml'))
    print(f"\nChunk '{_chunk}': {len(_xmls)} XML(s) → {_xmls[:6]}")
    for _xname in _xmls[:4]:
        _xp = os.path.join(_chunk_path, _xname)
        _size_kb = os.path.getsize(_xp) / 1e3
        try:
            _root = ET.parse(_xp).getroot()
            _mnem = _root.find(f".//{NS}mnemonicList") or _root.find(".//mnemonicList")
            _data = list(_root.iter(f"{NS}data")) or list(_root.iter("data"))
            _lci  = list(_root.iter(f"{NS}logCurveInfo")) or list(_root.iter("logCurveInfo"))
            print(f"  {_xname} ({_size_kb:.0f} KB): "
                  f"logCurveInfo={len(_lci)}  "
                  f"mnemonicList={'FOUND' if _mnem is not None else 'MISSING'}  "
                  f"data_rows={len(_data)}")
            if _mnem is not None and _mnem.text:
                print(f"    mnemonicList → {_mnem.text[:120]}")
            if _data and _data[0].text:
                print(f"    first data   → {_data[0].text[:120]}")
        except Exception as _e:
            print(f"  {_xname}: ERROR {_e}")

# COMMAND ----------
# MAGIC %md ## Step 3: Parse log XML chunks and write to Delta
# MAGIC
# MAGIC Processes one log object at a time. Each log object folder contains
# MAGIC numbered chunk directories, each with 000XX.xml data files.
# MAGIC These are concatenated in order and written to the Delta table.

# COMMAND ----------

import pandas as pd
import time as time_mod


def parse_log_object(log_obj_path, canon_map):
    """
    Parse all chunk XMLs in a log object folder.

    SiteCom exports never write mnemonicList. Column order is defined by the
    logCurveInfo sequence, which can change per-file (e.g. 50 → 47 channels).
    We re-read logCurveInfo from every file and cache only as a last resort.

    IMPORTANT: uses explicit `is None` checks throughout — never `or` on
    ElementTree Element objects (leaf elements are falsy even when found).

    Returns a pandas DataFrame with ts + canonical channel columns,
    or None if no usable data found.
    """
    all_rows = []
    cached_file_col = None  # last-resort fallback if a file has no logCurveInfo

    for chunk in sorted(os.listdir(log_obj_path)):
        chunk_path = os.path.join(log_obj_path, chunk)
        if not os.path.isdir(chunk_path):
            continue

        for xml_file in sorted(f for f in os.listdir(chunk_path) if f.endswith('.xml')):
            xml_path = os.path.join(chunk_path, xml_file)
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()

                file_col = None

                # 1. Try mnemonicList (standard WITSML logData — absent in SiteCom)
                mnem_el = root.find(f".//{NS}mnemonicList")
                if mnem_el is None:
                    mnem_el = root.find(".//mnemonicList")
                if mnem_el is not None and mnem_el.text:
                    file_mnems = [m.strip() for m in mnem_el.text.split(",")]
                    col = {canon: file_mnems.index(src)
                           for canon, src in canon_map.items()
                           if src in file_mnems}
                    if col:
                        file_col = col

                # 2. Fall back to logCurveInfo ordering (SiteCom always provides this).
                #    Re-read every file so varying column counts are handled correctly.
                if file_col is None:
                    lcis = list(root.iter(f"{NS}logCurveInfo"))
                    if not lcis:
                        lcis = list(root.iter("logCurveInfo"))
                    if lcis:
                        lci_mnems = []
                        for lci in lcis:
                            m = lci.find(f"{NS}mnemonic")
                            if m is None:
                                m = lci.find("mnemonic")
                            if m is not None and m.text:
                                lci_mnems.append(m.text.strip())
                        col = {canon: lci_mnems.index(src)
                               for canon, src in canon_map.items()
                               if src in lci_mnems}
                        if col:
                            file_col = col

                # 3. Last resort: reuse previous file's mapping
                if file_col is None:
                    file_col = cached_file_col
                else:
                    cached_file_col = file_col

                if not file_col:
                    continue

                # Parse data rows
                data_els = list(root.iter(f"{NS}data"))
                if not data_els:
                    data_els = list(root.iter("data"))
                for data_el in data_els:
                    if not data_el.text:
                        continue
                    vals = [v.strip() for v in data_el.text.split(",")]
                    if not vals:
                        continue

                    row = {"ts_raw": vals[0]}
                    for canon, idx in file_col.items():
                        if idx < len(vals) and vals[idx] not in ("", "null", "NULL"):
                            try:
                                row[canon] = float(vals[idx])
                            except ValueError:
                                row[canon] = None
                        else:
                            row[canon] = None
                    all_rows.append(row)

            except Exception as e:
                print(f"    Warning: {os.path.basename(xml_path)}: {e}")

    if not all_rows:
        return None

    pdf = pd.DataFrame(all_rows)
    pdf["ts"] = pd.to_datetime(pdf["ts_raw"], errors="coerce", utc=True)
    pdf = pdf.dropna(subset=["ts"]).drop(columns=["ts_raw"])

    # Keep only the Volve active drilling window (2007–2009).
    # Depth-indexed log objects store depth in metres (e.g. 2007.87) in column 0.
    # Integer depths like "2007", "2008", "2009" are parsed by pd.to_datetime as
    # the year 2007/2008/2009 → exactly midnight on 1 Jan of that year.  Those
    # sentinel rows are removed by the month-boundary exclusion below.
    # Float depths (e.g. "2007.87") → NaT → already dropped by the dropna above.
    pdf = pdf[(pdf["ts"] >= "2007-01-01") & (pdf["ts"] < "2010-01-01")]

    # Drop the surviving sentinel rows: depth integers parsed as round year-dates
    # land exactly on a month boundary (day=1, time=00:00:00.000).
    # Real drilling timestamps always have a non-zero time component.
    is_month_boundary = (
        (pdf["ts"].dt.day == 1) &
        (pdf["ts"].dt.hour == 0) &
        (pdf["ts"].dt.minute == 0) &
        (pdf["ts"].dt.second == 0) &
        (pdf["ts"].dt.microsecond == 0)
    )
    pdf = pdf[~is_month_boundary]

    pdf = pdf.sort_values("ts").reset_index(drop=True)

    # Ensure all canonical columns exist (fill missing with NaN)
    for canon in CHANNEL_MAP:
        if canon not in pdf.columns:
            pdf[canon] = float("nan")

    return pdf


# Drop and recreate the table so the schema matches the new CHANNEL_MAP.
# saveAsTable will infer the schema correctly from the first write.
spark.sql(f"DROP TABLE IF EXISTS {OUTPUT_TABLE}")

total_rows  = 0
wells_done  = set()
t0          = time_mod.time()

for i, lo in enumerate(candidate_objects):
    label = f"[{i+1}/{len(candidate_objects)}] {lo['well'][:55]}"
    print(f"\n{label}")

    pdf = parse_log_object(lo["log_obj_path"], lo["canon_map"])

    if pdf is None:
        print(f"  Skipped (no data parsed — mnemonicList not found in any XML)")
        continue
    if len(pdf) < MIN_ROWS:
        print(f"  Skipped ({len(pdf)} rows — depth-indexed or too small)")
        continue

    pdf["well_name"] = lo["well"]

    sdf = spark.createDataFrame(pdf)
    sdf.write.format("delta").mode("append").saveAsTable(OUTPUT_TABLE)

    total_rows += len(pdf)
    wells_done.add(lo["well"])
    elapsed = time_mod.time() - t0
    print(f"  {len(pdf):>8,} rows written  "
          f"(running total: {total_rows:,}  |  {elapsed:.0f}s elapsed)")

print(f"\n{'='*60}")
print(f"Parsing complete.")
print(f"Rows written : {total_rows:,}")
print(f"Wells written: {len(wells_done)}")
print(f"Output table : {OUTPUT_TABLE}")

# COMMAND ----------
# MAGIC %md ## Step 5: Post-parse cleanup and verify
# MAGIC
# MAGIC Although `parse_log_object` already applies both filters at parse time,
# MAGIC this step re-applies them as a safety net and verifies the final table.
# MAGIC Two filters are used (matching the logic inside `parse_log_object`):
# MAGIC
# MAGIC 1. **Drilling window**: keep only 2007–2009 (Volve active period).
# MAGIC 2. **Month-boundary exclusion**: depth integers like "2007", "2008", "2009"
# MAGIC    are parsed by `pd.to_datetime` as the year → exactly midnight on 1 Jan.
# MAGIC    Real timestamps always have a non-zero time component, so rows landing
# MAGIC    exactly on a month boundary are sentinel/garbage and are dropped.

# COMMAND ----------

df_raw   = spark.read.table(OUTPUT_TABLE)
df_clean = df_raw.filter(
    "ts >= '2007-01-01' AND ts < '2010-01-01'"
    " AND NOT (day(ts) = 1 AND hour(ts) = 0 AND minute(ts) = 0"
    "          AND second(ts) = 0 AND date_trunc('MONTH', ts) = ts)"
)

n_before = df_raw.count()
n_after  = df_clean.count()
print(f"Rows before filter : {n_before:,}")
print(f"Rows after  filter : {n_after:,}")
print(f"Rows removed       : {n_before - n_after:,}")

df_clean.write.format("delta").mode("overwrite").saveAsTable(OUTPUT_TABLE)
print(f"\nTable overwritten. {n_after:,} clean rows in {OUTPUT_TABLE}")

# COMMAND ----------
# MAGIC %md ## Step 4: Verify Delta table

# COMMAND ----------

from pyspark.sql.functions import col, count, when, min as spark_min, max as spark_max

df = spark.read.table(OUTPUT_TABLE)
n  = df.count()

print(f"Total rows: {n:,}\n")

print("Rows and date range per well:")
(df.groupBy("well_name")
   .agg(count("*").alias("rows"),
        spark_min("ts").alias("first_ts"),
        spark_max("ts").alias("last_ts"))
   .orderBy("well_name")
   .show(30, truncate=False))

print("\nNull % per canonical channel:")
ALL_CHANNELS = list(CHANNEL_MAP.keys())
null_exprs = [
    (count(when(col(c).isNull(), c)) * 100.0 / count("*")).alias(c)
    for c in ALL_CHANNELS
]
df.select(count("*").alias("total_rows"), *null_exprs).show(truncate=False)
