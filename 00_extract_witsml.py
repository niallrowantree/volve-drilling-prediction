# Databricks notebook source
# MAGIC %md
# MAGIC # Volve WITSML — Notebook 00: Extract & Inventory
# MAGIC
# MAGIC Extracts the 2.7 GB WITSML zip from the read-only Volve Data Village volume
# MAGIC to `workspace.volve_ml.witsml_raw`, then audits the channel inventory across
# MAGIC all wells to determine whether an ROP prediction model is viable.

# COMMAND ----------

ZIP_SRC  = "/Volumes/equinor_asa_volve_data_village/public/volve/WITSML Realtime drilling data/Volve - Real Time drilling data 13.05.2018.zip"
DEST_DIR = "/Volumes/workspace/volve_ml/witsml_raw"

# COMMAND ----------
# MAGIC %md ## Step 1: Extract the zip

# COMMAND ----------

import zipfile, os, time

print(f"Source : {ZIP_SRC}")
print(f"Dest   : {DEST_DIR}")
print(f"Size   : {os.path.getsize(ZIP_SRC) / 1e9:.2f} GB compressed")

t0 = time.time()
with zipfile.ZipFile(ZIP_SRC, "r") as zf:
    members = zf.namelist()
    print(f"Zip contains {len(members):,} entries")
    zf.extractall(DEST_DIR)

elapsed = time.time() - t0
print(f"\nExtraction complete in {elapsed:.0f}s")
print(f"Extracted to: {DEST_DIR}")

# COMMAND ----------
# MAGIC %md ## Step 2: Inventory top-level structure

# COMMAND ----------

import os

top_entries = sorted(os.listdir(DEST_DIR))
print(f"Top-level entries ({len(top_entries)}):")
for e in top_entries:
    print(f"  {e}")

# COMMAND ----------
# MAGIC %md ## Step 3: Walk all wells and catalogue available log objects

# COMMAND ----------

from collections import defaultdict

well_inventory = {}

for well_folder in sorted(os.listdir(DEST_DIR)):
    well_path = os.path.join(DEST_DIR, well_folder)
    if not os.path.isdir(well_path):
        continue

    objects = {}
    for obj_type in ["log", "mudLog", "trajectory", "bhaRun", "message", "tubular", "rig"]:
        obj_path = os.path.join(well_path, obj_type)
        if os.path.isdir(obj_path):
            # count sub-directories (each is one WITSML object instance)
            subs = [d for d in os.listdir(obj_path) if os.path.isdir(os.path.join(obj_path, d))]
            objects[obj_type] = len(subs)

    well_inventory[well_folder] = objects

print(f"{'Well':<55} {'log':>4} {'mudLog':>7} {'traj':>5} {'bhaRun':>7} {'msg':>4}")
print("-" * 85)
for well, objs in sorted(well_inventory.items()):
    print(f"{well:<55} {objs.get('log',0):>4} {objs.get('mudLog',0):>7} "
          f"{objs.get('trajectory',0):>5} {objs.get('bhaRun',0):>7} {objs.get('message',0):>4}")

# COMMAND ----------
# MAGIC %md ## Step 4: Inventory log channels (mnemonics) across all wells
# MAGIC
# MAGIC For each well, read the MetaFileInfo.txt inside log/1 (depth) and log/2 (time)
# MAGIC to catalogue what named log objects exist.

# COMMAND ----------

import re

log_inventory = {}   # well -> {depth: [log_names], time: [log_names]}

for well_folder in sorted(os.listdir(DEST_DIR)):
    well_path = os.path.join(DEST_DIR, well_folder)
    if not os.path.isdir(well_path):
        continue

    log_path = os.path.join(well_path, "log")
    if not os.path.isdir(log_path):
        continue

    well_logs = {}
    for index_type, label in [("1", "depth"), ("2", "time")]:
        meta = os.path.join(log_path, index_type, "MetaFileInfo.txt")
        if not os.path.exists(meta):
            continue
        with open(meta, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        # Each line format: <number> <name>
        names = []
        for line in content.strip().splitlines():
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                names.append(parts[1])
        well_logs[label] = names

    if well_logs:
        log_inventory[well_folder] = well_logs

print("Log object names by well:")
print("=" * 80)
for well, logs in sorted(log_inventory.items()):
    print(f"\n{well}")
    for idx_type, names in logs.items():
        print(f"  [{idx_type}] ({len(names)} objects): {', '.join(names[:10])}" +
              (f" ... +{len(names)-10} more" if len(names) > 10 else ""))

# COMMAND ----------
# MAGIC %md ## Step 5: Read channel mnemonics from a composite log XML
# MAGIC
# MAGIC Parse the logCurveInfo elements from the GenTime (time-indexed composite)
# MAGIC and DrillDepth (depth-indexed composite) log XMLs for the richest well.
# MAGIC This tells us exactly what sensor channels are recorded.

# COMMAND ----------

import xml.etree.ElementTree as ET

def get_log_channels(xml_path):
    """Extract mnemonic + unit pairs from a WITSML log XML file."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        ns = {"w": "http://www.witsml.org/schemas/1series"}

        channels = []
        # Try both namespaced and non-namespaced
        for lci in root.iter("{http://www.witsml.org/schemas/1series}logCurveInfo"):
            mnem = lci.find("{http://www.witsml.org/schemas/1series}mnemonic")
            unit = lci.find("{http://www.witsml.org/schemas/1series}unit")
            desc = lci.find("{http://www.witsml.org/schemas/1series}curveDescription")
            channels.append({
                "mnemonic": mnem.text if mnem is not None else "?",
                "unit":     unit.text if unit is not None else "",
                "desc":     desc.text if desc is not None else "",
            })

        if not channels:
            # Try without namespace
            for lci in root.iter("logCurveInfo"):
                mnem = lci.find("mnemonic")
                unit = lci.find("unit")
                channels.append({
                    "mnemonic": mnem.text if mnem is not None else "?",
                    "unit":     unit.text if unit is not None else "",
                    "desc":     "",
                })
        return channels
    except Exception as e:
        return [{"mnemonic": f"ERROR: {e}", "unit": "", "desc": ""}]


# Find wells that have the composite drilling log objects
TARGET_LOG_NAMES = ["GenTime", "DrillDepth", "CompositeLogData", "Composite", "Drill"]

print("Searching for composite drilling log XMLs...\n")

found_composites = []
for well_folder in sorted(os.listdir(DEST_DIR)):
    well_path = os.path.join(DEST_DIR, well_folder)
    if not os.path.isdir(well_path):
        continue

    for idx_type, label in [("1", "depth"), ("2", "time")]:
        log_base = os.path.join(well_path, "log", idx_type)
        if not os.path.isdir(log_base):
            continue

        for log_dir in os.listdir(log_base):
            for target in TARGET_LOG_NAMES:
                if target.lower() in log_dir.lower():
                    xml_candidates = [
                        f for f in os.listdir(os.path.join(log_base, log_dir))
                        if f.endswith(".xml")
                    ]
                    for xml_file in xml_candidates[:1]:
                        full_path = os.path.join(log_base, log_dir, xml_file)
                        found_composites.append({
                            "well": well_folder,
                            "index": label,
                            "log": log_dir,
                            "path": full_path,
                            "size_kb": os.path.getsize(full_path) // 1024,
                        })

print(f"Found {len(found_composites)} composite log XML files:\n")
for fc in found_composites[:20]:
    print(f"  {fc['well'][:40]:<40}  [{fc['index']}]  {fc['log']:<30}  {fc['size_kb']} KB")

# COMMAND ----------
# MAGIC %md ## Step 6: Parse channels from the largest composite logs

# COMMAND ----------

# Sort by size descending — largest files likely have the most data
found_composites.sort(key=lambda x: x["size_kb"], reverse=True)

all_mnemonics = defaultdict(set)   # mnemonic -> set of wells it appears in
channel_details = {}               # well+log -> list of channels

for fc in found_composites[:10]:   # check top 10 largest
    channels = get_log_channels(fc["path"])
    key = f"{fc['well'][:30]} | {fc['log'][:25]}"
    channel_details[key] = channels
    for ch in channels:
        all_mnemonics[ch["mnemonic"]].add(fc["well"])

print("Channel inventory from top composite log XMLs:")
print("=" * 90)
for key, channels in list(channel_details.items())[:5]:
    print(f"\n{key}")
    print(f"  {len(channels)} channels: " +
          ", ".join(f"{c['mnemonic']}({c['unit']})" for c in channels[:20]) +
          ("..." if len(channels) > 20 else ""))

print("\n\nChannels present in 3+ wells (robust across dataset):")
print("-" * 60)
for mnem, wells in sorted(all_mnemonics.items()):
    if len(wells) >= 3:
        print(f"  {mnem:<20}  {len(wells)} wells")

# COMMAND ----------
# MAGIC %md ## Step 7: Read actual data rows from a composite log
# MAGIC
# MAGIC Parse logData from the largest GenTime or DrillDepth XML to see sample rows
# MAGIC and confirm we have the ML-critical channels: ROP, WOB, RPM, TORQUE, SPP, FLOW.

# COMMAND ----------

def read_log_data_sample(xml_path, n_rows=20):
    """Read the logData mnemonics and first n data rows from a WITSML log XML."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Get column header
        header_el = root.find(".//{http://www.witsml.org/schemas/1series}mnemonicList")
        if header_el is None:
            header_el = root.find(".//mnemonicList")
        if header_el is None:
            return None, None

        header = [h.strip() for h in header_el.text.split(",")]

        # Get data rows
        rows = []
        for data_el in root.iter("{http://www.witsml.org/schemas/1series}data"):
            if data_el.text:
                rows.append([v.strip() for v in data_el.text.split(",")])
            if len(rows) >= n_rows:
                break

        if not rows:
            for data_el in root.iter("data"):
                if data_el.text:
                    rows.append([v.strip() for v in data_el.text.split(",")])
                if len(rows) >= n_rows:
                    break

        return header, rows
    except Exception as e:
        return None, str(e)


# Try the largest composite log we found
if found_composites:
    best = found_composites[0]
    print(f"Reading data from: {best['well']}")
    print(f"Log: {best['log']}  ({best['size_kb']} KB)\n")

    header, rows = read_log_data_sample(best["path"], n_rows=5)
    if header:
        print(f"Columns ({len(header)}): {', '.join(header[:30])}" +
              (f" ...+{len(header)-30}" if len(header) > 30 else ""))
        print(f"\nSample rows:")
        for r in (rows or [])[:5]:
            row_dict = dict(zip(header, r))
            # Print key drilling channels
            key_ch = ["DEPTH", "DBTM", "ROP", "WOB", "RPM", "TRQ", "TORQUE",
                      "SPP", "SPPA", "FLOW", "FLOWIN", "HKLD", "ECD", "BDENS"]
            for k in key_ch:
                if k in row_dict:
                    print(f"  {k:<12}: {row_dict[k]}")
            print()

# COMMAND ----------
# MAGIC %md ## Step 8: Summary — ML viability assessment

# COMMAND ----------

print("=" * 70)
print("WITSML ROP MODEL VIABILITY ASSESSMENT")
print("=" * 70)
print(f"\nWells with log objects:   {sum(1 for w in well_inventory.values() if w.get('log',0) > 0)}")
print(f"Wells with mudLog:        {sum(1 for w in well_inventory.values() if w.get('mudLog',0) > 0)}")
print(f"Wells with bhaRun:        {sum(1 for w in well_inventory.values() if w.get('bhaRun',0) > 0)}")
print(f"Composite log XMLs found: {len(found_composites)}")
print(f"\nKey ML channels confirmed across 3+ wells:")

rop_channels   = {m for m in all_mnemonics if "ROP" in m.upper()}
wob_channels   = {m for m in all_mnemonics if "WOB" in m.upper()}
rpm_channels   = {m for m in all_mnemonics if "RPM" in m.upper() or "ROTA" in m.upper()}
spp_channels   = {m for m in all_mnemonics if "SPP" in m.upper() or "SPPA" in m.upper()}
flow_channels  = {m for m in all_mnemonics if "FLOW" in m.upper() or "PUMPOUT" in m.upper()}
torque_channels= {m for m in all_mnemonics if "TRQ" in m.upper() or "TORQ" in m.upper()}
ecd_channels   = {m for m in all_mnemonics if "ECD" in m.upper()}

for label, ch_set in [("ROP", rop_channels), ("WOB", wob_channels), ("RPM", rpm_channels),
                       ("SPP", spp_channels), ("FLOW", flow_channels),
                       ("TORQUE", torque_channels), ("ECD", ecd_channels)]:
    if ch_set:
        print(f"  {label:<10}: {', '.join(sorted(ch_set)[:5])}")
    else:
        print(f"  {label:<10}: NOT FOUND in checked XMLs")
