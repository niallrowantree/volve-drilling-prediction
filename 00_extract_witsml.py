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
print(f"Top-level entries under DEST_DIR ({len(top_entries)}):")
for e in top_entries:
    full = os.path.join(DEST_DIR, e)
    kind = "dir" if os.path.isdir(full) else "file"
    print(f"  [{kind}]  {e}")

# Determine actual wells root.
# If the zip extracted to a single subdirectory, dive into it; otherwise use DEST_DIR.
top_dirs = [e for e in top_entries if os.path.isdir(os.path.join(DEST_DIR, e))]
if len(top_dirs) == 1:
    WELLS_ROOT = os.path.join(DEST_DIR, top_dirs[0])
    print(f"\nZip extracted into subdirectory → using WELLS_ROOT: {WELLS_ROOT}")
else:
    WELLS_ROOT = DEST_DIR
    print(f"\nUsing WELLS_ROOT = DEST_DIR ({len(top_dirs)} top-level dirs)")

# Show what WELLS_ROOT contains
wells_root_entries = sorted(os.listdir(WELLS_ROOT))
print(f"\nWELLS_ROOT contains {len(wells_root_entries)} entries (first 10):")
for e in wells_root_entries[:10]:
    print(f"  {e}")

# COMMAND ----------
# MAGIC %md ## Step 3: Walk all wells and catalogue available log objects
# MAGIC
# MAGIC Actual structure: WellName / {wellbore} / {objectType} / ...
# MAGIC The wellbore folder ("1") sits between the well folder and the WITSML object types.

# COMMAND ----------

from collections import defaultdict

well_inventory = {}

for well_folder in sorted(os.listdir(WELLS_ROOT)):
    well_path = os.path.join(WELLS_ROOT, well_folder)
    if not os.path.isdir(well_path):
        continue

    objects = defaultdict(int)
    # Wellbore folders are numeric (e.g. "1"), skip _wellInfo etc.
    wellbore_dirs = [d for d in os.listdir(well_path)
                     if os.path.isdir(os.path.join(well_path, d)) and not d.startswith('_')]
    for wb in wellbore_dirs:
        wb_path = os.path.join(well_path, wb)
        for obj_type in ["log", "mudLog", "trajectory", "bhaRun", "message"]:
            if os.path.isdir(os.path.join(wb_path, obj_type)):
                objects[obj_type] += 1

    well_inventory[well_folder] = dict(objects)

print(f"{'Well':<55} {'log':>4} {'mudLog':>7} {'traj':>5} {'bhaRun':>7}")
print("-" * 78)
for well, objs in sorted(well_inventory.items()):
    print(f"{well:<55} {objs.get('log',0):>4} {objs.get('mudLog',0):>7} "
          f"{objs.get('trajectory',0):>5} {objs.get('bhaRun',0):>7}")

# COMMAND ----------
# MAGIC %md ## Step 4: Find all data XML files across all wells
# MAGIC
# MAGIC Data files follow the pattern 00001.xml, 00002.xml etc. and are the
# MAGIC largest files in the hierarchy (4–6 MB each). Walk all wells to find them.

# COMMAND ----------

print("Scanning for data XML files (this may take 30–60s)...\n")
data_xmls = []  # list of (well_name, full_path, size_bytes)

for well_folder in sorted(os.listdir(WELLS_ROOT)):
    well_path = os.path.join(WELLS_ROOT, well_folder)
    if not os.path.isdir(well_path):
        continue
    for root, dirs, files in os.walk(well_path):
        dirs[:] = [d for d in dirs if not d.startswith('_')]
        for f in sorted(files):
            if f.endswith('.xml') and f[0].isdigit():
                full_path = os.path.join(root, f)
                size = os.path.getsize(full_path)
                data_xmls.append((well_folder, full_path, size))

data_xmls.sort(key=lambda x: -x[2])
print(f"Found {len(data_xmls)} data XML files across all wells")
print(f"\nLargest 15:")
for well, path, size in data_xmls[:15]:
    rel = os.path.relpath(path, os.path.join(WELLS_ROOT, well))
    print(f"  {well[:35]:<35}  {size/1e6:6.1f} MB  {rel}")

# COMMAND ----------
# MAGIC %md ## Step 5: Parse channel mnemonics from data XMLs (one per well)

# COMMAND ----------

import xml.etree.ElementTree as ET

def get_log_channels(xml_path):
    """Extract mnemonic + unit pairs from a WITSML log XML file."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        channels = []
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


all_mnemonics = defaultdict(set)   # mnemonic -> set of wells it appears in
channel_details = {}               # key -> list of channels

# Sample one XML per well (the largest for that well) for diversity
seen_wells = set()
sample_xmls = []
for well, path, size in data_xmls:
    if well not in seen_wells:
        sample_xmls.append((well, path, size))
        seen_wells.add(well)
    if len(sample_xmls) >= 15:
        break

print(f"Parsing channels from {len(sample_xmls)} XMLs (largest per well):\n")
for well, path, size in sample_xmls:
    channels = get_log_channels(path)
    rel = os.path.relpath(path, os.path.join(WELLS_ROOT, well))
    key = f"{well[:30]} | {rel}"
    channel_details[key] = channels
    for ch in channels:
        all_mnemonics[ch["mnemonic"]].add(well)
    print(f"  {well[:45]:<45}  {len(channels)} channels")

print("\n\nChannels present in 3+ wells:")
print("-" * 60)
for mnem, wells in sorted(all_mnemonics.items()):
    if len(wells) >= 3:
        print(f"  {mnem:<25}  {len(wells)} wells")

# COMMAND ----------
# MAGIC %md ## Step 6: Sample data rows from the largest XML

# COMMAND ----------

def read_log_data_sample(xml_path, n_rows=5):
    """Read the logData mnemonics and first n data rows from a WITSML log XML."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        header_el = root.find(".//{http://www.witsml.org/schemas/1series}mnemonicList")
        if header_el is None:
            header_el = root.find(".//mnemonicList")
        if header_el is None:
            return None, None
        header = [h.strip() for h in header_el.text.split(",")]
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


best_well, best_path, best_size = data_xmls[0]
print(f"Reading data from: {best_well}")
print(f"File: {os.path.relpath(best_path, os.path.join(WELLS_ROOT, best_well))}  ({best_size/1e6:.1f} MB)\n")

header, rows = read_log_data_sample(best_path, n_rows=3)
if header:
    print(f"Columns ({len(header)}): {', '.join(header[:30])}" +
          (f" ...+{len(header)-30}" if len(header) > 30 else ""))
    KEY_CH = ["DEPTH", "DBTM", "ROP", "WOB", "RPM", "TRQ", "TORQUE",
              "SPP", "SPPA", "FLOW", "FLOWIN", "HKLD", "ECD", "BDENS", "DMEA", "TIME"]
    print("\nKey drilling channels in first row:")
    if rows:
        row_dict = dict(zip(header, rows[0]))
        for k in KEY_CH:
            if k in row_dict:
                print(f"  {k:<12}: {row_dict[k]}")
else:
    print(f"Could not parse header. rows={rows}")

# COMMAND ----------
# MAGIC %md ## Step 7: Summary — ML viability assessment

# COMMAND ----------

print("=" * 70)
print("WITSML ROP MODEL VIABILITY ASSESSMENT")
print("=" * 70)
print(f"\nWells with log objects:   {sum(1 for w in well_inventory.values() if w.get('log',0) > 0)}")
print(f"Wells with mudLog:        {sum(1 for w in well_inventory.values() if w.get('mudLog',0) > 0)}")
print(f"Wells with bhaRun:        {sum(1 for w in well_inventory.values() if w.get('bhaRun',0) > 0)}")
print(f"Data XML files found:     {len(data_xmls)}")
print(f"Total data volume:        {sum(s for _,_,s in data_xmls)/1e9:.2f} GB")
print(f"\nKey ML channels confirmed across 3+ wells:")

for label, keywords in [("ROP",    ["ROP"]),
                         ("WOB",    ["WOB"]),
                         ("RPM",    ["RPM", "ROTA"]),
                         ("SPP",    ["SPP", "SPPA"]),
                         ("FLOW",   ["FLOW", "PUMPOUT"]),
                         ("TORQUE", ["TRQ", "TORQ"]),
                         ("ECD",    ["ECD"])]:
    found = {m for m in all_mnemonics for kw in keywords if kw in m.upper()}
    if found:
        print(f"  {label:<10}: {', '.join(sorted(found)[:5])}")
    else:
        print(f"  {label:<10}: NOT FOUND")
