# Volve Drilling Intelligence — ROP Prediction & Parameter Benchmarking

An end-to-end drilling analytics pipeline built on 15 million 30-second WITSML sensor records from Equinor's open [Volve North Sea dataset](https://www.equinor.com/energy/volve-data-sharing). The project runs entirely on Databricks (Unity Catalog, Delta Lake, serverless compute) using PySpark and XGBoost.

**[View the interactive analysis →](https://niallrowantree.com/volve-drilling.html)**

---

## What it does

Two complementary analyses answer different questions about drilling performance across the Volve campaign (14 wells, 2007–2009):

**Within-well efficiency scoring (NB06)**
Trains an XGBoost model on the first 80% of each well's drilling timeline and evaluates on the final 20%. Compares actual ROP against what the model expected given the applied parameters — the ratio is an *efficiency score*. Values below 1.0 indicate the crew was drilling slower than the physics of their own well suggested they should.

**Formation benchmarking (NB07)**
For each of 13 geological formations, splits all on-bottom rows into ROP quartiles and compares median drilling parameters in the fastest 25% vs slowest 25%. Answers the question: given this rock, what parameter combinations consistently produced the best ROP across the whole campaign?

---

## Key findings

- **Formation type dominates ROP** far more than drilling parameters. The spread across formations (1.5–102 m/hr median) dwarfs the spread within formations.
- **Skagerrak is WOB-dominant:** the fastest quartile used ~15 kN WOB at low RPM and achieved median ROP of 102 m/hr.
- **Ty is RPM-dominant:** ~160 RPM at light WOB (~4 kN) drove median ROP of 29 m/hr.
- **Deep reservoir counterintuition:** the fastest quartile in Aasgard and Draupne used *less* mud circulation (2,090 vs 3,540 L/min) than the slowest. High ROP in these formations reflects natural fractures — aggressive parameters don't help and may hurt.
- **F-14 is the standout well:** efficiency score of 1.42× — consistently drilling 42% faster than its own model predicted.
- **F-15S underperforms (0.36×):** a sidetrack with difficult hole conditions and a short drilling campaign that limits what the within-well model can learn.

---

## Pipeline

| Notebook | Purpose |
|----------|---------|
| `00_extract_witsml.py` | Extracts the 2.7 GB WITSML zip from Equinor's Data Village volume; audits channel inventory across all wells |
| `01_parse_witsml.py` | Walks all 23+ wells, normalises ~30 channel mnemonics to a canonical 14-channel schema, writes `drilling_raw` Delta table (15 M rows) |
| `02_labels_and_split.py` | Resamples to 30 s grid, drops null-ROP rows, performs train/test split by well (F-12 and F-15S held out entirely) |
| `03_feature_engineering.py` | Adds `spm_total`, joins formation labels, forward-fills sensor dropout, clips outliers to train percentiles, computes 5-min and 30-min rolling stats + t−1/t−5 lag features |
| `04_train_and_evaluate.py` | Trains cross-well XGBoost model, evaluates on held-out wells, logs to MLflow |
| `05_formation_tops.py` | Parses Volve EDM XML to extract formation top depths per well; normalises naming conventions; writes `formation_tops` interval table |
| `06_within_well_retrospective.py` | Per-well XGBoost with 80/20 time-split; on-bottom filter (WOB > 2 kN, RPM > 20, ROP > 1 m/hr); writes `retrospective` table with per-row efficiency scores |
| `07_parameter_benchmarking.py` | Quartile benchmarking by formation; compares Q1 vs Q4 median parameters; writes `param_benchmarks` and `well_formation_perf` tables |

---

## Delta tables produced

| Table | Description |
|-------|-------------|
| `workspace.volve_ml.drilling_raw` | Unified 15 M-row sensor table, all wells, canonical channels |
| `workspace.volve_ml.train_raw` | 12 training wells at 30 s resolution |
| `workspace.volve_ml.test_raw` | 2 held-out wells (F-12, F-15S) at 30 s resolution |
| `workspace.volve_ml.train_features` | Engineered feature matrix for training |
| `workspace.volve_ml.test_features` | Engineered feature matrix for evaluation |
| `workspace.volve_ml.formation_tops` | Formation depth intervals per well |
| `workspace.volve_ml.retrospective` | Per-row efficiency scores (on-bottom rows only) |
| `workspace.volve_ml.param_benchmarks` | Median parameters by formation × ROP quartile |
| `workspace.volve_ml.well_formation_perf` | Per-well efficiency by formation |

---

## Data quality issues surfaced

**Depth as a cross-well proxy:** including `dmea` as a feature in the cross-well model caused all F-10 test rows to be capped at 3,793 m — the 99th-percentile ceiling from training — because F-10 is shallower than most training wells. Removed from cross-well features; retained in the within-well model where it carries genuine geological meaning.

**F-15S units mismatch:** WITSML records `dmea` for F-15S in feet (max ~24,939) while all other wells use metres (~3,500). Formation tops join corrects for this before matching depth intervals.

**On-bottom filter essential:** without filtering to active drilling rows (WOB > 2 kN, RPM > 20, ROP > 1 m/hr), tripping, reaming, and surface operations inflate low-ROP counts and distort both the efficiency model and the quartile benchmarks.

---

## Stack

- **Platform:** Databricks (Unity Catalog, Delta Lake, serverless compute)
- **Processing:** PySpark, Python
- **Modelling:** XGBoost, MLflow
- **Data source:** [Equinor Volve Open Dataset](https://www.equinor.com/energy/volve-data-sharing) — WITSML real-time drilling logs + EDM formation tops
