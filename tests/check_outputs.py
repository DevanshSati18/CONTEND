"""Smoke check: every expected output exists and the per-frame files follow schema 1.0."""
import glob
import gzip
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from harness import COLUMNS  # noqa: E402

root = sys.argv[1] if len(sys.argv) > 1 else "results/smoke"
need = ["raw/runs.csv", "derived/g1.json", "derived/g2.json", "derived/g3.json", "derived/choose_config.json",
        "derived/run_summary.csv", "derived/results.json", "derived/rq2_stage_sensitivity.csv",
        "derived/rq3_deadlines.csv", "figures/fig2_rq1_prediction.pdf", "figures/fig3_rq2_stage_sensitivity.pdf",
        "figures/fig4_rq3_trace.pdf"]
missing = [n for n in need if not os.path.exists(os.path.join(root, n))]
runs = glob.glob(os.path.join(root, "raw", "E*", "*.csv.gz"))
bad = []
for f in runs:
    cols = list(pd.read_csv(f, nrows=1).columns)
    if cols != COLUMNS:
        bad.append(f)
man = pd.read_csv(os.path.join(root, "raw", "runs.csv"))
r = json.load(open(os.path.join(root, "derived", "results.json")))
print("block runs: %d  manifest rows: %d  schema-mismatch files: %d  missing outputs: %s" %
      (len(runs), len(man), len(bad), missing or "none"))
print("RQ1 blocks analysed: %d  RQ3 rows: %d  device_id: %s" % (len(r["rq1"]), len(r["rq3"]), r.get("device_id")))
ok = not missing and not bad and len(runs) == 24 and len(r["rq1"]) == 2 and r.get("device_id") == "smoke"
sys.exit(0 if ok else 1)
