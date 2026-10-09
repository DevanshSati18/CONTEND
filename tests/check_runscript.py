"""Smoke check for runscript.py: lock frozen, every session complete, repeat-aware statistics."""
import json
import os
import sys

root = sys.argv[1] if len(sys.argv) > 1 else "results/smoke_rs"
r = json.load(open(os.path.join(root, "derived", "results.json")))
lock = open(os.path.join(root, "locked_config.yaml")).read()
rep = os.path.exists(os.path.join(root, "REPORT.md"))
rq1 = r["rq1"]
print("RQ1 blocks: %d  sessions: %s  robust flags: %s  report: %s  frozen: %s" % (
    len(rq1), [x.get("n_sessions") for x in rq1], [x.get("robust_across_sessions") for x in rq1], rep, "frozen_at" in lock))
ok = (rep and "frozen_at" in lock and len(rq1) == 2 and all(x.get("n_sessions") == 2 for x in rq1)
      and all(len(x.get("ratio_by_session", [])) == 2 and x.get("ratio_ci_cluster") for x in rq1))
sys.exit(0 if ok else 1)
