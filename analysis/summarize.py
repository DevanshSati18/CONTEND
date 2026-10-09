"""Per-run summaries (the run is the experimental unit; frames are never pooled as independent).

  python analysis/summarize.py [--raw results/raw] [--out results/derived/run_summary.csv]

Closed-loop runs: trimmed-mean stage times, service time, P50/P95/P99.
Open-loop runs: deadline-miss rate (dropped frames count as misses), drop rate, max consecutive
misses, P95 tardiness, mean queue depth.
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from common import load_config, p  # noqa: E402

STAGES = ["t_pre_ms", "t_inf_ms", "t_nms_ms", "t_d2h_ms"]


def max_run(mask):
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def summarize_run(df, deadline_ms, trim):
    done = df[df["dropped"] == 0]
    row = {}
    for c in STAGES + ["service_ms", "latency_ms", "n_pre_nms", "n_post_nms"]:
        row[c] = float(stats.trim_mean(done[c].astype(float), trim)) if len(done) else np.nan
    row["other_ms"] = row["service_ms"] - sum(row[s] for s in STAGES)
    for q in (50, 95, 99):
        row["lat_p%d" % q] = float(np.percentile(done["latency_ms"], q)) if len(done) else np.nan
    df = df.sort_values("frame_idx")
    miss = (df["dropped"] == 1) | (df["latency_ms"].astype(float) > deadline_ms)
    tard = np.maximum(0.0, done["latency_ms"].astype(float) - deadline_ms)
    row.update(frames=len(df), miss_rate=float(miss.mean()), drop_rate=float((df["dropped"] == 1).mean()),
               max_consec_miss=int(max_run(miss.values)), tard_p95_ms=float(np.percentile(tard, 95)) if len(tard) else np.nan,
               queue_mean=float(df["queue_depth"].mean()), temp_mean_c=float(df["soc_temp_c"].astype(float).mean()),
               time_limit_hits=int(done["time_limit_hit"].sum()))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--include", nargs="*", default=None, help="blocks to include (default: all except G2/PILOT)")
    a = ap.parse_args()
    cfg = load_config()
    raw = a.raw or p(cfg["paths"]["raw"])
    deadline = 1000.0 / cfg["deadline"]["fps"]
    rows = []
    files = sorted(glob.glob(os.path.join(raw, "*", "*.csv")) + glob.glob(os.path.join(raw, "*", "*.csv.gz")))
    for fn in files:
        block = os.path.basename(os.path.dirname(fn))
        if a.include is not None and block not in a.include:
            continue
        if a.include is None and block in ("G2", "PILOT"):
            continue
        df = pd.read_csv(fn)
        if df.empty:
            continue
        r0 = df.iloc[0]
        row = dict(schema_version=r0.get("schema_version", ""), device_id=r0.get("device_id", cfg["device_id"]),
                   backend=r0.get("backend", ""), precision=r0.get("precision", ""), imgsz=r0.get("imgsz", ""),
                   fps_target=r0.get("fps_target", cfg["deadline"]["fps"]), run_id=r0["run_id"], block=block, cell=r0["cell"], input=r0["input"],
                   contention_run=r0["contention"], mode=r0["mode"], model=r0["model"])
        row.update(summarize_run(df, deadline, cfg["analysis"]["trim"]))
        rows.append(row)
    out = pd.DataFrame(rows)
    man = os.path.join(raw, "runs.csv")
    if os.path.exists(man) and len(out):
        m = pd.read_csv(man).drop_duplicates("run_id", keep="last")
        out = out.merge(m[["run_id", "order", "start_temp_c"]], on="run_id", how="left")
    if len(out):   # a block's contention type = the co-runner used in its loaded cells (C, D)
        out["contention"] = out.groupby("block")["contention_run"].transform(
            lambda s: s[s != "none"].iloc[0] if (s != "none").any() else "none")
    dst = a.out or p(os.path.join(cfg["paths"]["derived"], "run_summary.csv"))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    out.to_csv(dst, index=False)
    print("summarised %d runs -> %s" % (len(out), dst))


if __name__ == "__main__":
    main()
