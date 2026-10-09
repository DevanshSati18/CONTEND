"""Synthetic per-frame logs with a KNOWN answer, to prove the analysis detects what it should.

  python analysis/simulate.py --scenario h0 --out /tmp/sim_h0     # proportional slowdown only
  python analysis/simulate.py --scenario h1 --out /tmp/sim_h1     # adversarial NMS 40% more sensitive
then
  python analysis/summarize.py --raw /tmp/sim_h1/raw --out /tmp/sim_h1/summary.csv
  python analysis/analyze.py --summary /tmp/sim_h1/summary.csv --outdir /tmp/sim_h1 --figdir /tmp/sim_h1/fig
Expected decisions: h0 -> "H0: predictable", h1 -> "H1: amplified".
"""
import argparse
import os

import numpy as np
import pandas as pd

BASE = {"t_pre_ms": 2.0, "t_inf_ms": 14.0, "t_nms_ms": 1.5, "t_d2h_ms": 0.1}
ADV_NMS = 9.0                      # adversarial NMS time without contention
K = {"t_pre_ms": 1.25, "t_inf_ms": 1.10, "t_nms_ms": 1.30, "t_d2h_ms": 1.0}   # clean contention slowdown


def frames(cell, scen, rng, n, run_jitter):
    adv, load = cell in "BD", cell in "CD"
    rows = {}
    for s, v in BASE.items():
        v = ADV_NMS if (adv and s == "t_nms_ms") else v
        k = K[s] if load else 1.0
        if scen == "h1" and adv and load and s == "t_nms_ms":
            k *= 1.40                                   # the effect the study looks for
        rows[s] = v * k * run_jitter * rng.lognormal(0, 0.05, n)
    df = pd.DataFrame(rows)
    df["service_ms"] = df.sum(axis=1) + 0.2
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["h0", "h1"], default="h1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--frames", type=int, default=800)
    a = ap.parse_args()
    rng = np.random.RandomState(7)
    for block, mode in (("E1", "closed"), ("E3", "open")):
        d = os.path.join(a.out, "raw", block)
        os.makedirs(d, exist_ok=True)
        for cell in "ABCD":
            for r in range(a.runs):
                df = frames(cell, a.scenario, rng, a.frames, rng.lognormal(0, 0.01))
                n = len(df)
                df["latency_ms"] = df["service_ms"]
                if mode == "open":   # simple queue: latency accumulates when service > period
                    period, backlog, lat = 1000.0 / 30, 0.0, []
                    for s in df["service_ms"]:
                        backlog = max(0.0, backlog + s - period)
                        lat.append(s + backlog)
                    df["latency_ms"] = lat
                df["t_release_ms"] = np.arange(n) * 1000.0 / 30
                run_id = "%s_%s_mem_high_r%02d" % (block, cell, r)
                df = df.assign(run_id=run_id, cell=cell, input="adv" if cell in "BD" else "clean",
                               contention="mem_high" if cell in "CD" else "none", mode=mode, model="sim",
                               frame_idx=np.arange(n), image_idx=np.arange(n) % 500, n_pre_nms=0, n_post_nms=0,
                               time_limit_hit=0, queue_depth=0, dropped=0, soc_temp_c=50.0)
                df.to_csv(os.path.join(d, run_id + ".csv"), index=False)
    print("simulated scenario %s -> %s/raw" % (a.scenario, a.out))


if __name__ == "__main__":
    main()
