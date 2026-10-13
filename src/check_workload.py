"""Gate G1: does the perturbation inflate the NMS workload on the chosen configuration?

Runs clean and adversarial frames through a backend and compares pre-NMS candidates,
post-NMS boxes and NMS time. Works on the host (torch/ort) and on the board (trt).

  python src/check_workload.py --backend torch --model models/yolov8n.pt --imgsz 416 --n 200
  python src/check_workload.py --backend trt   --model models/yolov8n_416.engine --imgsz 416

G1 passes if mean candidates rise >= 10x AND the 95% CI of the NMS-time increase is above 0.
"""
import argparse
import json
import os

import numpy as np

from backends import make_backend
from common import adv_frames, frame_cache, load_config, p


def boot_ci(diff, n=5000, seed=0):
    rng = np.random.RandomState(seed)
    m = [rng.choice(diff, len(diff)).mean() for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="torch", choices=["trt", "ort", "torch"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--imgsz", type=int, required=True)
    ap.add_argument("--delta", default=None)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = load_config()
    frames = frame_cache(cfg, a.imgsz)[:a.n]
    name = cfg["model"]["name"]
    delta = np.load(a.delta or os.path.join(p(cfg["paths"]["perturbations"]), "%s_%d.npy" % (name, a.imgsz))).astype(np.float32)
    be = make_backend(a.backend, a.model, dict(cfg["nms"]))
    frames_adv = adv_frames(frames, delta)                   # same pre-processing for both inputs
    for fr in frames[:20]:                                   # warm-up
        be.process(fr, None)
    res = {"clean": [], "adv": []}
    for fr, fa in zip(frames, frames_adv):                   # paired: same frame, both inputs
        for key, x in (("clean", fr), ("adv", fa)):
            info, _ = be.process(x, None)
            res[key].append(info)
    out = {}
    for key in res:
        out[key] = {k: float(np.mean([r[k] for r in res[key]])) for k in
                    ["n_pre_nms", "n_post_nms", "t_inf_ms", "t_nms_ms", "t_pre_ms", "t_d2h_ms"]}
    ratio = out["adv"]["n_pre_nms"] / max(out["clean"]["n_pre_nms"], 1e-9)
    dn = np.array([r["t_nms_ms"] for r in res["adv"]]) - np.array([r["t_nms_ms"] for r in res["clean"]])
    lo, hi = boot_ci(dn)
    out["candidate_ratio"] = ratio
    out["nms_increase_ms_mean"] = float(dn.mean())
    out["nms_increase_ms_ci95"] = [lo, hi]
    out["G1_pass"] = bool(ratio >= 10 and lo > 0)
    print(json.dumps(out, indent=2))
    print("G1 %s: candidates x%.1f, NMS +%.2f ms (95%% CI %.2f to %.2f)" %
          ("PASS" if out["G1_pass"] else "FAIL", ratio, dn.mean(), lo, hi))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
