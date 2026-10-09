"""P0 rule: pick the model/size and frame rate whose CLEAN latency sits at 50-70% of the
per-frame deadline. Each device gets its own choice (Jetson: GPU, Pi: CPU).

  python src/choose_config.py                      # device defaults: models/*.<model_ext>
  python src/choose_config.py --models "models/yolov8n_*.onnx"

Prints mean clean closed-loop service time per model against 30/15/10/5/2 FPS deadlines and
recommends the highest frame rate that satisfies the rule. Copy the recommendation into
configs/devices/<device>.yaml (model.name, model.imgsz, deadline.fps).
"""
import argparse
import glob
import os

import numpy as np

from backends import make_backend
from common import frame_cache, load_config, pin_to_cores, save_json, p

FPS = [30, 15, 10, 5, 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=None)
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--frames", type=int, default=200)
    a = ap.parse_args()
    cfg = load_config()
    backend = a.backend or cfg["runtime"]["backend"]
    pin_to_cores(cfg["platform"]["detector_cores"])
    pats = a.models or ["models/*.%s" % cfg["runtime"]["model_ext"]]
    paths = sorted(set(sum([glob.glob(p(x)) for x in pats], [])))
    rows = []
    print("device: %s   backend: %s" % (cfg["device_id"], backend))
    print("%-24s %9s  %s" % ("model", "mean ms", "  ".join("%%@%dfps" % f for f in FPS)))
    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        imgsz = int(stem.split("_")[-1])
        frames = frame_cache(cfg, imgsz)
        be = make_backend(backend, path, dict(cfg["nms"]))
        for k in range(20):
            be.process(frames[k % len(frames)])
        ms = float(np.mean([sum(be.process(frames[k % len(frames)])[0][s]
                                for s in ("t_pre_ms", "t_inf_ms", "t_nms_ms", "t_d2h_ms"))
                            for k in range(a.frames)]))
        pct = [100 * ms * f / 1000.0 for f in FPS]
        ok = [f for f, q in zip(FPS, pct) if 50 <= q <= 70]
        rows.append(dict(model=stem, mean_ms=ms, ok_fps=ok))
        print("%-24s %9.2f  %s   %s" % (stem, ms, "  ".join("%7.0f%%" % q for q in pct),
                                        ("OK at %d FPS" % ok[0]) if ok else ""))
    best = sorted([r for r in rows if r["ok_fps"]], key=lambda r: (-r["ok_fps"][0], -r["mean_ms"]))
    if best:
        b = best[0]
        name, size = b["model"].rsplit("_", 1)
        print("\nRECOMMENDED for %s: model.name=%s  model.imgsz=%s  deadline.fps=%d  (clean %.1f ms)" %
              (cfg["device_id"], name, size, b["ok_fps"][0], b["mean_ms"]))
    else:
        print("\nNo model meets 50-70%% of a standard deadline; pick the closest and record it in CHANGELOG.")
    save_json(rows, os.path.join(p(cfg["paths"]["derived"]), "choose_config.json"))


if __name__ == "__main__":
    main()
