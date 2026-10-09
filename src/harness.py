"""Frame-replay timing harness: one CSV row per frame.

Modes
  closed : frames back to back; latency = service time (RQ1, RQ2)
  open   : frames released on a fixed clock (fps) into a bounded drop-oldest queue,
           one worker; latency = completion - release (RQ3: misses, backlog, drops)

Single trial from the command line (the orchestrator `run_experiment.py` calls run_trial):
  python src/harness.py --backend trt --model models/yolov8n_416.engine --imgsz 416 \
      --mode closed --input adv --out results/raw/test.csv
"""
import argparse
import collections
import csv
import os
import time

import numpy as np

from common import frame_cache, load_config, p, pin_to_cores, soc_temp_c

now = time.perf_counter

COLUMNS = ["schema_version", "device_id", "backend", "precision", "imgsz", "fps_target", "run_id", "block", "cell", "input", "contention", "mode", "model", "frame_idx", "image_idx",
           "t_release_ms", "t_start_ms", "t_done_ms", "t_pre_ms", "t_inf_ms", "t_nms_ms", "t_d2h_ms",
           "service_ms", "latency_ms", "n_pre_nms", "n_nms_in", "n_post_nms", "time_limit_hit",
           "queue_depth", "dropped", "soc_temp_c"]


def _row(meta, k, img_idx, rel, start, done, info, depth, dropped, temp, t0):
    r = dict(meta)
    r.update(frame_idx=k, image_idx=img_idx, t_release_ms=(rel - t0) * 1e3,
             t_start_ms=(start - t0) * 1e3 if start else "", t_done_ms=(done - t0) * 1e3 if done else "",
             queue_depth=depth, dropped=dropped, soc_temp_c=temp)
    if info:
        r.update(t_pre_ms=info["t_pre_ms"], t_inf_ms=info["t_inf_ms"], t_nms_ms=info["t_nms_ms"],
                 t_d2h_ms=info["t_d2h_ms"], service_ms=(done - start) * 1e3, latency_ms=(done - rel) * 1e3,
                 n_pre_nms=info["n_pre_nms"], n_nms_in=info["n_nms_in"], n_post_nms=info["n_post_nms"],
                 time_limit_hit=info["time_limit_hit"])
    return r


def run_trial(backend, frames, delta, mode, out_csv, meta, n_frames=3000, seconds=300, fps=30,
              queue_len=4, warmup=300):
    """Run one trial and write its per-frame CSV. Returns the number of logged frames."""
    n_img = len(frames)
    for k in range(warmup):                       # warm-up: clocks, caches, allocator
        backend.process(frames[k % n_img], delta)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    f = open(out_csv, "w")
    w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
    w.writeheader()
    temp = soc_temp_c()
    t0 = now()
    logged = 0
    if mode == "closed":
        for k in range(n_frames):
            if k % 100 == 0:
                temp = soc_temp_c()
            start = now()
            info, _ = backend.process(frames[k % n_img], delta)
            done = now()
            w.writerow(_row(meta, k, k % n_img, start, start, done, info, 0, 0, temp, t0))
            logged += 1
    elif mode == "open":
        period = 1.0 / fps
        q = collections.deque()
        nxt = 0
        n_total = int(seconds * fps)
        while True:
            t = now()
            while nxt < n_total and t0 + nxt * period <= t:      # release due frames
                q.append(nxt)
                nxt += 1
                if len(q) > queue_len:                           # drop oldest
                    d = q.popleft()
                    w.writerow(_row(meta, d, d % n_img, t0 + d * period, None, None, None, len(q), 1, temp, t0))
                    logged += 1
            if not q:
                if nxt >= n_total:
                    break
                time.sleep(max(0.0, t0 + nxt * period - now()))
                continue
            k = q.popleft()
            if k % 100 == 0:
                temp = soc_temp_c()
            start = now()
            info, _ = backend.process(frames[k % n_img], delta)
            done = now()
            w.writerow(_row(meta, k, k % n_img, t0 + k * period, start, done, info, len(q), 0, temp, t0))
            logged += 1
    else:
        raise ValueError(mode)
    f.close()
    return logged


def load_delta(path):
    return None if not path else np.load(path).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=None, choices=["trt", "ort", "torch"], help="default: device runtime.backend")
    ap.add_argument("--model", required=True)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--mode", default="closed", choices=["closed", "open"])
    ap.add_argument("--input", default="clean", choices=["clean", "adv"])
    ap.add_argument("--delta", default=None, help="perturbation .npy (default from config)")
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--seconds", type=int, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = load_config()
    imgsz = a.imgsz or cfg["model"]["imgsz"]
    pin_to_cores(cfg["platform"]["detector_cores"])
    frames = frame_cache(cfg, imgsz)
    delta = None
    if a.input == "adv":
        delta = load_delta(a.delta or os.path.join(p(cfg["paths"]["perturbations"]), "%s_%d.npy" % (cfg["model"]["name"], imgsz)))
    nms_cfg = dict(cfg["nms"])
    a.backend = a.backend or cfg["runtime"]["backend"]
    be = __import__("backends").make_backend(a.backend, a.model, nms_cfg)
    from common import SCHEMA_VERSION
    meta = dict(schema_version=SCHEMA_VERSION, device_id=cfg["device_id"], backend=a.backend,
                precision=cfg["runtime"]["precision"], imgsz=imgsz, fps_target=cfg["deadline"]["fps"],
                run_id=os.path.splitext(os.path.basename(a.out))[0], block="manual", cell="manual",
                input=a.input, contention="none", mode=a.mode, model=os.path.basename(a.model))
    st = cfg["stream"]
    n = run_trial(be, frames, delta, a.mode, a.out, meta,
                  n_frames=a.frames or st["closed_frames"], seconds=a.seconds or st["open_seconds"],
                  fps=cfg["deadline"]["fps"], queue_len=st["queue_len"], warmup=min(st["warmup_frames"], 50))
    print("logged %d frames -> %s" % (n, a.out))


if __name__ == "__main__":
    main()
