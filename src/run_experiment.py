"""Orchestrator: gates G2/G3 and the randomised factorial blocks.

  python src/run_experiment.py g2    --backend trt --model models/yolov8n_416.engine
  python src/run_experiment.py pilot --backend trt --model models/yolov8n_416.engine
  python src/run_experiment.py block E1 --backend trt --model models/yolov8n_416.engine

A block runs cells A (clean, no load), B (adv, no load), C (clean, load), D (adv, load)
`runs_per_cell` times each, in a seeded random order, with a cool-down between runs.
Every run writes results/raw/<block>/<run_id>.csv (+ co-runner and tegrastats logs) and a row
in results/raw/runs.csv. Re-running the same command resumes: finished runs are skipped.
"""
import argparse
import csv
import os
import random
import subprocess
import time

import numpy as np

from backends import make_backend
from common import SCHEMA_VERSION, env_info, frame_cache, load_config, p, pin_to_cores, save_json, soc_temp_c
import gzip
import shutil
from corunner import CoRunner
from harness import run_trial

CELLS = {"A": ("clean", False), "B": ("adv", False), "C": ("clean", True), "D": ("adv", True)}
MANIFEST_COLS = ["schema_version", "device_id", "backend", "precision", "imgsz", "fps_target", "run_id", "block", "cell", "input", "contention", "mode", "rep", "order", "model",
                 "start_time", "start_temp_c", "end_temp_c", "frames", "corunner_throughput", "status"]


def wait_cool(cfg, max_wait=600):
    limit = cfg["platform"]["max_start_temp_c"]
    t0 = time.time()
    while soc_temp_c() == soc_temp_c() and soc_temp_c() > limit and time.time() - t0 < max_wait:
        time.sleep(10)


class Runner(object):
    def __init__(self, a):
        self.cfg = cfg = load_config()
        pin_to_cores(cfg["platform"]["detector_cores"])
        self.imgsz = a.imgsz or cfg["model"]["imgsz"]
        self.frames = frame_cache(cfg, self.imgsz)
        dpath = a.delta or os.path.join(p(cfg["paths"]["perturbations"]), "%s_%d.npy" % (cfg["model"]["name"], self.imgsz))
        self.delta = np.load(dpath).astype(np.float32)
        a.backend = a.backend or cfg["runtime"]["backend"]
        self.be = make_backend(a.backend, a.model, dict(cfg["nms"]))
        self.common = dict(schema_version=SCHEMA_VERSION, device_id=cfg["device_id"], backend=a.backend,
                           precision=cfg["runtime"]["precision"], imgsz=self.imgsz, fps_target=cfg["deadline"]["fps"])
        self.model = os.path.basename(a.model)
        self.raw = p(cfg["paths"]["raw"])
        os.makedirs(self.raw, exist_ok=True)
        save_json(dict(env_info(), config=cfg, model=a.model, delta=dpath, backend=a.backend),
                  os.path.join(self.raw, "env_%s.json" % time.strftime("%Y%m%d_%H%M%S")))

    def trial(self, block, cell, contention, mode, rep, order, n_frames=None, seconds=None):
        inp, loaded = CELLS[cell]
        cont = contention if loaded else "none"
        run_id = "%s_%s_%s_r%02d" % (block, cell, contention, rep)
        out = os.path.join(self.raw, block, run_id + ".csv.gz")
        if os.path.exists(out):
            print("skip (done)", run_id)
            return out
        wait_cool(self.cfg)
        st = self.cfg["stream"]
        logd = os.path.join(self.raw, block, run_id + "_logs")
        teg = None
        try:
            os.makedirs(logd, exist_ok=True)
            teg = subprocess.Popen(["tegrastats", "--interval", "100"],
                                   stdout=open(os.path.join(logd, "tegrastats.log"), "w"), stderr=subprocess.STDOUT)
        except Exception:
            teg = None
        co = CoRunner(self.cfg, cont, logd).start()
        t_start, temp0 = time.strftime("%Y-%m-%dT%H:%M:%S"), soc_temp_c()
        meta = dict(self.common, run_id=run_id, block=block, cell=cell, input=inp, contention=cont, mode=mode,
                    model=self.model)
        status, n = "failed", 0
        try:
            n = run_trial(self.be, self.frames, self.delta if inp == "adv" else None, mode, out + ".part", meta,
                          n_frames=n_frames or st["closed_frames"], seconds=seconds or st["open_seconds"],
                          fps=self.cfg["deadline"]["fps"], queue_len=st["queue_len"], warmup=st["warmup_frames"])
            with open(out + ".part", "rb") as fi, gzip.open(out, "wb") as fo:   # standard: gzip CSV
                shutil.copyfileobj(fi, fo)
            os.remove(out + ".part")
            status = "ok"
        except KeyboardInterrupt:
            status = "interrupted"
            raise
        finally:
            co.stop()
            if teg:
                teg.terminate()
            thr = CoRunner.summarize(logd, cont) if cont != "none" else []
            self._manifest(dict(self.common, run_id=run_id, block=block, cell=cell, input=inp, contention=cont, mode=mode,
                                rep=rep, order=order, model=self.model, start_time=t_start, start_temp_c=temp0,
                                end_temp_c=soc_temp_c(), frames=n if status == "ok" else 0,
                                corunner_throughput=";".join("%.3f" % v for v in thr), status=status))
        print("%-28s %4d frames  temp %.1f C  co-runner %s" % (run_id, n, temp0, thr))
        time.sleep(self.cfg["platform"]["cooldown_s"])
        return out

    def _manifest(self, row):
        fn = os.path.join(self.raw, "runs.csv")
        new = not os.path.exists(fn)
        with open(fn, "a") as f:
            w = csv.DictWriter(f, fieldnames=MANIFEST_COLS)
            if new:
                w.writeheader()
            w.writerow(row)

    def block(self, name, runs=None):
        b = self.cfg["experiment"]["blocks"][name]
        reps = runs or b["runs_per_cell"]
        sched = [(c, r) for c in "ABCD" for r in range(reps)]
        random.Random(self.cfg["experiment"]["seed"] + sum(map(ord, name))).shuffle(sched)
        for order, (cell, rep) in enumerate(sched):
            self.trial(name, cell, b["contention"], b["mode"], rep, order)


def g2(r, n_frames=None):
    """Gate G2: each co-runner must slow clean closed-loop latency by >= 10%."""
    import pandas as pd
    n_frames = n_frames or min(500, r.cfg["stream"]["closed_frames"])
    res = {}
    for name in [k for k in r.cfg["contention"] if k != "none"]:
        base = r.trial("G2", "A", name, "closed", 0, 0, n_frames=n_frames)
        load = r.trial("G2", "C", name, "closed", 0, 1, n_frames=n_frames)
        tb = pd.read_csv(base)["latency_ms"].mean()
        tl = pd.read_csv(load)["latency_ms"].mean()
        kind = r.cfg["contention"][name]["kind"]
        ok = bool(tl / tb - 1 >= 0.10)
        # bandwidth co-runners must load the detector; the CPU co-runner is the CONTROL
        # (separates DRAM contention from CPU scheduling), so its slowdown is only reported.
        verdict = ("PASS" if ok else "FAIL") if kind == "bw" else "CONTROL (reported, no threshold)"
        res[name] = dict(kind=kind, clean_ms=tb, loaded_ms=tl, slowdown=tl / tb - 1,
                         G2_pass=ok if kind == "bw" else None)
        print("G2 %-8s clean %.2f ms -> loaded %.2f ms  (%+.1f%%)  %s" % (name, tb, tl, 100 * (tl / tb - 1), verdict))
    save_json(res, os.path.join(p(r.cfg["paths"]["derived"]), "g2.json"))


def pilot(r, reps=3):
    """Gate G3: 2x2 with mem_high, `reps` runs per cell; between-run CV of mean latency."""
    import pandas as pd
    for cell in "ABCD":
        for k in range(reps):
            r.trial("PILOT", cell, "mem_high", "closed", k, 0, n_frames=min(1000, r.cfg["stream"]["closed_frames"]))
    rows = []
    for cell in "ABCD":
        ms = [pd.read_csv(os.path.join(r.raw, "PILOT", "PILOT_%s_mem_high_r%02d.csv.gz" % (cell, k)))["latency_ms"].mean()
              for k in range(reps)]
        cv = float(np.std(ms, ddof=1) / np.mean(ms))
        rows.append(dict(cell=cell, mean_ms=float(np.mean(ms)), cv=cv))
        print("cell %s  mean %.2f ms  between-run CV %.2f%%" % (cell, np.mean(ms), 100 * cv))
    worst = max(x["cv"] for x in rows)
    # runs per cell so that the CI half-width of log-ratio R is under half the 5% margin
    need = int(np.ceil((1.645 * np.sqrt(4) * worst / (np.log(1.05) / 2)) ** 2))
    print("G3: worst CV %.2f%% -> about %d runs per cell needed (minimum 6)" % (100 * worst, max(need, 6)))
    save_json(dict(cells=rows, runs_per_cell_needed=max(need, 6)), os.path.join(p(r.cfg["paths"]["derived"]), "g3.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["g2", "pilot", "block"])
    ap.add_argument("name", nargs="?")
    ap.add_argument("--backend", default=None, choices=["trt", "ort", "torch"], help="default: device runtime.backend")
    ap.add_argument("--model", required=True)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--delta", default=None)
    ap.add_argument("--runs", type=int, default=None, help="override runs per cell")
    a = ap.parse_args()
    r = Runner(a)
    if a.cmd == "g2":
        g2(r)
    elif a.cmd == "pilot":
        pilot(r, a.runs or 3)
    else:
        r.block(a.name, a.runs)


if __name__ == "__main__":
    main()
