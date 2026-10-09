#!/usr/bin/env python3
"""CONTEND: one command runs the whole study on one device and stores its results in
results/<device>/. Re-running the same command resumes where it stopped.

  python3 runscript.py --pi          # Raspberry Pi            -> results/raspberry_pi/
  python3 runscript.py --nvidia      # Jetson Nano             -> results/jetson_nano/
  python3 runscript.py --dev         # Mac / dev VM (dry run)  -> results/dev_vm/  (never evidence)
  python3 runscript.py --smoke       # 5-minute end-to-end code check (tiny settings)

  python3 runscript.py --prepare     # Mac/host, once: ONNX models + attack perturbations to commit
  python3 runscript.py --compile     # Mac: git pull, analyse every device, combined tables + figures
  python3 runscript.py --pi --status # what is done, what is left, time estimate

Useful options: --sessions N (independent repetitions of every block, default 3), --push (commit
and push this device's results at the end), --force (continue after a failed gate), --skip-platform
(no sudo clock/governor settings; not for real data), --blocks E1 E2 E3.

Steps for a device (each one is skipped when its output already exists):
  1 environment   packages (runs the device setup script once), co-runner binaries
  2 platform      fixed clocks / performance governor (sudo), before every invocation
  3 images        the 500 seeded COCO evaluation images (downloaded one by one, ~80 MB)
  4 models        committed ONNX + perturbations; TensorRT engines on the Jetson
  5 select        50-70 % deadline rule -> model, input size, FPS (written to the lock file)
  6 gates         G1 workload, G2 contention, G3 pilot -> runs per cell; lock is frozen
  7 main          blocks E1-E3, repeated for N sessions in a fresh random order each time
  8 analyse       per-device statistics, figures and REPORT.md after every session

Uses only the Python standard library itself (it runs before the environment exists, and on the
Jetson's Python 3.6); the pipeline steps run in the device's environment.
"""
from __future__ import print_function

import argparse
import datetime
import glob
import json
import math
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
DEVICES = {"pi": "raspberry_pi", "nvidia": "jetson_nano", "dev": "dev_vm", "smoke": "smoke"}
SETUP = {"raspberry_pi": "scripts/40_pi_setup.sh", "jetson_nano": "scripts/10_nano_setup.sh",
         "dev_vm": "scripts/50_dev_setup.sh"}
PLATFORM = {"raspberry_pi": "scripts/41_pi_session.sh", "jetson_nano": "scripts/11_nano_session.sh"}
NEED = {"jetson_nano": "numpy cv2 torch torchvision tensorrt yaml pandas scipy",
        "default": "numpy cv2 torch torchvision onnxruntime yaml pandas scipy matplotlib"}
COCO_URL = "http://images.cocodataset.org/val2017/%012d.jpg"
PT_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/%s.pt"
LOG = [None]


# ----------------------------------------------------------------------------- helpers
def say(msg=""):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg) if msg else ""
    print(line)
    sys.stdout.flush()
    if LOG[0]:
        LOG[0].write(line + "\n")
        LOG[0].flush()


def banner(title):
    say("")
    say("=" * 78)
    say(title)
    say("=" * 78)


def die(msg, code=1):
    say("STOP: " + msg)
    sys.exit(code)


def run(cmd, env=None, check=True, quiet=False):
    """Run a command, stream its output to the terminal and to the run log."""
    say("$ " + " ".join(cmd))
    pr = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          universal_newlines=True, bufsize=1)
    tail = []
    for line in pr.stdout:
        line = line.rstrip("\n")
        tail = (tail + [line])[-40:]
        if not quiet:
            print("    " + line)
            sys.stdout.flush()
        if LOG[0]:
            LOG[0].write("    " + line + "\n")
    pr.wait()
    if LOG[0]:
        LOG[0].flush()
    if check and pr.returncode != 0:
        if quiet:
            for t in tail:
                print("    " + t)
        die("command failed (exit %d): %s" % (pr.returncode, " ".join(cmd)))
    return pr.returncode, tail


def jload(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def git(*args):
    try:
        return subprocess.check_output(["git"] + list(args), cwd=ROOT, stderr=subprocess.STDOUT,
                                       universal_newlines=True).strip()
    except Exception:
        return ""


class Ctx(object):
    """Everything a step needs: device id, python, environment, paths."""

    def __init__(self, device, args):
        self.dev, self.a = device, args
        self.res = os.path.join(ROOT, "results", device)
        self.der = os.path.join(self.res, "derived")
        self.lock_path = os.path.join(self.res, "locked_config.yaml")
        self.py = self.pick_python()
        self.env = dict(os.environ, CONTEND_DEVICE=device, PYTHONUNBUFFERED="1", PYTHONWARNINGS="ignore")

    def pick_python(self):
        if os.environ.get("CONTEND_PYTHON"):
            return os.environ["CONTEND_PYTHON"]
        venv = os.path.join(ROOT, ".venv", "bin", "python")
        if self.dev != "jetson_nano" and os.path.exists(venv):
            return venv
        return sys.executable or "python3"

    def py_run(self, *args, **kw):
        return run([self.py] + list(args), env=kw.pop("env", self.env), **kw)

    def cfg(self):
        """The merged config (base + device overlay + lock) as JSON, from the device python."""
        code = ("import sys, json; sys.path.insert(0, 'src'); from common import load_config; "
                "print(json.dumps(load_config()))")
        out = subprocess.check_output([self.py, "-c", code], cwd=ROOT, env=self.env, universal_newlines=True)
        return json.loads(out.strip().splitlines()[-1])

    def missing(self):
        mods = NEED.get(self.dev, NEED["default"]).split()
        code = ("import importlib, sys\nmiss=[]\nfor m in %r:\n    try:\n        importlib.import_module(m)\n"
                "    except Exception:\n        miss.append(m)\nprint('MISSING:' + ' '.join(miss))" % mods)
        try:
            with open(os.devnull, "w") as null:
                out = subprocess.check_output([self.py, "-c", code], cwd=ROOT, universal_newlines=True, stderr=null)
            got = [l for l in out.splitlines() if l.startswith("MISSING:")]
            return [got[-1][len("MISSING:"):].strip() if got else "(python not usable)"]
        except Exception:
            return ["(python not usable)"]


# ----------------------------------------------------------------------------- lock file (yaml subset)
def write_lock(ctx, lock):
    """Small YAML writer (no PyYAML needed here). Values: str/int/float/bool/None, dicts, lists."""
    def emit(obj, ind):
        out = []
        for k, v in obj.items():
            if isinstance(v, dict):
                out.append("%s%s:" % ("  " * ind, k))
                out += emit(v, ind + 1)
            else:
                out.append("%s%s: %s" % ("  " * ind, k, json.dumps(v)))
        return out
    os.makedirs(ctx.res, exist_ok=True)
    head = ["# Written by runscript.py on this device. 'config' overrides configs/contend.yaml +",
            "# configs/devices/%s.yaml. Frozen before the main runs = pre-registration record;" % ctx.dev,
            "# change only with a dated note in docs/CHANGELOG.md (or delete it to start over)."]
    with open(ctx.lock_path, "w") as f:
        f.write("\n".join(head + emit(lock, 0)) + "\n")


def read_lock(ctx):
    if not os.path.exists(ctx.lock_path):
        return {}
    code = "import yaml, json, sys; print(json.dumps(yaml.safe_load(open(sys.argv[1])) or {}))"
    try:
        return json.loads(subprocess.check_output([ctx.py, "-c", code, ctx.lock_path], universal_newlines=True))
    except Exception:
        return {}


# ----------------------------------------------------------------------------- steps
def step_environment(ctx):
    banner("1/8  Environment for %s" % ctx.dev)
    miss = ctx.missing()[0]
    if miss:
        say("missing Python packages: %s" % miss)
        if ctx.a.no_setup or ctx.dev not in SETUP:
            die("install them (see README) or run without --no-setup")
        say("running the one-time setup script %s (asks for your sudo password)" % SETUP[ctx.dev])
        run(["bash", SETUP[ctx.dev]])
        ctx.py = ctx.pick_python()
        miss = ctx.missing()[0]
        if miss:
            die("still missing after setup: %s" % miss)
    say("python: %s - all packages present" % ctx.py)
    for name in ("bwhog", "cpuhog"):   # always rebuilt: a copied checkout may hold another OS's binary
        b = os.path.join(ROOT, "corunners", name)
        rc, _ = run(["gcc", "-O2", "-o", b, b + ".c"], check=False, quiet=True)
        if rc != 0 and not os.path.exists(b):
            die("cannot compile corunners/%s.c (install gcc)" % name)
    say("co-runners compiled")


def step_platform(ctx):
    banner("2/8  Platform control (fixed clocks / governor)")
    script = PLATFORM.get(ctx.dev)
    if not script:
        say("development machine: no platform control (results are not evidence)")
        return
    if ctx.a.skip_platform:
        say("WARNING: --skip-platform: clocks are NOT fixed; do not use these numbers in the paper")
        return
    rc, _ = run(["bash", script], check=False)
    if rc != 0:
        die("platform control failed. Fix it (sudo, power supply, cooling) or pass --skip-platform for a test run")


def fetch_images(ids, img_dir, threads=8):
    """Download only the needed COCO val2017 images (the full zip is 778 MB)."""
    try:
        from urllib.request import urlopen
    except ImportError:
        from urllib2 import urlopen
    from concurrent.futures import ThreadPoolExecutor
    os.makedirs(img_dir, exist_ok=True)
    todo = [i for i in ids if not os.path.exists(os.path.join(img_dir, "%012d.jpg" % int(i)))]
    if not todo:
        return 0

    def get(i):
        dst = os.path.join(img_dir, "%012d.jpg" % int(i))
        for attempt in range(4):
            try:
                data = urlopen(COCO_URL % int(i), timeout=30).read()
                with open(dst + ".part", "wb") as f:
                    f.write(data)
                os.rename(dst + ".part", dst)
                return True
            except Exception:
                time.sleep(2 * (attempt + 1))
        return False
    say("downloading %d COCO images..." % len(todo))
    with ThreadPoolExecutor(threads) as ex:
        ok = list(ex.map(get, todo))
    if not all(ok):
        die("%d images failed to download; check the network and re-run" % ok.count(False))
    return len(todo)


def step_images(ctx, with_attack=False):
    banner("3/8  Evaluation images (seeded subset)")
    cfg = ctx.cfg()
    ids_f = os.path.join(ROOT, cfg["paths"]["subset_ids"])
    if not os.path.exists(ids_f):
        die("%s is missing: it is committed in the repo (git pull) or made by `python3 runscript.py "
            "--prepare` on the Mac" % cfg["paths"]["subset_ids"])
    ids = [l.strip() for l in open(ids_f) if l.strip()]
    if with_attack:
        af = os.path.join(os.path.dirname(ids_f), "attack_ids.txt")
        ids += [l.strip() for l in open(af) if l.strip()] if os.path.exists(af) else []
    n = fetch_images(ids, os.path.join(ROOT, cfg["paths"]["coco_images"]))
    say("%d images present (%d downloaded now)" % (len(ids), n))


def candidates(ctx, cfg):
    """Models that can be used on this device: ONNX file with a matching perturbation."""
    out = []
    for onnx in sorted(glob.glob(os.path.join(ROOT, cfg["paths"]["models"], "*.onnx"))):
        stem = os.path.splitext(os.path.basename(onnx))[0]
        if os.path.exists(os.path.join(ROOT, cfg["paths"]["perturbations"], stem + ".npy")):
            out.append(stem)
    return out


def model_file(cfg, stem):
    return os.path.join(cfg["paths"]["models"], "%s.%s" % (stem, cfg["runtime"]["model_ext"]))


def step_models(ctx):
    banner("4/8  Models and perturbations")
    cfg = ctx.cfg()
    cand = candidates(ctx, cfg)
    if not cand:
        die("no models/<name>_<size>.onnx with perturbations/<name>_<size>.npy. Run "
            "`python3 runscript.py --prepare` on the Mac, commit, push, then git pull here")
    say("candidates (ONNX + perturbation): %s" % ", ".join(cand))
    if cfg["runtime"]["model_ext"] == "engine":
        lk = read_lock(ctx).get("config", {}).get("model")
        md = cfg["paths"]["models"]
        todo = ["%s/%s_%s.onnx" % (md, lk["name"], lk["imgsz"])] if lk else ["%s/%s.onnx" % (md, c) for c in cand]
        todo = [t for t in todo if not os.path.exists(os.path.join(ROOT, t[:-5] + ".engine"))]
        if todo:
            say("building TensorRT FP16 engines (~5-10 min each): %s" % " ".join(todo))
            run(["bash", "scripts/12_build_engines.sh"] + todo)
    say("models ready")


def step_select(ctx):
    banner("5/8  Configuration choice (clean latency at 50-70 % of the frame deadline)")
    lock = read_lock(ctx)
    if lock.get("config", {}).get("model"):
        c = lock["config"]
        say("already locked: %s @ %s, %s FPS (delete %s to choose again)" % (
            c["model"]["name"], c["model"]["imgsz"], c["deadline"]["fps"], os.path.relpath(ctx.lock_path, ROOT)))
        return
    cfg = ctx.cfg()
    files = [model_file(cfg, s) for s in candidates(ctx, cfg)]
    files = [f for f in files if os.path.exists(os.path.join(ROOT, f))]
    ctx.py_run("src/choose_config.py", "--models", *files)
    rows = jload(os.path.join(ctx.der, "choose_config.json"), [])
    fps_list = [30, 15, 10, 5, 2]
    best, rule = None, True
    ok = sorted([r for r in rows if r["ok_fps"]], key=lambda r: (-r["ok_fps"][0], -r["mean_ms"]))
    if ok:
        best, fps = ok[0], ok[0]["ok_fps"][0]
    else:   # nothing inside 50-70 %: closest to 60 % (recorded; goes into CHANGELOG)
        rule = False
        best, fps = min(((r, f) for r in rows for f in fps_list), key=lambda x: (abs(x[0]["mean_ms"] * x[1] / 10.0 - 60), -x[1]))
    name, size = best["model"].rsplit("_", 1)
    share = best["mean_ms"] * fps / 1000.0
    lock = dict(device=ctx.dev, selected_at=datetime.datetime.now().isoformat(timespec="seconds"),
                rule_met=rule, clean_ms=round(best["mean_ms"], 2), deadline_share="%.0f %%" % (100 * share),
                config=dict(model=dict(name=name, imgsz=int(size)), deadline=dict(fps=fps)))
    write_lock(ctx, lock)
    say("LOCKED: %s @ %s, %d FPS (clean %.1f ms = %.0f %% of the deadline)%s" % (
        name, size, fps, best["mean_ms"], 100 * share, "" if rule else "  [50-70 % rule not met: closest; note it in CHANGELOG]"))


def plan_runs(cfg, g3, sessions, blocks):
    """Runs per cell per session. Total per cell = max(G3 requirement, configured runs_per_cell),
    spread over the sessions (at least 2 per session so every session can be analysed alone)."""
    need = int((g3 or {}).get("runs_per_cell_needed", 6))
    plan = {}
    for b in blocks:
        spec = cfg["experiment"]["blocks"][b]
        total = max(need, int(spec["runs_per_cell"])) if spec["mode"] == "closed" else int(spec["runs_per_cell"])
        plan[b] = max(2, int(math.ceil(total / float(sessions))))
    return plan


def step_gates(ctx):
    banner("6/8  Gates G1 (workload), G2 (contention), G3 (pilot)")
    cfg = ctx.cfg()
    m = cfg["model"]
    stem = "%s_%s" % (m["name"], m["imgsz"])
    mf = model_file(cfg, stem)
    delta = os.path.join(cfg["paths"]["perturbations"], stem + ".npy")
    g1f = os.path.join(ctx.der, "g1_device.json")
    if not os.path.exists(g1f):
        ctx.py_run("src/check_workload.py", "--backend", cfg["runtime"]["backend"], "--model", mf,
                   "--imgsz", str(m["imgsz"]), "--delta", delta, "--n", str(min(200, cfg["subset"]["n_images"])),
                   "--out", g1f)
    if not os.path.exists(os.path.join(ctx.der, "g2.json")):
        ctx.py_run("src/run_experiment.py", "g2", "--model", mf)
    if not os.path.exists(os.path.join(ctx.der, "g3.json")):
        ctx.py_run("src/run_experiment.py", "pilot", "--runs", "3", "--model", mf)
    g1, g2, g3 = (jload(os.path.join(ctx.der, n), {}) for n in ("g1_device.json", "g2.json", "g3.json"))
    fails = []
    if not g1.get("G1_pass"):
        fails.append("G1 (attack does not inflate the NMS workload enough on this device: candidates x%.1f)"
                     % g1.get("candidate_ratio", 0))
    for k, v in (g2 or {}).items():
        if v.get("G2_pass") is False:
            fails.append("G2 %s (co-runner slows clean latency only %+.1f %%, needs >= 10 %%)" % (k, 100 * v["slowdown"]))
    say("G1 %s | G2 %s | G3 %d runs per cell needed" % (
        "PASS" if g1.get("G1_pass") else "FAIL",
        ", ".join("%s %s" % (k, "control" if v.get("G2_pass") is None else ("PASS" if v["G2_pass"] else "FAIL"))
                  for k, v in sorted((g2 or {}).items())),
        g3.get("runs_per_cell_needed", -1)))
    if fails and ctx.a.force:
        say("WARNING: continuing despite a failed gate (--force): %s. Record this in docs/CHANGELOG.md." % "; ".join(fails))
    if fails and not ctx.a.force:
        report(ctx)
        die("gate failed: %s.\nSee README 'If a gate fails' (e.g. nms.device: cpu, stronger co-runner). "
            "Re-run with --force to continue anyway (record it in docs/CHANGELOG.md)." % "; ".join(fails))
    # freeze: this is the pre-registration point for this device
    lock = read_lock(ctx)
    blocks = ctx.a.blocks
    if not lock.get("frozen_at") or ctx.a.sessions != lock.get("sessions") or sorted(blocks) != sorted(lock.get("blocks", [])):
        plan = plan_runs(cfg, g3, ctx.a.sessions, blocks)
        if lock.get("frozen_at"):   # keep the frozen per-session runs; only the session count / blocks change
            old = lock["config"].get("experiment", {}).get("blocks", {})
            plan = dict((b, int(old[b]["runs_per_cell"]) if b in old else plan[b]) for b in blocks)
            lock.setdefault("changes", "")
            lock["changes"] = (lock["changes"] + " | " if lock["changes"] else "") + "%s: sessions %s -> %d, blocks %s" % (
                datetime.date.today().isoformat(), lock.get("sessions"), ctx.a.sessions, " ".join(blocks))
        lock["config"]["experiment"] = dict(blocks=dict((b, dict(runs_per_cell=plan[b])) for b in blocks))
        lock["sessions"], lock["blocks"] = ctx.a.sessions, blocks
        lock["g3_runs_needed"] = g3.get("runs_per_cell_needed")
        lock["gates"] = dict(G1=bool(g1.get("G1_pass")), G2=dict((k, v.get("G2_pass")) for k, v in (g2 or {}).items()),
                             forced=bool(fails))
        lock.setdefault("frozen_at", datetime.datetime.now().isoformat(timespec="seconds"))
        lock.setdefault("git_commit", git("rev-parse", "--short", "HEAD") or "none")
        write_lock(ctx, lock)
    say("frozen: %d sessions x {%s} runs per cell per session" % (
        lock["sessions"], ", ".join("%s: %d" % (b, v["runs_per_cell"]) for b, v in sorted(lock["config"]["experiment"]["blocks"].items()))))


def done_runs(ctx, block, session):
    return len(glob.glob(os.path.join(ctx.res, "raw", block, "%s_*_s%d_r*.csv.gz" % (block, session))))


def eta_hours(ctx, cfg, lock, remaining_only=True):
    g3 = jload(os.path.join(ctx.der, "g3.json"), {})
    ms = [c["mean_ms"] for c in g3.get("cells", [])] or [100.0]
    lat = sum(ms) / len(ms) / 1000.0
    st, pl = cfg["stream"], cfg["platform"]
    over = 3 + pl["cooldown_s"] + 4
    t = 0.0
    for s in range(1, lock["sessions"] + 1):
        for b, v in lock["config"]["experiment"]["blocks"].items():
            n = 4 * v["runs_per_cell"] - (done_runs(ctx, b, s) if remaining_only else 0)
            mode = cfg["experiment"]["blocks"][b]["mode"]
            per = (st["warmup_frames"] + st["closed_frames"]) * lat if mode == "closed" else \
                st["open_seconds"] + st["warmup_frames"] * lat
            t += max(n, 0) * (per + over)
    return t / 3600.0


def step_main(ctx):
    banner("7/8  Main blocks, %d independent sessions" % ctx.a.sessions)
    cfg, lock = ctx.cfg(), read_lock(ctx)
    mf = model_file(cfg, "%s_%s" % (cfg["model"]["name"], cfg["model"]["imgsz"]))
    say("estimated time left: %.1f h (rerun the same command to resume after any stop)" % eta_hours(ctx, cfg, lock))
    blocks = lock["blocks"]
    for s in range(1, lock["sessions"] + 1):
        runs_sess = dict((b, lock["config"]["experiment"]["blocks"][b]["runs_per_cell"]) for b in blocks)
        if all(done_runs(ctx, b, s) >= 4 * runs_sess[b] for b in blocks):
            say("session %d already complete" % s)
            continue
        say("---- session %d/%d  (%s)" % (s, lock["sessions"], ", ".join("%s %d/cell" % (b, runs_sess[b]) for b in blocks)))
        if s > 1 and PLATFORM.get(ctx.dev) and not ctx.a.skip_platform:
            run(["bash", PLATFORM[ctx.dev]], check=False)   # re-assert clocks between sessions
        for b in blocks:
            ctx.py_run("src/run_experiment.py", "block", b, "--session", str(s), "--runs", str(runs_sess[b]), "--model", mf)
        say("session %d complete - updating the analysis" % s)
        analyse(ctx)
        if ctx.a.push_each:
            push(ctx, "session %d" % s)


def analyse(ctx, quiet=True):
    rc, _ = ctx.py_run("analysis/summarize.py", check=False, quiet=quiet)
    if rc == 0:
        rc, tail = ctx.py_run("analysis/analyze.py", check=False, quiet=quiet)
        for line in tail:
            if line.strip().startswith(("mem_", "cpu", "RQ", "sessions")):
                say("  " + line.strip())
    if rc != 0:
        say("analysis step failed on this device (missing package?). The data is safe: run "
            "`python3 runscript.py --compile` on the Mac after pulling.")
    report(ctx)


def report(ctx):
    ctx.py_run("analysis/report.py", check=False, quiet=True)


def push(ctx, what):
    if not git("rev-parse", "--is-inside-work-tree"):
        say("not a git checkout: copy results/%s to the Mac by hand" % ctx.dev)
        return
    run(["git", "add", "-A", "results/%s" % ctx.dev], check=False)
    run(["git", "commit", "-m", "results: %s %s (%s)" % (ctx.dev, what, time.strftime("%Y-%m-%d %H:%M"))], check=False)
    run(["git", "pull", "--rebase"], check=False)
    rc, _ = run(["git", "push"], check=False)
    if rc != 0:
        say("push failed (credentials?). Results are committed locally; run `git push` yourself.")


def status(ctx):
    banner("Status: %s" % ctx.dev)
    lock = read_lock(ctx)
    if not lock:
        say("nothing locked yet: the run has not reached step 5")
        return
    c = lock["config"]
    say("model %s @ %s, %s FPS; frozen: %s" % (c["model"]["name"], c["model"]["imgsz"], c["deadline"]["fps"],
                                              lock.get("frozen_at", "no (gates not finished)")))
    if lock.get("frozen_at"):
        for s in range(1, lock["sessions"] + 1):
            say("session %d: %s" % (s, ", ".join("%s %d/%d" % (b, done_runs(ctx, b, s), 4 * v["runs_per_cell"])
                                               for b, v in sorted(c["experiment"]["blocks"].items()))))
        say("estimated time left: %.1f h" % eta_hours(ctx, ctx.cfg(), lock))
    rep = os.path.join(ctx.res, "REPORT.md")
    if os.path.exists(rep):
        say("report: %s" % os.path.relpath(rep, ROOT))


# ----------------------------------------------------------------------------- modes
def device_mode(dev, a):
    if not a.profile:   # remember who this checkout is (git-ignored)
        with open(os.path.join(ROOT, ".device"), "w") as f:
            f.write(dev + "\n")
    ctx = Ctx(dev, a)
    os.makedirs(os.path.join(ctx.res, "logs"), exist_ok=True)
    if a.status:
        return status(ctx)
    LOG[0] = open(os.path.join(ctx.res, "logs", "runscript_%s.log" % time.strftime("%Y%m%d_%H%M%S")), "w")
    t0 = time.time()
    say("CONTEND runscript: device=%s  sessions=%d  blocks=%s  git=%s" % (
        dev, a.sessions, " ".join(a.blocks), git("rev-parse", "--short", "HEAD") or "-"))
    step_environment(ctx)
    step_platform(ctx)
    step_images(ctx)
    step_models(ctx)
    step_select(ctx)
    step_models(ctx)          # Jetson: make sure the engine of the locked model exists
    step_gates(ctx)
    step_main(ctx)
    banner("8/8  Final analysis")
    analyse(ctx, quiet=False)
    say("done (this invocation: %.0f min). Report: results/%s/REPORT.md" % ((time.time() - t0) / 60.0, dev))
    if a.push or a.push_each:
        push(ctx, "complete")
    else:
        say("Next: git add results/%s && git commit -m 'results: %s' && git pull --rebase && git push" % (dev, dev))
        say("      (or re-run with --push). Then on the Mac: python3 runscript.py --compile")


def prepare_mode(a):
    """Mac/host: shared assets every device uses (commit + push them)."""
    ctx = Ctx("dev_vm", a)
    ctx.env["CONTEND_DEVICE"] = "dev_vm"
    banner("Prepare shared assets (host)")
    step_environment(ctx)
    rc, _ = ctx.py_run("-c", "import ultralytics", check=False, quiet=True)
    if rc != 0:
        ctx.py_run("-m", "pip", "install", "ultralytics", "onnx", "onnxslim")
    cfg = ctx.cfg()
    if not os.path.exists(os.path.join(ROOT, cfg["paths"]["subset_ids"])):
        die("data/subset/image_ids.txt missing: needs COCO annotations; run scripts/01_download_coco.sh "
            "and python3 src/make_subset.py")
    step_images(ctx, with_attack=True)
    try:
        from urllib.request import urlretrieve
    except ImportError:
        from urllib import urlretrieve
    for m in a.models:
        pt = os.path.join(ROOT, "models", m + ".pt")
        if not os.path.exists(pt):
            say("downloading %s.pt" % m)
            urlretrieve(PT_URL % m, pt)
        need = [s for s in a.sizes if not os.path.exists(os.path.join(ROOT, "models", "%s_%d.onnx" % (m, s)))]
        if need:
            ctx.py_run("src/export_models.py", "--models", m, "--sizes", *[str(s) for s in need])
        for s in a.sizes:
            npy = os.path.join(ROOT, "perturbations", "%s_%d.npy" % (m, s))
            if not os.path.exists(npy):
                say("attack %s @ %d (300 steps; resumable)" % (m, s))
                ctx.py_run("src/attack_uap.py", "--model", "models/%s.pt" % m, "--imgsz", str(s))
    say("Commit and push the shared assets:")
    say("  git add models/*.onnx perturbations data/subset/*.txt && git commit -m 'shared assets' && git push")


def compile_mode(a):
    """Mac: pull every device's results, analyse each device, combine."""
    banner("Compile all devices")
    if not a.no_pull and git("remote"):
        run(["git", "pull", "--rebase"], check=False)
    devs = sorted(os.path.basename(os.path.dirname(d)) for d in glob.glob(os.path.join(ROOT, "results", "*", "raw")))
    devs = [d for d in devs if not d.startswith("smoke") or a.include_smoke]
    if not devs:
        die("no results/<device>/raw folders yet")
    for d in devs:
        ctx = Ctx(d, a)
        say("---- %s" % d)
        analyse(ctx, quiet=True)
    ctx = Ctx("dev_vm", a)
    ctx.py_run("analysis/combine.py", check=False)
    ctx.py_run("analysis/report.py", "--combined", check=False, quiet=True)
    say("Per device: results/<device>/REPORT.md   Combined: results/combined/REPORT.md, contend_*.csv, fig5")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pi", action="store_true", help="Raspberry Pi -> results/raspberry_pi/")
    g.add_argument("--nvidia", "--jetson", dest="nvidia", action="store_true", help="Jetson Nano -> results/jetson_nano/")
    g.add_argument("--dev", action="store_true", help="development machine (never evidence) -> results/dev_vm/")
    g.add_argument("--smoke", action="store_true", help="5-minute end-to-end code check")
    g.add_argument("--prepare", action="store_true", help="host: ONNX models + perturbations to commit")
    g.add_argument("--compile", action="store_true", help="host: pull, analyse all devices, combine")
    g.add_argument("--profile", default=None, help="advanced: any configs/devices/<name>.yaml (e.g. smoke_rs)")
    ap.add_argument("--sessions", type=int, default=3, help="independent repetitions of every block (default 3)")
    ap.add_argument("--blocks", nargs="+", default=["E1", "E2", "E3"], help="blocks to run (E4 optional)")
    ap.add_argument("--status", action="store_true", help="show progress and time left, run nothing")
    ap.add_argument("--push", action="store_true", help="commit + push results/<device> at the end")
    ap.add_argument("--push-each", action="store_true", help="also commit + push after every session")
    ap.add_argument("--force", action="store_true", help="continue even if a gate fails")
    ap.add_argument("--skip-platform", action="store_true", help="no sudo clock settings (test runs only)")
    ap.add_argument("--no-setup", action="store_true", help="never run the setup script")
    ap.add_argument("--no-pull", action="store_true", help="--compile without git pull")
    ap.add_argument("--include-smoke", action="store_true", help="--compile: also analyse results/smoke")
    ap.add_argument("--models", nargs="+", default=["yolov8n"], help="--prepare: models")
    ap.add_argument("--sizes", nargs="+", type=int, default=[320, 416, 640], help="--prepare: input sizes")
    a = ap.parse_args()
    if a.sessions < 1:
        die("--sessions must be >= 1")
    os.chdir(ROOT)
    if a.smoke:
        sys.exit(subprocess.call(["bash", "scripts/smoke_test.sh"], cwd=ROOT))
    if a.prepare:
        return prepare_mode(a)
    if a.compile:
        return compile_mode(a)
    dev = a.profile or DEVICES["pi" if a.pi else "nvidia" if a.nvidia else "dev"]
    try:
        device_mode(dev, a)
    except KeyboardInterrupt:
        say("interrupted - re-run the same command to resume (finished runs are kept)")
        sys.exit(130)


if __name__ == "__main__":
    main()
