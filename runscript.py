#!/usr/bin/env python3
"""CONTEND: one command runs the whole study on one device and stores its results in
results/<device>/. Re-running the same command resumes where it stopped.

  python3 runscript.py --pi          # Raspberry Pi: one complete INSTANCE -> results/raspberry_pi/instance<k>/
  python3 runscript.py --nvidia      # Jetson Nano             -> results/jetson_nano/
  python3 runscript.py --dev         # Mac / dev VM (dry run)  -> results/dev_vm/  (never evidence)
  python3 runscript.py --smoke       # 5-minute end-to-end code check (tiny settings)

  python3 runscript.py --prepare     # Mac/host, once: ONNX models + attack perturbations to commit
  python3 runscript.py --compile     # Mac: git pull, analyse every device, combined tables + figures
  python3 runscript.py --pi --status # what is done, what is left, time estimate

An INSTANCE is one complete pass: gates G1-G3, then every condition of E1-E3 measured once, recorded
in results/<device>/instance<k>/. Each new command adds the next instance (instance1, instance2, ...);
the device report averages over all instances, and every instance re-checks the gates.

Useful options: --instances N (run N instances back to back), --runs R (runs per condition per
instance, default 1), --push / --push-each (commit and push after the last / every instance),
--force (continue after a failed gate), --skip-platform (no sudo clock settings; test only).

Steps for a device (each one is skipped when its output already exists):
  1 environment   packages (runs the device setup script once), co-runner binaries
  2 platform      fixed clocks / performance governor (sudo), before every invocation
  3 images        the 500 seeded COCO evaluation images (downloaded one by one, ~80 MB)
  4 models        committed ONNX + perturbations; TensorRT engines on the Jetson
  5 select        50-70 % deadline rule -> model, input size, FPS (written to the lock file)
  6 gates         G1 workload, G2 contention, G3 pilot, for THIS instance (config frozen at instance 1)
  7 main          blocks E1-E3, every condition once (or --runs R), in a fresh random order
  8 analyse       statistics over all instances, figures and REPORT.md after every instance

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
SETUP = {"raspberry_pi": "setup.sh --pi", "jetson_nano": "setup.sh --nvidia", "dev_vm": "setup.sh --linux"}
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
        self.raw = os.path.join(self.res, "raw")
        self.lock_path = os.path.join(self.res, "locked_config.yaml")
        self.py = self.pick_python()
        self.env = dict(os.environ, CONTEND_DEVICE=device, PYTHONUNBUFFERED="1", PYTHONWARNINGS="ignore")
        self.env.pop("CONTEND_INSTANCE", None)
        self.env_device = dict(self.env)          # device level: analysis pools all instances
        self.inst = None

    def set_instance(self, k):
        self.inst = k
        self.inst_dir = os.path.join(self.res, "instance%d" % k)
        self.der = os.path.join(self.inst_dir, "derived")
        self.raw = os.path.join(self.inst_dir, "raw")
        self.env = dict(self.env_device, CONTEND_INSTANCE=str(k))
        for d in (self.der, self.raw, os.path.join(self.inst_dir, "logs")):
            if not os.path.isdir(d):
                os.makedirs(d)

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
        run(["bash"] + SETUP[ctx.dev].split())
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
        ctx.py_run("src/run_experiment.py", "pilot", "--runs", "2", "--model", mf)
    g1, g2, g3 = (jload(os.path.join(ctx.der, n), {}) for n in ("g1_device.json", "g2.json", "g3.json"))
    fails = []
    if not g1.get("G1_pass"):
        fails.append("G1 (attack does not inflate the NMS workload enough on this device: candidates x%.1f)"
                     % g1.get("candidate_ratio", 0))
    used = set(cfg["experiment"]["blocks"][b]["contention"] for b in ctx.a.blocks)
    for k, v in (g2 or {}).items():
        if v.get("G2_pass") is False and k not in used:
            say("note: G2 %s %+.1f %% (< 10 %%) - not used by blocks %s, reported only" % (
                k, 100 * v["slowdown"], " ".join(ctx.a.blocks)))
        elif v.get("G2_pass") is False:
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
    # record this instance's gates; the configuration is frozen once, at the first instance
    with open(os.path.join(ctx.inst_dir, "gates.json"), "w") as f:
        json.dump(dict(instance=ctx.inst, G1_pass=bool(g1.get("G1_pass")), candidate_ratio=g1.get("candidate_ratio"),
                       nms_increase_ms=g1.get("nms_increase_ms_mean"), nms_increase_ci95=g1.get("nms_increase_ms_ci95"),
                       G2=dict((k, dict(slowdown=v.get("slowdown"), passed=v.get("G2_pass"))) for k, v in (g2 or {}).items()),
                       G3_worst_cv=max([c["cv"] for c in g3.get("cells", [])] or [None]), forced=bool(fails),
                       time=datetime.datetime.now().isoformat(timespec="seconds")), f, indent=2)
    lock = read_lock(ctx)
    if not lock.get("frozen_at"):
        lock["config"]["experiment"] = dict(blocks=dict((b, dict(runs_per_cell=ctx.a.runs)) for b in ctx.a.blocks))
        lock["blocks"], lock["runs_per_instance"] = ctx.a.blocks, ctx.a.runs
        lock["frozen_at"] = datetime.datetime.now().isoformat(timespec="seconds")
        lock["git_commit"] = git("rev-parse", "--short", "HEAD") or "none"
        write_lock(ctx, lock)
    say("instance %d: gates recorded; every condition of %s measured %d time(s)" % (
        ctx.inst, " ".join(lock["blocks"]), lock["runs_per_instance"]))


def done_runs(ctx, block, inst):
    raw = os.path.join(ctx.res, "instance%d" % inst, "raw", block)
    return len(glob.glob(os.path.join(raw, "%s_*_s%d_r*.csv.gz" % (block, inst))))


def instance_complete(ctx, k):
    return os.path.exists(os.path.join(ctx.res, "instance%d" % k, "COMPLETE"))


def instances(ctx):
    ks = []
    for d in glob.glob(os.path.join(ctx.res, "instance*")):
        try:
            ks.append(int(os.path.basename(d)[8:]))
        except ValueError:
            pass
    return sorted(ks)


def eta_hours(ctx, cfg, lock, k):
    g3 = jload(os.path.join(ctx.res, "instance%d" % k, "derived", "g3.json"), {})
    ms = [c["mean_ms"] for c in g3.get("cells", [])] or [100.0]
    lat = sum(ms) / len(ms) / 1000.0
    st, pl = cfg["stream"], cfg["platform"]
    t = 0.0
    for b in lock["blocks"]:
        n = 4 * lock["runs_per_instance"] - done_runs(ctx, b, k)
        mode = cfg["experiment"]["blocks"][b]["mode"]
        per = (st["warmup_frames"] + st["closed_frames"]) * lat if mode == "closed" else \
            st["open_seconds"] + st["warmup_frames"] * lat
        t += max(n, 0) * (per + 3 + pl["cooldown_s"] + 4)
    return t / 3600.0


def step_main(ctx):
    banner("7/8  Main blocks, instance %d (every condition %d time(s))" % (ctx.inst, ctx.a.runs))
    cfg, lock = ctx.cfg(), read_lock(ctx)
    mf = model_file(cfg, "%s_%s" % (cfg["model"]["name"], cfg["model"]["imgsz"]))
    eta = eta_hours(ctx, cfg, lock, ctx.inst)
    say("estimated time for this instance: %.1f h (rerun the same command to resume after any stop)" % eta)
    if eta > ctx.a.max_hours and not ctx.a.force:
        die("instance needs %.0f h (> --max-hours %.0f): the platform is probably unstable (heat, power: "
            "`vcgencmd get_throttled`) or runs are too long." % (eta, ctx.a.max_hours))
    for b in lock["blocks"]:
        ctx.py_run("src/run_experiment.py", "block", b, "--session", str(ctx.inst),
                   "--runs", str(lock["runs_per_instance"]), "--model", mf)
    with open(os.path.join(ctx.inst_dir, "COMPLETE"), "w") as f:
        f.write(datetime.datetime.now().isoformat(timespec="seconds") + "\n")
    say("instance %d complete - updating the analysis over all instances" % ctx.inst)


def analyse(ctx, quiet=True):
    env = ctx.env_device
    rc, _ = ctx.py_run("analysis/summarize.py", check=False, quiet=quiet, env=env)
    if rc == 0:
        rc, tail = ctx.py_run("analysis/analyze.py", check=False, quiet=quiet, env=env)
        for line in tail:
            if line.strip().startswith(("mem_", "cpu", "RQ", "instances")):
                say("  " + line.strip())
    if rc != 0:
        say("analysis step failed on this device (missing package?). The data is safe: run "
            "`python3 runscript.py --compile` on the Mac after pulling.")
    report(ctx)


def report(ctx):
    ctx.py_run("analysis/report.py", check=False, quiet=True, env=ctx.env_device)


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
    if not lock.get("config", {}).get("model"):
        say("nothing locked yet: instance 1 has not reached step 5")
        return
    c = lock["config"]
    say("model %s @ %s, %s FPS; frozen: %s" % (c["model"]["name"], c["model"]["imgsz"], c["deadline"]["fps"],
                                              lock.get("frozen_at", "no (gates of instance 1 not finished)")))
    for k in instances(ctx):
        g = jload(os.path.join(ctx.res, "instance%d" % k, "gates.json"), {})
        prog = ", ".join("%s %d/%d" % (b, done_runs(ctx, b, k), 4 * lock.get("runs_per_instance", 1))
                         for b in lock.get("blocks", []))
        say("instance %d: %s | gates %s | %s" % (k, "COMPLETE" if instance_complete(ctx, k) else "in progress",
            ("G1 %s x%.1f" % ("PASS" if g.get("G1_pass") else "FAIL", g.get("candidate_ratio") or 0)) if g else "not yet",
            prog))
    rep = os.path.join(ctx.res, "REPORT.md")
    if os.path.exists(rep):
        say("report: %s" % os.path.relpath(rep, ROOT))


# ----------------------------------------------------------------------------- modes
def device_mode(dev, a):
    if not a.profile:   # remember who this checkout is (git-ignored)
        with open(os.path.join(ROOT, ".device"), "w") as f:
            f.write(dev + "\n")
    ctx = Ctx(dev, a)
    if a.status:
        return status(ctx)
    have = instances(ctx)
    todo = [k for k in have if not instance_complete(ctx, k)]
    start = a.instance or (todo[0] if todo else (max(have) + 1 if have else 1))
    t0 = time.time()
    for n, k in enumerate(range(start, start + a.instances)):
        ctx.set_instance(k)
        if LOG[0]:
            LOG[0].close()
        LOG[0] = open(os.path.join(ctx.inst_dir, "logs", "runscript_%s.log" % time.strftime("%Y%m%d_%H%M%S")), "w")
        say("CONTEND runscript: device=%s  INSTANCE %d  blocks=%s  git=%s" % (
            dev, k, " ".join(a.blocks), git("rev-parse", "--short", "HEAD") or "-"))
        if instance_complete(ctx, k):
            say("instance %d is already complete" % k)
            continue
        if n == 0:
            step_environment(ctx)
            step_images(ctx)
            step_models(ctx)
        step_platform(ctx)
        step_select(ctx)
        step_models(ctx)          # Jetson: make sure the engine of the locked model exists
        step_gates(ctx)
        step_main(ctx)
        banner("8/8  Analysis over all instances")
        analyse(ctx, quiet=False)
        if a.push_each:
            push(ctx, "instance %d" % k)
    say("done (this command: %.0f min). Report: results/%s/REPORT.md   Next instance: run the same command again"
        % ((time.time() - t0) / 60.0, dev))
    if a.push and not a.push_each:
        push(ctx, "instance %d" % k)
    elif not a.push_each:
        say("Push: git add results/%s && git commit -m 'results: %s' && git pull --rebase && git push" % (dev, dev))


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
    devs = sorted(set(os.path.basename(os.path.dirname(d)) for d in glob.glob(os.path.join(ROOT, "results", "*", "raw")))
                  | set(os.path.basename(os.path.dirname(os.path.dirname(d)))
                        for d in glob.glob(os.path.join(ROOT, "results", "*", "instance*", "raw"))))
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
    ap.add_argument("--instances", type=int, default=1, help="complete instances to run back to back (default 1)")
    ap.add_argument("--instance", type=int, default=None, help="run/resume this instance number (default: next)")
    ap.add_argument("--runs", type=int, default=1, help="runs per condition in each instance (default 1)")
    ap.add_argument("--blocks", nargs="+", default=["E1", "E2", "E3"], help="blocks to run (E4 optional)")
    ap.add_argument("--status", action="store_true", help="show progress and time left, run nothing")
    ap.add_argument("--max-hours", type=float, default=30, help="stop before the main blocks if they would take longer")
    ap.add_argument("--push", action="store_true", help="commit + push results/<device> at the end")
    ap.add_argument("--push-each", action="store_true", help="commit + push after every instance")
    ap.add_argument("--force", action="store_true", help="continue even if a gate fails")
    ap.add_argument("--skip-platform", action="store_true", help="no sudo clock settings (test runs only)")
    ap.add_argument("--no-setup", action="store_true", help="never run the setup script")
    ap.add_argument("--no-pull", action="store_true", help="--compile without git pull")
    ap.add_argument("--include-smoke", action="store_true", help="--compile: also analyse results/smoke")
    ap.add_argument("--models", nargs="+", default=["yolov8n"], help="--prepare: models")
    ap.add_argument("--sizes", nargs="+", type=int, default=[320, 416, 640], help="--prepare: input sizes")
    a = ap.parse_args()
    if a.instances < 1 or a.runs < 1:
        die("--instances and --runs must be >= 1")
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
