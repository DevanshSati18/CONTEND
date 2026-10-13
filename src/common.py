"""Shared helpers: config, paths, letterbox pre-processing, frame cache.

Kept compatible with Python 3.6 (Jetson Nano / JetPack 4.6): no f-string '=' specifiers,
no walrus operator, no dataclasses, no typing generics.
"""
import json
import os
import random
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


SCHEMA_VERSION = "1.0"


def device_id():
    """Which device profile this checkout runs as: $CONTEND_DEVICE, else the git-ignored
    file `.device` in the repo root (written by the device setup script), else 'dev'."""
    d = os.environ.get("CONTEND_DEVICE")
    if not d and os.path.exists(os.path.join(ROOT, ".device")):
        with open(os.path.join(ROOT, ".device")) as f:
            d = f.read().strip()
    return d or "dev"


def _merge(base, over):
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def _overlay(cfg, dev, yaml, depth=0):
    """Merge configs/devices/<dev>.yaml; an overlay may start from another one (inherit: <name>)."""
    over = os.path.join(ROOT, "configs", "devices", dev + ".yaml")
    if not os.path.exists(over) or depth > 3:
        return
    with open(over) as f:
        y = yaml.safe_load(f) or {}
    parent = y.pop("inherit", None)
    if parent:
        _overlay(cfg, parent, yaml, depth + 1)
    _merge(cfg, y)


def load_config(path=None):
    """Base config + device overlay configs/devices/<device>.yaml. Result paths are
    namespaced per device (results/<device>/...), so devices never overwrite each other."""
    import yaml
    path = path or os.environ.get("CONTEND_CONFIG") or os.path.join(ROOT, "configs", "contend.yaml")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    dev = device_id()
    _overlay(cfg, dev, yaml)
    # choices fixed ON this device by runscript.py (model/size/fps from the 50-70 % rule, runs per
    # cell from G3). Lives with the device's results, so it is committed and pushed with them.
    lock = os.path.join(ROOT, "results", dev, "locked_config.yaml")
    if os.path.exists(lock) and not os.environ.get("CONTEND_NO_LOCK"):
        with open(lock) as f:
            _merge(cfg, (yaml.safe_load(f) or {}).get("config", {}))
    cfg["device_id"] = dev
    for k in ("raw", "derived", "figures"):
        cfg["paths"][k] = cfg["paths"][k].replace("{device}", dev)
    return cfg


def p(rel):
    """Absolute path inside the project."""
    return rel if os.path.isabs(rel) else os.path.join(ROOT, rel)


def letterbox(img, size, color=114):
    """Resize keeping aspect ratio and pad to a square `size` (Ultralytics style)."""
    import cv2
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    if (nh, nw) != (h, w):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (size - nh) // 2
    left = (size - nw) // 2
    out = np.full((size, size, 3), color, dtype=np.uint8)
    out[top:top + nh, left:left + nw] = img
    return out


def read_rgb(path):
    import cv2
    img = cv2.imread(path)
    if img is None:
        raise IOError("cannot read " + path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def subset_ids(cfg):
    with open(p(cfg["paths"]["subset_ids"])) as f:
        return [line.strip() for line in f if line.strip()]


def image_path(cfg, image_id):
    return os.path.join(p(cfg["paths"]["coco_images"]), "%012d.jpg" % int(image_id))


def frame_cache(cfg, imgsz, ids=None, rebuild=False):
    """Pre-letterboxed uint8 frames, shape (N, S, S, 3). Built once, then memory-mapped."""
    path = p(cfg["paths"]["subset_cache"].format(imgsz=imgsz))
    if os.path.exists(path) and not rebuild:
        return np.load(path, mmap_mode="r")
    ids = ids or subset_ids(cfg)
    frames = np.stack([letterbox(read_rgb(image_path(cfg, i)), imgsz) for i in ids])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, frames)
    return frames


def adv_frames(frames, delta):
    """Apply the universal perturbation ONCE, ahead of time, and quantise to uint8 - as a camera
    would deliver the perturbed scene. Clean and adversarial frames then go through identical
    per-frame pre-processing, so the attack cannot add CPU work of its own (measured on the
    Jetson: adding delta per frame cost +2.7 ms of pre-processing, an artifact)."""
    d = np.ascontiguousarray(delta.transpose(1, 2, 0) * 255.0, dtype=np.float32)
    out = np.empty(frames.shape, dtype=np.uint8)
    for i in range(len(frames)):
        out[i] = np.clip(np.rint(frames[i].astype(np.float32) + d), 0, 255).astype(np.uint8)
    return out


def to_input(frame_u8, delta=None):
    """uint8 HWC frame -> float32 NCHW in [0,1], optionally + perturbation (CHW, same size)."""
    x = frame_u8.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    if delta is not None:
        x = np.clip(x + delta[None], 0.0, 1.0)
    return np.ascontiguousarray(x, dtype=np.float32)


def soc_temp_c():
    """Highest thermal-zone temperature (Linux). Returns NaN where unavailable."""
    best = float("nan")
    base = "/sys/class/thermal"
    if not os.path.isdir(base):
        return best
    for z in os.listdir(base):
        f = os.path.join(base, z, "temp")
        try:
            with open(f) as fh:
                t = int(fh.read().strip()) / 1000.0
            best = t if best != best else max(best, t)
        except Exception:
            pass
    return best


def pin_to_cores(cores):
    """Pin this process AND size thread pools to the pinned cores.
    Without the thread cap, torch/ORT spawn one thread per machine core and oversubscribe the
    pinned cores: measured 5.6x slower CPU NMS (37.9 vs 6.8 ms) - an artifact that would
    also interact with the co-runners and contaminate the experiment."""
    if not cores:
        return
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(cores))
    else:   # macOS has no core pinning: still cap threads; results are not core-isolated
        sys.stderr.write("WARNING: core pinning not supported on this OS; threads capped only\n")
    n = len(cores)
    os.environ["OMP_NUM_THREADS"] = str(n)
    try:
        import torch
        torch.set_num_threads(n)
    except Exception:
        pass


def n_pinned_cores():
    try:
        return len(os.sched_getaffinity(0))
    except Exception:
        return int(os.environ["OMP_NUM_THREADS"]) if os.environ.get("OMP_NUM_THREADS") else None


def env_info():
    """Software/hardware versions written next to every run (reproducibility)."""
    info = {"python": sys.version.split()[0], "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    for mod in ["numpy", "torch", "torchvision", "onnxruntime", "tensorrt", "cv2", "ultralytics"]:
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            pass
    for path in ["/etc/nv_tegra_release", "/proc/device-tree/model"]:
        try:
            with open(path) as f:
                key = "board" if path.endswith("model") else os.path.basename(path)   # not "model": clashes with the detector name
                info[key] = f.read().strip().replace("\x00", "")
        except Exception:
            pass
    try:
        info["nvpmodel"] = subprocess.check_output(["nvpmodel", "-q"], stderr=subprocess.STDOUT).decode().strip()
    except Exception:
        pass
    return info


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def seeded(seed):
    random.seed(seed)
    np.random.seed(seed)
