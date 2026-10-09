"""Universal adversarial perturbation that inflates NMS candidates (HOST, needs a GPU).

Re-implements the max-objects + bounding-box-area objective of Phantom Sponges [1]
(Shapira et al., WACV 2023) for the Ultralytics YOLOv8 head. This is NOT a new attack:
it only produces the workload factor of the 2x2 design, and gate G1 checks it works.

  L = mean_i relu(logit(tau) + m - logit(s_i,k))  +  lambda * mean_{i: s_i,k > tau} (w_i * h_i) / S^2
  (hinge taken in logit space with margin m = 0.5; see objective())
  delta <- clip_[-eps, eps](Adam step on L),  input = clip_[0,1](x + delta)

where s_i,k is candidate i's score for the target class k. One delta is shared by all frames
(universal), optimised on data/subset/attack_ids.txt, which is disjoint from the evaluation set.

Usage:  python src/attack_uap.py --model models/yolov8n.pt --imgsz 416
Output: perturbations/<model>_<imgsz>.npy  (float32, 3xSxS, [-eps, eps]) + .json metadata
"""
import argparse
import hashlib
import os
import time

import numpy as np
import torch

from common import image_path, letterbox, load_config, p, read_rgb, save_json, seeded


def load_attack_frames(cfg, imgsz, limit=None):
    with open(p("data/subset/attack_ids.txt")) as f:
        ids = [x.strip() for x in f if x.strip()][:limit]
    frames = np.stack([letterbox(read_rgb(image_path(cfg, i)), imgsz) for i in ids])
    return torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0


def objective(pred, imgsz, target_class, tau, lam):
    """pred: (B, 4+nc, N) after sigmoid. Returns loss, mean candidate count."""
    scores = pred[:, 4:]
    s = scores.amax(1) if target_class is None else scores[:, target_class]
    # margin in logit space: the probability-space hinge has vanishing sigmoid gradients
    # for background boxes (s ~ 1e-4), so optimisation would never start
    sc = s.clamp(1e-6, 1 - 1e-6)
    logit, t_logit = torch.log(sc / (1 - sc)), float(np.log(tau / (1 - tau)))
    l_obj = torch.relu(t_logit + 0.5 - logit).mean()
    above = s > tau
    wh = pred[:, 2] * pred[:, 3] / float(imgsz * imgsz)
    l_area = (wh * above).sum() / above.sum().clamp(min=1)
    n_cand = (scores.amax(1) > tau).sum(1).float().mean().item()
    return l_obj + lam * l_area, n_cand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Ultralytics .pt weights")
    ap.add_argument("--imgsz", type=int, required=True)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="use only N attack images (debug)")
    ap.add_argument("--target-class", type=int, default=2, help="COCO class index (2=car, as in [1]); -1 = any")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="stop after N s and checkpoint; re-run the same command to resume (exit code 3)")
    a = ap.parse_args()
    cfg = load_config()
    ac = cfg["attack"]
    seeded(cfg["subset"]["seed"])
    torch.manual_seed(cfg["subset"]["seed"])
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    from ultralytics import YOLO
    model = YOLO(a.model).model.to(dev).eval()
    for q in model.parameters():
        q.requires_grad_(False)
    X = load_attack_frames(cfg, a.imgsz, a.limit)
    eps, steps = float(ac["eps"]), a.steps or int(ac["steps"])
    tgt = None if a.target_class < 0 else a.target_class
    name = os.path.splitext(os.path.basename(a.model))[0]
    out = a.out or p("perturbations/%s_%d.npy" % (name, a.imgsz))
    ckpt = out.replace(".npy", ".ckpt.pt")
    delta = torch.zeros(1, 3, a.imgsz, a.imgsz, device=dev, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=float(ac["lr"]))
    bs = min(int(ac["batch"]), len(X))
    t0 = time.time()
    hist, start, spent = [], 0, 0.0
    if os.path.exists(ckpt):                       # resume an interrupted optimisation
        st = torch.load(ckpt, map_location=dev, weights_only=False)
        with torch.no_grad():
            delta.copy_(st["delta"])
        opt.load_state_dict(st["opt"])
        torch.set_rng_state(st["rng"])
        hist, start, spent = st["hist"], st["step"], st["spent"]
        print("resumed at step %d" % start)
    for step in range(start, steps):
        if a.max_seconds and time.time() - t0 > a.max_seconds:
            os.makedirs(os.path.dirname(ckpt), exist_ok=True)
            torch.save({"delta": delta.detach(), "opt": opt.state_dict(), "rng": torch.get_rng_state(),
                        "hist": hist, "step": step, "spent": spent + time.time() - t0}, ckpt)
            print("checkpoint at step %d/%d -> re-run to continue" % (step, steps))
            raise SystemExit(3)
        idx = torch.randint(0, len(X), (bs,))
        x = X[idx].to(dev)
        y = model((x + delta).clamp(0, 1))
        y = y[0] if isinstance(y, (list, tuple)) else y
        loss, n_cand = objective(y, a.imgsz, tgt, float(ac["target_conf"]), float(ac["lambda_area"]))
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            delta.clamp_(-eps, eps)
        if step % 25 == 0 or step == steps - 1:
            hist.append({"step": step, "loss": float(loss), "mean_candidates": n_cand})
            print("step %4d  loss %.4f  mean candidates %.0f  (%.0fs)" % (step, float(loss), n_cand, time.time() - t0))
    d = delta.detach().cpu().numpy()[0].astype(np.float32)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.save(out, d)
    sha = hashlib.sha256(d.tobytes()).hexdigest()
    save_json({"model": a.model, "imgsz": a.imgsz, "eps": eps, "steps": steps, "target_class": tgt,
               "lambda_area": ac["lambda_area"], "lr": ac["lr"], "n_train_images": len(X),
               "sha256": sha, "history": hist, "seconds": spent + time.time() - t0}, out.replace(".npy", ".json"))
    if os.path.exists(ckpt):
        os.remove(ckpt)
    print("saved %s  sha256=%s" % (out, sha[:16]))


if __name__ == "__main__":
    main()
