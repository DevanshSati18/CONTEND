"""NMS identical in behaviour to Ultralytics `non_max_suppression` (single-label, class-aware).

Vendored so the same code runs on the host (new Ultralytics) and on the Jetson Nano
(Python 3.6, torch 1.10), where current Ultralytics cannot be installed.
`tests/test_nms.py` checks it against the installed Ultralytics function.

Steps (as in Ultralytics): confidence filter -> best class per box -> sort by score and
truncate to max_nms -> class-offset boxes -> torchvision NMS -> truncate to max_det.
"""
import time

import torch
import torchvision


def xywh2xyxy(x):
    y = x.clone()
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def count_candidates(pred, conf_thres=0.25):
    """Pre-NMS candidate count: boxes whose best class score exceeds conf_thres.
    pred: (1, 4+nc, N) raw YOLOv8 head output."""
    return int((pred[0, 4:].amax(0) > conf_thres).sum().item())


def nms(pred, conf_thres=0.25, iou_thres=0.7, max_det=300, max_nms=30000,
        agnostic=False, max_wh=7680, time_limit_s=2.05):
    """pred: tensor (1, 4+nc, N). Returns (det[n,6] xyxy,conf,cls ; info dict)."""
    t0 = time.time()
    x = pred[0].transpose(0, 1)                      # (N, 4+nc)
    scores_all = x[:, 4:]
    keep = scores_all.amax(1) > conf_thres
    x = x[keep]
    n_pre = int(x.shape[0])
    info = {"n_pre_nms": n_pre, "n_nms_in": 0, "n_post_nms": 0, "time_limit_hit": 0}
    if n_pre == 0:
        return x.new_zeros((0, 6)), info
    box = xywh2xyxy(x[:, :4])
    conf, j = x[:, 4:].max(1, keepdim=True)
    det = torch.cat((box, conf, j.float()), 1)
    det = det[conf.view(-1) > conf_thres]
    if det.shape[0] > max_nms:
        det = det[det[:, 4].argsort(descending=True)[:max_nms]]
    info["n_nms_in"] = int(det.shape[0])
    c = det[:, 5:6] * (0 if agnostic else max_wh)
    i = torchvision.ops.nms(det[:, :4] + c, det[:, 4], iou_thres)
    i = i[:max_det]
    out = det[i]
    info["n_post_nms"] = int(out.shape[0])
    if (time.time() - t0) > time_limit_s:
        info["time_limit_hit"] = 1
    return out, info
