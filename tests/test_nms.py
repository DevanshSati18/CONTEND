"""Check the vendored NMS against Ultralytics on random and real model outputs (host only)."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from nms import nms  # noqa: E402


def main():
    from ultralytics.utils.nms import non_max_suppression
    torch.manual_seed(0)
    n_ok = 0
    for trial in range(20):
        n = 2100 if trial % 2 else 8400
        pred = torch.rand(1, 84, n)
        pred[:, :2] *= 416
        pred[:, 2:4] = pred[:, 2:4] * 60 + 2
        pred[:, 4:] = pred[:, 4:] ** (3 + trial % 5)      # vary how many pass conf
        ours, info = nms(pred, 0.25, 0.7, 300)
        ref = non_max_suppression(pred, 0.25, 0.7, max_det=300)[0]
        same = ours.shape == ref.shape and torch.allclose(ours, ref, atol=1e-4)
        n_ok += int(same)
        if not same:
            print("MISMATCH trial", trial, ours.shape, ref.shape)
    print("NMS parity: %d/20 trials identical" % n_ok)
    assert n_ok == 20


if __name__ == "__main__":
    main()
