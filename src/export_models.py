"""Export YOLO models to ONNX on the HOST (the Nano cannot run current Ultralytics).

Usage:  python src/export_models.py --models yolov8n yolov8s --sizes 640 416 320
Output: models/<model>_<size>.onnx   (opset 12: accepted by TensorRT 8.2 on JetPack 4.6)
Then copy models/*.onnx to the board and build engines with scripts/build_engines.sh.
"""
import argparse
import os
import shutil

from common import p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["yolov8n", "yolov8s"])
    ap.add_argument("--sizes", nargs="+", type=int, default=[640, 416, 320])
    ap.add_argument("--opset", type=int, default=12)
    a = ap.parse_args()
    from ultralytics import YOLO
    os.makedirs(p("models"), exist_ok=True)
    for m in a.models:
        pt = p("models/%s.pt" % m)
        model = YOLO(pt if os.path.exists(pt) else m + ".pt")
        if not os.path.exists(pt) and os.path.exists(m + ".pt"):
            shutil.move(m + ".pt", pt)
        for s in a.sizes:
            out = model.export(format="onnx", imgsz=s, opset=a.opset, simplify=True, dynamic=False, batch=1)
            dst = p("models/%s_%d.onnx" % (m, s))
            shutil.move(out, dst)
            print("exported", dst)


if __name__ == "__main__":
    main()
