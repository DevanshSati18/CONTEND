#!/usr/bin/env bash
# HOST PC: subset -> ONNX exports -> frame caches -> universal perturbations -> gate G1 (workload).
# Generates candidates for every model x size, because the Nano picks the final one in P0.
#   source .venv/bin/activate && bash scripts/02_host_prepare.sh
set -euo pipefail
[ -f .venv/bin/activate ] && source .venv/bin/activate
cd "$(dirname "$0")/.."
MODELS="${MODELS:-yolov8n yolov8s}"
SIZES="${SIZES:-320 416 640}"

python src/make_subset.py
python src/export_models.py --models $MODELS --sizes $SIZES
for s in $SIZES; do
  python -c "import sys; sys.path.insert(0,'src'); from common import *; c=load_config(); print(frame_cache(c,$s).shape)"
done
for m in $MODELS; do
  for s in $SIZES; do
    python src/attack_uap.py --model models/$m.pt --imgsz $s
    # G1 on the host: the candidate count is a property of model + input, not of the hardware
    python src/check_workload.py --backend torch --model models/$m.pt --imgsz $s \
        --delta perturbations/${m}_$s.npy --n 200 --out results/host_prep/g1_host_${m}_$s.json
  done
done
echo "Copy to the Nano: models/*.onnx, perturbations/, data/subset/ (frames_*.npy, *.txt)"
