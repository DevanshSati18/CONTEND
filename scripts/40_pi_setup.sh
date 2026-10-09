#!/usr/bin/env bash
# RASPBERRY PI (64-bit Raspberry Pi OS Bookworm). Run once after cloning:
#   bash scripts/40_pi_setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."
echo raspberry_pi > .device
cat /proc/device-tree/model; echo; uname -m
[ "$(uname -m)" = "aarch64" ] || { echo "Needs the 64-bit OS (aarch64)."; exit 1; }
sudo apt-get update
sudo apt-get install -y python3-venv python3-dev gcc git libopenblas-dev unzip curl
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu || pip install torch torchvision
pip install onnxruntime opencv-python-headless numpy pandas scipy statsmodels matplotlib pyyaml
python -c "import torch, torchvision, onnxruntime, cv2; print('torch', torch.__version__, 'tv', torchvision.__version__, 'ort', onnxruntime.__version__)"
python tests/test_nms.py 2>/dev/null || echo "(NMS parity test needs ultralytics; skipped on the Pi)"
gcc -O2 -o corunners/bwhog corunners/bwhog.c
gcc -O2 -o corunners/cpuhog corunners/cpuhog.c
./corunners/bwhog 256 4 | tail -1
echo "Pi ready. Next: bash scripts/01_download_coco.sh && python src/make_subset.py"
