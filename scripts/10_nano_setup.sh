#!/usr/bin/env bash
# JETSON NANO (JetPack 4.6.x, Ubuntu 18.04, Python 3.6). Run once after flashing:
#   bash scripts/10_nano_setup.sh
# TensorRT and OpenCV Python bindings ship with JetPack. PyTorch must be NVIDIA's Jetson wheel.
set -euo pipefail
cd "$(dirname "$0")/.."
echo jetson_nano > .device

echo "== platform"; cat /etc/nv_tegra_release; cat /proc/device-tree/model; echo

sudo apt-get update
sudo apt-get install -y python3-pip python3-yaml python3-pandas python3-scipy libopenblas-base \
     libopenmpi-dev libjpeg-dev zlib1g-dev git htop

# --- PyTorch 1.10 for JetPack 4.6 (Python 3.6) ------------------------------------------
# Download torch-1.10.0-cp36-cp36m-linux_aarch64.whl from NVIDIA's "PyTorch for Jetson"
# forum thread (JetPack 4.6 section) into ~/wheels/, then:
if ! python3 -c "import torch" 2>/dev/null; then
  pip3 install --user numpy==1.19.4 "$HOME"/wheels/torch-1.10.0-cp36-cp36m-linux_aarch64.whl
fi
# --- torchvision 0.11.1 built against that torch (CUDA NMS kernel); takes ~45-60 min ------
if ! python3 -c "import torchvision" 2>/dev/null; then
  git clone --depth 1 --branch v0.11.1 https://github.com/pytorch/vision ~/torchvision
  (cd ~/torchvision && export BUILD_VERSION=0.11.1 && python3 setup.py install --user)
fi
python3 -c "import torch, torchvision, tensorrt, cv2; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), '| torchvision', torchvision.__version__, '| TRT', tensorrt.__version__, '| cv2', cv2.__version__)"
python3 -c "import torch, torchvision; b=torch.rand(100,4).cuda()*100; b[:,2:]+=b[:,:2]; print('CUDA NMS ok:', torchvision.ops.nms(b, torch.rand(100).cuda(), 0.5).numel()>0)"

gcc -O2 -o corunners/bwhog corunners/bwhog.c
gcc -O2 -o corunners/cpuhog corunners/cpuhog.c
echo "== co-runner sanity (5 s each)"
./corunners/bwhog 256 5 | tail -1
./corunners/cpuhog 3 | tail -1
echo "setup done. Before EVERY session run: bash scripts/11_nano_session.sh"
