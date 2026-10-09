#!/usr/bin/env bash
# HOST PC (Linux/WSL with an NVIDIA GPU). Creates the Python env used for data prep,
# perturbation generation and analysis.   Run from the project root:  bash scripts/00_host_setup.sh
set -euo pipefail
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# PyTorch with CUDA: pick the wheel matching your driver from https://pytorch.org/get-started/locally/
pip install torch torchvision
pip install -r requirements-host.txt
python -c "import torch; print('torch', torch.__version__, 'CUDA available:', torch.cuda.is_available())"
gcc -O2 -o corunners/bwhog corunners/bwhog.c
gcc -O2 -o corunners/cpuhog corunners/cpuhog.c
echo "host env ready: source .venv/bin/activate"
