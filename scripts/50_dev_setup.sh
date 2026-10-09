#!/usr/bin/env bash
# DEV VM / any Linux or macOS dev machine: full toolchain incl. ultralytics (export + attack).
#   bash scripts/50_dev_setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .device ] || echo dev_vm > .device
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision
pip install -r requirements-host.txt
python tests/test_nms.py
gcc -O2 -o corunners/bwhog corunners/bwhog.c && gcc -O2 -o corunners/cpuhog corunners/cpuhog.c
