#!/usr/bin/env bash
# ANY DEVICE (Jetson, Raspberry Pi, dev VM): gates G1 (on this device), G2, G3 pilot.
#   nohup bash scripts/20_run_all.sh > results/run_all_$(cat .device).log 2>&1 &
set -euo pipefail
source "$(dirname "$0")/_device.sh"
D="results/$DEVICE/derived"; mkdir -p "$D"
echo "== G1: workload on $DEVICE"
$PY src/check_workload.py --backend "$BACKEND" --model "$MODEL_FILE" --imgsz "$IMGSZ" --n 200 --out "$D/g1_device.json"
echo "== G2: contention validation"
$PY src/run_experiment.py g2 --model "$MODEL_FILE"
echo "== G3: pilot (3 runs per cell)"
$PY src/run_experiment.py pilot --model "$MODEL_FILE"
echo ">>> If G1-G3 PASS: set experiment.blocks runs_per_cell (see $D/g3.json) in"
echo ">>> configs/devices/$DEVICE.yaml, commit (pre-registration), then: bash scripts/21_main_blocks.sh"
