#!/usr/bin/env bash
# Sourced by the run scripts: exports DEVICE, BACKEND, EXT, MODEL_NAME, IMGSZ, MODEL_FILE, PY
# from configs/contend.yaml + configs/devices/<device>.yaml (device = .device file).
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi
PY=$(command -v python3)
eval "$($PY - <<'PYEOF'
import sys; sys.path.insert(0, 'src')
from common import load_config
c = load_config()
r, m = c['runtime'], c['model']
print('export DEVICE=%s BACKEND=%s EXT=%s MODEL_NAME=%s IMGSZ=%s' % (c['device_id'], r['backend'], r['model_ext'], m['name'], m['imgsz']))
PYEOF
)"
export MODEL_FILE="models/${MODEL_NAME}_${IMGSZ}.${EXT}"
export PY
echo "== device=$DEVICE backend=$BACKEND model=$MODEL_FILE"
