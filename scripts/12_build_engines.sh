#!/usr/bin/env bash
# JETSON NANO: build FP16 TensorRT engines from the host-exported ONNX files (~5-10 min each).
#   bash scripts/12_build_engines.sh            # all models/*.onnx
#   bash scripts/12_build_engines.sh models/yolov8n_416.onnx
set -euo pipefail
cd "$(dirname "$0")/.."
TRTEXEC=/usr/src/tensorrt/bin/trtexec
FILES="${*:-models/*.onnx}"
for onnx in $FILES; do
  eng="${onnx%.onnx}.engine"
  [ -f "$eng" ] && { echo "exists: $eng"; continue; }
  echo "== building $eng"
  $TRTEXEC --onnx="$onnx" --saveEngine="$eng" --fp16 --workspace=1024 2>&1 | tail -4
done
ls -la models/*.engine
