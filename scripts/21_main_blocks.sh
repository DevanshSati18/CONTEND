#!/usr/bin/env bash
# ANY DEVICE: main factorial blocks. Resumable (finished runs are skipped).
#   nohup bash scripts/21_main_blocks.sh > results/main_$(cat .device).log 2>&1 &
set -euo pipefail
source "$(dirname "$0")/_device.sh"
for B in ${BLOCKS:-E1 E2 E3}; do
  echo "== $DEVICE block $B  $(date)"
  $PY src/run_experiment.py block "$B" --model "$MODEL_FILE"
done
echo "== done $(date). Next: bash scripts/30_analyze.sh && git add results/$DEVICE && git commit && git push"
