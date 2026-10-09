#!/usr/bin/env bash
# Summarise + analyse EVERY device folder under results/, then build the combined dataset.
# Run on any machine after `git pull` (each device pushes its own results/<device>/).
#   bash scripts/30_analyze.sh
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate
for dir in results/*/raw; do
  dev=$(basename "$(dirname "$dir")")
  echo "================ $dev"
  CONTEND_DEVICE=$dev python3 analysis/summarize.py
  CONTEND_DEVICE=$dev python3 analysis/analyze.py
done
python3 analysis/combine.py
echo "Per device: results/<device>/derived/results.json, results/<device>/figures/"
echo "Combined:   results/combined/contend_*.csv, fig5_cross_device_rq1.pdf"
