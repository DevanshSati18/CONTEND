#!/usr/bin/env bash
# SMOKE TEST: run EVERY stage of CONTEND end to end with tiny settings (~5-10 min on a laptop /
# Pi CPU), so we know the code works before the full runs. Isolated: writes only to
# data/coco128, data/subset_smoke, models_smoke, perturbations_smoke, results/smoke.
#   bash scripts/smoke_test.sh            # all stages
#   KEEP=1 bash scripts/smoke_test.sh     # keep previous smoke outputs (default: start clean)
# Needs: python3 with the host requirements (ultralytics only for steps 3-4; on a device without
# it, copy models_smoke/ and perturbations_smoke/ from a machine that has it).
set -uo pipefail
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate
export CONTEND_DEVICE=smoke PYTHONWARNINGS=ignore
PY=python3
LOG=results/smoke/smoke.log
[ "${KEEP:-0}" = 1 ] || rm -rf results/smoke data/subset_smoke perturbations_smoke
mkdir -p results/smoke
: > "$LOG"
declare -a NAMES STATUS
step() {   # step <name> <command...>
  local name="$1"; shift
  local t0=$(date +%s)
  printf '\n=== %s\n' "$name" | tee -a "$LOG"
  if "$@" >>"$LOG" 2>&1; then st=PASS; else st=FAIL; fi
  NAMES+=("$name"); STATUS+=("$st ($(( $(date +%s) - t0 ))s)")
  printf '%s  %s\n' "$st" "$name"
  tail -n 3 "$LOG" | sed 's/^/    /'
}
has_ultra() { $PY -c "import ultralytics" 2>/dev/null; }

# 0. data: 128 COCO images from the Ultralytics release (no full COCO needed)
if [ ! -d data/coco128/images/train2017 ]; then
  mkdir -p data && curl -sSL -o data/coco128.zip https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip \
    && (cd data && unzip -q -o coco128.zip && rm coco128.zip)
fi
[ -f models/yolov8n.pt ] || curl -sSL -o models/yolov8n.pt https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt
gcc -O2 -o corunners/bwhog corunners/bwhog.c && gcc -O2 -o corunners/cpuhog corunners/cpuhog.c

if has_ultra; then step "1  NMS parity vs Ultralytics"     $PY tests/test_nms.py; fi
step "2  seeded subset (32 eval / 32 attack)"                $PY src/make_subset.py
if has_ultra; then
  step "3  ONNX export (yolov8n 320)"                         $PY src/export_models.py --models yolov8n --sizes 320 --outdir models_smoke
  step "4  attack perturbation (20 steps)"                    $PY src/attack_uap.py --model models/yolov8n.pt --imgsz 320
fi
M=models_smoke/yolov8n_320.onnx
step "5  frame cache"                                         $PY -c "import sys; sys.path.insert(0,'src'); from common import *; print(frame_cache(load_config(),320).shape)"
step "6  G1 workload check (result not judged)"               $PY src/check_workload.py --backend ort --model $M --imgsz 320 --n 32 --out results/smoke/derived/g1.json
step "7  choose_config (50-70% rule)"                         $PY src/choose_config.py --models "$M" --frames 30
step "8  single harness trial, closed loop"                   $PY src/harness.py --model $M --imgsz 320 --mode closed --input adv --frames 40 --out results/smoke/raw/manual/closed.csv
step "9  single harness trial, open loop + queue"             $PY src/harness.py --model $M --imgsz 320 --mode open --input adv --seconds 2 --out results/smoke/raw/manual/open.csv
step "10 co-runner validation (bw + cpu, 3 s)"                bash -c "$PY src/corunner.py --name mem_high --seconds 3 && $PY src/corunner.py --name cpu --seconds 3"
step "11 G2 contention gate"                                  $PY src/run_experiment.py g2 --model $M
step "12 G3 pilot (2 runs/cell)"                              $PY src/run_experiment.py pilot --runs 2 --model $M
step "13 block E1 closed / mem_high (randomised)"             $PY src/run_experiment.py block E1 --model $M
step "14 block E2 closed / cpu"                               $PY src/run_experiment.py block E2 --model $M
step "15 block E3 open-loop 30 FPS / mem_high"                $PY src/run_experiment.py block E3 --model $M
step "16 resume: re-run E1 skips finished runs"               bash -c "$PY src/run_experiment.py block E1 --model $M | grep -c 'skip (done)'"
step "17 summarize per-run table"                             $PY analysis/summarize.py
step "18 analyze RQ1-RQ3 + figures"                           $PY analysis/analyze.py
step "19 analysis validation: proportional -> H0"             bash -c "$PY analysis/simulate.py --scenario h0 --out results/smoke/sim_h0 --runs 6 --frames 200 && $PY analysis/summarize.py --raw results/smoke/sim_h0/raw --out results/smoke/sim_h0/s.csv && $PY analysis/analyze.py --summary results/smoke/sim_h0/s.csv --outdir results/smoke/sim_h0 --figdir results/smoke/sim_h0/fig --nboot 300 | grep -q 'H0: predictable'"
step "20 analysis validation: planted +40% NMS -> H1"         bash -c "$PY analysis/simulate.py --scenario h1 --out results/smoke/sim_h1 --runs 6 --frames 200 && $PY analysis/summarize.py --raw results/smoke/sim_h1/raw --out results/smoke/sim_h1/s.csv && $PY analysis/analyze.py --summary results/smoke/sim_h1/s.csv --outdir results/smoke/sim_h1 --figdir results/smoke/sim_h1/fig --nboot 300 | grep -q 'H1: amplified'"
step "21 combine all devices (smoke excluded from fig)"       $PY analysis/combine.py
step "22 output files present + schema check"                $PY tests/check_outputs.py results/smoke

{
  echo "# CONTEND smoke test — $(date '+%Y-%m-%d %H:%M')"
  echo; echo "Host: $(uname -srm), Python $($PY -c 'import sys;print(sys.version.split()[0])')"
  echo "Tiny settings (configs/devices/smoke.yaml). Numbers are meaningless; only PASS/FAIL matters."
  echo; echo "| Step | Result |"; echo "|---|---|"
  for i in "${!NAMES[@]}"; do echo "| ${NAMES[$i]} | ${STATUS[$i]} |"; done
} > results/smoke/SMOKE_REPORT.md
echo; cat results/smoke/SMOKE_REPORT.md
grep -q "| FAIL" results/smoke/SMOKE_REPORT.md && { echo "SMOKE TEST: FAILURES (see $LOG)"; exit 1; } || echo "SMOKE TEST: ALL PASSED"
