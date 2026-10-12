# CONTEND results: jetson_nano

_NVIDIA Jetson Nano 4GB (Maxwell GPU, 4x Cortex-A57, shared LPDDR4)_  
Generated 2026-10-12 19:09 by `analysis/report.py` (re-run after every session; numbers come from `derived/results.json`).

## Configuration (locked on this device)

| Item | Value |
|---|---|
| Backend / precision | trt / fp16 |
| Model / input size | yolov8n / 416 |
| Deadline | 15 FPS (66.7 ms); clean latency 36.58 ms = 55 % of the deadline |
| 50-70 % rule met | True |
| Sessions planned | n/a |
| Runs per cell per session |  |
| Frozen (pre-registration) | not yet (git ?) |

## Gates

| Gate | Result |
|---|---|
| G1 workload | FAIL: candidates x32.3, NMS +-4.69 ms (95% CI -15.13 to 1.10) |
| G2 cpu | CONTROL: clean 37.56 -> loaded 36.88 ms (-1.8 %) |
| G2 mem_high | PASS: clean 37.53 -> loaded 51.21 ms (+36.5 %) |
| G2 mem_low | PASS: clean 38.03 -> loaded 45.42 ms (+19.4 %) |
| G3 pilot | worst between-run CV 0.94 % -> 6 runs per cell needed |

## Files

- `raw/<block>/<run_id>.csv.gz` per-frame logs; `raw/runs.csv` run manifest
- `derived/run_summary.csv` one row per run; `derived/results.json` all statistics
- `figures/fig2..fig4` (PDF + PNG)
