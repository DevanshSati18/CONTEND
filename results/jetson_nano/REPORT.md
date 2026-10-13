# CONTEND results: jetson_nano

_NVIDIA Jetson Nano 4GB (Maxwell GPU, 4x Cortex-A57, shared LPDDR4)_  
Generated 2026-10-14 00:14 by `analysis/report.py` (re-run after every instance; numbers come from `derived/results.json`).

## Configuration (locked on this device)

| Item | Value |
|---|---|
| Backend / precision | trt / fp16 |
| Model / input size | yolov8n / 416 |
| Deadline | 15 FPS (66.7 ms); clean latency 36.66 ms = 55 % of the deadline |
| 50-70 % rule met | True |
| Runs per condition per instance | n/a |
| Frozen (pre-registration) | not yet (git ?) |

## Instances (each = gates + every condition once)

| Instance | Status | G1 candidates | G1 NMS increase (95% CI) | G2 memory high / low | CPU control | G3 worst CV |
|---|---|---|---|---|---|---|
| instance1 | in progress | - | - | - / - | - | - |
| instance2 | in progress | - | - | - / - | - | - |
| instance3 | in progress | - | - | - / - | - | - |
| instance4 | in progress | - | - | - / - | - | - |

## Files

- `instance<k>/raw/<block>/<run_id>.csv.gz` per-frame logs; `instance<k>/derived/` that instance's gates
- `derived/run_summary.csv` one row per run; `derived/results.json` all statistics
- `figures/fig2..fig4` (PDF + PNG)
