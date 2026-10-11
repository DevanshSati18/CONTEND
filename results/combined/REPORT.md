# CONTEND combined results (all devices)

Generated 2026-10-09 20:56 by `analysis/report.py --combined` from `results/combined/*.csv`. Devices with platform_class dev/smoke are development data, never evidence.

## RQ1 per device x contention type

| Device | Class | Contention | Sessions | Runs/cell | Ratio | 95 % CI | Decision | Robust across sessions |
|---|---|---|---|---|---|---|---|---|
| dev_vm | dev | mem_high | 1 | 2 | 1.106 | [0.915, 1.330] | inconclusive | nan |
| smoke | smoke | mem_high | 1 | 2 | 1.005 | [0.997, 1.014] | H0: predictable (equivalent within margin) | nan |
| smoke | smoke | cpu | 1 | 2 | 1.002 | [0.994, 1.010] | H0: predictable (equivalent within margin) | nan |
| smoke_rs | smoke | mem_high | 2 | 6 | 1.009 | [0.986, 1.031] | H0: predictable (equivalent within margin) | True |
| smoke_rs | smoke | cpu | 2 | 6 | 0.978 | [0.957, 0.995] | H0: predictable (equivalent within margin) | True |

## RQ2 NMS-stage sensitivity

| Device | Contention | k clean | k adv | ratio | 95 % CI |
|---|---|---|---|---|---|
| dev_vm | mem_high | 2.700 | 2.521 | 0.934 | [0.796, 1.085] |
| smoke | mem_high | 3.125 | 3.023 | 0.967 | [0.925, 1.009] |
| smoke | cpu | 0.962 | 1.004 | 1.044 | [0.935, 1.159] |
| smoke_rs | mem_high | 2.584 | 2.699 | 1.044 | [0.878, 1.243] |
| smoke_rs | cpu | 1.152 | 0.898 | 0.779 | [0.649, 0.924] |

Files: `contend_runs.csv` (one row per run, all devices), `contend_rq1.csv`, `contend_rq2_nms.csv`, `contend_rq3.csv`, `fig5_cross_device_rq1.pdf`.
