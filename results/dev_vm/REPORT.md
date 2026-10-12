# CONTEND results: dev_vm

_Linux arm64 VM on Apple-silicon Mac (development only)_  
Generated 2026-10-09 20:56 by `analysis/report.py` (re-run after every session; numbers come from `derived/results.json`).

## Gates

| Gate | Result |
|---|---|
| G2 cpu | FAIL: clean 16.44 -> loaded 17.95 ms (+9.1 %) |
| G2 mem_high | PASS: clean 16.40 -> loaded 24.72 ms (+50.8 %) |
| G2 mem_low | PASS: clean 15.92 -> loaded 19.71 ms (+23.8 %) |
| G3 pilot | worst between-run CV 6.77 % -> 84 runs per cell needed |

## Runs completed (block x session)

| Block | S1 |
|---|---|
| E1 | 13 |

## RQ1: is the combined latency predictable? (measured / predicted latency of cell D)

Margin ±5 %. Pooled CI = run-level bootstrap (pre-registered). Two-stage CI also resamples whole sessions; *robust* = both give the same decision.

| Contention | Sessions | Runs/cell | Ratio | Pooled CI | Decision | Per-session ratios | Between-session SD | Two-stage CI | Robust |
|---|---|---|---|---|---|---|---|---|---|
| Memory BW (high) | 1 | 2 | 1.106 | [0.915, 1.330] | inconclusive | 1.106 | n/a | n/a | n/a (1 session) |

Additive interaction I_T (reported for contrast only; positive by construction under proportional slowdown): mem_high +4.02 ms

## RQ2: per-stage contention sensitivity (adversarial k / clean k)

| Contention | Stage | k clean | k adv | ratio | 95 % CI |
|---|---|---|---|---|---|
| mem_high | t_pre_ms | 2.399 | 3.100 | 1.292 | [1.093, 1.514] |
| mem_high | t_inf_ms | 1.795 | 2.028 | 1.130 | [0.920, 1.367] |
| mem_high | t_nms_ms | 2.700 | 2.521 | 0.934 | [0.796, 1.085] |

## Files

- `raw/<block>/<run_id>.csv.gz` per-frame logs; `raw/runs.csv` run manifest
- `derived/run_summary.csv` one row per run; `derived/results.json` all statistics
- `figures/fig2..fig4` (PDF + PNG)
