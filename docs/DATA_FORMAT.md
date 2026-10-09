# CONTEND data format (schema 1.0)

Every device writes into its own folder, so the Jetson, the Raspberry Pi and the dev VM can push to
the same git repo without conflicts. Results are committed to git (gzip CSV, a few MB per device).

```
results/
├── <device_id>/                 jetson_nano | raspberry_pi | dev_vm
│   ├── raw/
│   │   ├── runs.csv             manifest: one row per run (all blocks, incl. failed/interrupted)
│   │   ├── env_<timestamp>.json software/hardware versions + full resolved config per session
│   │   └── <block>/<run_id>.csv.gz      per-frame records (table A)
│   │       <block>/<run_id>_logs/       co-runner throughput logs, tegrastats (Jetson)
│   ├── derived/
│   │   ├── g1_device.json g2.json g3.json choose_config.json   gate results
│   │   ├── run_summary.csv      one row per run (table B) - the experimental unit
│   │   ├── results.json         RQ1/RQ2/RQ3 statistics for this device
│   │   ├── rq2_stage_sensitivity.csv  rq3_deadlines.csv
│   └── figures/                 fig2-fig4 for this device
└── combined/                    built by analysis/combine.py from all devices
    ├── contend_runs.csv         table B for all devices + device_label, platform_class, deadline_ms
    ├── contend_rq1.csv contend_rq2_nms.csv contend_rq3.csv
    └── fig5_cross_device_rq1.pdf
```

`run_id` = `<block>_<cell>_<contention>_r<rep>`; it is unique within a device. The global key is
(`device_id`, `run_id`).

## Table A — per-frame record (`raw/<block>/<run_id>.csv.gz`)

| Column | Unit | Meaning |
|---|---|---|
| schema_version | – | "1.0" |
| device_id, backend, precision | – | e.g. raspberry_pi, ort, fp32 |
| imgsz, fps_target | px, frames/s | network input size; deadline = 1000/fps_target ms |
| run_id, block, cell | – | cell A clean/none, B adv/none, C clean/load, D adv/load |
| input | clean / adv | |
| contention | none / mem_high / mem_low / cpu | co-runner active in THIS run |
| mode | closed / open | closed = back-to-back; open = fixed-rate release, bounded queue |
| model | – | model file name |
| frame_idx, image_idx | – | frame order in the run; index into data/subset/image_ids.txt |
| t_release_ms, t_start_ms, t_done_ms | ms from run start | release, service start, completion |
| t_pre_ms, t_inf_ms, t_nms_ms, t_d2h_ms | ms | stage times (synchronised) |
| service_ms | ms | t_done − t_start |
| latency_ms | ms | t_done − t_release (includes queueing in open mode) |
| n_pre_nms, n_nms_in, n_post_nms | count | candidates above conf; boxes into the NMS kernel; detections out |
| time_limit_hit | 0/1 | Ultralytics NMS time guard exceeded |
| queue_depth | frames | queue length when the frame started |
| dropped | 0/1 | frame dropped by the bounded queue (counts as a deadline miss) |
| soc_temp_c | °C | hottest thermal zone, sampled every 100 frames |

## Table B — per-run summary (`derived/run_summary.csv`, `combined/contend_runs.csv`)

Device fields as in table A, plus: 5 %-trimmed means of every stage, `service_ms`, `latency_ms`,
`other_ms` (service minus stages), `lat_p50/p95/p99`, `n_pre_nms`, `n_post_nms`, `frames`,
`miss_rate` (latency > deadline or dropped), `drop_rate`, `max_consec_miss`, `tard_p95_ms`,
`queue_mean`, `temp_mean_c`, `time_limit_hits`, `order` (randomised position), `start_temp_c`,
and `contention` = the block's co-runner type.

## Rules
* Never edit files in `raw/`. Re-analysis always starts from `raw/`.
* A device's numbers are only comparable within that device (different deadline and backend);
  the cross-device comparison is of RQ1 ratios and decisions (`contend_rq1.csv`), not of milliseconds.
* `platform_class: dev` rows (dev VM) are development data and never used as evidence in the paper.
