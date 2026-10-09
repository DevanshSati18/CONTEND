# Status checkpoint — 10 Oct 2026

## Done (dev_vm = Linux arm64 VM on the Mac; development data only, never evidence)
- Repo + multi-device pipeline committed; data format in docs/DATA_FORMAT.md.
- COCO val2017 in data/coco (git-ignored); seeded subset: 500 eval / 200 attack images.
- ONNX: yolov8n 320/416/640 (committed); yolov8s 320/416/640 (local only, git-ignored).
- Perturbations: yolov8n_320, yolov8n_416 (committed; sha256 in the .json files).
- G1 (dev_vm, ORT, yolov8n_320, 200 held-out images): candidates x22.1, NMS +1.55 ms (95% CI 1.38-1.70) -> PASS.
- G2 (dev_vm): mem_low +23.8 % PASS; mem_high +50.8 % PASS; cpu (control) +9.1 %.
- G3 pilot (dev_vm, 3 runs/cell, fixed order): A 16.05, B 16.87, C 27.07, D 33.60 ms; worst CV 6.8 %.
  Stage model: measured/predicted D = 1.168 (95% CI 1.077-1.264). The excess is in the INFERENCE
  stage (C 24.1 ms vs D 29.0 ms); the NMS-stage ratio is ~1.02. Preliminary: VM, 3 runs, fixed order.
- E1 dry run (randomised): 13 of 24 runs done; resumable.
## Added 10 Oct
- `runscript.py`: one command per device (`--pi`, `--nvidia`, `--dev`), resumable; `--compile` on the Mac.
- Sessions: every block repeated in N independent sessions (default 3); repeat-aware statistics
  (per-session ratios, between-session SD, two-stage bootstrap, robust flag) in results.json and REPORT.md.
- Smoke test step 23 runs runscript.py end to end with 2 sessions (verified on the dev VM).

## Next
1. Mac: `python3 runscript.py --smoke` (expect 23/23 PASS), commit, then
   `gh repo create CONTEND --private --source . --push`.
2. Raspberry Pi: `git clone ...; cd CONTEND; tmux new -s contend; python3 runscript.py --pi --push-each`.
3. Jetson Nano (after flashing + torch wheel in ~/wheels): same with `--nvidia`.
4. Mac: `python3 runscript.py --compile` -> results/combined/REPORT.md; fill the paper TODOs.


Watch item: check whether the inference-stage excess under memory contention reproduces on the
Pi/Jetson with randomised order (RQ2 per-stage table: rq2_stage_sensitivity.csv, not only NMS).
