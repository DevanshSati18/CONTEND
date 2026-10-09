# Status checkpoint — 9 Oct 2026

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

## Next
1. Push to GitHub from the Mac Terminal:  cd ~/Development/CONTEND && gh repo create CONTEND --private --source . --push
2. Raspberry Pi: git clone; bash scripts/40_pi_setup.sh; bash scripts/01_download_coco.sh;
   python src/make_subset.py; bash scripts/41_pi_session.sh; python src/choose_config.py
   -> set model/imgsz/fps in configs/devices/raspberry_pi.yaml.
3. Pi gates: nohup bash scripts/20_run_all.sh > results/gates_raspberry_pi.log 2>&1 &
   then fill docs/PREREGISTRATION.md, commit, nohup bash scripts/21_main_blocks.sh > results/main_raspberry_pi.log 2>&1 &
4. Optional: finish the dev E1 dry run (resumes):
   CONTEND_DEVICE=dev_vm python src/run_experiment.py block E1 --runs 6 --model models/yolov8n_320.onnx
5. Jetson Nano when it arrives: README "Step 4" onward (scripts/10_nano_setup.sh).
6. After devices push results: bash scripts/30_analyze.sh -> results/combined/.

Watch item: check whether the inference-stage excess under memory contention reproduces on the
Pi/Jetson with randomised order (RQ2 per-stage table: rq2_stage_sensitivity.csv, not only NMS).
