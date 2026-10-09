# CONTEND
**CONTention × adversarial Evaluation of NMS Deadlines**

Does background resource contention *amplify* NMS latency attacks on an edge SoC, or does it just
slow everything proportionally? CONTEND runs a controlled 2×2 factorial experiment
(clean/adversarial input × no-load/co-runner) on an NVIDIA Jetson Nano, predicts the combined
latency from isolated runs with a stage-wise proportional-slowdown model, and tests the
prediction with a pre-registered ±5 % equivalence margin (frozen objective v1.0, 9 Oct 2026).

Target: MoSICom 2026 — submission deadline **14 Oct 2026**.
Paper: *Does Resource Contention Amplify NMS Latency Attacks? A Controlled Factorial Study on Edge Devices*

## 0. Multi-device workflow (Raspberry Pi + Jetson Nano, one repo)

Each device clones the same repo, declares who it is in a git-ignored `.device` file, and writes
only to `results/<device>/`. Results are committed and pushed; any machine then merges them.

| Device | `.device` | Backend | NMS | Setup | Per-session |
|---|---|---|---|---|---|
| Raspberry Pi (64-bit OS) | `raspberry_pi` | ONNX Runtime FP32 (CPU) | CPU | `scripts/40_pi_setup.sh` | `scripts/41_pi_session.sh` |
| Jetson Nano (JetPack 4.6) | `jetson_nano` | TensorRT FP16 (GPU) | GPU | `scripts/10_nano_setup.sh` | `scripts/11_nano_session.sh` |
| Mac dev VM (arm64 Linux) | `dev_vm` | ONNX Runtime FP32 | CPU | `scripts/50_dev_setup.sh` | – (development only, never evidence) |

Device-specific settings (backend, cores, model size, deadline) live in `configs/devices/<device>.yaml`
and override `configs/contend.yaml`. The data format is defined in `docs/DATA_FORMAT.md`.

```bash
# on every device
git clone <repo-url> CONTEND && cd CONTEND
bash scripts/40_pi_setup.sh            # or 10_nano_setup.sh on the Jetson
bash scripts/01_download_coco.sh && python3 src/make_subset.py   # identical seeded subset everywhere
python3 src/choose_config.py           # 50-70 % rule -> edit configs/devices/<device>.yaml
nohup bash scripts/20_run_all.sh  > results/gates_$(cat .device).log 2>&1 &   # G1-G3
nohup bash scripts/21_main_blocks.sh > results/main_$(cat .device).log 2>&1 &  # E1-E3
git add results/$(cat .device) configs/devices && git commit -m "results: $(cat .device)" && git pull --rebase && git push
# on any machine, after pulling everyone's results
bash scripts/30_analyze.sh             # per-device stats + results/combined/ + fig5
```
Shared assets made once on the dev machine and committed: `data/subset/*.txt`,
`models/*.onnx` (yolov8n), `perturbations/*.npy` (+ sha256 json). The Pi and Jetson rebuild the
frame caches from their own COCO copy using the same image IDs.


---

## 1. Project structure

```
CONTEND/
├── README.md                    this file: structure, commands, 6-day plan
├── requirements-host.txt        Python packages for the host PC
├── configs/
│   └── contend.yaml             single source of truth (model, NMS, attack, co-runners, blocks)
├── scripts/                     run in numeric order
│   ├── 00_host_setup.sh         HOST  venv + packages + compile co-runners
│   ├── 01_download_coco.sh      HOST  COCO val2017 images + annotations
│   ├── 02_host_prepare.sh       HOST  subset, ONNX export, frame caches, perturbations, gate G1
│   ├── 10_nano_setup.sh         NANO  PyTorch/torchvision, checks, co-runners (once)
│   ├── 11_nano_session.sh       NANO  MAXN, locked clocks, fan (before EVERY session)
│   ├── 12_build_engines.sh      NANO  ONNX -> FP16 TensorRT engines
│   ├── 20_run_all.sh            NANO  gates G1 (board), G2, G3 pilot
│   ├── 21_main_blocks.sh        NANO  main factorial blocks E1-E3 (E4 optional)
│   └── 30_analyze.sh            HOST  copy results, statistics, figures
├── src/
│   ├── common.py                config, letterbox, frame cache, core pinning, env capture
│   ├── nms.py                   Ultralytics-identical NMS (runs on Python 3.6 / torch 1.10)
│   ├── backends.py              TensorRT / ONNX Runtime / PyTorch with identical stage timing
│   ├── harness.py               closed-loop and 30 FPS open-loop frame replay, per-frame CSV
│   ├── make_subset.py           seeded 500-image eval set + disjoint 200-image attack set
│   ├── export_models.py         YOLO -> ONNX (opset 12, TensorRT 8.2 compatible)
│   ├── attack_uap.py            universal perturbation, Phantom Sponges objective [re-implementation]
│   ├── check_workload.py        gate G1: candidates x10 and NMS-time increase
│   ├── choose_config.py         P0 rule: clean latency at 50-70 % of the deadline
│   ├── corunner.py              start/stop/validate co-runners pinned to spare cores
│   └── run_experiment.py        gates G2/G3 and randomised, resumable factorial blocks
├── corunners/
│   ├── bwhog.c                  memory-bandwidth co-runner (prints GB/s)
│   └── cpuhog.c                 cache-resident CPU co-runner, the control (prints Mit/s)
├── analysis/
│   ├── summarize.py             per-run summaries (the run is the unit, not the frame)
│   ├── analyze.py               RQ1 prediction test, RQ2 per-stage k, RQ3 deadlines, figures
│   └── simulate.py              synthetic logs with known answers (validates the analysis)
├── tests/test_nms.py            vendored NMS == Ultralytics NMS (20/20 identical)
├── paper/
│   ├── main.tex                 IEEE conference draft (Intro, Related Work, Method written)
│   └── refs.bib                 21 references (entries marked VERIFY need checking)
├── docs/
│   ├── PREREGISTRATION.md       fill after the pilot, commit before main runs
│   ├── CHANGELOG.md             every deviation from the plan, dated
│   └── session_log.md           room/board temperature per session
├── data/coco/                   COCO val2017 (downloaded)          [git-ignored]
├── data/subset/                 image_ids.txt, attack_ids.txt, frames_<size>.npy
├── models/                      .pt (host), .onnx (host -> Nano), .engine (Nano)
├── perturbations/               <model>_<size>.npy + .json (sha256, training curve)
├── results/raw/<block>/         per-frame CSV per run + co-runner/tegrastats logs, runs.csv
├── results/derived/             g1/g2/g3 gate JSON, run_summary.csv, results.json
└── figures/                     fig2 (RQ1), fig3 (RQ2), fig4 (RQ3) as PDF + PNG
```

## 2. What the experiment measures

| Cell | Input | Co-runner | Role |
|---|---|---|---|
| A | clean | none | baseline |
| B | adversarial | none | attack alone |
| C | clean | yes | contention alone |
| D | adversarial | yes | combined (tested) |

* **RQ1** `R = log(T_D / T̂_D)` with `T̂_D = Σ_s k_s · T_s(B)` and `k_s = T_s(C) / T_s(A)`.
  H0 *predictable* if the TOST CI of R is inside ±log(1.05); H1 *amplified* if the CI is above +log(1.05).
  The naive additive interaction `T_D − T_B − T_C + T_A` is reported only for contrast: it is
  positive by construction whenever contention slows everything proportionally.
* **RQ2** per-stage `k_adv = T_s(D)/T_s(B)` vs `k_clean`, memory-bandwidth vs CPU co-runner.
* **RQ3** 30 FPS open-loop: miss rate, absolute miss increase (D−C vs B−A), drops, max consecutive misses.

| Block | Mode | Co-runner | Runs/cell | Approx. board time |
|---|---|---|---|---|
| E1 | closed | mem_high (2× bwhog) | 8 | ~2.5 h |
| E2 | closed | cpu (2× cpuhog) | 8 | ~2.5 h |
| E3 | open, 30 FPS, 5 min | mem_high | 4 | ~2 h |
| E4 (optional) | closed | mem_low (1× bwhog) | 6 | ~1.5 h |

## 3. Step-by-step commands

### Step 1 — Host PC setup (now, ~15 min)
```bash
cd ~/CONTEND
bash scripts/00_host_setup.sh
source .venv/bin/activate
python tests/test_nms.py                # expect: NMS parity: 20/20 trials identical
```

### Step 2 — Dataset (now, ~10-20 min download)
```bash
bash scripts/01_download_coco.sh        # expect: images: 5000
```

### Step 3 — Host preparation + gate G1 on the host (~1-2 h with a GPU)
```bash
bash scripts/02_host_prepare.sh
cat results/derived/g1_host_*.json | grep -E "candidate_ratio|G1_pass"
```
G1 (host) must show `candidate_ratio >= 10`. The candidate count is a property of the model and
input, so a host failure means it will also fail on the board.

### Step 4 — Flash and set up the Nano (when the board arrives, ~2 h, mostly waiting)
1. Flash **JetPack 4.6.1** (Jetson Nano Developer Kit SD image) with balenaEtcher; boot, create user.
2. Fit the J48 jumper and use the 5 V / 4 A barrel supply; attach the fan; connect Ethernet.
3. Download NVIDIA's `torch-1.10.0-cp36-cp36m-linux_aarch64.whl` ("PyTorch for Jetson" forum
   thread, JetPack 4.6 section) into `~/wheels/` on the Nano.
4. Copy the project to the Nano (from the host):
```bash
rsync -av --exclude .venv --exclude data/coco --exclude 'models/*.pt' ~/CONTEND/ <user>@<nano-ip>:~/CONTEND/
```
5. On the Nano:
```bash
cd ~/CONTEND
bash scripts/10_nano_setup.sh           # torchvision build takes ~45-60 min: start it and wait
```

### Step 5 — Engines and configuration choice (Nano, ~1 h)
```bash
bash scripts/11_nano_session.sh         # before EVERY session
bash scripts/12_build_engines.sh        # all models/*.onnx -> .engine
python3 src/choose_config.py --backend trt --engines "models/*.engine"
```
Pick the row marked `USE`; write `model.name`, `model.imgsz` and `deadline.fps` into
`configs/contend.yaml`.

### Step 6 — Gates on the board (~1.5 h)
```bash
nohup bash scripts/20_run_all.sh > results/run_all.log 2>&1 &
tail -f results/run_all.log
```
* **G1** `G1 PASS: candidates x.., NMS +.. ms` (CI above zero). If it FAILS on GPU NMS: set
  `nms.device: cpu` in the config (pre-planned fallback) and re-run; record it in `docs/CHANGELOG.md`.
* **G2** each co-runner `PASS` (≥ 10 % slower clean latency). If mem fails: raise `instances`/`buffer_mb`.
* **G3** prints the runs per cell needed. Use max(that, 6) within the time you have.

### Step 7 — Pre-register (10 min)
Fill `docs/PREREGISTRATION.md`, set `runs_per_cell` in the config, then
```bash
git init && git add -A && git commit -m "Pre-registration: CONTEND v1.0"
```

### Step 8 — Main experiments (Nano, overnight, ~7 h)
```bash
bash scripts/11_nano_session.sh
nohup bash scripts/21_main_blocks.sh > results/main.log 2>&1 &
# optional when time allows:  BLOCKS=E4 nohup bash scripts/21_main_blocks.sh > results/e4.log 2>&1 &
```
Resumable: if anything stops, run the same command again; finished runs are skipped.

### Step 9 — Analysis (host, ~10 min)
```bash
NANO=<user>@<nano-ip> bash scripts/30_analyze.sh
```
Prints RQ1 decisions, RQ2 NMS sensitivity and RQ3 deadline effects; writes
`results/derived/results.json` and `figures/fig2..fig4.pdf`.

### Step 10 — Paper
Upload `paper/` and `figures/*.pdf` to Overleaf (IEEE conference template is built in),
replace every red `\TODO{...}` with numbers from `results.json`, verify the `VERIFY` references.

## 4. Six-day plan to 14 October

| Day | Date | Goal | Done when |
|---|---|---|---|
| 0 | Fri 9 Oct | Steps 1-3 on the host; board arrives: Step 4 (flash, start torchvision build) | G1 host passes; Nano boots with torch + torchvision |
| 1 | Sat 10 Oct | Steps 5-7: engines, config choice, G1/G2/G3 on board, pre-register; start Step 8 overnight | gates pass; E1-E3 running |
| 2 | Sun 11 Oct | Main runs finish; Step 9 analysis; E4 overnight if time | results.json + figures exist |
| 3 | Mon 12 Oct | Write Method, Setup, Results; insert figures | all TODOs except Discussion filled |
| 4 | Tue 13 Oct | Discussion, Abstract, Conclusion; supervisor review; verify references | full draft reviewed |
| 5 | Wed 14 Oct | Formatting, page limit, anonymisation if required, PDF check, submit | submission confirmation saved |

**If a gate fails:** G1 → try `nms.device: cpu`, then the larger input size; G2 → stronger
co-runner; G3 → fewer blocks with more runs (keep E1 and E3; drop E2/E4). Never change the
hypotheses or margin after pre-registration; log any deviation in `docs/CHANGELOG.md`.

## 5. Verified in development (sandbox, x86 CPU)
* Vendored NMS identical to Ultralytics NMS: 20/20 random trials.
* Perturbation on YOLOv8n @320, 300 steps on 64 COCO images: **21.9×** pre-NMS candidates on
  held-out images (G1 logic passes).
* Harness, co-runners (bwhog ≈ 8-10 GB/s, cpuhog), G2, randomised blocks, open-loop queue with
  drops, summaries and analysis ran end to end on the ONNX Runtime backend.
* Analysis validated on synthetic data: proportional slowdown → "H0: predictable" (while the naive
  additive interaction was "significant" +2.2 ms); planted 40 % NMS amplification → "H1: amplified",
  recovered NMS ratio 1.39.
* Thread pools are capped to pinned cores (an uncapped torch pool made CPU NMS 5.6× slower).

**Not testable here:** the TensorRT backend and the Jetson scripts (no Jetson in the sandbox).
Run `python3 src/harness.py --backend trt --model models/<name>.engine --mode closed --input clean
--frames 100 --out results/raw/smoke.csv` first on the Nano and send me any error.
