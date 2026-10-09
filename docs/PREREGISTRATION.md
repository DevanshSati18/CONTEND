# CONTEND pre-registration

Fill the blanks after the pilot (gates G1-G3) and BEFORE the main blocks start, then commit:

    git add configs/contend.yaml docs/PREREGISTRATION.md
    git commit -m "Pre-registration: CONTEND v1.0"

The commit hash and date are cited in the paper. With `runscript.py` the device-specific part is
frozen automatically in `results/<device>/locked_config.yaml` (timestamp + git commit) right after
the gates and before the first main run; copy its values into the table below.

| Item | Value |
|---|---|
| Objective version | v1.0 (frozen 9 Oct 2026) |
| Platform | NVIDIA Jetson Nano ____ GB, JetPack ____, MAXN, clocks locked |
| Model / input size / deadline | ______ / ______ / ____ FPS (chosen by the 50-70% rule, `src/choose_config.py`) |
| NMS | conf 0.25, IoU 0.7, max_det 300, device ______ |
| Perturbation | Phantom Sponges objective, L-inf 16/255, file `perturbations/________.npy`, sha256 ________ |
| G1 (board) | candidates x____ ; NMS +____ ms (95% CI ____ to ____) |
| G2 | mem_high +____ % ; mem_low +____ % ; cpu +____ % |
| G3 | worst between-run CV ____ % -> runs per cell ____ |
| Primary test (RQ1) | R = log(T_D / T_D_pred); H0 if TOST CI inside +/-log(1.05); H1 if CI above +log(1.05); Bonferroni over contention types |
| Secondary (RQ2) | per-stage k_adv / k_clean with 95% bootstrap CI |
| RQ3 | open-loop miss rate, absolute miss increase D-C vs B-A, drops, max consecutive misses |
| Exclusion rules | start temperature > max_start_temp_c (waits, then logs); clocks not locked; warm-up frames |
| Planned run counts | E1 ____ /cell, E2 ____ /cell, E3 ____ /cell, E4 (optional) ____ /cell (total = sessions x runs per session; see `results/<device>/locked_config.yaml`) |
| Repetition (sessions) | ____ independent sessions per block, each in a fresh seeded random order. Primary: pooled run-level bootstrap (stratified by session). Robustness: per-session ratios, between-session SD, two-stage (session, run) bootstrap; a decision is called *robust* only if both bootstraps agree |

Any later change: add a dated line to `docs/CHANGELOG.md` and report it in the paper.
