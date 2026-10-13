# Changelog (deviations from the pre-registered plan)

| Date | Change | Reason |
|---|---|---|
| 2026-10-09 | Compressed schedule for the 14 Oct deadline: 6-8 runs per cell instead of >=10; YOLOv5 contrast and Raspberry Pi block moved to the journal extension | Board available only from 9 Oct |
| 2026-10-09 | Attack objective hinge taken in logit space (margin 0.5) | Probability-space hinge has vanishing sigmoid gradients; verified that candidates then rise (30 -> ~700 on a CPU test) |
| 2026-10-09 | Multiple-comparison correction: Bonferroni (planned: Holm) | Simpler with bootstrap CIs; more conservative than Holm |
| 2026-10-09 | dev_vm: E1 dry run stopped at a clean checkpoint (13/24 runs; same command resumes) | session limit / device cool-down |
| 2026-10-09 | G2: CPU co-runner treated as control (no 10 % threshold) | isolates CPU scheduling from DRAM contention by design |
| 2026-10-10 | Repetition: every block is run in N independent sessions (default 3), each in a fresh seeded random order; total runs per cell = max(G3 requirement, planned), spread over the sessions | Run-to-run and day-to-day variation: the answer must not depend on one sitting. Primary analysis unchanged (run-level bootstrap, now stratified by session); added per-session ratios, between-session SD and a two-stage bootstrap as a robustness check |
| 2026-10-10 | One-command runner `runscript.py` (--pi / --nvidia / --dev); device choices frozen in results/<device>/locked_config.yaml; only the 500 subset images are downloaded (not the 778 MB zip); side logs gzipped | Reproducible, resumable runs per device; results small enough to push |
| 2026-10-13 | Perturbation applied once to the frames (uint8, as a camera would deliver it) instead of per frame during pre-processing | Jetson pilot: per-frame `x + delta` + clip added +2.7 ms of pre-processing to cells B/D - CPU work that belongs to the harness, not to the attack. Now clean and adversarial frames get identical pre-processing; quantisation changes the perturbation by <= 0.5 grey levels. Runs before this fix are not mixed with runs after it |
| 2026-10-13 | Jetson Nano: NMS moved to the CPU (`nms.device: cpu`, pre-planned fallback) | G1 failed with GPU NMS: candidates x32.3 but NMS time unchanged (pilot medians 5.8 -> 6.4 ms; paired G1 mean -4.7 ms, 95 % CI -15.1 to +1.1). GPU NMS on the Maxwell is dominated by fixed launch/sync cost, consistent with EVADE. The GPU-NMS gate data is kept as results/jetson_nano_gpu_nms_G1fail |
