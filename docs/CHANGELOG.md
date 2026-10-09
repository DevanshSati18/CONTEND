# Changelog (deviations from the pre-registered plan)

| Date | Change | Reason |
|---|---|---|
| 2026-10-09 | Compressed schedule for the 14 Oct deadline: 6-8 runs per cell instead of >=10; YOLOv5 contrast and Raspberry Pi block moved to the journal extension | Board available only from 9 Oct |
| 2026-10-09 | Attack objective hinge taken in logit space (margin 0.5) | Probability-space hinge has vanishing sigmoid gradients; verified that candidates then rise (30 -> ~700 on a CPU test) |
| 2026-10-09 | Multiple-comparison correction: Bonferroni (planned: Holm) | Simpler with bootstrap CIs; more conservative than Holm |
| 2026-10-09 | dev_vm: E1 dry run stopped at a clean checkpoint (13/24 runs; same command resumes) | session limit / device cool-down |
| 2026-10-09 | G2: CPU co-runner treated as control (no 10 % threshold) | isolates CPU scheduling from DRAM contention by design |
