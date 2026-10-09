"""Human-readable result reports (Markdown), regenerated after every session.

  python analysis/report.py                 # this device  -> results/<device>/REPORT.md
  python analysis/report.py --combined      # all devices  -> results/combined/REPORT.md

Reads only files the pipeline already wrote (locked_config.yaml, g1/g2/g3 JSON, run_summary.csv,
results.json, results/combined/*.csv); it never changes any number.
"""
import argparse
import json
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from common import ROOT, load_config, p  # noqa: E402

NICE = {"mem_high": "Memory BW (high)", "mem_low": "Memory BW (low)", "cpu": "CPU compute (control)"}


def jload(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def ci(v, fmt="%.3f"):
    return ("[" + fmt + ", " + fmt + "]") % (v[0], v[1]) if v else "n/a"


def device_report(cfg):
    dev = cfg["device_id"]
    base = os.path.join(ROOT, "results", dev)
    der = p(cfg["paths"]["derived"])
    L = ["# CONTEND results: %s" % dev, "",
         "_%s_  " % cfg.get("device_label", dev),
         "Generated %s by `analysis/report.py` (re-run after every session; numbers come from "
         "`derived/results.json`)." % time.strftime("%Y-%m-%d %H:%M"), ""]
    lock = os.path.join(base, "locked_config.yaml")
    if os.path.exists(lock):
        import yaml
        with open(lock) as f:
            lk = yaml.safe_load(f) or {}
        c = lk.get("config", {})
        L += ["## Configuration (locked on this device)", "",
              "| Item | Value |", "|---|---|",
              "| Backend / precision | %s / %s |" % (cfg["runtime"]["backend"], cfg["runtime"]["precision"]),
              "| Model / input size | %s / %s |" % (c.get("model", {}).get("name"), c.get("model", {}).get("imgsz")),
              "| Deadline | %s FPS (%.1f ms); clean latency %s ms = %s of the deadline |" % (
                  c.get("deadline", {}).get("fps"), 1000.0 / float(c.get("deadline", {}).get("fps", 1)),
                  lk.get("clean_ms", "n/a"), lk.get("deadline_share", "n/a")),
              "| 50-70 %% rule met | %s |" % lk.get("rule_met", "n/a"),
              "| Sessions planned | %s |" % lk.get("sessions", "n/a"),
              "| Runs per cell per session | %s |" % ", ".join("%s: %s" % (b, v.get("runs_per_cell"))
                                                          for b, v in sorted(c.get("experiment", {}).get("blocks", {}).items())),
              "| Frozen (pre-registration) | %s (git %s) |" % (lk.get("frozen_at", "not yet"), lk.get("git_commit", "?")),
              ""]
    g1, g2, g3 = (jload(os.path.join(der, n)) for n in ("g1_device.json", "g2.json", "g3.json"))
    L += ["## Gates", "", "| Gate | Result |", "|---|---|"]
    if g1:
        L.append("| G1 workload | %s: candidates x%.1f, NMS +%.2f ms (95%% CI %.2f to %.2f) |" % (
            "PASS" if g1.get("G1_pass") else "FAIL", g1.get("candidate_ratio", float("nan")),
            g1.get("nms_increase_ms_mean", float("nan")), *(g1.get("nms_increase_ms_ci95") or [float("nan")] * 2)))
    for k, v in sorted((g2 or {}).items()):
        verdict = "CONTROL" if v.get("G2_pass") is None else ("PASS" if v["G2_pass"] else "FAIL")
        L.append("| G2 %s | %s: clean %.2f -> loaded %.2f ms (%+.1f %%) |" % (
            k, verdict, v["clean_ms"], v["loaded_ms"], 100 * v["slowdown"]))
    if g3:
        L.append("| G3 pilot | worst between-run CV %.2f %% -> %d runs per cell needed |" % (
            100 * max(c["cv"] for c in g3["cells"]), g3["runs_per_cell_needed"]))
    L.append("")
    sf = os.path.join(der, "run_summary.csv")
    if os.path.exists(sf):
        s = pd.read_csv(sf)
        if "session" not in s:
            s["session"] = 1
        t = s.groupby(["block", "session"]).size().unstack(fill_value=0)
        L += ["## Runs completed (block x session)", "", "| Block | " + " | ".join("S%d" % c for c in t.columns) + " |",
              "|---|" + "---|" * len(t.columns)]
        for b, row in t.iterrows():
            L.append("| %s | %s |" % (b, " | ".join(str(int(x)) for x in row.values)))
        L.append("")
    r = jload(os.path.join(der, "results.json"))
    if r:
        L += ["## RQ1: is the combined latency predictable? (measured / predicted latency of cell D)", "",
              "Margin ±%d %%. Pooled CI = run-level bootstrap (pre-registered). Two-stage CI also resamples whole "
              "sessions; *robust* = both give the same decision." % round(100 * (r["rq1"][0]["margin"] if r["rq1"] else 0.05)), "",
              "| Contention | Sessions | Runs/cell | Ratio | Pooled CI | Decision | Per-session ratios | Between-session SD | Two-stage CI | Robust |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for x in r["rq1"]:
            sd = x.get("between_session_sd_pct")
            L.append("| %s | %s | %d | %.3f | %s | %s | %s | %s | %s | %s |" % (
                NICE.get(x["contention"], x["contention"]), x.get("n_sessions", 1), x["runs_per_cell"], x["ratio"],
                ci(x["ratio_ci_two_sided"]), x["decision"],
                ", ".join("%.3f" % v for v in x.get("ratio_by_session", [])) or "n/a",
                ("%.2f %%" % sd) if sd is not None else "n/a", ci(x.get("ratio_ci_cluster")),
                {True: "yes", False: "NO", None: "n/a (1 session)"}[x.get("robust_across_sessions")]))
        L += ["", "Additive interaction I_T (reported for contrast only; positive by construction under proportional slowdown): " +
              "; ".join("%s %+.2f ms" % (x["contention"], x["I_T_ms"]) for x in r["rq1"]), ""]
        L += ["## RQ2: per-stage contention sensitivity (adversarial k / clean k)", "",
              "| Contention | Stage | k clean | k adv | ratio | 95 % CI |", "|---|---|---|---|---|---|"]
        for x in r["rq2"]:
            if x["stage"] in ("t_pre_ms", "t_inf_ms", "t_nms_ms"):
                L.append("| %s | %s | %.3f | %.3f | %.3f | %s |" % (x["contention"], x["stage"], x["k_clean"], x["k_adv"],
                                                                 x["k_ratio"], ci(x["k_ratio_ci"])))
        L.append("")
        eff = [x for x in r["rq3"] if "effect" in str(x.get("cell"))]
        if eff:
            L += ["## RQ3: open-loop deadline misses (%.0f ms deadline)" % r.get("deadline_ms", 0), "",
                  "| Contention | Effect | Miss-rate change | 95 % CI |", "|---|---|---|---|"]
            for x in eff:
                L.append("| %s | %s | %+.3f | %s |" % (x["contention"], x["cell"], x["miss_rate"], ci(x["miss_rate_ci95"], "%+.3f")))
            L.append("")
    L += ["## Files", "", "- `raw/<block>/<run_id>.csv.gz` per-frame logs; `raw/runs.csv` run manifest",
          "- `derived/run_summary.csv` one row per run; `derived/results.json` all statistics",
          "- `figures/fig2..fig4` (PDF + PNG)", ""]
    out = os.path.join(base, "REPORT.md")
    with open(out, "w") as f:
        f.write("\n".join(L))
    print("wrote %s" % out)


def combined_report():
    d = os.path.join(ROOT, "results", "combined")
    L = ["# CONTEND combined results (all devices)", "",
         "Generated %s by `analysis/report.py --combined` from `results/combined/*.csv`. Devices with "
         "platform_class dev/smoke are development data, never evidence." % time.strftime("%Y-%m-%d %H:%M"), ""]
    f1 = os.path.join(d, "contend_rq1.csv")
    if os.path.exists(f1) and os.path.getsize(f1) > 2:
        q = pd.read_csv(f1)
        L += ["## RQ1 per device x contention type", "",
              "| Device | Class | Contention | Sessions | Runs/cell | Ratio | 95 % CI | Decision | Robust across sessions |",
              "|---|---|---|---|---|---|---|---|---|"]
        for _, x in q.iterrows():
            L.append("| %s | %s | %s | %s | %s | %.3f | [%.3f, %.3f] | %s | %s |" % (
                x["device_id"], x.get("platform_class", ""), x["contention"], x.get("n_sessions", 1), x["runs_per_cell"],
                x["ratio"], x["ratio_ci_lo"], x["ratio_ci_hi"], x["decision"], x.get("robust_across_sessions", "")))
        L.append("")
    f2 = os.path.join(d, "contend_rq2_nms.csv")
    if os.path.exists(f2) and os.path.getsize(f2) > 2:
        q = pd.read_csv(f2)
        L += ["## RQ2 NMS-stage sensitivity", "", "| Device | Contention | k clean | k adv | ratio | 95 % CI |",
              "|---|---|---|---|---|---|"]
        for _, x in q.iterrows():
            L.append("| %s | %s | %.3f | %.3f | %.3f | [%.3f, %.3f] |" % (x["device_id"], x["contention"], x["k_clean"],
                                                                      x["k_adv"], x["k_ratio"], x["k_ratio_ci_lo"], x["k_ratio_ci_hi"]))
        L.append("")
    L += ["Files: `contend_runs.csv` (one row per run, all devices), `contend_rq1.csv`, `contend_rq2_nms.csv`, "
          "`contend_rq3.csv`, `fig5_cross_device_rq1.pdf`.", ""]
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "REPORT.md"), "w") as f:
        f.write("\n".join(L))
    print("wrote %s" % os.path.join(d, "REPORT.md"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combined", action="store_true")
    a = ap.parse_args()
    if a.combined:
        combined_report()
    else:
        device_report(load_config())


if __name__ == "__main__":
    main()
