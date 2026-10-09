"""Merge every device's results into one cross-device dataset (schema in docs/DATA_FORMAT.md).

  python analysis/combine.py          # after each device ran summarize.py + analyze.py

Reads   results/<device>/derived/run_summary.csv and results.json for every device folder
Writes  results/combined/contend_runs.csv      one row per run, all devices (the analysis unit)
        results/combined/contend_rq1.csv       RQ1 decision per device x contention type
        results/combined/contend_rq2_nms.csv   NMS-stage sensitivity per device x contention
        results/combined/contend_rq3.csv       deadline metrics per device x cell
        results/combined/fig5_cross_device_rq1.(pdf|png)
Devices whose platform_class is 'dev' or 'smoke' are kept in the CSVs (flagged) but left out of the figure.
"""
import glob
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from common import ROOT  # noqa: E402

NICE = {"mem_high": "Memory BW (high)", "mem_low": "Memory BW (low)", "cpu": "CPU compute"}
DEV_COLORS = {"jetson_nano": "#2a78d6", "raspberry_pi": "#eb6834"}   # categorical slots 1-2 (validated)


def main():
    res_dir = os.path.join(ROOT, "results")
    out = os.path.join(res_dir, "combined")
    os.makedirs(out, exist_ok=True)
    runs, rq1, rq2, rq3 = [], [], [], []
    for jf in sorted(glob.glob(os.path.join(res_dir, "*", "derived", "results.json"))):
        dev_dir = os.path.dirname(os.path.dirname(jf))
        dev = os.path.basename(dev_dir)
        if dev == "combined":
            continue
        with open(jf) as f:
            r = json.load(f)
        meta = dict(device_id=r.get("device_id", dev), device_label=r.get("device_label", dev),
                    platform_class=r.get("platform_class", ""), deadline_ms=r.get("deadline_ms"))
        sf = os.path.join(dev_dir, "derived", "run_summary.csv")
        if os.path.exists(sf):
            d = pd.read_csv(sf)
            for k, v in meta.items():
                d[k] = v
            runs.append(d)
        for x in r.get("rq1", []):
            rq1.append(dict(meta, contention=x["contention"], runs_per_cell=x["runs_per_cell"],
                            T_A_ms=x["T_A"], T_B_ms=x["T_B"], T_C_ms=x["T_C"], T_D_ms=x["T_D"],
                            T_D_pred_ms=x["T_D_pred"], ratio=x["ratio"],
                            ratio_ci_lo=x["ratio_ci_two_sided"][0], ratio_ci_hi=x["ratio_ci_two_sided"][1],
                            decision=x["decision"], I_T_ms=x["I_T_ms"]))
        for x in r.get("rq2", []):
            if x["stage"] == "t_nms_ms":
                rq2.append(dict(meta, contention=x["contention"], k_clean=x["k_clean"], k_adv=x["k_adv"],
                                k_ratio=x["k_ratio"], k_ratio_ci_lo=x["k_ratio_ci"][0],
                                k_ratio_ci_hi=x["k_ratio_ci"][1], nms_share_B=x["share_B"]))
        for x in r.get("rq3", []):
            row = dict(meta)
            row.update({k: (v if not isinstance(v, list) else ";".join("%.4f" % z for z in v)) for k, v in x.items()})
            rq3.append(row)
    if runs:
        pd.concat(runs, ignore_index=True).to_csv(os.path.join(out, "contend_runs.csv"), index=False)
    pd.DataFrame(rq1).to_csv(os.path.join(out, "contend_rq1.csv"), index=False)
    pd.DataFrame(rq2).to_csv(os.path.join(out, "contend_rq2_nms.csv"), index=False)
    pd.DataFrame(rq3).to_csv(os.path.join(out, "contend_rq3.csv"), index=False)
    print("devices: %s" % sorted(set(x["device_id"] for x in rq1)))
    print(pd.DataFrame(rq1)[["device_id", "contention", "runs_per_cell", "ratio", "ratio_ci_lo", "ratio_ci_hi",
                             "decision"]].to_string(index=False) if rq1 else "no RQ1 results yet")
    figure(pd.DataFrame(rq1), out)


def figure(df, out):
    """Fig. 5: RQ1 ratio per contention type, one marker per edge device (dev excluded)."""
    if df.empty:
        return
    df = df[~df["platform_class"].isin(["dev", "smoke"])]
    if df.empty:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    conts = [c for c in ["mem_high", "mem_low", "cpu"] if c in set(df["contention"])]
    devs = sorted(df["device_id"].unique())
    fig, ax = plt.subplots(figsize=(3.5, 0.6 + 0.45 * len(conts)))
    ax.axvspan(0.95, 1.05, color="#e9e9e6", lw=0, zorder=0)
    ax.axvline(1, color="#888", lw=0.6)
    for j, dv in enumerate(devs):
        off = (j - (len(devs) - 1) / 2) * 0.22
        for i, c in enumerate(conts):
            r = df[(df["device_id"] == dv) & (df["contention"] == c)]
            if r.empty:
                continue
            r = r.iloc[0]
            col = DEV_COLORS.get(dv, "#555")
            ax.plot([r["ratio_ci_lo"], r["ratio_ci_hi"]], [i + off, i + off], color=col, lw=1.2)
            ax.plot(r["ratio"], i + off, "o" if j == 0 else "s", ms=4.5, color=col, mec="white", mew=0.8,
                    label=r["device_id"].replace("_", " ") if i == 0 else None)
    ax.set_yticks(range(len(conts)))
    ax.set_yticklabels([NICE.get(c, c) for c in conts])
    ax.set_xlabel("Measured / predicted latency of cell D (95% CI)")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig5_cross_device_rq1.pdf"))
    fig.savefig(os.path.join(out, "fig5_cross_device_rq1.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
