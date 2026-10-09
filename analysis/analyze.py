"""Pre-registered analysis for RQ1-RQ3. Reads results/derived/run_summary.csv.

  python analysis/analyze.py [--summary ...] [--outdir results/derived] [--figdir figures]

RQ1  per closed-loop block (one contention type):
       k_s   = mean T_s(C) / mean T_s(A)                    (clean contention sensitivity)
       T^_D  = sum_s k_s * mean T_s(B)                      (stage-wise proportional prediction)
       R     = log(mean T_D / T^_D)                         (measured vs predicted)
     run-level bootstrap within cells. Decision with margin m = log(1 + margin):
       H0 predictable  : (1-2a') CI of R inside [-m, +m]   (TOST)
       H1 amplified    : (1-a') CI of R entirely above +m
       less-than-pred. : (1-a') CI of R entirely below -m
       otherwise inconclusive;   a' = 0.05 / (number of blocks)  (Bonferroni)
RQ2  per stage: k_s(adv) = T_s(D)/T_s(B) vs k_s(clean) = T_s(C)/T_s(A), ratio + CI;
     plus OLS on run-level log service time: input * load (+ run order, start temperature), HC3.
     The additive interaction I_T = T_D - T_B - T_C + T_A is reported for transparency only:
     under proportional slowdown it is positive by construction.
RQ3  open-loop blocks: miss rate, absolute miss increase (D-C and B-A), drops, max consecutive
     misses, P95 tardiness, with run-level bootstrap CIs.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from common import load_config, p  # noqa: E402

STAGES = ["t_pre_ms", "t_inf_ms", "t_nms_ms", "t_d2h_ms", "other_ms"]
STAGE_LABEL = {"t_pre_ms": "Pre-process", "t_inf_ms": "Inference", "t_nms_ms": "NMS",
               "t_d2h_ms": "Copy to host", "other_ms": "Other"}
CLEAN, ADV = "#2a78d6", "#eb6834"    # categorical slots 1-2, validated (light surface)
NICE = {"mem_high": "Memory BW (high)", "mem_low": "Memory BW (low)", "cpu": "CPU compute", "none": "None"}


def cell_means(d):
    """dict cell -> DataFrame of per-run stage means for one block."""
    return {c: d[d["cell"] == c][STAGES].reset_index(drop=True) for c in "ABCD"}


def stat_R(cm):
    mA, mB, mC, mD = (cm[c].mean() for c in "ABCD")
    k = (mC / mA).replace([np.inf], np.nan).fillna(1.0)
    pred = float((k * mB).sum())
    meas = float(mD.sum())
    return np.log(meas / pred), meas, pred, k


def boot(cm, fn, n, seed):
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(n):
        s = {c: cm[c].iloc[rng.randint(0, len(cm[c]), len(cm[c]))] for c in "ABCD"}
        out.append(fn(s))
    return np.array(out)


def rq1_rq2(df, cfg, nboot):
    closed = df[df["mode"] == "closed"]
    blocks = sorted(closed["block"].unique())
    m = np.log(1 + cfg["analysis"]["equivalence_margin"])
    a_adj = 0.05 / max(len(blocks), 1)
    res1, res2 = [], []
    for b in blocks:
        d = closed[closed["block"] == b]
        cm = cell_means(d)
        if any(len(cm[c]) < 2 for c in "ABCD"):
            print("block %s: fewer than 2 runs in a cell, skipped" % b)
            continue
        R, meas, pred, k = stat_R(cm)
        bs = boot(cm, lambda s: stat_R(s)[0], nboot, 1)
        tost = np.percentile(bs, [100 * a_adj, 100 * (1 - a_adj)])
        two = np.percentile(bs, [100 * a_adj / 2, 100 * (1 - a_adj / 2)])
        if -m < tost[0] and tost[1] < m:
            dec = "H0: predictable (equivalent within margin)"
        elif two[0] > m:
            dec = "H1: amplified beyond prediction"
        elif two[1] < -m:
            dec = "below prediction"
        else:
            dec = "inconclusive"
        mA, mB, mC, mD = (cm[c].sum(axis=1).mean() for c in "ABCD")
        it = boot(cm, lambda s: s["D"].sum(axis=1).mean() - s["B"].sum(axis=1).mean() - s["C"].sum(axis=1).mean() + s["A"].sum(axis=1).mean(),
                  nboot, 2)
        res1.append(dict(block=b, contention=d["contention"].iloc[0], runs_per_cell=min(len(cm[c]) for c in "ABCD"),
                         T_A=mA, T_B=mB, T_C=mC, T_D=mD, T_D_pred=pred, ratio=float(np.exp(R)),
                         ratio_ci_tost=[float(np.exp(tost[0])), float(np.exp(tost[1]))],
                         ratio_ci_two_sided=[float(np.exp(two[0])), float(np.exp(two[1]))],
                         margin=cfg["analysis"]["equivalence_margin"], alpha_adj=a_adj, decision=dec,
                         I_T_ms=mD - mB - mC + mA, I_T_ci95=[float(np.percentile(it, 2.5)), float(np.percentile(it, 97.5))]))
        for s in STAGES:
            def ks(sm, s=s):
                kc = sm["C"][s].mean() / sm["A"][s].mean()
                ka = sm["D"][s].mean() / sm["B"][s].mean()
                return kc, ka, ka / kc
            pt = ks(cm)
            bb = boot(cm, ks, nboot, 3)
            ci = np.nanpercentile(bb, [2.5, 97.5], axis=0)
            res2.append(dict(block=b, contention=d["contention"].iloc[0], stage=s, k_clean=pt[0], k_adv=pt[1],
                             k_ratio=pt[2], k_clean_ci=list(ci[:, 0]), k_adv_ci=list(ci[:, 1]),
                             k_ratio_ci=list(ci[:, 2]), share_B=float(cm["B"][s].mean() / cm["B"].sum(axis=1).mean())))
    return res1, res2, blocks


def regression(df):
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        return {}
    out = {}
    for b, d in df[df["mode"] == "closed"].groupby("block"):
        d = d.copy()
        d["log_service"] = np.log(d["service_ms"])
        d["adv"] = (d["input"] == "adv").astype(int)
        d["load"] = d["cell"].isin(["C", "D"]).astype(int)
        terms = "adv * load"
        if "order" in d and d["order"].notna().all():
            terms += " + order"
        if "start_temp_c" in d and d["start_temp_c"].notna().all() and d["start_temp_c"].std() > 0:
            terms += " + start_temp_c"
        fit = smf.ols("log_service ~ " + terms, data=d).fit(cov_type="HC3")
        out[b] = {k: dict(coef=float(fit.params[k]), ci=[float(x) for x in fit.conf_int().loc[k]], p=float(fit.pvalues[k]))
                  for k in fit.params.index}
    return out


def rq3(df, nboot):
    op = df[df["mode"] == "open"]
    rows = []
    for b, d in op.groupby("block"):
        cm = {c: d[d["cell"] == c].reset_index(drop=True) for c in "ABCD"}
        if any(len(cm[c]) == 0 for c in "ABCD"):
            continue
        for c in "ABCD":
            rows.append(dict(block=b, contention=d["contention"].iloc[0], cell=c, runs=len(cm[c]),
                             miss_rate=float(cm[c]["miss_rate"].mean()), drop_rate=float(cm[c]["drop_rate"].mean()),
                             max_consec_miss=float(cm[c]["max_consec_miss"].mean()),
                             tard_p95_ms=float(cm[c]["tard_p95_ms"].mean()), lat_p99=float(cm[c]["lat_p99"].mean())))
        rng = np.random.RandomState(4)
        def inc(x, y):
            bs = [rng.choice(cm[x]["miss_rate"], len(cm[x])).mean() - rng.choice(cm[y]["miss_rate"], len(cm[y])).mean()
                  for _ in range(nboot)]
            return float(cm[x]["miss_rate"].mean() - cm[y]["miss_rate"].mean()), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        for name, (x, y) in {"attack_effect_no_load(B-A)": ("B", "A"), "attack_effect_under_load(D-C)": ("D", "C")}.items():
            v, ci = inc(x, y)
            rows.append(dict(block=b, contention=d["contention"].iloc[0], cell=name, miss_rate=v, miss_rate_ci95=ci))
    return rows


def figures(r1, r2, figdir, raw):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": "#888", "axes.linewidth": 0.6,
                         "xtick.color": "#444", "ytick.color": "#444", "pdf.fonttype": 42})
    os.makedirs(figdir, exist_ok=True)
    W = 3.5   # IEEE single column (inches)
    if r1:   # Fig. 2: measured / predicted with CIs and the equivalence band
        fig, ax = plt.subplots(figsize=(W, 0.45 + 0.35 * len(r1)))
        m = r1[0]["margin"]
        ax.axvspan(1 - m, 1 + m, color="#e9e9e6", zorder=0, lw=0)
        ax.axvline(1, color="#888", lw=0.6, zorder=1)
        for i, r in enumerate(r1):
            lo, hi = r["ratio_ci_two_sided"]
            ax.plot([lo, hi], [i, i], color="#222", lw=1.2, solid_capstyle="round")
            ax.plot(r["ratio"], i, "o", ms=5, color="#222", mec="white", mew=1)
        ax.set_yticks(range(len(r1)))
        ax.set_yticklabels([NICE.get(r["contention"], r["contention"]) for r in r1])
        ax.set_xlabel("Measured / predicted latency of cell D (95% CI)")
        ax.text(1 + m, len(r1) - 0.4, " ±%d%% band" % round(100 * m), fontsize=7, color="#555", va="center")
        ax.set_ylim(-0.6, len(r1) - 0.2)
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, "fig2_rq1_prediction.pdf"))
        fig.savefig(os.path.join(figdir, "fig2_rq1_prediction.png"), dpi=200)
        plt.close(fig)
    if r2:   # Fig. 3: per-stage contention sensitivity, clean vs adversarial, one panel per block
        blocks = sorted(set(r["block"] for r in r2))
        fig, axes = plt.subplots(1, len(blocks), figsize=(W if len(blocks) == 1 else 7.16, 2.0), sharey=True, squeeze=False)
        for ax, b in zip(axes[0], blocks):
            rows = [r for r in r2 if r["block"] == b and r["stage"] in ("t_pre_ms", "t_inf_ms", "t_nms_ms")]
            x = np.arange(len(rows))
            for off, key, col, hatch, lab in ((-0.19, "k_clean", CLEAN, None, "Clean"), (0.19, "k_adv", ADV, "////", "Adversarial")):
                vals = [r[key] for r in rows]
                err = np.array([[r[key] - r[key + "_ci"][0], r[key + "_ci"][1] - r[key]] for r in rows]).T
                ax.bar(x + off, vals, 0.36, color=col, hatch=hatch, edgecolor="white", lw=0.8, label=lab, zorder=2)
                ax.errorbar(x + off, vals, yerr=err, fmt="none", ecolor="#333", elinewidth=0.7, capsize=1.5, zorder=3)
            ax.axhline(1, color="#888", lw=0.6, zorder=1)
            ax.set_xticks(x)
            ax.set_xticklabels([STAGE_LABEL[r["stage"]] for r in rows])
            ax.set_title(NICE.get(rows[0]["contention"], rows[0]["contention"]) if rows else b, fontsize=8)
        axes[0][0].set_ylabel("Contention slowdown k (×)")
        axes[0][0].legend(frameon=False, fontsize=7, loc="upper left")
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, "fig3_rq2_stage_sensitivity.pdf"))
        fig.savefig(os.path.join(figdir, "fig3_rq2_stage_sensitivity.png"), dpi=200)
        plt.close(fig)


def trace_figure(raw, figdir, fps):
    """Fig. 4: per-frame latency of one open-loop C and D run against the deadline."""
    import glob
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    runs = {}
    for c in "CD":
        g = sorted(glob.glob(os.path.join(raw, "*", "*_%s_*_r00.csv" % c)) +
                   glob.glob(os.path.join(raw, "*", "*_%s_*_r00.csv.gz" % c)))
        g = [f for f in g if pd.read_csv(f, nrows=1)["mode"].iloc[0] == "open"]
        if g:
            runs[c] = pd.read_csv(g[0])
    if len(runs) < 2:
        return
    fig, ax = plt.subplots(figsize=(3.5, 1.7))
    for c, col, lab in (("C", CLEAN, "Clean + contention (C)"), ("D", ADV, "Adversarial + contention (D)")):
        d = runs[c][runs[c]["dropped"] == 0]
        ax.plot(d["t_release_ms"] / 1000.0, d["latency_ms"], lw=0.8, color=col, label=lab)
    ax.axhline(1000.0 / fps, color="#222", lw=0.8, ls="--")
    ax.text(ax.get_xlim()[1], 1000.0 / fps, " deadline", fontsize=7, va="bottom", ha="right", color="#222")
    ax.set_xlabel("Stream time (s)")
    ax.set_ylabel("Frame latency (ms)")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "fig4_rq3_trace.pdf"))
    fig.savefig(os.path.join(figdir, "fig4_rq3_trace.png"), dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=None)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--figdir", default=None)
    ap.add_argument("--raw", default=None)
    ap.add_argument("--nboot", type=int, default=None)
    a = ap.parse_args()
    cfg = load_config()
    outdir = a.outdir or p(cfg["paths"]["derived"])
    df = pd.read_csv(a.summary or os.path.join(outdir, "run_summary.csv"))
    nboot = a.nboot or cfg["analysis"]["bootstrap"]
    r1, r2, _ = rq1_rq2(df, cfg, nboot)
    reg = regression(df)
    r3 = rq3(df, nboot)
    res = dict(rq1=r1, rq2=r2, regression=reg, rq3=r3)
    os.makedirs(outdir, exist_ok=True)
    res["device_id"] = cfg["device_id"]
    res["device_label"] = cfg.get("device_label", cfg["device_id"])
    res["platform_class"] = cfg.get("platform_class", "")
    res["deadline_ms"] = 1000.0 / cfg["deadline"]["fps"]
    with open(os.path.join(outdir, "results.json"), "w") as f:
        json.dump(res, f, indent=2, default=float)
    pd.DataFrame(r2).to_csv(os.path.join(outdir, "rq2_stage_sensitivity.csv"), index=False)
    pd.DataFrame(r3).to_csv(os.path.join(outdir, "rq3_deadlines.csv"), index=False)
    figdir = a.figdir or p(cfg["paths"]["figures"])
    figures(r1, r2, figdir, a.raw)
    trace_figure(a.raw or p(cfg["paths"]["raw"]), figdir, cfg["deadline"]["fps"])
    print("\nRQ1  measured vs stage-wise predicted latency of cell D")
    for r in r1:
        print("  %-9s  runs/cell %d  T_D %.2f ms  predicted %.2f ms  ratio %.3f  95%% CI %.3f-%.3f  -> %s" % (
            r["contention"], r["runs_per_cell"], r["T_D"], r["T_D_pred"], r["ratio"],
            r["ratio_ci_two_sided"][0], r["ratio_ci_two_sided"][1], r["decision"]))
        print("             (additive interaction I_T %+.2f ms, 95%% CI %+.2f to %+.2f; positive by construction under"
              " proportional slowdown)" % (r["I_T_ms"], r["I_T_ci95"][0], r["I_T_ci95"][1]))
    print("\nRQ2  NMS-stage contention sensitivity (adv / clean)")
    for r in r2:
        if r["stage"] == "t_nms_ms":
            print("  %-9s  k_clean %.3f  k_adv %.3f  ratio %.3f  95%% CI %.3f-%.3f" % (
                r["contention"], r["k_clean"], r["k_adv"], r["k_ratio"], r["k_ratio_ci"][0], r["k_ratio_ci"][1]))
    if r3:
        print("\nRQ3  open-loop deadline misses")
        for r in r3:
            if "ci95" in str(r) and "effect" in r["cell"]:
                print("  %-9s %-30s %+.3f  95%% CI %+.3f to %+.3f" % (r["contention"], r["cell"], r["miss_rate"],
                                                                     r["miss_rate_ci95"][0], r["miss_rate_ci95"][1]))
    print("\nwrote %s/results.json and figures in %s" % (outdir, figdir))


if __name__ == "__main__":
    main()
