"""
paper/make_figures.py
=====================
Regenerate every figure used in ``paper/paper.md`` from material already in the
repository. Run from the project root:

    PYTHONPATH=. python paper/make_figures.py

Outputs are written to ``paper/figures/``:

  fig_crisis_run.png         one representative crisis run (prices / fire sales /
                             capital / leverage) via the shipped plot_crisis().
  fig_outcome_distribution.png  bucket-fraction bars at -15% vs -20% (regenerated
                             at a moderate N so the script is self-contained).
  fig_sensitivity.png        standardised LHS main-effect coefficients on the
                             asset-0 price crash, read from the existing LHS
                             samples CSV (no heavy re-run).
  fig_contagion_decomp.png   per-agent contagion share of holding value on a
                             never-shocked asset, from the existing decomposition
                             summary CSV.

Nothing here mutates model code or the canonical outputs/ artefacts; the outcome
distribution is recomputed into paper/figures only.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bookstaber_abm.simulation.engine import Simulation
from bookstaber_abm.analysis.plots import history_to_df, plot_crisis
from bookstaber_abm.analysis.buckets import classify_run, summarize_runs, bucket_counts
from experiments.robustness import BASE  # shipped production calibration

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Monte Carlo seeds. N_DIST: outcome-distribution figure; SWEEP_N: crowding /
# leverage sweeps (per level). Both default to 1000 (verified run); override via env.
N_DIST = int(os.environ.get("N_DIST", "1000"))
SWEEP_N = int(os.environ.get("SWEEP_N", "1000"))
ROOT = os.path.dirname(HERE)

BUCKETS = ["no_default", "hf0_only", "partial", "all_default"]
BUCKET_LABELS = {
    "no_default": "No default",
    "hf0_only": "HF0 only",
    "partial": "Partial cascade",
    "all_default": "Full cascade",
}
C_15 = "#185FA5"
C_20 = "#E24B4A"


# ---------------------------------------------------------------------------- #
# Figure 1 — a single narrated crisis run
# ---------------------------------------------------------------------------- #
def fig_crisis_run() -> None:
    # A -20% run; scan seeds for one that produces a rich (partial/all) cascade.
    cfg = replace(BASE, shock_size=-0.20)
    chosen = None
    for seed in range(20):
        history = Simulation(replace(cfg, seed=seed)).run()
        bucket = classify_run(history)["bucket"]
        if bucket in ("partial", "all_default"):
            chosen = (seed, history, bucket)
            break
    if chosen is None:  # fall back to seed 0 whatever it produced
        history = Simulation(replace(cfg, seed=0)).run()
        chosen = (0, history, classify_run(history)["bucket"])

    seed, history, bucket = chosen
    df = history_to_df(history)
    fig = plot_crisis(df, cfg)
    fig.savefig(os.path.join(FIG_DIR, "fig_crisis_run.png"),
                bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[fig_crisis_run] seed={seed} bucket={bucket}")


# ---------------------------------------------------------------------------- #
# Figure 2 — outcome distribution across shock severity
# ---------------------------------------------------------------------------- #
def _bucket_fracs(shock: float, n: int) -> dict:
    hist = [Simulation(replace(BASE, shock_size=shock, seed=s)).run()
            for s in range(n)]
    df = summarize_runs(hist, shock)
    counts = bucket_counts(df)
    return {b: counts[b] / n for b in BUCKETS}


def fig_outcome_distribution() -> None:
    f15 = _bucket_fracs(-0.15, N_DIST)
    f20 = _bucket_fracs(-0.20, N_DIST)

    x = np.arange(len(BUCKETS))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w / 2, [f15[b] for b in BUCKETS], w, label="-15% shock", color=C_15)
    ax.bar(x + w / 2, [f20[b] for b in BUCKETS], w, label="-20% shock", color=C_20)
    ax.set_xticks(x)
    ax.set_xticklabels([BUCKET_LABELS[b] for b in BUCKETS])
    ax.set_ylabel("Fraction of runs")
    ax.set_title(f"Crisis-outcome distribution across shock severity (N={N_DIST} seeds each)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    for xi, b in zip(x, BUCKETS):
        ax.text(xi - w / 2, f15[b] + 0.01, f"{f15[b]*100:.0f}", ha="center", fontsize=8)
        ax.text(xi + w / 2, f20[b] + 0.01, f"{f20[b]*100:.0f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_outcome_distribution.png"),
                bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[fig_outcome_distribution] -15%={f15} -20%={f20}")


# ---------------------------------------------------------------------------- #
# Figure 3 — global sensitivity (standardised LHS main effects)
# ---------------------------------------------------------------------------- #
PREDICTORS = [
    "beta", "noise_std", "cp_haircut_normal", "phi_cw", "phi_hc",
    "hf0_shock_weight", "cp_max_loan", "bd_liq_ratio_min",
    "hf_lev_target", "bd_liq_rate",
]
PRETTY = {
    "hf0_shock_weight": "HF0 concentration",
    "hf_lev_target": "HF leverage target",
    "beta": "Price impact (β)",
    "noise_std": "Price noise",
    "cp_max_loan": "Loan cap",
    "phi_cw": "CW sensitivity",
    "phi_hc": "Haircut sensitivity",
    "cp_haircut_normal": "Base haircut",
    "bd_liq_ratio_min": "Liq-ratio floor",
    "bd_liq_rate": "BD liq reserve rate",
}


def emit_regression_table() -> None:
    """Render the full LHS regression (main effects + pairwise interactions) on
    the asset-0 price crash as a Markdown table fragment for the manuscript.
    Reads the tidy coefficient CSV written by experiments/sensitivity_regression.py."""
    coef_csv = os.path.join(ROOT, "outputs", "sensitivity_lhs_2026_05_30",
                            "regression_coefs_15pct.csv")
    if not os.path.exists(coef_csv):
        print(f"[regression_table] SKIP — {coef_csv} not found "
              f"(run experiments/sensitivity_regression.py first)")
        return
    df = pd.read_csv(coef_csv)
    df["stars"] = df["stars"].fillna("")  # ns terms have no stars
    d = df[df["outcome"] == "price_change_asset0"].copy()

    mains = d[~d["is_interaction"]].copy()
    mains["abs"] = mains["estimate"].abs()
    mains = mains.sort_values("abs", ascending=False)

    inters = d[d["is_interaction"]].copy()
    sig = inters[inters["pvalue"] < 0.10].copy()
    sig["abs"] = sig["estimate"].abs()
    sig = sig.sort_values("abs", ascending=False).head(6)

    def fmt(name):
        return PRETTY.get(name, name)

    lines = []
    lines.append("**Main effects** (standardized, SD units):\n")
    lines.append("| Parameter | Estimate | p | Sig |")
    lines.append("|---|---:|---:|:--:|")
    for _, r in mains.iterrows():
        lines.append(f"| {fmt(r['term'])} | {r['estimate']:+.4f} | "
                     f"{r['pvalue']:.3f} | {r['stars']} |")
    lines.append("")
    lines.append("**Top significant pairwise interactions** (p < 0.10):\n")
    if sig.empty:
        lines.append("_None significant at p < 0.10._")
    else:
        lines.append("| Interaction | Estimate | p | Sig |")
        lines.append("|---|---:|---:|:--:|")
        for _, r in sig.iterrows():
            a, b = r["term"].split(":")
            lines.append(f"| {fmt(a)} × {fmt(b)} | {r['estimate']:+.4f} | "
                         f"{r['pvalue']:.3f} | {r['stars']} |")
    out = os.path.join(FIG_DIR, "regression_table_15pct.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[regression_table] wrote {out} "
          f"({len(mains)} mains, {len(sig)} sig interactions shown)")


# ---------------------------------------------------------------------------- #
# Figure 4 — contagion decomposition (never-shocked-asset holding loss)
# ---------------------------------------------------------------------------- #
def fig_contagion_decomp() -> None:
    csv = os.path.join(
        os.path.dirname(HERE),
        "outputs", "contagion_decomposition", "summary_20pct.csv",
    )
    df = pd.read_csv(csv)
    # holdings rows on a non-shock asset; average the contagion % per agent
    h = df[(df["level"] == "holding") & (df["is_shock_asset"] == 0)]
    g = (h.groupby("agent")["contagion_pct_mean"].mean()
         .reindex(["HF0", "HF1", "HF2", "HF3", "BD0", "BD1"]))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#888"] + [C_20] * 3 + [C_15] * 2
    ax.bar(g.index, g.values, color=colors)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Mean contagion effect on holding value (%)")
    ax.set_title("Loss attributable to contagion alone, never-shocked assets (-20%)")
    ax.grid(alpha=0.3, axis="y")
    for xi, v in enumerate(g.values):
        if not np.isnan(v):
            ax.text(xi, v - 2, f"{v:.0f}%", ha="center", va="top", fontsize=8, color="white")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_contagion_decomp.png"),
                bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("[fig_contagion_decomp]", dict(g.round(1)))


# ---------------------------------------------------------------------------- #
# Figure 5 — crowding sweep: effect on agent capital and asset prices
# ---------------------------------------------------------------------------- #
AGENTS = ["HF0", "HF1", "HF2", "HF3", "BD0", "BD1"]


def _final_metrics(history: list[dict]) -> dict:
    """Final-step mean HF capital, mean BD capital, and asset-0 price for one run."""
    f = history[-1]
    return {
        "hf_capital": float(np.mean(f["hf_capitals"])),
        "bd_capital": float(np.mean(f["bd_capitals"])),
        "price0": float(f["prices"][0]),
    }


def _sweep(cfg_variant_fn, levels, n: int) -> dict:
    """For each level, run n seeds and collect per-run final metrics. Returns
    {level: {metric: np.array(n)}}."""
    out = {}
    for lvl in levels:
        cfg = cfg_variant_fn(lvl)
        hf, bd, p0 = [], [], []
        for s in range(n):
            m = _final_metrics(Simulation(replace(cfg, seed=s)).run())
            hf.append(m["hf_capital"]); bd.append(m["bd_capital"]); p0.append(m["price0"])
        out[lvl] = {"hf_capital": np.array(hf), "bd_capital": np.array(bd),
                    "price0": np.array(p0)}
    return out


def _plot_sweep(swept: dict, levels, xlabel: str, title: str, fname: str) -> None:
    """Two-panel: (left) mean final HF & BD capital ±IQR; (right) mean asset-0 price ±IQR."""
    def band(metric):
        mean = [swept[l][metric].mean() for l in levels]
        lo = [np.percentile(swept[l][metric], 25) for l in levels]
        hi = [np.percentile(swept[l][metric], 75) for l in levels]
        return np.array(mean), np.array(lo), np.array(hi)

    fig, (axc, axp) = plt.subplots(1, 2, figsize=(13, 4.5))
    for metric, color, lbl in [("hf_capital", C_20, "Hedge funds"),
                               ("bd_capital", C_15, "Bank/dealers")]:
        m, lo, hi = band(metric)
        axc.plot(levels, m, "o-", color=color, lw=1.8, label=lbl)
        axc.fill_between(levels, lo, hi, color=color, alpha=0.15)
    axc.axhline(BASE.hf_initial_capital, color="black", lw=0.8, ls=":",
                label="Initial capital")
    axc.set_xlabel(xlabel); axc.set_ylabel("Mean final capital")
    axc.set_title("Agent capital"); axc.legend(fontsize=8); axc.grid(alpha=0.3)

    m, lo, hi = band("price0")
    axp.plot(levels, m, "o-", color="#1D9E75", lw=1.8)
    axp.fill_between(levels, lo, hi, color="#1D9E75", alpha=0.15)
    axp.axhline(BASE.initial_price, color="black", lw=0.8, ls=":", label="Initial price")
    axp.set_xlabel(xlabel); axp.set_ylabel("Mean final price")
    axp.set_title("Shocked asset (asset 0) price"); axp.legend(fontsize=8); axp.grid(alpha=0.3)

    fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, fname), bbox_inches="tight", dpi=150)
    plt.close(fig)


def fig_crowding() -> None:
    # crowding is only honoured when hf_allocations_hetero is EMPTY (engine.py:66).
    levels = [0.0, 0.25, 0.5, 0.75, 1.0]

    def variant(c):
        return replace(BASE, shock_size=-0.20, crowding=c,
                       hf_allocations_hetero=[], hf_allocation=[1.0 / 6] * 6)

    swept = _sweep(variant, levels, SWEEP_N)
    _plot_sweep(swept, levels, "Portfolio crowding (overlap)",
                f"Effect of portfolio crowding (−20% shock, N={SWEEP_N})",
                "fig_crowding.png")
    print("[fig_crowding] price0 means:",
          {l: round(float(swept[l]['price0'].mean()), 1) for l in levels})


def fig_leverage() -> None:
    # scale buffer/max proportionally off target (as robustness.py does).
    levels = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    ratio_buf = BASE.hf_lev_buffer / BASE.hf_lev_target
    ratio_max = BASE.hf_lev_max / BASE.hf_lev_target

    def variant(t):
        return replace(BASE, shock_size=-0.20, hf_lev_target=t,
                       hf_lev_buffer=t * ratio_buf, hf_lev_max=t * ratio_max)

    swept = _sweep(variant, levels, SWEEP_N)
    _plot_sweep(swept, levels, "Hedge-fund leverage target",
                f"Effect of hedge-fund leverage (−20% shock, N={SWEEP_N})",
                "fig_leverage.png")
    print("[fig_leverage] price0 means:",
          {l: round(float(swept[l]['price0'].mean()), 1) for l in levels})


# ---------------------------------------------------------------------------- #
# Figure 6 — when does contagion accrue (per-step trajectory)
# ---------------------------------------------------------------------------- #
def fig_contagion_timeline() -> None:
    csv = os.path.join(ROOT, "outputs", "contagion_decomposition", "traj_20pct.csv")
    if not os.path.exists(csv):
        print(f"[fig_contagion_timeline] SKIP — {csv} not found")
        return
    df = pd.read_csv(csv)
    df = df[(df["rel_step"] >= -1) & (df["rel_step"] <= 20)]
    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.cm.tab10(np.linspace(0, 0.6, len(AGENTS)))
    for agent, color in zip(AGENTS, cmap):
        sub = df[df["agent"] == agent].sort_values("rel_step")
        ax.plot(sub["rel_step"], sub["contagion_mean"], lw=1.8, label=agent, color=color)
    ax.axvline(0, color="#E24B4A", lw=1, ls="--", alpha=0.7, label="Shock")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Steps after shock")
    ax.set_ylabel("Mean contagion effect on capital ($)")
    ax.set_title("When contagion accrues, per agent (−20%)")
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_contagion_timeline.png"),
                bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("[fig_contagion_timeline] done")


# ---------------------------------------------------------------------------- #
# Figure 7 — event sequences (fire sale / default) for a sample of runs
# ---------------------------------------------------------------------------- #
def fig_event_sequence() -> None:
    csv = os.path.join(ROOT, "outputs", "contagion_decomposition", "events_20pct.csv")
    if not os.path.exists(csv):
        print(f"[fig_event_sequence] SKIP — {csv} not found")
        return
    df = pd.read_csv(csv)
    runs = sorted(df["run"].unique())[:12]   # up to 12 sample runs
    ncol = 3
    nrow = int(np.ceil(len(runs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 2.4 * nrow), squeeze=False)
    ymap = {a: i for i, a in enumerate(AGENTS)}
    for idx, run in enumerate(runs):
        ax = axes[idx // ncol][idx % ncol]
        sub = df[df["run"] == run]
        for _, r in sub.iterrows():
            y = ymap.get(r["agent"], -1)
            if r["event_type"] == "fire_sale":
                ax.scatter(r["rel_step"], y, marker="o", s=45,
                           facecolors="none", edgecolors="#EF9F27", lw=1.6, zorder=3)
            else:  # default
                ax.scatter(r["rel_step"], y, marker="x", s=55, color="#E24B4A",
                           lw=2.0, zorder=4)
        ax.set_yticks(range(len(AGENTS))); ax.set_yticklabels(AGENTS, fontsize=7)
        ax.set_xlim(-1, max(6, sub["rel_step"].max() + 1) if len(sub) else 6)
        ax.set_ylim(-0.5, len(AGENTS) - 0.5)
        ax.axvline(0, color="#E24B4A", lw=0.8, ls="--", alpha=0.5)
        ax.set_title(f"Run {run}", fontsize=9); ax.grid(alpha=0.25, axis="x")
        if idx % ncol == 0:
            ax.set_ylabel("agent", fontsize=8)
        if idx // ncol == nrow - 1:
            ax.set_xlabel("steps after shock", fontsize=8)
    # hide any unused axes
    for j in range(len(runs), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle("Cascade event sequences  (○ = enters fire sale,  ✕ = default)",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_event_sequence.png"),
                bbox_inches="tight", dpi=150)
    plt.close(fig)

    # also emit a compact Markdown event log for the first few runs (for prose)
    lines = ["| Run | Ordered events (step after shock) |", "|---|---|"]
    for run in runs[:5]:
        sub = df[df["run"] == run].sort_values(["step"])
        seq = ", ".join(
            f"{r['agent']} {'fire-sale' if r['event_type']=='fire_sale' else 'DEFAULT'} (+{int(r['rel_step'])})"
            for _, r in sub.iterrows()
        )
        lines.append(f"| {run} | {seq if seq else 'no events'} |")
    with open(os.path.join(FIG_DIR, "event_log_20pct.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[fig_event_sequence] {len(runs)} runs plotted; event log written")


def main() -> None:
    print(f"Writing figures to {FIG_DIR}")
    emit_regression_table()   # cheap (reads CSV)
    fig_contagion_decomp()    # cheap (reads CSV)
    fig_contagion_timeline()  # cheap (reads CSV)
    fig_event_sequence()      # cheap (reads CSV)
    fig_crisis_run()          # ~20 short sims
    fig_outcome_distribution()  # 2 * N_DIST sims
    fig_crowding()            # 5 * SWEEP_N sims
    fig_leverage()            # 6 * SWEEP_N sims
    print("Done.")


if __name__ == "__main__":
    main()
