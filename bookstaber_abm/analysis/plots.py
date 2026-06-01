"""
analysis/plots.py — Plotting utilities for the Bookstaber ABM.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bookstaber_abm.config import SimConfig
    from bookstaber_abm.simulation.monte_carlo import MonteCarloRunner


def history_to_df(history: list[dict]) -> pd.DataFrame:
    rows = []
    for r in history:
        row = {"t": r["t"]}
        for k in ("n_fire_sales", "n_active_hf", "n_defaults", "n_bd_defaults",
                  "total_forced_flow", "portfolio_overlap",
                  "deriv_exposure", "deriv_losses", "shock_active"):
            row[k] = r.get(k, 0)
        for m, p in enumerate(r["prices"]): row[f"price_{m}"] = p
        for m, f in enumerate(r["net_forced_flow"]): row[f"flow_{m}"] = f
        for n, c in enumerate(r["hf_capitals"]): row[f"hf_cap_{n}"] = c
        for n, lev in enumerate(r["hf_leverages"]): row[f"hf_lev_{n}"] = lev
        for k2, c in enumerate(r["bd_capitals"]): row[f"bd_cap_{k2}"] = c
        for k2, hc in enumerate(r.get("haircuts", [])): row[f"haircut_bd{k2}"] = hc
        rows.append(row)
    return pd.DataFrame(rows).set_index("t")


def plot_crisis(df: pd.DataFrame, cfg, save_path=None):
    fig = plt.figure(figsize=(13, 9))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)
    price_cols = sorted(c for c in df.columns if c.startswith("price_"))
    cap_cols   = sorted(c for c in df.columns if c.startswith("hf_cap_"))
    lev_cols   = sorted(c for c in df.columns if c.startswith("hf_lev_"))
    hc_cols    = sorted(c for c in df.columns if c.startswith("haircut_"))
    t = cfg.shock_step

    ax1 = fig.add_subplot(gs[0, 0])
    colors = plt.cm.tab10(np.linspace(0, 0.6, len(price_cols)))
    for col, c in zip(price_cols, colors):
        ax1.plot(df.index, df[col], lw=1.5, label=col.replace("price_", "Asset "), color=c)
    ax1.axvline(t, color="#E24B4A", lw=1, ls="--", alpha=0.7, label="Shock")
    ax1.set_title("Asset prices", fontsize=11); ax1.set_xlabel("Step"); ax1.set_ylabel("Price")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(df.index, df["total_forced_flow"].abs(), color="#E24B4A", alpha=0.65, width=1.0)
    ax2.axvline(t, color="#E24B4A", lw=1, ls="--", alpha=0.7)
    ax2.set_ylabel("Total forced sales", fontsize=9)
    if hc_cols:
        ax2b = ax2.twinx()
        for col in hc_cols:
            ax2b.plot(df.index, df[col], lw=1.5, ls=":", color="#185FA5", alpha=0.9)
        ax2b.set_ylabel("Haircut", fontsize=9, color="#185FA5")
        ax2b.tick_params(axis='y', colors="#185FA5"); ax2b.set_ylim(0, 0.35)
    ax2.set_title("Fire sales & haircut", fontsize=11); ax2.set_xlabel("Step"); ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 0])
    for col in cap_cols:
        ax3.plot(df.index, df[col], lw=1.2, label=col.replace("hf_cap_", "HF "))
    ax3.axvline(t, color="#E24B4A", lw=1, ls="--", alpha=0.7)
    ax3.axhline(0, color="black", lw=0.8, ls=":")
    ax3.set_title("HF capital", fontsize=11); ax3.set_xlabel("Step"); ax3.set_ylabel("Capital")
    ax3.legend(fontsize=7, ncol=2); ax3.grid(alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 1])
    for col in lev_cols:
        s = df[col].replace([np.inf, -np.inf], np.nan)
        ax4.plot(df.index, s, lw=1.2, label=col.replace("hf_lev_", "HF "))
    ax4.axvline(t, color="#E24B4A", lw=1, ls="--", alpha=0.7)
    ax4.axhline(cfg.hf_lev_max,    color="#E24B4A", lw=1.0, ls="--", alpha=0.8, label="Max")
    ax4.axhline(cfg.hf_lev_buffer, color="#EF9F27", lw=1.0, ls="--", alpha=0.8, label="Buffer")
    ax4.axhline(cfg.hf_lev_target, color="#1D9E75", lw=1.0, ls="--", alpha=0.8, label="Target")
    ax4.set_title("HF leverage", fontsize=11); ax4.set_xlabel("Step"); ax4.set_ylabel("Leverage")
    ax4.legend(fontsize=7, ncol=2); ax4.grid(alpha=0.3)

    fig.suptitle(
        f"Bookstaber ABM  |  shock={cfg.shock_size*100:.0f}% at t={t}  "
        f"β={cfg.beta}  N={cfg.n_hedge_funds} HFs  crowding={cfg.crowding}",
        fontsize=11, y=1.01,
    )
    if save_path: plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.tight_layout()
    return fig


def plot_monte_carlo(mc, cfg, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    t_idx = np.arange(cfg.n_steps)

    b = mc.bands("price_0")
    ax = axes[0]
    ax.fill_between(t_idx, b["p5"], b["p95"], alpha=0.2, color="#185FA5", label="5–95th pct")
    ax.fill_between(t_idx, b["p25"], b["p75"], alpha=0.35, color="#185FA5", label="25–75th pct")
    ax.plot(t_idx, b["mean"], lw=1.8, color="#185FA5", label="Mean")
    ax.plot(t_idx, b["median"], lw=1.2, color="#0C447C", ls="--", label="Median")
    ax.axvline(cfg.shock_step, color="#E24B4A", lw=1, ls="--", alpha=0.7, label="Shock")
    ax.set_title(f"Asset 0 price  (n={cfg.mc_runs} runs)", fontsize=11)
    ax.set_xlabel("Step"); ax.set_ylabel("Price"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    dist = mc.default_distribution()
    ax = axes[1]
    bins = np.arange(-0.5, max(dist.max() + 1.5, 2))
    ax.hist(dist, bins=bins, color="#185FA5", alpha=0.75, edgecolor="white")
    ax.axvline(dist.mean(), color="#E24B4A", lw=1.5, ls="--", label=f"Mean={dist.mean():.1f}")
    ax.set_title("HF default distribution", fontsize=11)
    ax.set_xlabel("Defaults"); ax.set_ylabel("Count"); ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")

    p_crisis = mc.crisis_probability()
    ax = axes[2]
    ax.plot(t_idx, p_crisis, lw=2, color="#E24B4A")
    ax.fill_between(t_idx, 0, p_crisis, alpha=0.15, color="#E24B4A")
    ax.axvline(cfg.shock_step, color="#E24B4A", lw=1, ls="--", alpha=0.7)
    ax.set_ylim(0, 1.05)
    ax.set_title("P(fire sale ≥ 1) per step", fontsize=11)
    ax.set_xlabel("Step"); ax.set_ylabel("Probability"); ax.grid(alpha=0.3)

    fig.suptitle(
        f"Monte Carlo  |  {cfg.mc_runs} runs  |  shock={cfg.shock_size*100:.0f}%"
        f"  β={cfg.beta}  lev_target={cfg.hf_lev_target}x  crowding={cfg.crowding}",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    if save_path: plt.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


def plot_crowding_comparison(base_cfg, crowding_levels=None, n_runs=15, save_path=None):
    from dataclasses import replace
    from bookstaber_abm.simulation.monte_carlo import MonteCarloRunner

    if crowding_levels is None:
        crowding_levels = [0.0, 0.25, 0.5, 0.75, 1.0]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(crowding_levels)))
    t_idx = np.arange(base_cfg.n_steps)

    for crowding, color in zip(crowding_levels, colors):
        cfg = replace(base_cfg, crowding=crowding, mc_runs=n_runs)
        mc = MonteCarloRunner(cfg).run()
        b = mc.bands("price_0")
        p = mc.crisis_probability()
        label = f"crowding={crowding:.2f}"
        axes[0].plot(t_idx, b["median"], lw=1.8, color=color, label=label)
        axes[0].fill_between(t_idx, b["p25"], b["p75"], alpha=0.10, color=color)
        axes[1].plot(t_idx, p, lw=1.8, color=color, label=label)

    for ax in axes:
        ax.axvline(base_cfg.shock_step, color="black", lw=1, ls="--", alpha=0.4)
        ax.grid(alpha=0.3); ax.legend(fontsize=8)

    axes[0].set_title("Median price ±IQR", fontsize=11)
    axes[0].set_xlabel("Step"); axes[0].set_ylabel("Price")
    axes[1].set_title("P(fire sale ≥ 1)", fontsize=11)
    axes[1].set_xlabel("Step"); axes[1].set_ylabel("Probability"); axes[1].set_ylim(0, 1.05)

    fig.suptitle(
        f"Crowding comparison  |  shock={base_cfg.shock_size*100:.0f}%"
        f"  β={base_cfg.beta}  {n_runs} runs each",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    if save_path: plt.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


def plot_phase_space(results, x_label="shock_size", y_label="beta",
                     metric="max_fire_sales", save_path=None):
    xs = sorted(set(k[0] for k in results))
    ys = sorted(set(k[1] for k in results))
    grid = np.zeros((len(ys), len(xs)))
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            hist = results.get((x, y), [])
            if not hist: continue
            if metric == "max_fire_sales":
                grid[i, j] = max(r["n_fire_sales"] for r in hist)
            elif metric == "n_defaults":
                grid[i, j] = hist[-1]["n_defaults"]
            elif metric == "price_drop":
                p0 = hist[0]["prices"][0]
                grid[i, j] = (min(r["prices"][0] for r in hist) - p0) / p0 * 100

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(grid, aspect="auto", origin="lower", cmap="RdYlGn_r")
    ax.set_xticks(range(len(xs))); ax.set_xticklabels([f"{x:.3f}" for x in xs], rotation=45, fontsize=9)
    ax.set_yticks(range(len(ys))); ax.set_yticklabels([f"{y:.3f}" for y in ys], fontsize=9)
    ax.set_xlabel(x_label); ax.set_ylabel(y_label); ax.set_title(f"Phase space: {metric}")
    plt.colorbar(im, ax=ax); plt.tight_layout()
    if save_path: plt.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig
