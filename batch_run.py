"""
batch_run.py — run many simulations, produce per-run dashboards and a summary.

Run from the project root:
    PYTHONPATH=. python batch_run.py

Outputs
-------
outputs/runs/run_NNN.png   — per-run agent dashboards  (if SAVE_PER_RUN=True)
outputs/runs/run_NNN.csv   — per-run flat DataFrame (prices, capitals, leverages, …)
outputs/runs/run_NNN.json  — per-run full history including nested arrays (holdings, forced flows)
outputs/summary.png        — cross-run summary dashboard
"""
import os, sys, json, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*empty slice.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*All-NaN.*")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from dataclasses import replace

from bookstaber_abm.config import SimConfig
from bookstaber_abm.simulation.engine import Simulation
from bookstaber_abm.analysis.plots import history_to_df
from bookstaber_abm.analysis.buckets import summarize_runs, bucket_counts
from experiments.sweep import hetero_4hf_6assets_gradient, asym_2bd_6assets
from dashboard import make_dashboard

os.makedirs("outputs/runs", exist_ok=True)

# ── Experiment settings ────────────────────────────────────────────────────────
N_RUNS = 5

# Save a full per-run dashboard PNG for every run.
# With N_RUNS=100 this adds ~2-3 min and 100 image files — set False for speed.
SAVE_PER_RUN = True

# When True, all runs share the same HF portfolio weight vectors (generated once
# from BASE.seed).  Only the noise/execution randomness differs across runs.
# When False, each run also re-samples portfolio weights (crowding model).
FIXED_ALLOCATIONS = False

# ── Shared config: joint −15%/−20% calibration with diversified funding ──────
# Strict paper-classifier ("hf0_only" requires no other HF or BD even fire-sells).
# Cells EE_ship_n003_l070_15_n1000 / EE_ship_n003_l070_20_n1000 in sweep.py.
# Paper / This (N=1000, 2026-05-27 ship):
#   -15%:  no_default  hf0_only  partial  all_default
#   paper       0.4%       25%     68.6%        6%
#   this        0.0%     66.5%     22.3%     11.2%   ← inverted (Pareto trade-off)
#   -20%:
#   paper       2.4%       51%       30%       17%
#   this        0.0%     49.3%     35.0%     15.7%   ← three buckets within 5pp ✓
# Combined with the new rate-limited default-liquidation in hedge_fund.py
# (HF default no longer dumps all holdings in one step) and the lifted
# hf_max_liq_frac=0.70 / noise_std=0.003 settings. See CLAUDE.md
# "Stage A/B parameter sweep (2026-05-27)" for the full analysis.
BASE = SimConfig(
    n_assets=6, n_hedge_funds=4, n_bank_dealers=2, n_cash_providers=1,
    n_steps=200, shock_step=50, shock_asset=0, shock_size=-0.15,
    # Higher beta for stronger price-impact cascade. Necessary because HF0 is
    # the only entity whose default we want to drive — the cascade then propagates
    # to BDs/HFs via market prices (not via shared-BD funding).
    beta=0.55, beta1=0.0, normalise_beta=True, noise_std=0.003,
    hf_max_liq_frac=0.70, bd_max_liq_frac=0.70,
    fire_sale_shock_concentration=1.0,
    # HF leverage hierarchy — HF0 (f=0.60) defaults at -15% (post-shock lev ≈ 26).
    hf_lev_target=8.0, hf_lev_buffer=14.0, hf_lev_max=20.0,
    # BD leverage hierarchy lowered relative to prior calibration to keep BDs
    # robust during -15% cascades (so hf0_only mass survives).
    bd_lev_target=5.0, bd_lev_buffer=10.0, bd_lev_max=15.0,
    # Treasury thresholds matched to steady-state LiqRatio for this BD leverage.
    bd_liq_ratio_min=0.025, bd_liq_ratio_target=0.035,
    phi_cw=1000.0,
    cp_cw_smoothing_alpha=0.5,
    crowding=0.0,
    # Soft HF vulnerability gradient. Only HF0 is meant to default directly
    # from the shock; HF1/2/3 have very low shock-asset exposure so they can
    # only fail via cascade (market impact or BD failure).
    #   HF0 (canary) = 0.60 → post-shock lev ≈ 26 at -15%, always defaults
    #   HF1 = 0.20, HF2 = 0.18, HF3 = 0.18 → post-shock lev ≈ 11, survive directly
    hf_allocations_hetero=hetero_4hf_6assets_gradient(0.60, 0.20, 0.18, 0.18),
    hf_allocation=[1.0 / 6] * 6,
    bd_allocations_hetero=asym_2bd_6assets(tilt=0.02),
    # Diversified funding: HF0 routed entirely through BD0 (alone on BD0).
    # HF1/2/3 all routed through BD1. This isolates HF0's failure from HF1-3's
    # funding chain — HF1-3 can only fail via market price impact, not via
    # haircut tightening on the BD that HF0 is dragging down.
    hf_bd_funding_weights=[[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
    hf_funding_squeeze_threshold=0.02,
    cp_max_loan=300000.0,
    enable_derivatives_desk=False,
    seed=50,
)


# ── Run simulations ────────────────────────────────────────────────────────────
print(f"Running {N_RUNS} simulations  |  n_assets={BASE.n_assets}  "
      f"save_per_run={SAVE_PER_RUN}  fixed_alloc={FIXED_ALLOCATIONS}")
all_histories = []
for i in range(N_RUNS):
    cfg_i   = replace(BASE, seed=i)
    history = Simulation(cfg_i).run()
    all_histories.append(history)

    # CSV: flat per-step DataFrame (prices, capitals, leverages, haircuts, …)
    history_to_df(history).to_csv(f"outputs/runs/run_{i:03d}.csv")

    # JSON: full history including nested arrays (hf_holdings, hf_forced_flows, …)
    with open(f"outputs/runs/run_{i:03d}.json", "w") as f:
        json.dump(history, f)

    if SAVE_PER_RUN:
        out = f"outputs/runs/run_{i:03d}.png"
        make_dashboard(history, cfg_i, out, run_label=f"Run {i:03d}")
        print(f"  [{i+1:3d}/{N_RUNS}] saved {out}")
    else:
        print(f"  [{i+1:3d}/{N_RUNS}] done", end="\r")

print(f"\nAll {N_RUNS} runs done. Building summary dashboard...")

# ── Aggregate data across runs ─────────────────────────────────────────────────
N_STEPS = len(all_histories[0])
T       = np.arange(N_STEPS)
N_HF    = BASE.n_hedge_funds
N_BD    = BASE.n_bank_dealers
M       = BASE.n_assets

prices_all      = np.array([[r["prices"]           for r in h] for h in all_histories])  # (R,T,M)
n_fs_all        = np.array([[r["n_fire_sales"]      for r in h] for h in all_histories])  # (R,T)
n_def_all       = np.array([[r["n_defaults"]        for r in h] for h in all_histories])  # (R,T)
total_forced_all= np.array([[r["total_forced_flow"] for r in h] for h in all_histories])  # (R,T)
hf_cap_all      = np.array([[r["hf_capitals"]       for r in h] for h in all_histories])  # (R,T,N_HF)
hf_lev_all      = np.array([[r["hf_leverages"]      for r in h] for h in all_histories], dtype=float)
hf_fs_all       = np.array([[r["hf_in_fire_sale"]   for r in h] for h in all_histories])  # (R,T,N_HF)
bd_cap_all      = np.array([[r["bd_capitals"]        for r in h] for h in all_histories])  # (R,T,N_BD)
bd_lev_all      = np.array([[r["bd_leverages"]       for r in h] for h in all_histories], dtype=float)
haircuts_all    = np.array([[r["haircuts"]            for r in h] for h in all_histories])  # (R,T,N_BD)
overlap_all     = np.array([[r["portfolio_overlap"]   for r in h] for h in all_histories])  # (R,T)

hf_lev_all[~np.isfinite(hf_lev_all)] = np.nan
bd_lev_all[~np.isfinite(bd_lev_all)] = np.nan

# ── Derived cross-run metrics ──────────────────────────────────────────────────
hf_defaulted    = hf_cap_all[:, -1, :] <= 0          # (R, N_HF) bool
hf_default_prob = hf_defaulted.mean(axis=0)           # (N_HF,)

hf_default_step = np.full((N_RUNS, N_HF), np.nan)
for r in range(N_RUNS):
    for n in range(N_HF):
        steps = np.where(hf_cap_all[r, :, n] <= 0)[0]
        if len(steps):
            hf_default_step[r, n] = steps[0]

bd_defaulted    = bd_cap_all[:, -1, :] <= 0
bd_default_prob = bd_defaulted.mean(axis=0)

hf_fs_steps  = hf_fs_all.sum(axis=1)     # (R, N_HF)
final_defaults = n_def_all[:, -1]         # (R,)

# ── Bucket classification (compare to paper's outcome distribution) ───────────
distribution_df = summarize_runs(all_histories, shock_size=BASE.shock_size)
distribution_df.to_csv("outputs/distribution_summary.csv", index=False)
counts = bucket_counts(distribution_df)
total = sum(counts.values())
print("\nOutcome buckets (shock={:.0%}, N={}):".format(BASE.shock_size, total))
for name in ["no_default", "hf0_only", "partial", "all_default"]:
    n = counts[name]
    pct = (n / total * 100) if total else 0.0
    print(f"  {name:12s} : {n:4d}  ({pct:5.1f}%)")

# ── Palettes (scale with agent/asset counts) ───────────────────────────────────
HF_C    = (plt.cm.tab20(np.linspace(0.0, 0.95, max(N_HF, 2))) if N_HF > 10
           else plt.cm.tab10(np.linspace(0.0, 0.9,  max(N_HF, 1))))
BD_C    = plt.cm.Set2(np.linspace(0.0, 0.8, max(N_BD, 2)))
ASSET_C = (plt.cm.tab20(np.linspace(0.0, 0.95, max(M, 2))) if M > 10
           else plt.cm.Set1(np.linspace(0.0, 0.6,  max(M, 1))))
SHOCK_KW  = dict(color="#d62728", lw=1.0, ls="--", alpha=0.65)
RUN_ALPHA = max(0.04, min(0.35, 4.0 / N_RUNS))

def vl(ax):
    ax.axvline(BASE.shock_step, **SHOCK_KW)

def sty(ax, title, ylabel="", xlabel="Step", legend=True):
    ax.set_title(title, fontsize=8.5, fontweight="bold", pad=3)
    ax.set_ylabel(ylabel, fontsize=7.5); ax.set_xlabel(xlabel, fontsize=7.5)
    ax.tick_params(labelsize=6.5); ax.grid(alpha=0.25, lw=0.5)
    if legend: ax.legend(fontsize=6, loc="best", framealpha=0.6)

def confidence_band(ax, data_RT, color, label=""):
    mean = np.nanmean(data_RT, axis=0)
    p10  = np.nanpercentile(data_RT, 10, axis=0)
    p90  = np.nanpercentile(data_RT, 90, axis=0)
    for r in range(N_RUNS):
        ax.plot(T, data_RT[r], color=color, lw=0.4, alpha=RUN_ALPHA)
    ax.fill_between(T, p10, p90, color=color, alpha=0.18)
    ax.plot(T, mean, color=color, lw=2.0, label=label or "Mean")

# ── Summary figure (5 rows × 3 cols) ──────────────────────────────────────────
fig = plt.figure(figsize=(20, 28))
fig.patch.set_facecolor("#f0f0f0")
gs  = gridspec.GridSpec(5, 3, figure=fig, hspace=0.58, wspace=0.35,
                        top=0.95, bottom=0.04, left=0.07, right=0.97)
fig.suptitle(
    f"ABM Summary — {N_RUNS} runs  |  shock={BASE.shock_size*100:.0f}% at t={BASE.shock_step}  "
    f"β={BASE.beta}  n_assets={M}  lev_max={BASE.hf_lev_max}x  "
    f"crowding={BASE.crowding}  fixed_alloc={FIXED_ALLOCATIONS}",
    fontsize=10, y=0.975, fontweight="bold",
)

# ── Row 0: Asset price paths ─────────────────────────────────────────────────
if M <= 3:
    # One panel per asset (original layout)
    for m in range(M):
        ax = fig.add_subplot(gs[0, m])
        confidence_band(ax, prices_all[:, :, m], ASSET_C[m], label=f"Mean A{m}")
        vl(ax); sty(ax, f"Asset {m} Price — all runs", ylabel="Price")
else:
    # Panel 0: Mean price line per asset (no individual run lines — too many)
    ax = fig.add_subplot(gs[0, 0])
    for m in range(M):
        mean_p = np.nanmean(prices_all[:, :, m], axis=0)
        ax.plot(T, mean_p, lw=0.9, color=ASSET_C[m], alpha=0.8)
    vl(ax)
    if M > 10:
        sm = plt.cm.ScalarMappable(cmap="tab20", norm=plt.Normalize(0, M - 1))
        sm.set_array([]); plt.colorbar(sm, ax=ax, label="Asset index", pad=0.01)
    sty(ax, f"Mean Price per Asset (all {M})", ylabel="Price", legend=False)

    # Panel 1: Shocked asset with full CI band
    shock_m = BASE.shock_asset
    ax = fig.add_subplot(gs[0, 1])
    confidence_band(ax, prices_all[:, :, shock_m], ASSET_C[shock_m],
                    label=f"Asset {shock_m} (shocked)")
    vl(ax); sty(ax, f"Shocked Asset {shock_m} — all runs", ylabel="Price")

    # Panel 2: Market price index (mean across assets)
    ax = fig.add_subplot(gs[0, 2])
    price_index = prices_all.mean(axis=2)   # (R, T)
    confidence_band(ax, price_index, "#5d5d5d", label="Market index (mean)")
    vl(ax); sty(ax, "Market Price Index (Σ assets)", ylabel="Avg price")

# ── Row 1: Fire-sale heatmap | HF default probability | Default timing ────────
ax = fig.add_subplot(gs[1, 0])
im = ax.imshow(n_fs_all, aspect="auto", origin="lower",
               cmap="Reds", interpolation="nearest",
               extent=[0, N_STEPS-1, -0.5, N_RUNS-0.5])
ax.axvline(BASE.shock_step, color="white", lw=1.2, ls="--", alpha=0.8)
plt.colorbar(im, ax=ax, label="# agents fire-selling", pad=0.02)
ax.set_xlabel("Step", fontsize=7.5); ax.set_ylabel("Run", fontsize=7.5)
ax.tick_params(labelsize=6.5)
ax.set_title("Fire-Sale Activity Heatmap (run × step)", fontsize=8.5,
             fontweight="bold", pad=3)

ax = fig.add_subplot(gs[1, 1])
x_hf = np.arange(N_HF)
bars = ax.bar(x_hf, hf_default_prob * 100, color=HF_C, alpha=0.85, edgecolor="white")
for b, p in zip(bars, hf_default_prob):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1.0,
            f"{p*100:.0f}%", ha="center", va="bottom", fontsize=6)
ax.set_xticks(x_hf)
ax.set_xticklabels([f"HF {n}" for n in range(N_HF)], fontsize=6, rotation=45)
ax.set_ylabel("Default rate (%)", fontsize=7.5)
ax.set_ylim(0, 110); ax.tick_params(labelsize=6.5); ax.grid(axis="y", alpha=0.3)
for k in range(N_BD):
    ax.axhline(bd_default_prob[k] * 100, color=BD_C[k], lw=1.5, ls=":",
               alpha=0.8, label=f"BD {k} rate")
ax.set_title("Default Probability per Agent", fontsize=8.5, fontweight="bold", pad=3)
ax.legend(fontsize=6, framealpha=0.6)

ax = fig.add_subplot(gs[1, 2])
rng_jitter = np.random.default_rng(99)
for n in range(N_HF):
    steps = hf_default_step[:, n]
    valid = steps[~np.isnan(steps)]
    if len(valid):
        jitter = rng_jitter.uniform(-0.3, 0.3, len(valid))
        ax.scatter(valid, np.full(len(valid), n) + jitter,
                   color=HF_C[n], s=15, alpha=0.75, zorder=3)
ax.set_yticks(range(N_HF))
ax.set_yticklabels([f"HF {n}" for n in range(N_HF)], fontsize=6)
ax.set_xlabel("Step of first default", fontsize=7.5)
ax.axvline(BASE.shock_step, **SHOCK_KW)
ax.tick_params(labelsize=6.5); ax.grid(alpha=0.25, lw=0.5)
ax.set_title("HF Default Timing (one dot = one run)", fontsize=8.5,
             fontweight="bold", pad=3)
if hf_default_step[~np.isnan(hf_default_step)].size == 0:
    ax.text(0.5, 0.5, "No defaults", ha="center", va="center",
            transform=ax.transAxes, fontsize=10, color="gray")

# ── Row 2: HF capital paths | HF final capital box | BD capital paths ─────────
ax = fig.add_subplot(gs[2, 0])
for n in range(N_HF):
    confidence_band(ax, hf_cap_all[:, :, n], HF_C[n], label=f"HF {n}")
ax.axhline(0, color="black", lw=0.7, ls=":")
vl(ax); sty(ax, "HF Capital — all runs (band=P10/P90)", ylabel="Capital ($)",
            legend=(N_HF <= 8))

ax = fig.add_subplot(gs[2, 1])
data_box = [hf_cap_all[:, -1, n] for n in range(N_HF)]
bp = ax.boxplot(data_box, patch_artist=True, notch=False,
                medianprops=dict(color="black", lw=1.5))
for patch, color in zip(bp["boxes"], HF_C):
    patch.set_facecolor(color); patch.set_alpha(0.7)
ax.axhline(0, color="#d62728", lw=0.9, ls="--", alpha=0.7)
ax.set_xticks(range(1, N_HF+1))
ax.set_xticklabels([f"HF {n}" for n in range(N_HF)], fontsize=6, rotation=45)
ax.set_ylabel("Final capital ($)", fontsize=7.5)
ax.tick_params(labelsize=6.5); ax.grid(axis="y", alpha=0.3)
ax.set_title("HF Final Capital Distribution", fontsize=8.5, fontweight="bold", pad=3)

ax = fig.add_subplot(gs[2, 2])
for k in range(N_BD):
    confidence_band(ax, bd_cap_all[:, :, k], BD_C[k], label=f"BD {k}")
ax.axhline(0, color="black", lw=0.7, ls=":")
vl(ax); sty(ax, "BD Capital — all runs (band=P10/P90)", ylabel="Capital ($)")

# ── Row 3: HF leverage | HF fire-sale steps | BD leverage ────────────────────
ax = fig.add_subplot(gs[3, 0])
for n in range(N_HF):
    confidence_band(ax, hf_lev_all[:, :, n], HF_C[n], label=f"HF {n}")
ax.axhline(BASE.hf_lev_max,    color="#d62728", lw=1.0, ls="--", alpha=0.8, label="Max")
ax.axhline(BASE.hf_lev_buffer, color="#ff7f0e", lw=1.0, ls="--", alpha=0.8, label="Buffer")
ax.axhline(BASE.hf_lev_target, color="#2ca02c", lw=1.0, ls="--", alpha=0.8, label="Target")
vl(ax); sty(ax, "HF Leverage — all runs", ylabel="Leverage", legend=(N_HF <= 6))

ax = fig.add_subplot(gs[3, 1])
mean_fs = hf_fs_steps.mean(axis=0)
std_fs  = hf_fs_steps.std(axis=0)
x_pos   = np.arange(N_HF)
ax.bar(x_pos, mean_fs, color=HF_C, alpha=0.8, edgecolor="white", label="Mean")
ax.errorbar(x_pos, mean_fs, yerr=std_fs, fmt="none",
            color="black", capsize=3, lw=1.2)
if N_RUNS <= 50:
    for n in range(N_HF):
        jitter = np.random.default_rng(n).uniform(-0.25, 0.25, N_RUNS)
        ax.scatter(n + jitter, hf_fs_steps[:, n],
                   color=HF_C[n], s=8, alpha=0.4, zorder=3)
ax.set_xticks(x_pos)
ax.set_xticklabels([f"HF {n}" for n in range(N_HF)], fontsize=6, rotation=45)
ax.set_ylabel("Steps in fire sale", fontsize=7.5)
ax.tick_params(labelsize=6.5); ax.grid(axis="y", alpha=0.3)
ax.set_title("HF Fire-Sale Steps per Run (mean ± std)", fontsize=8.5,
             fontweight="bold", pad=3)

ax = fig.add_subplot(gs[3, 2])
for k in range(N_BD):
    confidence_band(ax, bd_lev_all[:, :, k], BD_C[k], label=f"BD {k}")
ax.axhline(BASE.bd_lev_max,    color="#d62728", lw=1.0, ls="--", alpha=0.8, label="Max")
ax.axhline(BASE.bd_lev_buffer, color="#ff7f0e", lw=1.0, ls="--", alpha=0.8, label="Buffer")
ax.axhline(BASE.bd_lev_target, color="#2ca02c", lw=1.0, ls="--", alpha=0.8, label="Target")
vl(ax); sty(ax, "BD Leverage — all runs", ylabel="Leverage")

# ── Row 4: Forced flow | Haircut | Defaults distribution ─────────────────────
ax = fig.add_subplot(gs[4, 0])
confidence_band(ax, total_forced_all, "#d62728", label="Mean total forced flow")
vl(ax); sty(ax, "Total Forced Flow — all runs", ylabel="|qty| per step", legend=False)
ax.legend(fontsize=6, framealpha=0.6)

ax = fig.add_subplot(gs[4, 1])
for k in range(N_BD):
    confidence_band(ax, haircuts_all[:, :, k], BD_C[k], label=f"BD {k}")
ax.axhline(BASE.cp_haircut_normal,   color="#2ca02c", lw=0.9, ls=":", alpha=0.8)
ax.axhline(BASE.cp_haircut_stressed, color="#d62728", lw=0.9, ls=":", alpha=0.8)
vl(ax); sty(ax, "CP Haircut — all runs", ylabel="Haircut fraction")

ax = fig.add_subplot(gs[4, 2])
max_def = int(final_defaults.max()) if final_defaults.max() > 0 else 1
bins    = np.arange(-0.5, max_def + 1.5)
ax.hist(final_defaults, bins=bins, color="#5d5d5d", edgecolor="white", alpha=0.8)
ax.axvline(final_defaults.mean(), color="#d62728", lw=1.8, ls="--",
           label=f"Mean = {final_defaults.mean():.1f}")
ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax.set_xlabel("Total HF defaults at run end", fontsize=7.5)
ax.set_ylabel("# runs", fontsize=7.5)
ax.tick_params(labelsize=6.5); ax.grid(axis="y", alpha=0.3)
ax.set_title("HF Default Count Distribution", fontsize=8.5, fontweight="bold", pad=3)
p_any = (final_defaults >= 1).mean()
ax.text(0.97, 0.95, f"P(any default) = {p_any:.0%}",
        ha="right", va="top", transform=ax.transAxes, fontsize=7.5,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
ax.legend(fontsize=7, framealpha=0.6)

# ── Bottom legend (capped per class) ──────────────────────────────────────────
_mh = min(N_HF, 8); _mb = min(N_BD, 4); _ma = min(M, 6)
legend_elems = (
    [Line2D([0],[0], color=HF_C[n],    lw=1.5, label=f"HF {n}")    for n in range(_mh)]
  + ([Line2D([0],[0], color="gray", lw=1.0, label=f"…+{N_HF-_mh} HFs")] if N_HF > _mh else [])
  + [Line2D([0],[0], color=BD_C[k],    lw=1.5, label=f"BD {k}")    for k in range(_mb)]
  + [Line2D([0],[0], color=ASSET_C[m], lw=1.5, label=f"Asset {m}") for m in range(_ma)]
  + ([Line2D([0],[0], color="gray", lw=1.0, label=f"…+{M-_ma} assets")] if M > _ma else [])
  + [Line2D([0],[0], color="#d62728",  lw=1.2, ls="--", label="Shock/LevMax")]
  + [Line2D([0],[0], color="#ff7f0e",  lw=1.0, ls="--", label="LevBuffer")]
  + [Line2D([0],[0], color="#2ca02c",  lw=1.0, ls="--", label="LevTarget")]
  + [mpatches.Patch(facecolor="gray",  alpha=0.3, label="P10–P90 band")]
)
fig.legend(handles=legend_elems, loc="lower center", ncol=10,
           fontsize=6.5, framealpha=0.85, bbox_to_anchor=(0.5, 0.005))

plt.savefig("outputs/summary.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close(fig)
print("Saved: outputs/summary.png")
print(f"\nDone. Summary: outputs/summary.png"
      + (f"  |  Per-run dashboards: outputs/runs/" if SAVE_PER_RUN else ""))
