"""
dashboard.py — per-run agent variable dashboard for the Bookstaber ABM.

Standalone:
    PYTHONPATH=. python dashboard.py          → outputs/dashboard.png

As a module:
    from dashboard import make_dashboard
    make_dashboard(history, cfg, "outputs/run_00.png")
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D


def make_dashboard(history, cfg, save_path, run_label=""):
    """
    Render the full 6-row per-agent dashboard for one simulation run.

    Parameters
    ----------
    history   : list of step-dicts from Simulation.run()
    cfg       : SimConfig used for that run
    save_path : where to write the PNG
    run_label : optional string appended to the figure title (e.g. "Run 03")
    """
    N_STEPS = len(history)
    T       = np.arange(N_STEPS)
    SHOCK_T = cfg.shock_step
    N_HF    = cfg.n_hedge_funds
    N_BD    = cfg.n_bank_dealers
    N_CP    = cfg.n_cash_providers
    M       = cfg.n_assets
    bd_ids  = [f"BD_{k}" for k in range(N_BD)]

    # ── Extract arrays ────────────────────────────────────────────────────────
    prices       = np.array([r["prices"]          for r in history])
    forced_flow  = np.array([r["net_forced_flow"]  for r in history])
    n_fire_sales = np.array([r["n_fire_sales"]     for r in history])
    n_defaults   = np.array([r["n_defaults"]       for r in history])
    n_bd_def     = np.array([r["n_bd_defaults"]    for r in history])
    total_forced = np.array([r["total_forced_flow"] for r in history])
    overlap      = np.array([r["portfolio_overlap"] for r in history])

    hf_capitals  = np.array([r["hf_capitals"]      for r in history])
    hf_leverages = np.array([r["hf_leverages"]      for r in history], dtype=float)
    hf_fundings  = np.array([r["hf_fundings"]       for r in history])
    hf_holdings  = np.array([r["hf_holdings"]       for r in history])
    hf_fire_sale = np.array([r["hf_in_fire_sale"]   for r in history])
    hf_forced    = np.abs(np.array([r["hf_forced_flows"] for r in history]))
    bd_forced    = np.abs(np.array([r["bd_forced_flows"] for r in history]))

    bd_capitals  = np.array([r["bd_capitals"]         for r in history])
    bd_leverages = np.array([r["bd_leverages"]         for r in history], dtype=float)
    bd_fund_cp   = np.array([r["bd_fundings_from_cp"]  for r in history])
    bd_fire_sale = np.array([r["bd_in_fire_sale"]      for r in history])

    bd_liq_ratios   = np.array([r["bd_liq_ratios"]    for r in history])
    bd_cw           = np.array([r["bd_cw"]            for r in history])
    bd_liq_reserves = np.array([r["bd_liq_reserves"]  for r in history])
    bd_liq_debits   = np.array([r["bd_liq_debits"]    for r in history])

    haircuts     = np.array([r["haircuts"] for r in history])
    cp_loans     = np.array([
        [[r["cp_loans"][c].get(bid, 0.0) for bid in bd_ids] for c in range(N_CP)]
        for r in history
    ])

    hf_leverages[~np.isfinite(hf_leverages)] = np.nan
    bd_leverages[~np.isfinite(bd_leverages)] = np.nan

    # ── Palettes (scale with agent/asset counts) ──────────────────────────────
    HF_C    = (plt.cm.tab20(np.linspace(0.0, 0.95, max(N_HF, 2))) if N_HF > 10
               else plt.cm.tab10(np.linspace(0.0, 0.9,  max(N_HF, 1))))
    BD_C    = plt.cm.Set2(np.linspace(0.0, 0.8, max(N_BD, 2)))
    CP_C    = plt.cm.Dark2(np.linspace(0.0, 0.6, max(N_CP, 2)))
    ASSET_C = (plt.cm.tab20(np.linspace(0.0, 0.95, max(M, 2))) if M > 10
               else plt.cm.Set1(np.linspace(0.0, 0.6,  max(M, 1))))
    SHOCK_KW = dict(color="#d62728", lw=1.2, ls="--", alpha=0.7)
    HATCHES  = ["", "/", "x", "\\", "+", "o", "*", "-", "|"]  # cycles if M > len

    def vl(ax):  ax.axvline(SHOCK_T, **SHOCK_KW)

    def sty(ax, title, ylabel="", xlabel="Step", legend=True):
        ax.set_title(title, fontsize=8.5, fontweight="bold", pad=3)
        ax.set_ylabel(ylabel, fontsize=7.5); ax.set_xlabel(xlabel, fontsize=7.5)
        ax.tick_params(labelsize=6.5); ax.grid(alpha=0.25, lw=0.5)
        if legend: ax.legend(fontsize=6, loc="best", framealpha=0.6)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 34))
    fig.patch.set_facecolor("#f5f5f5")
    gs  = gridspec.GridSpec(8, 3, figure=fig, hspace=0.55, wspace=0.35,
                            top=0.95, bottom=0.04, left=0.07, right=0.97)
    title = (f"ABM Dashboard  |  shock={cfg.shock_size*100:.0f}% at t={SHOCK_T}  "
             f"β={cfg.beta}  lev_max={cfg.hf_lev_max}x  crowding={cfg.crowding}  "
             f"seed={cfg.seed}")
    if run_label:
        title = f"[{run_label}]  " + title
    fig.suptitle(title, fontsize=10, y=0.975, fontweight="bold")

    # Row 0 — Market
    ax = fig.add_subplot(gs[0, 0])
    for m in range(M):
        ax.plot(T, prices[:, m], lw=1.5, color=ASSET_C[m], label=f"Asset {m}")
    vl(ax); sty(ax, "Asset Prices", ylabel="Price")

    ax = fig.add_subplot(gs[0, 1])
    for m in range(M):
        ax.plot(T, forced_flow[:, m], lw=1.2, color=ASSET_C[m], label=f"Asset {m}")
    ax.axhline(0, color="black", lw=0.6, ls=":")
    vl(ax); sty(ax, "Net Forced Flow per Asset", ylabel="Qty (−=sell)")

    ax = fig.add_subplot(gs[0, 2])
    ax.bar(T, n_fire_sales, color="#d62728", alpha=0.65, width=1.0, label="# fire-selling")
    ax2 = ax.twinx()
    ax2.step(T, n_defaults, color="#7f7f7f", lw=1.5, where="post", label="HF def (cum)")
    ax2.step(T, n_bd_def,   color="#bcbd22", lw=1.5, where="post", label="BD def (cum)")
    ax2.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax2.set_ylabel("Cumulative defaults", fontsize=7); ax2.tick_params(labelsize=6.5)
    ax2.legend(fontsize=5.5, loc="upper left", framealpha=0.6)
    vl(ax); sty(ax, "Fire-Sale Count & Defaults", ylabel="# agents")

    # Row 1 — HF state
    ax = fig.add_subplot(gs[1, 0])
    for n in range(N_HF):
        ax.plot(T, hf_capitals[:, n], lw=1.3, color=HF_C[n], label=f"HF {n}")
    ax.axhline(0, color="black", lw=0.7, ls=":")
    vl(ax); sty(ax, "HF Capital", ylabel="Capital ($)")

    ax = fig.add_subplot(gs[1, 1])
    for n in range(N_HF):
        ax.plot(T, hf_leverages[:, n], lw=1.1, color=HF_C[n], label=f"HF {n}", alpha=0.85)
        fs = np.where(hf_fire_sale[:, n])[0]
        if len(fs):
            ax.scatter(fs, hf_leverages[fs, n], color=HF_C[n], marker="x", s=28, zorder=5)
    ax.axhline(cfg.hf_lev_max,    color="#d62728", lw=1.0, ls="--", alpha=0.85, label="Max")
    ax.axhline(cfg.hf_lev_buffer, color="#ff7f0e", lw=1.0, ls="--", alpha=0.85, label="Buffer")
    ax.axhline(cfg.hf_lev_target, color="#2ca02c", lw=1.0, ls="--", alpha=0.85, label="Target")
    vl(ax); sty(ax, "HF Leverage  (x = fire-sale step)", ylabel="Leverage")

    ax = fig.add_subplot(gs[1, 2])
    for n in range(N_HF):
        ax.plot(T, hf_fundings[:, n], lw=1.3, color=HF_C[n], label=f"HF {n}")
    ax.axhline(0, color="black", lw=0.7, ls=":")
    vl(ax); sty(ax, "HF Funding Needed", ylabel="Funding ($)")

    # Row 2 — HF holdings (consolidated: works for any M)
    # Panel [2,0]: total holdings per asset summed across all HFs
    ax = fig.add_subplot(gs[2, 0])
    holdings_by_asset = hf_holdings.sum(axis=1)   # (T, M) — Σ over HF
    for m in range(M):
        ax.plot(T, holdings_by_asset[:, m], lw=1.1, color=ASSET_C[m],
                alpha=0.8, label=f"A{m}" if M <= 8 else None)
    if M > 8:
        sm = plt.cm.ScalarMappable(
            cmap="tab20" if M > 10 else "Set1",
            norm=plt.Normalize(0, M - 1))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="Asset index", pad=0.01)
    vl(ax); sty(ax, "Total Holdings by Asset (Σ HFs)", ylabel="Shares", legend=(M <= 8))

    # Panel [2,1]: total holdings per HF summed across all assets
    ax = fig.add_subplot(gs[2, 1])
    for n in range(N_HF):
        hf_total = hf_holdings[:, n, :].sum(axis=1)   # (T,)
        ax.plot(T, hf_total, lw=1.1, color=HF_C[n], alpha=0.8,
                label=f"HF {n}" if N_HF <= 10 else None)
    if N_HF > 10:
        sm2 = plt.cm.ScalarMappable(
            cmap="tab20", norm=plt.Normalize(0, N_HF - 1))
        sm2.set_array([])
        plt.colorbar(sm2, ax=ax, label="HF index", pad=0.01)
    vl(ax); sty(ax, "Total Holdings by HF (Σ assets)", ylabel="Shares", legend=(N_HF <= 10))

    # Panel [2,2]: heatmap of final-step holdings (HF × Asset)
    ax = fig.add_subplot(gs[2, 2])
    final_h = hf_holdings[-1]   # (N_HF, M)
    im = ax.imshow(final_h, aspect="auto", origin="upper", cmap="YlOrRd",
                   extent=[-0.5, M - 0.5, N_HF - 0.5, -0.5])
    plt.colorbar(im, ax=ax, label="Shares", pad=0.01)
    ax.set_xlabel("Asset", fontsize=7.5); ax.set_ylabel("HF", fontsize=7.5)
    ax.tick_params(labelsize=6.5)
    ax.set_title("Final Holdings Heatmap (HF × Asset)", fontsize=8.5, fontweight="bold", pad=3)

    # Row 3 — BD state
    ax = fig.add_subplot(gs[3, 0])
    for k in range(N_BD):
        ax.plot(T, bd_capitals[:, k], lw=1.5, color=BD_C[k], label=f"BD {k}")
    ax.axhline(0, color="black", lw=0.7, ls=":")
    vl(ax); sty(ax, "Bank/Dealer Capital", ylabel="Capital ($)")

    ax = fig.add_subplot(gs[3, 1])
    for k in range(N_BD):
        ax.plot(T, bd_leverages[:, k], lw=1.3, color=BD_C[k], label=f"BD {k}", alpha=0.85)
        fs = np.where(bd_fire_sale[:, k])[0]
        if len(fs):
            ax.scatter(fs, bd_leverages[fs, k], color=BD_C[k], marker="x", s=35, zorder=5)
    ax.axhline(cfg.bd_lev_max,    color="#d62728", lw=1.0, ls="--", alpha=0.85, label="Max")
    ax.axhline(cfg.bd_lev_buffer, color="#ff7f0e", lw=1.0, ls="--", alpha=0.85, label="Buffer")
    ax.axhline(cfg.bd_lev_target, color="#2ca02c", lw=1.0, ls="--", alpha=0.85, label="Target")
    vl(ax); sty(ax, "BD Leverage  (x = fire-sale step)", ylabel="Leverage")

    ax = fig.add_subplot(gs[3, 2])
    for k in range(N_BD):
        ax.plot(T, bd_fund_cp[:, k], lw=1.5, color=BD_C[k], label=f"BD {k}")
    vl(ax); sty(ax, "BD Funding from Cash Providers", ylabel="Funding ($)")

    # Row 4 — Cash providers
    ax = fig.add_subplot(gs[4, 0])
    for k in range(N_BD):
        ax.plot(T, haircuts[:, k], lw=1.5, color=BD_C[k], label=f"HC->BD {k}")
    ax.axhline(cfg.cp_haircut_normal,   color="#2ca02c", lw=0.9, ls=":", alpha=0.8, label="Normal")
    ax.axhline(cfg.cp_haircut_stressed, color="#d62728", lw=0.9, ls=":", alpha=0.8, label="Stressed")
    vl(ax); sty(ax, "CP Haircut (Eq. 22 / LiqRatio deficit)", ylabel="Haircut")

    ax = fig.add_subplot(gs[4, 1])
    ls_sty = ["-", "--"]
    for c in range(N_CP):
        for k in range(N_BD):
            ax.plot(T, cp_loans[:, c, k], lw=1.3,
                    color=BD_C[k], ls=ls_sty[c % 2], label=f"CP{c}->BD{k}")
    vl(ax); sty(ax, "CP Loans to Bank/Dealers", ylabel="Loan ($)")

    ax = fig.add_subplot(gs[4, 2])
    ax.plot(T, overlap, lw=1.5, color="#8c564b")
    ax.fill_between(T, 0, overlap, alpha=0.15, color="#8c564b")
    ax.set_ylim(0, 1.05)
    vl(ax); sty(ax, "HF Portfolio Overlap (cosine)", ylabel="Similarity", legend=False)

    # Row 5 — Forced flow detail + derivatives
    ax = fig.add_subplot(gs[5, 0])
    bottom = np.zeros(N_STEPS)
    if M <= 5:
        # Color = agent, hatch = asset (readable for small M)
        asset_handles = []
        for n in range(N_HF):
            for m in range(M):
                ax.bar(T, hf_forced[:, n, m], bottom=bottom, width=1.0,
                       color=HF_C[n], hatch=HATCHES[m % len(HATCHES)],
                       alpha=0.85, edgecolor="white", linewidth=0.2)
                bottom += hf_forced[:, n, m]
                if n == 0:
                    asset_handles.append(mpatches.Patch(
                        facecolor="white", edgecolor="black",
                        hatch=HATCHES[m % len(HATCHES)], label=f"Asset {m}"))
        for k in range(N_BD):
            for m in range(M):
                ax.bar(T, bd_forced[:, k, m], bottom=bottom, width=1.0,
                       color=BD_C[k], hatch=HATCHES[m % len(HATCHES)],
                       alpha=0.85, edgecolor="white", linewidth=0.2)
                bottom += bd_forced[:, k, m]
        agent_h = ([mpatches.Patch(facecolor=HF_C[n], label=f"HF {n}") for n in range(N_HF)]
                   + [mpatches.Patch(facecolor=BD_C[k], label=f"BD {k}") for k in range(N_BD)])
        l1 = ax.legend(handles=agent_h, fontsize=5, loc="upper left",
                       framealpha=0.7, ncol=2, title="Agent", title_fontsize=5.5)
        ax.add_artist(l1)
        ax.legend(handles=asset_handles, fontsize=5, loc="upper right",
                  framealpha=0.7, title="Asset (hatch)", title_fontsize=5.5)
        title_ff = "Forced Flow by Agent x Asset"
    else:
        # Many assets: color = agent, sum over assets per bar (asset detail in heatmap below)
        for n in range(N_HF):
            flow = hf_forced[:, n, :].sum(axis=1)
            ax.bar(T, flow, bottom=bottom, width=1.0, color=HF_C[n], alpha=0.85, edgecolor="none")
            bottom += flow
        for k in range(N_BD):
            flow = bd_forced[:, k, :].sum(axis=1)
            ax.bar(T, flow, bottom=bottom, width=1.0, color=BD_C[k], alpha=0.85, edgecolor="none")
            bottom += flow
        agent_h = ([mpatches.Patch(facecolor=HF_C[n], label=f"HF {n}") for n in range(min(N_HF, 10))]
                   + [mpatches.Patch(facecolor=BD_C[k], label=f"BD {k}") for k in range(N_BD)])
        ax.legend(handles=agent_h, fontsize=5, loc="upper left",
                  framealpha=0.7, ncol=2, title="Agent", title_fontsize=5.5)
        ax.text(0.99, 0.99, f"Σ over {M} assets", ha="right", va="top",
                transform=ax.transAxes, fontsize=6, color="gray")
        title_ff = "Forced Flow by Agent (Σ assets)"
    vl(ax); sty(ax, title_ff, ylabel="|qty|", legend=False)

    ax = fig.add_subplot(gs[5, 1])
    deriv_exp    = np.array([r["deriv_exposure"] for r in history])
    deriv_losses = np.array([r["deriv_losses"]   for r in history])
    ax.plot(T, deriv_exp,    lw=1.5, color="#1f77b4", label="MTM exposure")
    ax.plot(T, deriv_losses, lw=1.5, color="#d62728", label="Losses")
    vl(ax); sty(ax, "Derivatives Desk", ylabel="$ exposure / loss")

    ax = fig.add_subplot(gs[5, 2])
    bottom = np.zeros(N_STEPS)
    for n in range(N_HF):
        vals = hf_fire_sale[:, n].astype(float)
        ax.bar(T, vals, bottom=bottom, color=HF_C[n], alpha=0.85,
               width=1.0, label=f"HF {n}")
        bottom += vals
    vl(ax); sty(ax, "HF Fire-Sale Activity (stacked)", ylabel="# HFs selling")

    # Row 6 — BD treasury state (LiqRatio, CW, Reserve/Debit)
    ax = fig.add_subplot(gs[6, 0])
    for k in range(N_BD):
        ax.plot(T, bd_liq_ratios[:, k], lw=1.3, color=BD_C[k],
                label=f"BD {k}", alpha=0.9)
    ax.axhline(cfg.bd_liq_ratio_min,    color="#d62728", lw=1.0, ls="--",
               alpha=0.85, label="Min")
    ax.axhline(cfg.bd_liq_ratio_target, color="#2ca02c", lw=1.0, ls="--",
               alpha=0.85, label="Target")
    ax.set_ylim(bottom=0)
    vl(ax); sty(ax, "BD Liquidity Ratio (LiqReserve / FTD)", ylabel="LiqRatio")

    ax = fig.add_subplot(gs[6, 1])
    for k in range(N_BD):
        ax.plot(T, bd_cw[:, k], lw=1.3, color=BD_C[k], label=f"BD {k}", alpha=0.9)
    ax.axhline(100, color="#2ca02c", lw=0.8, ls=":", alpha=0.7, label="Baseline (100)")
    ax.set_ylim(-5, 105)
    vl(ax); sty(ax, "BD Creditworthiness (CW, Eq. 21)", ylabel="CW (0-100)")

    ax = fig.add_subplot(gs[6, 2])
    for k in range(N_BD):
        ax.plot(T, bd_liq_reserves[:, k], lw=1.3, color=BD_C[k],
                label=f"Reserve BD{k}", alpha=0.9)
        ax.bar(T, bd_liq_debits[:, k], color=BD_C[k], alpha=0.35,
               edgecolor="none", width=1.0, label=f"Debit BD{k}")
    ax.axhline(0, color="black", lw=0.6, ls=":")
    vl(ax); sty(ax, "BD Liquidity Reserve (line) & Debit (bars)",
               ylabel="$ (reserve / debit)")

    # Row 7 — BD default focus (active timeline, new-default waves, capital w/ default markers)
    # Reconstruct per-step active flags (fall back to capital>0 if snapshot lacks the key)
    if "bd_active" in history[0]:
        bd_active = np.array([r["bd_active"] for r in history], dtype=bool)
    else:
        bd_active = bd_capitals > 0

    # First step each agent is inactive = default step
    def _first_inactive(active_arr):
        out = []
        for k in range(active_arr.shape[1]):
            idx = np.where(~active_arr[:, k])[0]
            out.append(int(idx[0]) if len(idx) else None)
        return out
    bd_def_step = _first_inactive(bd_active)

    # [7,0]: BD active-status Gantt — one row per BD, green=active red=defaulted
    ax = fig.add_subplot(gs[7, 0])
    for k in range(N_BD):
        active = bd_active[:, k]
        ax.fill_between(T, k - 0.4, k + 0.4, where=active,
                        color="#2ca02c", alpha=0.55, step="post", linewidth=0)
        ax.fill_between(T, k - 0.4, k + 0.4, where=~active,
                        color="#d62728", alpha=0.55, step="post", linewidth=0)
        if bd_def_step[k] is not None:
            ax.axvline(bd_def_step[k], color=BD_C[k], lw=1.2, alpha=0.9)
            ax.text(bd_def_step[k], k + 0.45, f"t={bd_def_step[k]}",
                    fontsize=6, color=BD_C[k], ha="left", va="bottom")
    ax.set_yticks(range(N_BD))
    ax.set_yticklabels([f"BD {k}" for k in range(N_BD)], fontsize=7)
    ax.set_ylim(-0.6, N_BD - 0.4)
    vl(ax)
    ax.set_title("BD Active Timeline (green=active, red=defaulted)",
                 fontsize=8.5, fontweight="bold", pad=3)
    ax.set_xlabel("Step", fontsize=7.5); ax.tick_params(labelsize=6.5)
    ax.grid(alpha=0.25, lw=0.5, axis="x")

    # [7,1]: New-default events per step (BD vs HF) — shows the cascade timing
    ax = fig.add_subplot(gs[7, 1])
    new_bd_def = np.diff(np.concatenate([[0], n_bd_def]))
    new_hf_def = np.diff(np.concatenate([[0], n_defaults]))
    ax.bar(T, new_hf_def, color="#7f7f7f", alpha=0.7, width=1.0, label="New HF defaults")
    ax.bar(T, new_bd_def, bottom=new_hf_def, color="#bcbd22",
           alpha=0.85, width=1.0, label="New BD defaults")
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    # Annotate time-since-shock for the first BD default
    bd_def_times = [d for d in bd_def_step if d is not None]
    if bd_def_times:
        first_bd = min(bd_def_times)
        ax.annotate(f"first BD: t={first_bd}\n(Δt={first_bd - SHOCK_T} after shock)",
                    xy=(first_bd, max(new_bd_def[first_bd], 1)),
                    xytext=(8, 12), textcoords="offset points",
                    fontsize=6.5, color="#bcbd22",
                    arrowprops=dict(arrowstyle="->", color="#bcbd22", lw=0.8))
    vl(ax); sty(ax, "New Defaults per Step (HF + BD)", ylabel="# new defaults")

    # [7,2]: BD capital with default-step vertical markers per BD
    ax = fig.add_subplot(gs[7, 2])
    for k in range(N_BD):
        ax.plot(T, bd_capitals[:, k], lw=1.5, color=BD_C[k], label=f"BD {k}")
        if bd_def_step[k] is not None:
            ds = bd_def_step[k]
            ax.axvline(ds, color=BD_C[k], lw=1.0, ls=":", alpha=0.8)
            ax.scatter([ds], [bd_capitals[ds, k]], color=BD_C[k],
                       marker="v", s=55, zorder=5, edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", lw=0.7, ls=":")
    vl(ax); sty(ax, "BD Capital w/ Default Step (▼)", ylabel="Capital ($)")

    # Bottom legend — cap per-class entries so it doesn't overflow
    _max_hf_leg = min(N_HF, 10)
    _max_bd_leg = min(N_BD, 4)
    _max_cp_leg = min(N_CP, 2)
    _max_as_leg = min(M, 8)
    legend_elems = (
        [Line2D([0],[0], color=HF_C[n],    lw=1.5, label=f"HF {n}") for n in range(_max_hf_leg)]
      + ([Line2D([0],[0], color="gray",     lw=1.0, label=f"…+{N_HF-_max_hf_leg} HFs")] if N_HF > _max_hf_leg else [])
      + [Line2D([0],[0], color=BD_C[k],    lw=1.5, label=f"BD {k}") for k in range(_max_bd_leg)]
      + [Line2D([0],[0], color=CP_C[c],    lw=1.5, label=f"CP {c}") for c in range(_max_cp_leg)]
      + [Line2D([0],[0], color=ASSET_C[m], lw=1.5, label=f"Asset {m}") for m in range(_max_as_leg)]
      + ([Line2D([0],[0], color="gray",     lw=1.0, label=f"…+{M-_max_as_leg} assets")] if M > _max_as_leg else [])
      + [Line2D([0],[0], color="#d62728",  lw=1.2, ls="--", label="Shock/LevMax")]
      + [Line2D([0],[0], color="#ff7f0e",  lw=1.0, ls="--", label="LevBuffer")]
      + [Line2D([0],[0], color="#2ca02c",  lw=1.0, ls="--", label="LevTarget")]
      + [Line2D([0],[0], marker="x", color="gray", lw=0, ms=6, label="Fire-sale step")]
    )
    fig.legend(handles=legend_elems, loc="lower center", ncol=10,
               fontsize=6.5, framealpha=0.85, bbox_to_anchor=(0.5, 0.005))

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    from bookstaber_abm.config import SimConfig
    from bookstaber_abm.simulation.engine import Simulation
    os.makedirs("outputs", exist_ok=True)

    CFG = SimConfig(
        n_assets=3, n_hedge_funds=5, n_bank_dealers=2, n_cash_providers=2,
        n_steps=120, shock_step=40, shock_asset=0, shock_size=-0.20,
        noise_std=0.001, hf_lev_target=5.0, hf_lev_buffer=6.0, hf_lev_max=7.0,
        beta=0.03, crowding=0.7, seed=42,
    )
    print("Running simulation...")
    sim     = Simulation(CFG)
    history = sim.run()
    make_dashboard(history, CFG, "outputs/dashboard.png")
    print("Saved: outputs/dashboard.png")
