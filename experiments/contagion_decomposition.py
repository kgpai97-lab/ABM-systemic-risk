"""
experiments/contagion_decomposition.py
--------------------------------------
Decompose the total effect of an exogenous price shock into a PRIMARY effect and
a CONTAGION effect, per agent and per asset, via a seed-matched counterfactual.

Method (counterfactual "suppress all endogenous response")
----------------------------------------------------------
For each seed we run the SAME config twice:

  (A) SUPPRESSED arm  (cfg.suppress_contagion=True):
        the exogenous shock is applied, agents mark-to-market and may even default
        from the direct leverage hit, BUT forced sales and default liquidations do
        not move prices and the funding chain is frozen at need — so no secondary
        price cascade and no funding-squeeze contagion. The RNG/noise path is
        identical to a normal run (same number of update_prices calls), so the two
        arms differ ONLY in the contagion channel.

  (B) NORMAL arm  (cfg.suppress_contagion=False):
        the full model — forced sales move prices, the cascade unfolds.

Then for any quantity X (a holding's value, an agent's capital, leverage):
        PRIMARY(X)   = X_suppressed - X_preshock     (the direct shock hit)
        CONTAGION(X) = X_normal     - X_suppressed   (everything the cascade adds)
        TOTAL(X)     = X_normal     - X_preshock      = PRIMARY + CONTAGION

We report CONTAGION explicitly — "how is a fund/bank's holding in an asset affected
by the contagion BESIDES the primary price shock". Note CONTAGION on a NON-shocked
asset is pure spillover (its primary effect is ~0, only noise), which is the cleanest
read on contagion.

Outputs (Monte Carlo over N_RUNS seeds, run separately at each shock in SHOCKS):
    outputs/contagion_decomposition/holdings_decomp_<tag>.csv
        one row per (seed, agent, asset): preshock / suppressed / normal holding
        VALUE (price*qty), plus primary / contagion / total deltas (abs and %).
    outputs/contagion_decomposition/agent_decomp_<tag>.csv
        one row per (seed, agent): capital & leverage in each arm + deltas, and the
        default bucket reached in suppressed vs normal arm.
    outputs/contagion_decomposition/summary_<tag>.csv
        MC distribution (mean / std / p5 / p50 / p95) of the contagion effect,
        aggregated per agent and per asset.

Usage:
    PYTHONPATH=. python experiments/contagion_decomposition.py            # both shocks, full
    PYTHONPATH=. python experiments/contagion_decomposition.py --smoke    # fast check
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from dataclasses import replace

from bookstaber_abm.simulation.engine import Simulation
from bookstaber_abm.analysis.buckets import classify_run

# Reuse the shipped production calibration verbatim as the baseline config.
from experiments.robustness import BASE


# ── Run configuration ────────────────────────────────────────────────────────
N_RUNS = int(os.environ.get("N_RUNS", "500"))   # Monte Carlo seeds per shock
SHOCKS = [-0.15, -0.20]          # run the decomposition at each
OUT_ROOT = "outputs/contagion_decomposition"
N_EVENT_SAMPLE = 20              # number of seeds to log full event sequences for


def _agent_labels(cfg) -> list[str]:
    hf = [f"HF{n}" for n in range(cfg.n_hedge_funds)]
    bd = [f"BD{k}" for k in range(cfg.n_bank_dealers)]
    return hf + bd


def _holdings_and_state(snapshot: dict, cfg) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    From a history snapshot return, stacked HF-then-BD:
        holdings  (n_agents, n_assets)
        capitals  (n_agents,)
        leverages (n_agents,)
        active    (n_agents,) bool
    """
    prices = np.array(snapshot["prices"])
    hf_h = np.array(snapshot["hf_holdings"])           # (n_hf, n_assets)
    bd_h = np.array(snapshot["bd_holdings"]) if "bd_holdings" in snapshot else None
    holdings = np.vstack([hf_h, bd_h]) if bd_h is not None else hf_h
    capitals = np.array(snapshot["hf_capitals"] + snapshot["bd_capitals"])
    leverages = np.array(snapshot["hf_leverages"] + snapshot["bd_leverages"])
    active = np.array(snapshot["hf_active"] + snapshot["bd_active"], dtype=bool)
    return holdings, capitals, leverages, active, prices


def _trajectory_rows(hist_supp, hist_norm, cfg, shock, seed) -> list[dict]:
    """Per-step contagion-to-capital per agent: for every step from the shock
    onward, contagion(t) = capital_normal(t) - capital_suppressed(t). Captures
    *when*, after the shock, the cascade subtracts capital from each agent."""
    labels = _agent_labels(cfg)
    rows = []
    n = min(len(hist_supp), len(hist_norm))
    for t in range(cfg.shock_step - 1, n):
        cap_supp = hist_supp[t]["hf_capitals"] + hist_supp[t]["bd_capitals"]
        cap_norm = hist_norm[t]["hf_capitals"] + hist_norm[t]["bd_capitals"]
        for a, label in enumerate(labels):
            rows.append({
                "shock": shock, "seed": seed, "step": t,
                "rel_step": t - cfg.shock_step, "agent": label,
                "capital_suppressed": float(cap_supp[a]),
                "capital_normal": float(cap_norm[a]),
                "contagion_delta": float(cap_norm[a] - cap_supp[a]),
            })
    return rows


def _event_rows(hist_norm, cfg, shock, seed) -> list[dict]:
    """Ordered event log of the NORMAL-arm cascade: the step each agent first
    *enters a fire sale* and the step each agent *defaults* (active -> inactive)."""
    labels = _agent_labels(cfg)
    n_hf = cfg.n_hedge_funds
    fired = [False] * len(labels)
    defaulted = [False] * len(labels)
    rows = []
    for t, snap in enumerate(hist_norm):
        in_fs = list(snap["hf_in_fire_sale"]) + list(snap["bd_in_fire_sale"])
        active = list(snap["hf_active"]) + list(snap["bd_active"])
        for a, label in enumerate(labels):
            if in_fs[a] and not fired[a]:
                fired[a] = True
                rows.append({"shock": shock, "run": seed, "step": t,
                             "rel_step": t - cfg.shock_step, "agent": label,
                             "event_type": "fire_sale"})
            if (not active[a]) and not defaulted[a]:
                defaulted[a] = True
                rows.append({"shock": shock, "run": seed, "step": t,
                             "rel_step": t - cfg.shock_step, "agent": label,
                             "event_type": "default"})
    return rows


def run_one_seed(cfg_base, shock: float, seed: int) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """
    Run the suppressed and normal arms at one seed; return
    (holdings_rows, agent_rows, traj_rows, event_rows).
    """
    cfg = replace(cfg_base, shock_size=shock, seed=seed)
    shock_step = cfg.shock_step

    hist_supp = Simulation(replace(cfg, suppress_contagion=True)).run()
    hist_norm = Simulation(replace(cfg, suppress_contagion=False)).run()

    labels = _agent_labels(cfg)

    # Pre-shock snapshot: last step BEFORE the shock lands (same in both arms — the
    # arms are bit-identical up to and including shock_step-1 because suppression
    # only changes post-shock price impact / funding).
    pre = hist_norm[shock_step - 1]
    h_pre, cap_pre, lev_pre, act_pre, p_pre = _holdings_and_state(pre, cfg)

    # Final snapshot of each arm (end of crisis).
    h_sup, cap_sup, lev_sup, act_sup, p_sup = _holdings_and_state(hist_supp[-1], cfg)
    h_nrm, cap_nrm, lev_nrm, act_nrm, p_nrm = _holdings_and_state(hist_norm[-1], cfg)

    bucket_sup = classify_run(hist_supp)["bucket"]
    bucket_nrm = classify_run(hist_norm)["bucket"]

    holdings_rows: list[dict] = []
    agent_rows: list[dict] = []

    for a, label in enumerate(labels):
        # ── Holding VALUE per asset (price * qty), valued at each arm's own prices
        # so the delta captures BOTH the qty change (forced sales) and the price
        # change (cascade) — i.e. the full mark-to-market effect of contagion. ──
        for m in range(cfg.n_assets):
            v_pre = float(p_pre[m] * h_pre[a, m])
            v_sup = float(p_sup[m] * h_sup[a, m])
            v_nrm = float(p_nrm[m] * h_nrm[a, m])
            primary = v_sup - v_pre
            contagion = v_nrm - v_sup
            total = v_nrm - v_pre
            holdings_rows.append({
                "shock": shock, "seed": seed, "agent": label, "asset": m,
                "is_shock_asset": int(m == cfg.shock_asset),
                "value_preshock": v_pre,
                "value_suppressed": v_sup,
                "value_normal": v_nrm,
                "primary_delta": primary,
                "contagion_delta": contagion,
                "total_delta": total,
                "contagion_pct_of_preshock": (contagion / v_pre * 100.0) if v_pre > 1e-9 else 0.0,
            })

        # ── Agent-level capital / leverage decomposition ──
        agent_rows.append({
            "shock": shock, "seed": seed, "agent": label,
            "capital_preshock": float(cap_pre[a]),
            "capital_suppressed": float(cap_sup[a]),
            "capital_normal": float(cap_nrm[a]),
            "capital_primary_delta": float(cap_sup[a] - cap_pre[a]),
            "capital_contagion_delta": float(cap_nrm[a] - cap_sup[a]),
            "leverage_preshock": float(lev_pre[a]),
            "leverage_suppressed": float(lev_sup[a]),
            "leverage_normal": float(lev_nrm[a]),
            "active_suppressed": int(bool(act_sup[a])),
            "active_normal": int(bool(act_nrm[a])),
            # default ATTRIBUTABLE to contagion: survives the primary shock but
            # dies in the full run.
            "default_from_contagion": int(bool(act_sup[a]) and not bool(act_nrm[a])),
            "default_from_primary": int(not bool(act_sup[a])),
            "bucket_suppressed": bucket_sup,
            "bucket_normal": bucket_nrm,
        })

    traj_rows = _trajectory_rows(hist_supp, hist_norm, cfg, shock, seed)
    event_rows = _event_rows(hist_norm, cfg, shock, seed)
    return holdings_rows, agent_rows, traj_rows, event_rows


def summarise(holdings_df: pd.DataFrame, agent_df: pd.DataFrame) -> pd.DataFrame:
    """MC distribution of the contagion effect, per agent×asset and per agent."""
    rows = []

    def pctl(s, q):
        return float(np.percentile(s, q)) if len(s) else float("nan")

    # Per agent × asset: contagion on holding value.
    g = holdings_df.groupby(["agent", "asset", "is_shock_asset"])
    for (agent, asset, is_shock), sub in g:
        c = sub["contagion_delta"]
        cp = sub["contagion_pct_of_preshock"]
        rows.append({
            "level": "holding", "agent": agent, "asset": int(asset),
            "is_shock_asset": int(is_shock),
            "contagion_mean": float(c.mean()), "contagion_std": float(c.std()),
            "contagion_p5": pctl(c, 5), "contagion_p50": pctl(c, 50), "contagion_p95": pctl(c, 95),
            "contagion_pct_mean": float(cp.mean()), "contagion_pct_p50": pctl(cp, 50),
        })

    # Per agent: capital contagion + default-from-contagion frequency.
    ga = agent_df.groupby("agent")
    for agent, sub in ga:
        c = sub["capital_contagion_delta"]
        rows.append({
            "level": "agent_capital", "agent": agent, "asset": -1, "is_shock_asset": -1,
            "contagion_mean": float(c.mean()), "contagion_std": float(c.std()),
            "contagion_p5": pctl(c, 5), "contagion_p50": pctl(c, 50), "contagion_p95": pctl(c, 95),
            "default_from_contagion_freq": float(sub["default_from_contagion"].mean()),
            "default_from_primary_freq": float(sub["default_from_primary"].mean()),
        })

    return pd.DataFrame(rows)


def run_shock(cfg_base, shock: float, n_runs: int) -> None:
    tag = f"{abs(shock) * 100:.0f}pct"
    print(f"\n[contagion] shock={shock:.0%}  N_RUNS={n_runs}  ({2 * n_runs} sims)")

    all_holdings: list[dict] = []
    all_agents: list[dict] = []
    all_traj: list[dict] = []
    all_events: list[dict] = []
    for seed in range(n_runs):
        h_rows, a_rows, t_rows, e_rows = run_one_seed(cfg_base, shock, seed)
        all_holdings.extend(h_rows)
        all_agents.extend(a_rows)
        all_traj.extend(t_rows)
        if seed < N_EVENT_SAMPLE:           # log full event sequences for a sample of runs
            all_events.extend(e_rows)
        if (seed + 1) % 50 == 0 or seed == n_runs - 1:
            print(f"  [{seed + 1:4d}/{n_runs}]", end="\r")

    holdings_df = pd.DataFrame(all_holdings)
    agent_df = pd.DataFrame(all_agents)
    summary_df = summarise(holdings_df, agent_df)

    holdings_df.to_csv(f"{OUT_ROOT}/holdings_decomp_{tag}.csv", index=False)
    agent_df.to_csv(f"{OUT_ROOT}/agent_decomp_{tag}.csv", index=False)
    summary_df.to_csv(f"{OUT_ROOT}/summary_{tag}.csv", index=False)

    # Per-step contagion-to-capital trajectory, aggregated across all seeds.
    traj_df = pd.DataFrame(all_traj)
    traj_summary = (
        traj_df.groupby(["rel_step", "agent"])["contagion_delta"]
        .agg(contagion_mean="mean",
             contagion_p5=lambda s: float(np.percentile(s, 5)),
             contagion_p50=lambda s: float(np.percentile(s, 50)),
             contagion_p95=lambda s: float(np.percentile(s, 95)))
        .reset_index()
    )
    traj_summary.insert(0, "shock", shock)
    traj_summary.to_csv(f"{OUT_ROOT}/traj_{tag}.csv", index=False)

    # Ordered fire-sale / default event log for the first N_EVENT_SAMPLE runs.
    events_df = pd.DataFrame(all_events)
    events_df.to_csv(f"{OUT_ROOT}/events_{tag}.csv", index=False)

    print(f"\n[contagion] wrote holdings_decomp/agent_decomp/summary/traj/events _{tag}.csv")

    _print_headline(summary_df, agent_df, shock)


def _print_headline(summary_df: pd.DataFrame, agent_df: pd.DataFrame, shock: float) -> None:
    """Console headline: per-agent contagion on shock vs non-shock holdings + defaults."""
    hold = summary_df[summary_df["level"] == "holding"]
    cap = summary_df[summary_df["level"] == "agent_capital"]

    print(f"\n  -- Contagion effect on holdings (mean % of pre-shock value), shock={shock:.0%} --")
    print(f"  {'agent':6s}  {'shock-asset':>14s}  {'non-shock avg':>14s}")
    for agent in hold["agent"].unique():
        sub = hold[hold["agent"] == agent]
        sa = sub[sub["is_shock_asset"] == 1]["contagion_pct_mean"]
        ns = sub[sub["is_shock_asset"] == 0]["contagion_pct_mean"]
        sa_v = float(sa.mean()) if len(sa) else float("nan")
        ns_v = float(ns.mean()) if len(ns) else float("nan")
        print(f"  {agent:6s}  {sa_v:13.2f}%  {ns_v:13.2f}%")

    print(f"\n  -- Default attribution (frequency across {agent_df['seed'].nunique()} seeds) --")
    print(f"  {'agent':6s}  {'from primary':>13s}  {'from contagion':>15s}")
    for _, r in cap.iterrows():
        print(f"  {r['agent']:6s}  {r['default_from_primary_freq']:12.1%}  "
              f"{r['default_from_contagion_freq']:14.1%}")


def main(argv: list[str]) -> None:
    os.makedirs(OUT_ROOT, exist_ok=True)
    smoke = "--smoke" in argv
    n_runs = 10 if smoke else N_RUNS
    shocks = [-0.20] if smoke else SHOCKS

    for shock in shocks:
        run_shock(BASE, shock, n_runs)


if __name__ == "__main__":
    main(sys.argv)
