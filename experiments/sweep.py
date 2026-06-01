"""
experiments/sweep.py
--------------------
Final shipped calibration cells for the joint −15% / −20% outcome-distribution
match (Bookstaber-Paddrik-Tivnan reproduction).

This file was trimmed for publication: it now contains only the two shipped
cells — ``EE_ship_n003_l070_15_n1000`` and ``EE_ship_n003_l070_20_n1000`` — that
reproduce the calibration shipped in ``batch_run.py`` (noise_std=0.003,
hf_max_liq_frac=bd_max_liq_frac=0.70, rate-limited default liquidation). The
~96 historical calibration cells from the parameter search were removed; see
CLAUDE.md for the full calibration history.

For each cell we:
  1. Run N simulations (N=1000 for the shipped cells).
  2. Classify each run into an outcome bucket
     (no_default / hf0_only / partial / all_default).
  3. Write outputs/cell_<id>/distribution_summary.csv + bucket_counts.json.
  4. Print bucket counts.

Usage:
    PYTHONPATH=. python experiments/sweep.py [cell_id ...]

With no arguments, runs every cell. A cell_id substring selects matching cells,
e.g. ``EE_ship`` runs both shipped cells.

The allocation helpers ``hetero_4hf_6assets_gradient`` and ``asym_2bd_6assets``
are also imported by ``batch_run.py``, ``experiments/robustness.py`` and
``experiments/sensitivity_lhs.py``.
"""
from __future__ import annotations

import os
import sys
import json
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bookstaber_abm.config import SimConfig
from bookstaber_abm.simulation.engine import Simulation
from bookstaber_abm.analysis.plots import history_to_df
from bookstaber_abm.analysis.buckets import summarize_runs, bucket_counts


# ── Substrate for replace(): the EE cells override essentially every field, so
# these starting values are immaterial; only the dataclass shape matters. ─────
BASE = SimConfig(
    n_assets=6, n_hedge_funds=4, n_bank_dealers=2, n_cash_providers=1,
    n_steps=60, shock_step=30, shock_asset=0, shock_size=-0.20,
    beta=0.1, beta1=0.02, normalise_beta=True, noise_std=0.005,
    hf_max_liq_frac=0.05, bd_max_liq_frac=0.05,
    hf_lev_target=5.0, hf_lev_buffer=5.3, hf_lev_max=5.5,
    bd_lev_target=8, bd_lev_buffer=10, bd_lev_max=13,
    bd_liq_ratio_min=0.025,
    bd_liq_ratio_target=0.035,
    phi_cw=4000.0,
    crowding=0.5,
    hf_funding_squeeze_threshold=1.10,
    cp_max_loan=100000.0,
    enable_derivatives_desk=False,
    seed=50,
)

SAVE_PER_RUN = False  # set True to also dump run_*.csv / run_*.json per cell


# ── Allocation helpers (also imported by batch_run.py / robustness.py / LHS) ──
def hetero_4hf_6assets_gradient(
    f0: float = 0.60, f1: float = 0.45, f2: float = 0.35, f3: float = 0.25
) -> list[list[float]]:
    """4 HFs over 6 assets with a gradient of shock-asset (asset 0) exposure.
    HF0 (canary) at f0, HF1 at f1, ..., HF3 at f3. Each HF spreads the
    remaining (1 - f) equally across assets 1-5. Defaults give post-shock leverage
    (lev_target=8, shock=-0.20): HF0≈176, HF1≈22, HF2≈14, HF3≈11.
    """
    n_assets = 6
    fs = [f0, f1, f2, f3]
    out = []
    for f in fs:
        rest = (1.0 - f) / (n_assets - 1)
        out.append([f] + [rest] * (n_assets - 1))
    return out


def asym_2bd_6assets(tilt: float = 0.02) -> list[list[float]]:
    """BD0 +tilt on shock asset, BD1 -tilt. Mean BD exposure to the shock asset
    is unchanged at 1/6.
    """
    base = 1.0 / 6
    bd0 = [base + tilt] + [(1.0 - base - tilt) / 5] * 5
    bd1 = [base - tilt] + [(1.0 - base + tilt) / 5] * 5
    return [bd0, bd1]


# ── Shipped calibration cells ─────────────────────────────────────────────────
# All cells built off batch_run.py BASE (the joint −15%/−20% diversified-funding
# calibration). Paper-distance metric (computed in main): Σ |bucket% - paper%|
# across 4 buckets at each shock, summed across both shocks.

def _ee_base_overrides(shock_size: float) -> dict:
    return dict(
        n_assets=6, n_hedge_funds=4, n_bank_dealers=2, n_cash_providers=1,
        n_steps=200, shock_step=50, shock_asset=0, shock_size=shock_size,
        beta=0.55, beta1=0.0, normalise_beta=True, noise_std=0.002,
        hf_max_liq_frac=0.20, bd_max_liq_frac=0.20,
        fire_sale_shock_concentration=1.0,
        hf_lev_target=8.0, hf_lev_buffer=14.0, hf_lev_max=20.0,
        bd_lev_target=5.0, bd_lev_buffer=10.0, bd_lev_max=15.0,
        bd_liq_rate=0.30,
        bd_liq_ratio_min=0.025, bd_liq_ratio_target=0.035,
        phi_cw=1000.0, cp_cw_smoothing_alpha=0.5,
        crowding=0.0,
        hf_allocations_hetero=hetero_4hf_6assets_gradient(0.60, 0.20, 0.18, 0.18),
        hf_allocation=[1.0/6]*6,
        bd_allocations_hetero=asym_2bd_6assets(tilt=0.02),
        hf_bd_funding_weights=[[1.0,0.0],[0.0,1.0],[0.0,1.0],[0.0,1.0]],
        hf_funding_squeeze_threshold=0.02,
        cp_max_loan=10_000_000.0,
        enable_derivatives_desk=False,
    )


def _ee_cell(cell_id: str, shock_size: float, n: int, **patch) -> dict:
    o = _ee_base_overrides(shock_size)
    o.update(patch)
    return dict(id=cell_id, n_runs=n, overrides=o)


CELLS: list[dict] = []

# Shipped calibration at N=1000 for both shocks. Matches what batch_run.py BASE
# uses (noise_std=0.003, liq=0.70 + rate-limited default liquidation).
for shk in (-0.15, -0.20):
    tag = f"{int(abs(shk)*100):02d}"
    CELLS.append(_ee_cell(
        f"EE_ship_n003_l070_{tag}_n1000", shk, 1000,
        noise_std=0.003, hf_max_liq_frac=0.70, bd_max_liq_frac=0.70))


# ── Cell runner ───────────────────────────────────────────────────────────────
def run_cell(cell: dict) -> dict[str, int]:
    """
    Run one cell, write outputs/cell_<id>/distribution_summary.csv, return
    bucket counts.
    """
    cell_id = cell["id"]
    n_runs = cell["n_runs"]
    overrides = cell["overrides"]

    out_dir = f"outputs/cell_{cell_id}"
    os.makedirs(out_dir, exist_ok=True)

    cfg = replace(BASE, **overrides)

    print(f"\n[cell {cell_id}] N={n_runs}  shock={cfg.shock_size:.0%}  "
          f"n_hf={cfg.n_hedge_funds}  n_assets={cfg.n_assets}  "
          f"derivatives={cfg.enable_derivatives_desk}")

    histories = []
    for i in range(n_runs):
        cfg_i = replace(cfg, seed=i)
        history = Simulation(cfg_i).run()
        histories.append(history)
        if SAVE_PER_RUN:
            history_to_df(history).to_csv(f"{out_dir}/run_{i:03d}.csv")
            with open(f"{out_dir}/run_{i:03d}.json", "w") as f:
                json.dump(history, f)
        print(f"  [{i+1:3d}/{n_runs}] done", end="\r")

    df = summarize_runs(histories, shock_size=cfg.shock_size)
    df.to_csv(f"{out_dir}/distribution_summary.csv", index=False)
    counts = bucket_counts(df)

    total = sum(counts.values())
    print(f"\n[cell {cell_id}] bucket counts (N={total}):")
    for name in ["no_default", "hf0_only", "partial", "all_default"]:
        n = counts[name]
        pct = (n / total * 100) if total else 0.0
        print(f"  {name:12s} : {n:4d}  ({pct:5.1f}%)")

    # Brief HF0 stats
    qd_hf0 = df.get("qdemand_hf0", []).sum() if "qdemand_hf0" in df.columns else 0
    def_hf0 = df.get("default_hf0", []).sum() if "default_hf0" in df.columns else 0
    print(f"  HF0 qDemand: {qd_hf0}/{total}  HF0 default: {def_hf0}/{total}")

    # Save bucket summary as JSON for easy programmatic comparison
    with open(f"{out_dir}/bucket_counts.json", "w") as f:
        json.dump({"cell": cell_id, "n_runs": total,
                   "shock_size": cfg.shock_size, "counts": counts}, f, indent=2)

    return counts


# ── Paper-match heuristic ─────────────────────────────────────────────────────
def matches_paper(counts: dict[str, int], total: int) -> bool:
    """
    Heuristic: a cell qualitatively matches the paper if (a) all 4 buckets are
    populated and (b) the all-four-default rate is >5%.
    """
    if total == 0:
        return False
    populated = sum(1 for v in counts.values() if v > 0)
    all_def_pct = counts["all_default"] / total
    return populated == 4 and all_def_pct >= 0.05


def main(argv: list[str]) -> None:
    os.makedirs("outputs", exist_ok=True)

    requested = set(argv[1:])  # by id substring
    cells_to_run = (
        [c for c in CELLS if any(s in c["id"] for s in requested)]
        if requested else
        list(CELLS)
    )

    all_counts: dict[str, dict[str, int]] = {}
    for cell in cells_to_run:
        counts = run_cell(cell)
        all_counts[cell["id"]] = counts

    # Write a top-level summary table across all cells run
    summary_rows = []
    for cid, counts in all_counts.items():
        total = sum(counts.values())
        row = {"cell": cid, "n_runs": total}
        row.update({k: counts[k] for k in
                    ["no_default", "hf0_only", "partial", "all_default"]})
        if total:
            for k in ["no_default", "hf0_only", "partial", "all_default"]:
                row[f"{k}_pct"] = round(counts[k] / total * 100, 1)
        summary_rows.append(row)

    import pandas as pd
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("outputs/sweep_summary.csv", index=False)
    print("\n[sweep] wrote outputs/sweep_summary.csv")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv)
