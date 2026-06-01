"""
experiments/robustness.py
--------------------------
OFAT robustness sweep matching the paper's Table 3 (Bookstaber-Paddrik-Tivnan).

Each cell varies one parameter across 3–4 levels bracketing the calibrated
baseline (batch_run.py CC4j calibration at -15% shock).  For every run we
record OLS-ready metrics (price changes, capital changes, forced-sale counts)
in addition to the standard bucket classification — enabling a direct
reproduction of the Table 3 regression.

Usage:
    PYTHONPATH=. python experiments/robustness.py              # run all cells
    PYTHONPATH=. python experiments/robustness.py shock beta   # run matching cells

Outputs:
    outputs/robustness/cell_<id>/distribution_summary.csv  — one row per run
    outputs/robustness/cell_<id>/bucket_counts.json
    outputs/robustness/robustness_summary.csv              — one row per cell
"""
from __future__ import annotations

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from dataclasses import replace

from bookstaber_abm.config import SimConfig
from bookstaber_abm.simulation.engine import Simulation
from bookstaber_abm.analysis.buckets import classify_run, bucket_counts

# Import allocation helpers from sweep.py (no copy-paste)
from experiments.sweep import hetero_4hf_6assets_gradient, asym_2bd_6assets


N_RUNS = 100
OUT_ROOT = "outputs/robustness_2026_05_27"


# ── BASE: 2026-05-27 Pareto-ship calibration (rate-limited default liq) ───────
# Matches batch_run.py BASE exactly. Differs from the prior (2026-05-26) BASE
# only in noise_std (0.002→0.003) and the liquidation caps (0.20→0.70), plus the
# structural rate-limit on HedgeFund.apply_default_liquidation. Those two config
# changes plus the structural fix give the new cascade dynamics this regression
# re-measures. See CLAUDE.md "Stage A/B parameter sweep (2026-05-27)".
BASE = SimConfig(
    n_assets=6, n_hedge_funds=4, n_bank_dealers=2, n_cash_providers=1,
    n_steps=200, shock_step=50, shock_asset=0, shock_size=-0.15,
    beta=0.55, beta1=0.0, normalise_beta=True, noise_std=0.003,
    hf_max_liq_frac=0.70, bd_max_liq_frac=0.70,
    fire_sale_shock_concentration=1.0,
    hf_lev_target=8.0, hf_lev_buffer=14.0, hf_lev_max=20.0,
    bd_lev_target=5.0, bd_lev_buffer=10.0, bd_lev_max=15.0,
    bd_liq_ratio_min=0.025, bd_liq_ratio_target=0.035,
    phi_cw=1000.0,
    phi_hc=0.10,
    cp_cw_smoothing_alpha=0.5,
    crowding=0.0,
    hf_allocations_hetero=hetero_4hf_6assets_gradient(0.60, 0.20, 0.18, 0.18),
    hf_allocation=[1.0 / 6] * 6,
    bd_allocations_hetero=asym_2bd_6assets(tilt=0.02),
    hf_bd_funding_weights=[[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
    hf_funding_squeeze_threshold=0.02,
    cp_haircut_normal=0.10,
    cp_haircut_stressed=0.25,
    cp_max_loan=10_000_000.0,
    enable_derivatives_desk=False,
    seed=50,
)


# ── OLS metric extractor ───────────────────────────────────────────────────────
def extract_run_metrics(history: list[dict], cfg: SimConfig) -> dict:
    """
    Extract per-run OLS-ready scalars from a simulation history.

    Price changes are computed relative to the pre-shock snapshot at shock_step.
    Capital changes likewise.  Forced-sale counts are step counts in fire-sale.

    Returns a flat dict suitable for a DataFrame row.
    """
    t0 = cfg.shock_step  # pre-shock reference
    snap0 = history[t0]
    snap_final = history[-1]

    n_hf = len(snap_final["hf_capitals"])
    n_bd = len(snap_final["bd_capitals"])
    n_price_assets = min(3, cfg.n_assets)  # track first 3 assets (paper Table 3)

    metrics: dict = {}

    # Price changes for first 3 assets
    for i in range(n_price_assets):
        p0 = snap0["prices"][i]
        pf = snap_final["prices"][i]
        metrics[f"price_change_asset{i}"] = (pf - p0) / p0 if p0 != 0 else 0.0

    # Capital changes per entity
    for n in range(n_hf):
        c0 = snap0["hf_capitals"][n]
        cf = snap_final["hf_capitals"][n]
        metrics[f"capital_change_hf{n}"] = (cf - c0) / c0 if c0 != 0 else 0.0

    for k in range(n_bd):
        c0 = snap0["bd_capitals"][k]
        cf = snap_final["bd_capitals"][k]
        metrics[f"capital_change_bd{k}"] = (cf - c0) / c0 if c0 != 0 else 0.0

    # Count steps each entity was in forced sale (post-shock only)
    for n in range(n_hf):
        metrics[f"n_forced_sales_hf{n}"] = sum(
            1 for snap in history[t0:] if snap["hf_in_fire_sale"][n]
        )

    for k in range(n_bd):
        metrics[f"n_forced_sales_bd{k}"] = sum(
            1 for snap in history[t0:] if snap["bd_in_fire_sale"][k]
        )

    return metrics


# ── Experiment grid ────────────────────────────────────────────────────────────
#
# Rescaling convention (all relative to calibrated BASE):
#   - Levels bracket the baseline value with ~0.5×, 1×, 2× (and 4× where useful)
#   - Ak(0) levels vary HF0 canary weight only; HF1-3 held fixed
#   - Leverage levels preserve buffer=1.75×target, max=2.5×target ratios
#
# Paper param      → config key(s)            → levels
# ─────────────────────────────────────────────────────
# Price shock       shock_size                 {-0.10, -0.13, -0.15, -0.20}  (baseline=-0.15)
# βm               beta                       {0.28, 0.55, 1.10}            (0.5×, 1×, 2×)
# HCc,k (normal)   cp_haircut_normal           {0.07, 0.10, 0.13, 0.16}
# ϕCW              phi_cw                     {500, 1000, 2000, 4000}
# ϕHC              phi_hc                     {0.05, 0.10, 0.20, 0.40}
# Ak(0)            hf_allocations_hetero(f0)  {0.10, 0.20, 0.40, 0.60}
# QMax             cp_max_loan                {1M, 10M, 50M}
# LiqRatioMin/Tgt  bd_liq_ratio_min/target    3 pairs
# HF lev target    hf_lev_{target,buffer,max} 3 triplets

def _alloc(f0: float) -> list[list[float]]:
    """Rebuild 4HF/6asset gradient allocation for a given HF0 canary weight f0."""
    return hetero_4hf_6assets_gradient(f0, 0.20, 0.18, 0.18)


CELLS: list[dict] = [

    # ── 1. Price shock ────────────────────────────────────────────────────────
    dict(id="shock_10pct",  label="shock=-10%",  param="shock_size",  value=-0.10,
         overrides=dict(shock_size=-0.10)),
    dict(id="shock_13pct",  label="shock=-13%",  param="shock_size",  value=-0.13,
         overrides=dict(shock_size=-0.13)),
    dict(id="shock_15pct",  label="shock=-15% (baseline)", param="shock_size", value=-0.15,
         overrides=dict()),  # baseline — no changes
    dict(id="shock_20pct",  label="shock=-20%",  param="shock_size",  value=-0.20,
         overrides=dict(shock_size=-0.20)),

    # ── 2. βm (price-impact sensitivity) ─────────────────────────────────────
    dict(id="beta_half",    label="beta=0.28 (0.5×)", param="beta", value=0.28,
         overrides=dict(beta=0.28)),
    dict(id="beta_base",    label="beta=0.55 (baseline)", param="beta", value=0.55,
         overrides=dict()),
    dict(id="beta_2x",      label="beta=1.10 (2×)",   param="beta", value=1.10,
         overrides=dict(beta=1.10)),

    # ── 3. HCc,k — initial / normal haircut ──────────────────────────────────
    dict(id="hc_007",  label="haircut_normal=0.07", param="cp_haircut_normal", value=0.07,
         overrides=dict(cp_haircut_normal=0.07)),
    dict(id="hc_010",  label="haircut_normal=0.10 (baseline)", param="cp_haircut_normal", value=0.10,
         overrides=dict()),
    dict(id="hc_013",  label="haircut_normal=0.13", param="cp_haircut_normal", value=0.13,
         overrides=dict(cp_haircut_normal=0.13)),
    dict(id="hc_016",  label="haircut_normal=0.16", param="cp_haircut_normal", value=0.16,
         overrides=dict(cp_haircut_normal=0.16)),

    # ── 4. ϕCW ────────────────────────────────────────────────────────────────
    dict(id="phicw_500",   label="phi_cw=500 (0.5×)",  param="phi_cw", value=500,
         overrides=dict(phi_cw=500.0)),
    dict(id="phicw_1000",  label="phi_cw=1000 (baseline)", param="phi_cw", value=1000,
         overrides=dict()),
    dict(id="phicw_2000",  label="phi_cw=2000 (2×)",   param="phi_cw", value=2000,
         overrides=dict(phi_cw=2000.0)),
    dict(id="phicw_4000",  label="phi_cw=4000 (4×)",   param="phi_cw", value=4000,
         overrides=dict(phi_cw=4000.0)),

    # ── 5. ϕHC ────────────────────────────────────────────────────────────────
    dict(id="phihc_005",  label="phi_hc=0.05 (0.5×)",  param="phi_hc", value=0.05,
         overrides=dict(phi_hc=0.05)),
    dict(id="phihc_010",  label="phi_hc=0.10 (baseline)", param="phi_hc", value=0.10,
         overrides=dict()),
    dict(id="phihc_020",  label="phi_hc=0.20 (2×)",    param="phi_hc", value=0.20,
         overrides=dict(phi_hc=0.20)),
    dict(id="phihc_040",  label="phi_hc=0.40 (4×)",    param="phi_hc", value=0.40,
         overrides=dict(phi_hc=0.40)),

    # ── 6. Ak(0) — HF0 shock-asset allocation ────────────────────────────────
    dict(id="alloc_f0_010",  label="HF0 shock-wt=0.10", param="hf0_shock_weight", value=0.10,
         overrides=dict(hf_allocations_hetero=_alloc(0.10))),
    dict(id="alloc_f0_020",  label="HF0 shock-wt=0.20", param="hf0_shock_weight", value=0.20,
         overrides=dict(hf_allocations_hetero=_alloc(0.20))),
    dict(id="alloc_f0_040",  label="HF0 shock-wt=0.40", param="hf0_shock_weight", value=0.40,
         overrides=dict(hf_allocations_hetero=_alloc(0.40))),
    dict(id="alloc_f0_060",  label="HF0 shock-wt=0.60 (baseline)", param="hf0_shock_weight", value=0.60,
         overrides=dict()),  # baseline f0=0.60

    # ── 7. QMax — cash-provider loan cap ─────────────────────────────────────
    dict(id="qmax_1M",   label="cp_max_loan=1M",           param="cp_max_loan", value=1_000_000,
         overrides=dict(cp_max_loan=1_000_000.0)),
    dict(id="qmax_10M",  label="cp_max_loan=10M (baseline)", param="cp_max_loan", value=10_000_000,
         overrides=dict()),
    dict(id="qmax_50M",  label="cp_max_loan=50M",          param="cp_max_loan", value=50_000_000,
         overrides=dict(cp_max_loan=50_000_000.0)),

    # ── 8. (LiqRatioMin, LiqRatioTarget) pairs ────────────────────────────────
    dict(id="liqratio_low",   label="LiqRatio=(0.015,0.025)", param="bd_liq_ratio_min", value=0.015,
         overrides=dict(bd_liq_ratio_min=0.015, bd_liq_ratio_target=0.025)),
    dict(id="liqratio_base",  label="LiqRatio=(0.025,0.035) (baseline)", param="bd_liq_ratio_min", value=0.025,
         overrides=dict()),
    dict(id="liqratio_high",  label="LiqRatio=(0.040,0.055)", param="bd_liq_ratio_min", value=0.040,
         overrides=dict(bd_liq_ratio_min=0.040, bd_liq_ratio_target=0.055)),

    # ── 9. HF leverage target (buffer=1.75×, max=2.5× ratio preserved) ───────
    dict(id="lev_low",   label="hf_lev=(5,9,13)",         param="hf_lev_target", value=5.0,
         overrides=dict(hf_lev_target=5.0, hf_lev_buffer=9.0, hf_lev_max=13.0)),
    dict(id="lev_base",  label="hf_lev=(8,14,20) (baseline)", param="hf_lev_target", value=8.0,
         overrides=dict()),
    dict(id="lev_high",  label="hf_lev=(10,16,22)",       param="hf_lev_target", value=10.0,
         overrides=dict(hf_lev_target=10.0, hf_lev_buffer=16.0, hf_lev_max=22.0)),

    # ── 10. noise_std (NEW 2026-05-27) ────────────────────────────────────────
    # The only Stage-A knob that reopened the cascade under the new regime.
    # Baseline = 0.003. Levels bracket 0.5×, 1×, 2×, 3×.
    dict(id="noise_0015", label="noise_std=0.0015 (0.5×)", param="noise_std", value=0.0015,
         overrides=dict(noise_std=0.0015)),
    dict(id="noise_003",  label="noise_std=0.003 (baseline)", param="noise_std", value=0.003,
         overrides=dict()),
    dict(id="noise_006",  label="noise_std=0.006 (2×)", param="noise_std", value=0.006,
         overrides=dict(noise_std=0.006)),
    dict(id="noise_009",  label="noise_std=0.009 (3×)", param="noise_std", value=0.009,
         overrides=dict(noise_std=0.009)),

    # ── 11. bd_liq_rate (NEW 2026-05-27) ──────────────────────────────────────
    # BD liquidity-buffer reserve fraction. Inert as a single knob in Stage A;
    # confirmed here under the regression rather than assumed. Baseline = 0.3.
    dict(id="bdliq_015", label="bd_liq_rate=0.15 (0.5×)", param="bd_liq_rate", value=0.15,
         overrides=dict(bd_liq_rate=0.15)),
    dict(id="bdliq_030", label="bd_liq_rate=0.30 (baseline)", param="bd_liq_rate", value=0.30,
         overrides=dict()),
    dict(id="bdliq_045", label="bd_liq_rate=0.45 (1.5×)", param="bd_liq_rate", value=0.45,
         overrides=dict(bd_liq_rate=0.45)),
    dict(id="bdliq_060", label="bd_liq_rate=0.60 (2×)", param="bd_liq_rate", value=0.60,
         overrides=dict(bd_liq_rate=0.60)),
]


# ── Cell runner ────────────────────────────────────────────────────────────────
def run_robustness_cell(cell: dict, base_cfg: SimConfig,
                        out_root: str, n_runs: int) -> dict:
    """
    Run one robustness cell, write CSV + JSON outputs, return bucket counts.
    """
    cell_id = cell["id"]
    overrides = cell["overrides"]
    out_dir = f"{out_root}/cell_{cell_id}"
    os.makedirs(out_dir, exist_ok=True)

    cfg_template = replace(base_cfg, **overrides)

    print(f"\n[robustness] cell={cell_id}  label={cell['label']}  N={n_runs}")
    print(f"  shock={cfg_template.shock_size:.0%}  beta={cfg_template.beta}"
          f"  phi_cw={cfg_template.phi_cw}  phi_hc={cfg_template.phi_hc}"
          f"  cp_max_loan={cfg_template.cp_max_loan:.0e}"
          f"  hf_lev={cfg_template.hf_lev_target}/{cfg_template.hf_lev_buffer}/{cfg_template.hf_lev_max}")

    rows = []
    for i in range(n_runs):
        cfg_i = replace(cfg_template, seed=i)
        history = Simulation(cfg_i).run()

        cls = classify_run(history)
        metrics = extract_run_metrics(history, cfg_i)

        row = {
            "run": i,
            "cell": cell_id,
            "param": cell["param"],
            "param_value": cell["value"],
            "shock_size": cfg_i.shock_size,
            "bucket": cls["bucket"],
        }
        # Per-entity default flags
        for n, v in enumerate(cls["default_hf"]):
            row[f"default_hf{n}"] = v
        for k, v in enumerate(cls["default_bd"]):
            row[f"default_bd{k}"] = v
        for n, v in enumerate(cls["qdemand_hf"]):
            row[f"qdemand_hf{n}"] = v
        for k, v in enumerate(cls["qdemand_bd"]):
            row[f"qdemand_bd{k}"] = v
        row.update(metrics)
        rows.append(row)
        print(f"  [{i+1:3d}/{n_runs}] {cls['bucket']}", end="\r")

    df = pd.DataFrame(rows)
    df.to_csv(f"{out_dir}/distribution_summary.csv", index=False)

    counts = bucket_counts(df)
    total = sum(counts.values())

    print(f"\n[{cell_id}] bucket counts (N={total}):")
    for name in ["no_default", "hf0_only", "partial", "all_default"]:
        n = counts[name]
        pct = (n / total * 100) if total else 0.0
        print(f"  {name:12s} : {n:4d}  ({pct:5.1f}%)")

    # Mean price change on shock asset as quick diagnostic
    if "price_change_asset0" in df.columns:
        print(f"  mean price_change_asset0 = {df['price_change_asset0'].mean():.4f}")

    with open(f"{out_dir}/bucket_counts.json", "w") as f:
        json.dump({"cell": cell_id, "label": cell["label"],
                   "param": cell["param"], "param_value": cell["value"],
                   "n_runs": total, "shock_size": cfg_template.shock_size,
                   "counts": counts}, f, indent=2)

    return counts


# ── Summary aggregator ─────────────────────────────────────────────────────────
def run_all(cells: list[dict], base_cfg: SimConfig,
            out_root: str, n_runs: int) -> None:
    """Run all cells and write a cross-cell summary CSV."""
    os.makedirs(out_root, exist_ok=True)

    summary_rows = []
    for cell in cells:
        counts = run_robustness_cell(cell, base_cfg, out_root, n_runs)
        total = sum(counts.values())
        row = {
            "cell": cell["id"],
            "label": cell["label"],
            "param": cell["param"],
            "param_value": cell["value"],
            "n_runs": total,
        }
        for bkt in ["no_default", "hf0_only", "partial", "all_default"]:
            row[f"{bkt}_count"] = counts[bkt]
            row[f"{bkt}_pct"] = round(counts[bkt] / total * 100, 1) if total else 0.0

        # Also load distribution_summary for aggregate stats
        csv_path = f"{out_root}/cell_{cell['id']}/distribution_summary.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            metric_cols = [c for c in df.columns if c.startswith(
                ("price_change_", "capital_change_", "n_forced_sales_"))]
            for col in metric_cols:
                row[f"{col}_mean"] = round(df[col].mean(), 6)
                row[f"{col}_std"] = round(df[col].std(), 6)

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = f"{out_root}/robustness_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[robustness] wrote {summary_path}")

    # Print compact table
    cols = ["cell", "param", "param_value",
            "hf0_only_pct", "partial_pct", "all_default_pct"]
    print(summary_df[[c for c in cols if c in summary_df.columns]].to_string(index=False))


# ── Entry point ────────────────────────────────────────────────────────────────
def main(argv: list[str]) -> None:
    os.makedirs(OUT_ROOT, exist_ok=True)

    requested = set(argv[1:])
    cells_to_run = (
        [c for c in CELLS if any(s in c["id"] for s in requested)]
        if requested else CELLS
    )

    if not cells_to_run:
        print(f"No cells matched: {requested}")
        print("Available cell ids:", [c["id"] for c in CELLS])
        return

    print(f"[robustness] Running {len(cells_to_run)} cell(s), "
          f"N_RUNS={N_RUNS} each -> {OUT_ROOT}/")

    run_all(cells_to_run, BASE, OUT_ROOT, N_RUNS)


if __name__ == "__main__":
    main(sys.argv)
