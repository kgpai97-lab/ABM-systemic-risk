"""
experiments/sensitivity_lhs.py
------------------------------
Global sensitivity via Latin Hypercube Sampling (LHS).

Unlike experiments/robustness.py (which is OFAT — one factor at a time around a
baseline), this varies ALL ~10 levers simultaneously over the joint parameter
space.  Each LHS point is a unique config; we run it for N_SEEDS seeds and store
the per-point mean of the 16 outcome metrics (reused from robustness.py) plus the
four bucket fractions.  The companion experiments/sensitivity_regression.py then
fits outcome ~ (all params)**2 (main effects + all pairwise interactions), which
the OFAT design structurally cannot estimate.

Shock is FIXED per run (not sampled): run once at -15% and once at -20% via the
SHOCK env var, writing shock-tagged CSVs.

Usage:
    PYTHONPATH=. python experiments/sensitivity_lhs.py              # -15%, full
    SHOCK=-0.20 PYTHONPATH=. python experiments/sensitivity_lhs.py  # -20%, full
    PYTHONPATH=. python experiments/sensitivity_lhs.py --smoke      # fast check
    SHOCK=-0.20 PYTHONPATH=. python experiments/sensitivity_lhs.py --smoke

Outputs:
    outputs/sensitivity_lhs_2026_05_27/lhs_samples_<tag>.csv
        one row per LHS point: sampled params + 16 mean outcomes + bucket fractions
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from dataclasses import replace
from scipy.stats import qmc

from bookstaber_abm.config import SimConfig
from bookstaber_abm.simulation.engine import Simulation
from bookstaber_abm.analysis.buckets import classify_run

# Reuse the OFAT baseline and the metric extractor verbatim — do not re-implement.
from experiments.robustness import BASE, extract_run_metrics
from experiments.sweep import hetero_4hf_6assets_gradient


# ── Run configuration ────────────────────────────────────────────────────────
N_LHS = 1500          # distinct LHS points (≈27 rows / coefficient at 55 coefs)
N_SEEDS = 8           # seeds per point, averaged to one regression row
LHS_SEED = 20260527   # reproducible scramble

SHOCK = float(os.environ.get("SHOCK", "-0.15"))
_TAG = f"{abs(SHOCK) * 100:.0f}pct"
OUT_ROOT = "outputs/sensitivity_lhs_2026_05_30"  # cp_max_loan floor lowered to 2e5 (credit-gate now reachable)
OUT_CSV = f"{OUT_ROOT}/lhs_samples_{_TAG}.csv"

BUCKETS = ["no_default", "hf0_only", "partial", "all_default"]


# ── Parameter space ──────────────────────────────────────────────────────────
# The 9 simply-mapped dims (name, lo, hi, scale).  Leverage (target + two
# positive offsets) is handled separately as dims 9–11 so the strict assertion
# hf_lev_target < buffer < max holds by construction (no rejection sampling).
#
# Dim order is fixed and load-bearing — build_param_row reads u_row by index.
PARAM_SPECS = [
    ("beta",              0.28,   1.10,   "linear"),   # 0
    ("noise_std",         0.0015, 0.009,  "linear"),   # 1
    ("cp_haircut_normal", 0.07,   0.16,   "linear"),   # 2
    ("phi_cw",            500.0,  4000.0, "log"),       # 3
    ("phi_hc",            0.05,   0.40,   "linear"),    # 4
    ("hf0_shock_weight",  0.10,   0.60,   "linear"),    # 5
    ("cp_max_loan",       2e5,    5e7,    "log"),       # 6  (floor lowered 1e6->2e5 2026-05-30: the CW loan gate only binds when cp_max_loan approaches the ~140-305k collateral-implied loan; the prior 1e6 floor sat ~3x above the ~300k activation threshold so the gate never bound across the whole LHS. New floor spans the 200k-500k activation band.)
    ("bd_liq_ratio_min",  0.015,  0.040,  "linear"),    # 7
    ("hf_lev_target",     5.0,    10.0,   "linear"),    # 8
]
# Dims 9, 10: hf_lev_buffer / hf_lev_max positive offsets (see build_param_row).
# Dim 11: bd_liq_rate.
D = 12

# The 10 canonical regressors the companion script regresses on (buffer/max are
# derived, not independent — excluded to avoid collinearity).
CANONICAL_PREDICTORS = [
    "beta", "noise_std", "cp_haircut_normal", "phi_cw", "phi_hc",
    "hf0_shock_weight", "cp_max_loan", "bd_liq_ratio_min", "hf_lev_target",
    "bd_liq_rate",
]


def _scale(u: float, lo: float, hi: float, scale: str) -> float:
    """Map a unit-interval value u∈[0,1] onto [lo, hi], linear or log."""
    if scale == "log":
        return float(np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo))))
    return float(lo + u * (hi - lo))


def sample_unit_hypercube(n: int, d: int, seed: int) -> np.ndarray:
    """Scrambled Latin Hypercube sample of n points in the d-dim unit cube."""
    sampler = qmc.LatinHypercube(d=d, scramble=True, seed=seed)
    return sampler.random(n)


def build_param_row(u_row: np.ndarray) -> dict:
    """
    Map one length-D unit vector to a dict of canonical config scalars plus the
    derived leverage hierarchy and the ratio target.

    Leverage reparam (zero rejection): buffer and max are strictly-positive
    offsets off target, so hf_lev_target < hf_lev_buffer < hf_lev_max always.
    Centered on the OFAT geometry (buffer ≈ 1.75×target, max ≈ 2.5×target).
    """
    p: dict = {}
    for i, (name, lo, hi, scale) in enumerate(PARAM_SPECS):
        p[name] = _scale(u_row[i], lo, hi, scale)

    # Leverage offsets (dims 9, 10). Offset ranges are fractions of the parent
    # value, so both offsets are strictly positive.
    target = p["hf_lev_target"]
    buf_offset = _scale(u_row[9], 0.55 * target, 0.95 * target, "linear")
    buffer = target + buf_offset
    max_offset = _scale(u_row[10], 0.55 * buffer, 0.85 * buffer, "linear")
    lev_max = buffer + max_offset
    p["hf_lev_buffer"] = buffer
    p["hf_lev_max"] = lev_max

    # bd_liq_rate (dim 11)
    p["bd_liq_rate"] = _scale(u_row[11], 0.15, 0.60, "linear")

    # Preserve the min<target liquidity-ratio gap (matches OFAT pairs, +0.010).
    p["bd_liq_ratio_target"] = p["bd_liq_ratio_min"] + 0.010

    return p


def make_config(params: dict, shock: float) -> SimConfig:
    """Build a SimConfig from a param row, inheriting everything else from BASE."""
    overrides = dict(
        shock_size=shock,
        beta=params["beta"],
        noise_std=params["noise_std"],
        cp_haircut_normal=params["cp_haircut_normal"],
        phi_cw=params["phi_cw"],
        phi_hc=params["phi_hc"],
        cp_max_loan=params["cp_max_loan"],
        bd_liq_ratio_min=params["bd_liq_ratio_min"],
        bd_liq_ratio_target=params["bd_liq_ratio_target"],
        hf_lev_target=params["hf_lev_target"],
        hf_lev_buffer=params["hf_lev_buffer"],
        hf_lev_max=params["hf_lev_max"],
        bd_liq_rate=params["bd_liq_rate"],
        # HF0 shock-asset weight rebuilds the allocation matrix (auto-normalised).
        hf_allocations_hetero=hetero_4hf_6assets_gradient(
            params["hf0_shock_weight"], 0.20, 0.18, 0.18
        ),
    )
    # __post_init__ validation fires here; the offset reparam guarantees it passes.
    return replace(BASE, **overrides)


def run_point(params: dict, shock: float, n_seeds: int) -> dict:
    """
    Run one LHS point over n_seeds; return params + mean of the 16 metrics + the
    four bucket fractions.  One row for the regression (the stochastic mean).
    """
    metric_rows = []
    bucket_tally = {b: 0 for b in BUCKETS}

    for seed in range(n_seeds):
        cfg = replace(make_config(params, shock), seed=seed)
        history = Simulation(cfg).run()
        metric_rows.append(extract_run_metrics(history, cfg))
        bucket_tally[classify_run(history)["bucket"]] += 1

    mdf = pd.DataFrame(metric_rows)
    mean_metrics = {col: float(mdf[col].mean()) for col in mdf.columns}
    bucket_fracs = {f"frac_{b}": bucket_tally[b] / n_seeds for b in BUCKETS}

    row = {**params, "shock_size": shock, "n_seeds": n_seeds}
    row.update(mean_metrics)
    row.update(bucket_fracs)
    return row


def main(argv: list[str]) -> None:
    os.makedirs(OUT_ROOT, exist_ok=True)

    smoke = "--smoke" in argv
    n_lhs = 20 if smoke else N_LHS
    n_seeds = 2 if smoke else N_SEEDS

    print(f"[lhs] shock={SHOCK:.0%}  N_LHS={n_lhs}  N_SEEDS={n_seeds}  "
          f"({n_lhs * n_seeds} sims)  -> {OUT_CSV}")

    U = sample_unit_hypercube(n_lhs, D, LHS_SEED)

    rows = []
    for i in range(n_lhs):
        params = build_param_row(U[i])
        rows.append(run_point(params, SHOCK, n_seeds))
        print(f"  [{i + 1:4d}/{n_lhs}] "
              f"beta={params['beta']:.2f} lev={params['hf_lev_target']:.1f}"
              f"/{params['hf_lev_buffer']:.1f}/{params['hf_lev_max']:.1f} "
              f"f0={params['hf0_shock_weight']:.2f}", end="\r")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n[lhs] wrote {OUT_CSV}  ({len(df)} rows, {len(df.columns)} cols)")

    # Quick diagnostic: mean shock-asset price change + bucket mix.
    if "price_change_asset0" in df.columns:
        print(f"  mean price_change_asset0 = {df['price_change_asset0'].mean():.4f}")
    frac_cols = [f"frac_{b}" for b in BUCKETS]
    print("  mean bucket fractions: "
          + "  ".join(f"{b}={df[f'frac_{b}'].mean():.2f}" for b in BUCKETS))


if __name__ == "__main__":
    main(sys.argv)
