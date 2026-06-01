"""
experiments/sensitivity_regression.py
--------------------------------------
Regression analysis of the LHS global-sensitivity sample
(experiments/sensitivity_lhs.py).  For each outcome metric we fit

    outcome ~ (z_p1 + z_p2 + ... + z_p10)**2

via statsmodels' formula interface.  In patsy, `(...)**2` expands to all main
effects PLUS all pairwise interactions (no quadratic self-terms) — 10 main +
C(10,2)=45 interaction = 55 terms.  Predictors are z-scored first, so coefficients
are in SD units (comparable across levers) and the interaction products are
centered, which sharply reduces multicollinearity.

This is the analysis the OFAT design (experiments/regression_analysis.py) cannot
do: it estimates how levers interact, not just their isolated main effects.

Shock is fixed per run via the SHOCK env var (reads the matching LHS CSV).

Usage:
    PYTHONPATH=. python experiments/sensitivity_regression.py              # -15%
    SHOCK=-0.20 PYTHONPATH=. python experiments/sensitivity_regression.py  # -20%

Outputs (under outputs/sensitivity_lhs_2026_05_27/):
    regression_report_<tag>.txt   — human-readable per-outcome tables
    regression_coefs_<tag>.csv    — tidy long table of every coefficient
"""
from __future__ import annotations

import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# Canonical predictor list lives with the sampler (single source of truth).
from experiments.sensitivity_lhs import CANONICAL_PREDICTORS, OUT_ROOT


SHOCK = float(os.environ.get("SHOCK", "-0.15"))
_TAG = f"{abs(SHOCK) * 100:.0f}pct"
IN_CSV = f"{OUT_ROOT}/lhs_samples_{_TAG}.csv"
REPORT = f"{OUT_ROOT}/regression_report_{_TAG}.txt"
COEF_CSV = f"{OUT_ROOT}/regression_coefs_{_TAG}.csv"

PREDICTORS = CANONICAL_PREDICTORS

# Outcomes to model (subset of the 16; the headline price/capital/forced-sale set).
OUTCOMES = [
    "price_change_asset0",
    "price_change_asset1",
    "capital_change_hf0",
    "capital_change_hf1",
    "capital_change_bd0",
    "n_forced_sales_hf0",
    "n_forced_sales_hf1",
]

# OFAT prior (from outputs/robustness_2026_05_27): which main effects were
# dominant vs inert, so flip_check can flag disagreements.
OFAT_DOMINANT = {"beta", "hf_lev_target", "hf0_shock_weight"}
OFAT_INERT = {
    "cp_haircut_normal", "phi_cw", "phi_hc", "cp_max_loan",
    "bd_liq_ratio_min", "bd_liq_rate", "noise_std",
}


def stars(pval: float) -> str:
    if pval < 0.01:  return "***"
    if pval < 0.05:  return "**"
    if pval < 0.10:  return "*"
    return ""


def load(in_csv: str) -> pd.DataFrame:
    if not os.path.exists(in_csv):
        raise FileNotFoundError(
            f"{in_csv} not found. Run the sampler first:\n"
            f"  SHOCK={SHOCK} PYTHONPATH=. python experiments/sensitivity_lhs.py"
        )
    df = pd.read_csv(in_csv)
    missing = [c for c in PREDICTORS if c not in df.columns]
    if missing:
        raise ValueError(f"{in_csv} missing predictor columns: {missing}")
    return df


def zscore(df: pd.DataFrame, predictors: list[str]) -> pd.DataFrame:
    """Add z_<p> columns (mean 0, sd 1) for each predictor."""
    out = df.copy()
    for p in predictors:
        mu = df[p].mean()
        sd = df[p].std(ddof=0)
        out[f"z_{p}"] = (df[p] - mu) / sd if sd > 0 else 0.0
    return out


def fit_one(df_z: pd.DataFrame, outcome: str) -> tuple[pd.DataFrame, dict]:
    """
    Fit outcome ~ (z_p1 + ... + z_pk)**2.  Return (coef_table, fit_stats).
    coef_table is sorted by |estimate| descending, with an is_interaction flag.
    """
    zcols = [f"z_{p}" for p in PREDICTORS]
    formula = f"{outcome} ~ ({' + '.join(zcols)})**2"
    model = smf.ols(formula, data=df_z).fit()

    rows = []
    for term in model.params.index:
        if term == "Intercept":
            continue
        rows.append({
            "outcome": outcome,
            "term": term.replace("z_", ""),  # strip z_ for readability
            "estimate": model.params[term],
            "std_err": model.bse[term],
            "t": model.tvalues[term],
            "pvalue": model.pvalues[term],
            "stars": stars(model.pvalues[term]),
            "is_interaction": ":" in term,
        })
    coef = pd.DataFrame(rows)
    coef["abs_est"] = coef["estimate"].abs()
    coef = coef.sort_values("abs_est", ascending=False).drop(columns="abs_est")

    stats = {
        "outcome": outcome,
        "rsquared": model.rsquared,
        "rsquared_adj": model.rsquared_adj,
        "fvalue": model.fvalue,
        "f_pvalue": model.f_pvalue,
        "nobs": int(model.nobs),
    }
    return coef, stats


def flip_check(coef: pd.DataFrame) -> list[str]:
    """Flag main effects whose significance disagrees with the OFAT prior."""
    notes = []
    mains = coef[~coef["is_interaction"]]
    for _, r in mains.iterrows():
        name, sig = r["term"], bool(r["stars"])
        if name in OFAT_DOMINANT and not sig:
            notes.append(f"    FLIP: {name} was OFAT-dominant but is ns here")
        if name in OFAT_INERT and sig:
            notes.append(f"    FLIP: {name} was OFAT-inert but is significant here "
                         f"(est={r['estimate']:.4f}{r['stars']})")
    return notes


def write_report(all_coefs: list[pd.DataFrame], all_stats: list[dict]) -> None:
    lines = []
    lines.append("=" * 92)
    lines.append(f"LHS GLOBAL SENSITIVITY — main effects + pairwise interactions   (shock={SHOCK:.0%})")
    lines.append(f"  formula: outcome ~ (z_{{{', '.join(PREDICTORS)}}})**2")
    lines.append(f"  {len(PREDICTORS)} main + {len(PREDICTORS)*(len(PREDICTORS)-1)//2} pairwise = "
                 f"{len(PREDICTORS) + len(PREDICTORS)*(len(PREDICTORS)-1)//2} terms")
    lines.append("  predictors z-scored; estimates in SD units.  * p<0.10  ** p<0.05  *** p<0.01")
    lines.append("=" * 92)
    lines.append("")

    for coef, st in zip(all_coefs, all_stats):
        lines.append("-" * 92)
        lines.append(f"OUTCOME: {st['outcome']}    "
                     f"R²={st['rsquared']:.3f}  adjR²={st['rsquared_adj']:.3f}  "
                     f"F={st['fvalue']:.1f} (p={st['f_pvalue']:.2e})  N={st['nobs']}")
        lines.append("-" * 92)

        mains = coef[~coef["is_interaction"]]
        inters = coef[coef["is_interaction"]]

        lines.append("  MAIN EFFECTS (by |estimate|):")
        lines.append(f"    {'param':<22} {'estimate':>10} {'std_err':>9} {'p':>8} {'sig':>4}")
        for _, r in mains.iterrows():
            lines.append(f"    {r['term']:<22} {r['estimate']:>10.4f} {r['std_err']:>9.4f} "
                         f"{r['pvalue']:>8.4f} {r['stars']:>4}")

        sig_inters = inters[inters["stars"] != ""].head(10)
        lines.append(f"  TOP SIGNIFICANT INTERACTIONS ({len(inters[inters['stars'] != ''])} sig of "
                     f"{len(inters)} total; top 10 by |estimate|):")
        if sig_inters.empty:
            lines.append("    (none significant)")
        else:
            for _, r in sig_inters.iterrows():
                lines.append(f"    {r['term']:<34} {r['estimate']:>10.4f} "
                             f"{r['pvalue']:>8.4f} {r['stars']:>4}")

        notes = flip_check(coef)
        if notes:
            lines.append("  vs OFAT prior:")
            lines.extend(notes)
        lines.append("")

    report = "\n".join(lines)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print(f"\n[regression] wrote {REPORT}")


def main(argv: list[str]) -> None:
    os.makedirs(OUT_ROOT, exist_ok=True)
    print(f"[regression] loading {IN_CSV}")
    df = load(IN_CSV)
    print(f"  {len(df)} LHS points, {len(PREDICTORS)} predictors")
    df_z = zscore(df, PREDICTORS)

    all_coefs, all_stats = [], []
    for outcome in OUTCOMES:
        if outcome not in df.columns:
            print(f"  [skip] {outcome} not in CSV")
            continue
        coef, st = fit_one(df_z, outcome)
        all_coefs.append(coef)
        all_stats.append(st)

    pd.concat(all_coefs, ignore_index=True).to_csv(COEF_CSV, index=False)
    print(f"[regression] wrote {COEF_CSV}")
    write_report(all_coefs, all_stats)


if __name__ == "__main__":
    main(sys.argv)
