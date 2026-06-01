"""
experiments/regression_analysis.py
------------------------------------
OLS regression analysis of the robustness sweep, mirroring Table 3 of
Bookstaber-Paddrik-Tivnan.

For each outcome variable (price_change_asset0/1/2, capital_change_hf0/1,
capital_change_bd0/1, n_forced_sales_hf0/1) we run a cross-cell OLS regression
with the parameter value as the regressor, grouped by parameter group.

Additionally runs a pooled regression across all cells with dummies for each
parameter group, analogous to the paper's Table 3 significance test.

Usage:
    PYTHONPATH=. python experiments/regression_analysis.py

Outputs:
    outputs/robustness/regression_table3.csv    -- per-outcome OLS results
    outputs/robustness/regression_pooled.csv    -- pooled regression
    outputs/robustness/regression_report.txt    -- human-readable summary
"""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import glob

# statsmodels for OLS + significance stars
try:
    import statsmodels.api as sm
    from statsmodels.regression.linear_model import OLS
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("[warn] statsmodels not found — install with: pip install statsmodels")
    print("       Falling back to numpy-based OLS (no p-values).")


OUT_ROOT = "outputs/robustness_2026_05_27"
REPORT_PATH = f"{OUT_ROOT}/regression_report.txt"

# ── Outcome variables matching Table 3 of the paper ──────────────────────────
OUTCOMES = [
    "price_change_asset0",
    "price_change_asset1",
    "price_change_asset2",
    "capital_change_hf0",
    "capital_change_hf1",
    "capital_change_bd0",
    "capital_change_bd1",
    "n_forced_sales_hf0",
    "n_forced_sales_hf1",
]

# Labels matching paper table rows
OUTCOME_LABELS = {
    "price_change_asset0": "Price change Asset 1",
    "price_change_asset1": "Price change Asset 2",
    "price_change_asset2": "Price change Asset 3",
    "capital_change_hf0":  "Capital change HF 1 (canary)",
    "capital_change_hf1":  "Capital change HF 2",
    "capital_change_bd0":  "Capital change BD 1",
    "capital_change_bd1":  "Capital change BD 2",
    "n_forced_sales_hf0":  "# forced sales HF 1",
    "n_forced_sales_hf1":  "# forced sales HF 2",
}

# Parameter group ordering for the pooled regression (matches paper's Table 3 columns)
PARAM_GROUPS = [
    "shock_size",
    "beta",
    "cp_haircut_normal",
    "phi_cw",
    "phi_hc",
    "hf0_shock_weight",
    "cp_max_loan",
    "bd_liq_ratio_min",
    "hf_lev_target",
    "noise_std",
    "bd_liq_rate",
]

PARAM_LABELS = {
    "shock_size":         "Shock size",
    "beta":               "beta_m (price impact)",
    "cp_haircut_normal":  "HC_c,k (haircut)",
    "phi_cw":             "phi_CW",
    "phi_hc":             "phi_HC",
    "hf0_shock_weight":   "A_k(0) (HF0 alloc)",
    "cp_max_loan":        "Q^Max_k(t) (loan cap)",
    "bd_liq_ratio_min":   "LiqRatio_Min",
    "hf_lev_target":      "Leverage target",
    "noise_std":          "noise_std (price noise)",
    "bd_liq_rate":        "BD liq-reserve rate",
}


def stars(pval: float) -> str:
    if pval < 0.01:  return "***"
    if pval < 0.05:  return "**"
    if pval < 0.10:  return "*"
    return ""


def load_all_runs() -> pd.DataFrame:
    """Load all per-cell distribution_summary CSVs into one pooled DataFrame."""
    paths = sorted(glob.glob(f"{OUT_ROOT}/cell_*/distribution_summary.csv"))
    if not paths:
        raise FileNotFoundError(
            f"No cell CSVs found in {OUT_ROOT}/cell_*/. "
            "Run experiments/robustness.py first."
        )
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def run_ols_by_param(data: pd.DataFrame, outcome: str) -> list[dict]:
    """
    For each parameter group: regress outcome ~ param_value (normalised).
    Returns one row per parameter group.
    """
    results = []
    for param in PARAM_GROUPS:
        subset = data[data["param"] == param].copy()
        if len(subset) < 4 or outcome not in subset.columns:
            results.append({
                "param": param, "outcome": outcome,
                "n": len(subset), "estimate": np.nan,
                "pvalue": np.nan, "stars": "", "adj_r2": np.nan
            })
            continue

        y = subset[outcome].values.astype(float)
        x_raw = subset["param_value"].values.astype(float)

        # Standardise x so coefficients are comparable across groups
        x_std = x_raw.std()
        if x_std == 0:
            results.append({
                "param": param, "outcome": outcome,
                "n": len(subset), "estimate": np.nan,
                "pvalue": np.nan, "stars": "—", "adj_r2": np.nan
            })
            continue
        x_norm = (x_raw - x_raw.mean()) / x_std

        if HAS_STATSMODELS:
            X = sm.add_constant(x_norm)
            try:
                model = OLS(y, X, missing="drop").fit()
                est  = model.params[1]
                pval = model.pvalues[1]
                adj_r2 = model.rsquared_adj
                st = stars(pval)
            except Exception as e:
                est, pval, adj_r2, st = np.nan, np.nan, np.nan, f"ERR:{e}"
        else:
            # numpy fallback: no p-values
            coeffs = np.polyfit(x_norm, y, 1)
            est = coeffs[0]
            ss_res = np.sum((y - (coeffs[0]*x_norm + coeffs[1]))**2)
            ss_tot = np.sum((y - y.mean())**2)
            r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
            adj_r2 = 1 - (1 - r2)*(len(y)-1)/(len(y)-2) if len(y) > 2 else np.nan
            pval, st = np.nan, ""

        results.append({
            "param": param, "outcome": outcome,
            "n": len(subset), "estimate": round(est, 6),
            "pvalue": round(pval, 4) if not np.isnan(pval) else np.nan,
            "stars": st, "adj_r2": round(adj_r2, 4) if not np.isnan(adj_r2) else np.nan
        })
    return results


def run_pooled_regression(data: pd.DataFrame, outcome: str) -> dict | None:
    """
    Pooled OLS across all cells: outcome ~ param_value + param_group_dummies.
    Standardise param_value within each group so coefficients are comparable.
    Returns a dict of {param: (estimate, pvalue, stars)}.
    """
    if outcome not in data.columns:
        return None

    rows = []
    for _, grp in data.groupby("param"):
        g = grp.copy()
        std = g["param_value"].std()
        if std > 0:
            g["x_norm"] = (g["param_value"] - g["param_value"].mean()) / std
        else:
            g["x_norm"] = 0.0
        rows.append(g)
    df_pool = pd.concat(rows, ignore_index=True)

    # Build feature matrix: one x_norm per param group (interaction), intercept
    dummies = pd.get_dummies(df_pool["param"], prefix="p", drop_first=False)
    # Interact each dummy with x_norm
    for col in dummies.columns:
        dummies[col] = dummies[col] * df_pool["x_norm"]

    y = df_pool[outcome].values.astype(float)
    X_df = pd.concat([pd.Series(np.ones(len(y)), name="const"), dummies], axis=1)
    X = X_df.values.astype(float)

    if not HAS_STATSMODELS:
        return None

    try:
        model = OLS(y, X, missing="drop").fit()
    except Exception:
        return None

    out = {"adj_r2": round(model.rsquared_adj, 4),
           "f_stat": round(model.fvalue, 2) if model.fvalue else np.nan}
    for i, col in enumerate(X_df.columns[1:], start=1):
        param_name = col[2:]  # strip "p_" prefix
        est  = model.params[i]
        pval = model.pvalues[i]
        out[param_name] = {"estimate": round(est,6), "pvalue": round(pval,4), "stars": stars(pval)}
    return out


def format_table3(by_param_rows: list[dict], pooled: dict | None) -> str:
    """Render a text table mirroring Table 3 layout."""
    lines = []
    lines.append("=" * 90)
    lines.append("TABLE 3 ANALOGUE: OLS significance of model parameters")
    lines.append("Outcome ~ standardised(param_value) | per parameter group")
    lines.append("=" * 90)
    lines.append(f"{'Param':<22} {'Outcome':<30} {'N':>5} {'Estimate':>10} {'p-val':>8} {'Sig':>4} {'AdjR2':>7}")
    lines.append("-" * 90)

    df = pd.DataFrame(by_param_rows)
    for param in PARAM_GROUPS:
        subset = df[df["param"] == param]
        for _, row in subset.iterrows():
            pval_str = f"{row['pvalue']:.4f}" if not pd.isna(row["pvalue"]) else "   n/a"
            adj_r2_str = f"{row['adj_r2']:.4f}" if not pd.isna(row["adj_r2"]) else "   n/a"
            lines.append(
                f"{PARAM_LABELS.get(row['param'], row['param']):<22} "
                f"{OUTCOME_LABELS.get(row['outcome'], row['outcome']):<30} "
                f"{int(row['n']):>5} "
                f"{row['estimate']:>10.4f} "
                f"{pval_str:>8} "
                f"{row['stars']:>4} "
                f"{adj_r2_str:>7}"
            )
        lines.append("")

    if pooled:
        lines.append("=" * 90)
        lines.append("POOLED REGRESSION (all params, interaction with group dummies)")
        lines.append(f"  Adj.R2={pooled.get('adj_r2','n/a')}  F-stat={pooled.get('f_stat','n/a')}")
        lines.append(f"{'Param':<22} {'Estimate':>12} {'p-val':>10} {'Sig':>5}")
        lines.append("-" * 55)
        for param in PARAM_GROUPS:
            # Match against pooled keys (param group name as stored)
            key = f"p_{param}"
            # statsmodels may strip/rename; search for it
            matched = {k: v for k, v in pooled.items()
                       if isinstance(v, dict) and param in k}
            if matched:
                k0 = list(matched.keys())[0]
                v = matched[k0]
                lines.append(
                    f"{PARAM_LABELS.get(param, param):<22} "
                    f"{v['estimate']:>12.6f} "
                    f"{v['pvalue']:>10.4f} "
                    f"{v['stars']:>5}"
                )
            else:
                lines.append(f"{PARAM_LABELS.get(param, param):<22}   (not in pooled model)")
        lines.append("")

    return "\n".join(lines)


def format_wide_table(all_rows: list[dict]) -> str:
    """
    Wide format: rows = outcomes, columns = param groups.
    Each cell shows estimate + significance stars. Mirrors exact Table 3 shape.
    """
    df = pd.DataFrame(all_rows)
    lines = []
    lines.append("")
    lines.append("WIDE TABLE (estimate [stars]) — columns = parameters, rows = outcomes")
    lines.append("  * p<0.10   ** p<0.05   *** p<0.01")
    lines.append("")

    # Header
    col_w = 18
    header = f"{'Outcome':<32}"
    for p in PARAM_GROUPS:
        label = PARAM_LABELS.get(p, p)[:col_w-1]
        header += f" {label:>{col_w}}"
    lines.append(header)
    lines.append("-" * (32 + (col_w+1)*len(PARAM_GROUPS)))

    for outcome in OUTCOMES:
        row_df = df[df["outcome"] == outcome]
        row_str = f"{OUTCOME_LABELS.get(outcome, outcome):<32}"
        for p in PARAM_GROUPS:
            match = row_df[row_df["param"] == p]
            if match.empty or pd.isna(match.iloc[0]["estimate"]):
                cell = "—"
            else:
                est = match.iloc[0]["estimate"]
                st  = match.iloc[0]["stars"]
                cell = f"{est:.4f}{st}"
            row_str += f" {cell:>{col_w}}"
        lines.append(row_str)

    return "\n".join(lines)


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    print(f"Loading run data from {OUT_ROOT}/cell_*/...")
    data = load_all_runs()
    print(f"  Loaded {len(data)} runs across {data['cell'].nunique()} cells.")

    # ── Per-param OLS ─────────────────────────────────────────────────────────
    all_rows = []
    for outcome in OUTCOMES:
        rows = run_ols_by_param(data, outcome)
        all_rows.extend(rows)

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(f"{OUT_ROOT}/regression_table3.csv", index=False)
    print(f"  Wrote {OUT_ROOT}/regression_table3.csv")

    # ── Pooled regression (one per outcome) ───────────────────────────────────
    pooled_results = {}
    pooled_rows = []
    for outcome in OUTCOMES:
        p = run_pooled_regression(data, outcome)
        pooled_results[outcome] = p
        if p:
            for param in PARAM_GROUPS:
                matched = {k: v for k, v in p.items()
                           if isinstance(v, dict) and param in k}
                if matched:
                    k0 = list(matched.keys())[0]
                    v = matched[k0]
                    pooled_rows.append({
                        "outcome": outcome, "param": param,
                        "estimate": v["estimate"], "pvalue": v["pvalue"],
                        "stars": v["stars"],
                        "adj_r2": p.get("adj_r2"), "f_stat": p.get("f_stat"),
                    })

    if pooled_rows:
        pd.DataFrame(pooled_rows).to_csv(f"{OUT_ROOT}/regression_pooled.csv", index=False)
        print(f"  Wrote {OUT_ROOT}/regression_pooled.csv")

    # ── Report ────────────────────────────────────────────────────────────────
    report_lines = []

    # Bucket summary first
    report_lines.append("=" * 90)
    report_lines.append("BUCKET DISTRIBUTION SUMMARY")
    report_lines.append("=" * 90)
    summary = (data.groupby(["param", "param_value"])["bucket"]
               .value_counts(normalize=True)
               .mul(100).round(1)
               .unstack(fill_value=0.0)
               .reset_index())
    report_lines.append(summary.to_string(index=False))
    report_lines.append("")

    # Mean outcomes per cell
    report_lines.append("=" * 90)
    report_lines.append("MEAN OUTCOMES PER CELL (key metrics)")
    report_lines.append("=" * 90)
    key_metrics = ["price_change_asset0", "capital_change_hf0",
                   "n_forced_sales_hf0", "n_forced_sales_hf1"]
    key_metrics = [m for m in key_metrics if m in data.columns]
    cell_means = (data.groupby(["param", "param_value"])[key_metrics]
                  .mean().round(4).reset_index())
    report_lines.append(cell_means.to_string(index=False))
    report_lines.append("")

    # Table 3 analogue
    report_lines.append(format_table3(all_rows, pooled_results.get(OUTCOMES[0])))
    report_lines.append(format_wide_table(all_rows))

    report_lines.append("")
    report_lines.append("=" * 90)
    report_lines.append("SENSITIVITY RANKING (by |estimate| in price_change_asset0 regression)")
    report_lines.append("=" * 90)
    pc0 = results_df[results_df["outcome"] == "price_change_asset0"].copy()
    pc0["abs_est"] = pc0["estimate"].abs()
    pc0 = pc0.sort_values("abs_est", ascending=False)
    for _, row in pc0.iterrows():
        pval_str = f"p={row['pvalue']:.4f}" if not pd.isna(row["pvalue"]) else "p=n/a"
        flag = row["stars"] if row["stars"] else "ns"
        print_est = f"{row['estimate']:.4f}" if not pd.isna(row["estimate"]) else "n/a"
        report_lines.append(
            f"  {PARAM_LABELS.get(row['param'], row['param']):<28}"
            f"  estimate={print_est:>9}  {pval_str}  {flag}"
        )

    report_lines.append("")
    report_lines.append("NOTES")
    report_lines.append("  - Estimates are from OLS of outcome on *standardised* param_value within each group.")
    report_lines.append("  - N=100 per cell, 2026-05-27 Pareto-ship BASE (noise_std=0.003, liq_frac=0.70,")
    report_lines.append("    rate-limited default liquidation). Shock = -15%.")
    report_lines.append("  - Parameters producing zero variance in outcomes are marked '—' (inert in this regime).")
    report_lines.append("  - See the SENSITIVITY RANKING above for which levers move outcomes in this regime;")
    report_lines.append("    compare against the 2026-05-26 ranking (beta, hf_lev_target dominant) in CLAUDE.md.")

    report = "\n".join(report_lines)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n[regression] wrote {REPORT_PATH}")
    print(report)


if __name__ == "__main__":
    main()
