"""
simulation/monte_carlo.py
-------------------------
Monte Carlo runner for the Bookstaber ABM.

Runs a SimConfig N times with different random seeds, then aggregates
results into per-step statistics: mean, std, and 5th/95th percentiles.

Usage
-----
    from bookstaber_abm.simulation.monte_carlo import MonteCarloRunner

    cfg = SimConfig(n_steps=150, shock_size=-0.20, mc_runs=30)
    mc  = MonteCarloRunner(cfg)
    mc.run()

    df_mean = mc.summary("mean")   # DataFrame: index=t, cols=metric
    bands   = mc.bands("price_0")  # dict with mean/p5/p95/std arrays
"""

from __future__ import annotations
import copy
import numpy as np
import pandas as pd
from dataclasses import replace

from bookstaber_abm.config import SimConfig
from bookstaber_abm.simulation.engine import Simulation

# Scalar metrics extracted from each history record
_SCALAR_METRICS = [
    "n_fire_sales",
    "n_active_hf",
    "n_defaults",
    "n_bd_defaults",
    "total_forced_flow",
    "portfolio_overlap",
    "deriv_exposure",
    "deriv_losses",
]


def _extract_scalars(record: dict) -> dict:
    """Flatten one history record into a plain dict of floats."""
    row: dict[str, float] = {"t": record["t"]}

    for key in _SCALAR_METRICS:
        row[key] = float(record.get(key, 0.0))

    for m, p in enumerate(record["prices"]):
        row[f"price_{m}"] = float(p)

    for m, f in enumerate(record["net_forced_flow"]):
        row[f"flow_{m}"] = float(f)

    for n, c in enumerate(record["hf_capitals"]):
        row[f"hf_cap_{n}"] = float(c)

    for n, lev in enumerate(record["hf_leverages"]):
        row[f"hf_lev_{n}"] = float(lev) if np.isfinite(lev) else float("nan")

    for k, c in enumerate(record["bd_capitals"]):
        row[f"bd_cap_{k}"] = float(c)

    for k, hc in enumerate(record.get("haircuts", [])):
        row[f"haircut_bd{k}"] = float(hc)

    return row


class MonteCarloRunner:
    """
    Runs cfg.mc_runs independent simulations with seeds
    cfg.seed, cfg.seed+1, ..., cfg.seed+mc_runs-1.

    Attributes
    ----------
    raw : list[list[dict]]
        raw[run_idx][step] = history record
    columns : list[str]
        All metric names extracted from records
    """

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.raw: list[list[dict]] = []
        self.columns: list[str] = []
        self._stats: dict[str, np.ndarray] | None = None   # cached

    # ------------------------------------------------------------------ #
    # Execution                                                            #
    # ------------------------------------------------------------------ #

    def run(self, verbose: bool = False) -> "MonteCarloRunner":
        """
        Execute all Monte Carlo runs.  Returns self for chaining.

        Each run uses a different seed so price paths and allocation
        draws are independent across runs.
        """
        self.raw = []
        for i in range(self.cfg.mc_runs):
            run_cfg = replace(self.cfg, seed=self.cfg.seed + i)
            sim = Simulation(run_cfg)
            history = sim.run()
            self.raw.append(history)
            if verbose:
                n_def = history[-1]["n_defaults"]
                print(f"  Run {i+1:3d}/{self.cfg.mc_runs}  seed={run_cfg.seed}"
                      f"  defaults={n_def}")

        # Determine column names from first run
        if self.raw:
            self.columns = list(_extract_scalars(self.raw[0][0]).keys())

        self._stats = None   # invalidate cache
        return self

    # ------------------------------------------------------------------ #
    # Statistics                                                           #
    # ------------------------------------------------------------------ #

    def _build_stats(self) -> dict[str, np.ndarray]:
        """
        Build a dict mapping each metric name → 2D array shape (n_steps, n_runs).
        """
        if not self.raw:
            raise RuntimeError("Call .run() before accessing statistics.")

        n_steps = len(self.raw[0])
        n_runs  = len(self.raw)

        # data[metric][step, run]
        data: dict[str, np.ndarray] = {}
        for col in self.columns:
            data[col] = np.full((n_steps, n_runs), np.nan)

        for run_idx, history in enumerate(self.raw):
            for step_idx, record in enumerate(history):
                row = _extract_scalars(record)
                for col, val in row.items():
                    if col in data:
                        data[col][step_idx, run_idx] = val

        return data

    def _get_stats(self) -> dict[str, np.ndarray]:
        if self._stats is None:
            self._stats = self._build_stats()
        return self._stats

    def summary(self, stat: str = "mean") -> pd.DataFrame:
        """
        Return a DataFrame (index=t, columns=metrics) for one statistic.

        stat: one of 'mean' | 'std' | 'p5' | 'p25' | 'median' | 'p75' | 'p95'
        """
        data = self._get_stats()
        n_steps = len(self.raw[0])

        funcs = {
            "mean":   lambda x: np.nanmean(x, axis=1),
            "std":    lambda x: np.nanstd(x, axis=1),
            "p5":     lambda x: np.nanpercentile(x, 5, axis=1),
            "p25":    lambda x: np.nanpercentile(x, 25, axis=1),
            "median": lambda x: np.nanpercentile(x, 50, axis=1),
            "p75":    lambda x: np.nanpercentile(x, 75, axis=1),
            "p95":    lambda x: np.nanpercentile(x, 95, axis=1),
        }
        if stat not in funcs:
            raise ValueError(f"stat must be one of {list(funcs)}")

        fn = funcs[stat]
        rows = {col: fn(data[col]) for col in self.columns if col != "t"}
        df = pd.DataFrame(rows, index=range(n_steps))
        df.index.name = "t"
        return df

    def bands(self, metric: str) -> dict[str, np.ndarray]:
        """
        Return confidence band arrays for a single metric.

        Returns dict with keys: mean, std, p5, p25, median, p75, p95
        Each value is a 1D array of length n_steps.
        """
        data = self._get_stats()
        if metric not in data:
            raise KeyError(f"Metric '{metric}' not found. "
                           f"Available: {list(data.keys())[:10]}...")
        x = data[metric]
        return {
            "mean":   np.nanmean(x, axis=1),
            "std":    np.nanstd(x, axis=1),
            "p5":     np.nanpercentile(x, 5, axis=1),
            "p25":    np.nanpercentile(x, 25, axis=1),
            "median": np.nanpercentile(x, 50, axis=1),
            "p75":    np.nanpercentile(x, 75, axis=1),
            "p95":    np.nanpercentile(x, 95, axis=1),
        }

    def crisis_probability(self, threshold: int = 1) -> np.ndarray:
        """
        P(n_fire_sales >= threshold) at each step, across all runs.
        Returns 1D array of length n_steps.
        """
        data = self._get_stats()
        x = data["n_fire_sales"]   # shape (n_steps, n_runs)
        return (x >= threshold).mean(axis=1)

    def default_distribution(self) -> np.ndarray:
        """
        Distribution of total HF defaults at end of run across all runs.
        Returns 1D array of length n_runs.
        """
        return np.array([h[-1]["n_defaults"] for h in self.raw], dtype=float)
