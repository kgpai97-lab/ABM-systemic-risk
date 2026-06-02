# An Agent-Based Model of Fire Sales and Systemic Risk

An agent-based model (ABM) of financial systemic risk. It simulates a small financial
system — cash providers, bank/dealers, and leveraged hedge funds trading in a
price-impactable asset market — and studies how an exogenous price shock propagates into a
crisis through fire sales and funding runs.

The headline finding of the accompanying write-up is that **the extent of a crisis is
governed by the system's *reaction* to a loss, not by the loss itself.** For the full
narrative, results, and figures, see **[`paper/paper.md`](paper/paper.md)**. This README
covers setup and how to run things.

---

## Requirements

- **Python 3.10+** (the code uses `X | None` type-union syntax). Developed on 3.13.
- Python packages: `numpy`, `pandas`, `matplotlib` (core); `scipy` and `statsmodels`
  (sensitivity experiments); `pytest` (tests). All pinned in
  [`requirements.txt`](requirements.txt).

No compiled extensions, GPU, or external services are needed — it is pure Python + NumPy.

## Setup

```bash
# 1. Clone, then create and activate a virtual environment
python -m venv .venv

# macOS / Linux:
source .venv/bin/activate
# Windows (PowerShell):
.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt
```

The package is **not** pip-installed; it is imported from the repo root. Every command
below is run from the project root with `PYTHONPATH` pointing at it.

- macOS / Linux: prefix commands with `PYTHONPATH=.`
- Windows (PowerShell): set it once per session with `$env:PYTHONPATH = "."`, then run the
  bare `python ...` command.

## Quick start

```bash
# Run the test suite (14 test classes covering config, leverage, price impact,
# funding, haircuts, defaults, derivatives)
PYTHONPATH=. python -m pytest bookstaber_abm/tests/test_mechanics.py -q

# Produce a batch of runs with per-run dashboards + a summary panel
PYTHONPATH=. python batch_run.py            # writes to outputs/runs/

# Render a single detailed per-agent dashboard
PYTHONPATH=. python dashboard.py            # writes outputs/dashboard.png
```

A minimal programmatic run:

```python
from dataclasses import replace
from bookstaber_abm.config import SimConfig
from bookstaber_abm.simulation.engine import Simulation

cfg = SimConfig(n_hedge_funds=4, n_assets=6, n_bank_dealers=2,
                shock_step=50, shock_asset=0, shock_size=-0.20, seed=0)
history = Simulation(cfg).run()      # list of per-step snapshot dicts
print("final asset prices:", history[-1]["prices"])
print("HF active at end:",    history[-1]["hf_active"])
```

## Project layout

```
bookstaber_abm/            # the model package
  config.py                # SimConfig dataclass — every parameter + validation
  simulation/
    engine.py              # the 12-step periodic event loop (the core)
    monte_carlo.py         # multi-run harness + summary statistics
  agents/
    hedge_fund.py          # leverage rules, two-phase fire sale, funding squeeze
    bank_dealer.py         # 4-desk dealer: finance / prime broker / trading / treasury
    cash_provider.py       # loan sizing, haircuts, creditworthiness gate
    derivatives_desk.py    # bilateral counterparty exposures (off by default)
  market/
    asset_market.py        # price impact:  beta_eff = beta0 + beta1*|flow|
  analysis/
    buckets.py             # outcome classifier (no_default / hf0_only / partial / all_default)
    plots.py               # crisis, Monte Carlo, crowding, phase-space plots
  tests/
    test_mechanics.py      # pytest suite

experiments/               # research drivers (each is a CLI script)
  sweep.py                 # outcome-distribution calibration cells (N=1000)
  robustness.py            # one-factor-at-a-time (OFAT) sensitivity sweep
  regression_analysis.py   # OLS on the OFAT sweep
  sensitivity_lhs.py       # global sensitivity: Latin Hypercube design
  sensitivity_regression.py# main-effects + pairwise-interaction regression on the LHS
  contagion_decomposition.py # primary-vs-contagion seed-matched counterfactual

paper/                     # the write-up
  paper.md                 # the manuscript
  make_figures.py          # regenerates every figure + Table 1 from outputs/
  figures/                 # generated PNGs + table fragments

batch_run.py               # production batch harness (root)
dashboard.py               # single-run 6-row agent dashboard (root)
outputs/                   # all generated CSVs, JSONs, and figures
```

## How the model works (one paragraph)

Each step runs a fixed **12-stage event loop** (see `simulation/engine.py`). Agents follow a
strict *pure-compute-then-apply* discipline: they all read the **same** price snapshot, compute
the orders they *would* place (pure, no mutation), and only then does the engine apply every
order and update state — so no agent gets a within-step ordering advantage. Hedge funds and
bank/dealers obey a leverage hierarchy (`target < buffer < max`); breaching `max` triggers a
rate-limited **forced sale** that concentrates on the shocked asset first. Forced sales — and
only forced sales — feed a **price-impact** rule that moves prices once per step on the
aggregated net flow, which marks down everyone's books and can trigger the next round of
breaches. In parallel, a **funding chain** (cash provider → dealer finance desk → prime broker →
hedge fund) tightens as collateral erodes: a dealer's creditworthiness gates how much it can
borrow, and a dealer can suffer a **liquidity default** even while solvent. The result is a
cascade whose size depends on how the agents react. See [`paper/paper.md`](paper/paper.md) §2
for the full specification.

## Key configuration parameters

All live in [`bookstaber_abm/config.py`](bookstaber_abm/config.py) (`SimConfig`). The most
important:

| Parameter | Meaning |
|---|---|
| `n_assets`, `n_hedge_funds`, `n_bank_dealers`, `n_steps`, `seed` | structure & RNG |
| `shock_step`, `shock_asset`, `shock_size` | exogenous shock (e.g. `-0.20` = −20%) |
| `hf_lev_target / _buffer / _max` | hedge-fund leverage hierarchy |
| `hf_max_liq_frac` | max fraction of holdings force-sold per step (rate limit) |
| `beta`, `beta1` | price-impact: `beta_eff = beta + beta1·\|flow\|` (`beta1=0` ⇒ linear) |
| `noise_std` | per-step Gaussian price noise |
| `crowding` | portfolio overlap across HFs (only active when `hf_allocations_hetero` is empty) |
| `hf_allocations_hetero` | per-HF portfolio weights (overrides `crowding`) |
| `enable_derivatives_desk` | counterparty-risk channel (off in the shipped config) |
| `suppress_contagion` | counterfactual switch used by the contagion experiment |

`SimConfig.__post_init__` validates leverage hierarchies, allocation vectors, and shock
bounds, so an invalid config fails fast with a clear assertion.

## Reproducing the paper's results

From the project root (these are the heavy runs; the figures script accepts `N_DIST` /
`SWEEP_N` env vars to run with fewer seeds for a quick pass):

```bash
# Outcome distributions, N=1000 per shock
PYTHONPATH=. python experiments/sweep.py

# Global sensitivity: LHS design (~11 min) then the interaction regression
PYTHONPATH=. python experiments/sensitivity_lhs.py
PYTHONPATH=. python experiments/sensitivity_regression.py

# Primary-vs-contagion decomposition + per-step trajectories + event log
N_RUNS=1000 PYTHONPATH=. python experiments/contagion_decomposition.py

# Regenerate every figure + Table 1 used in paper/paper.md
PYTHONPATH=. python paper/make_figures.py
```

Runs are deterministic given a `seed`, so results are reproducible. Outputs land under
`outputs/` and `paper/figures/`.

## Testing

```bash
PYTHONPATH=. python -m pytest bookstaber_abm/tests/test_mechanics.py -q       # all
PYTHONPATH=. python -m pytest bookstaber_abm/tests/test_mechanics.py -k Leverage -v  # one area
```

Two pre-existing failures are known
(`TestLeverageBreach::test_forced_sale_targets_buffer` and
`TestBDFundingSqueeze::test_squeeze_triggers_fire_sale_targeting_shock_asset`): they assert
pure shock-asset selling in fire sales, whereas the implementation uses a proportional-weights
fallback. They are not load-bearing for the results.

## Notes & gotchas

- **Always run with `PYTHONPATH=.`** — the package is imported from the repo root, not installed.
- **`crowding` vs `hf_allocations_hetero`**: if `hf_allocations_hetero` is set (as in the shipped
  config), the `crowding` parameter is ignored. To sweep crowding, leave the hetero allocations
  empty (see the crowding experiment in `paper/make_figures.py`).
- **Matplotlib backend**: the batch/figure scripts force the non-interactive `Agg` backend and
  write PNGs to disk; no display is required.

## Attribution

Research code implementing an agent-based model of financial vulnerability, inspired by the
Bookstaber–Paddrik–Tivnan financial-system ABM. This is an independent re-build and extension
(rate-limited liquidation, diversified funding networks, convex price impact, contagion
decomposition, and global sensitivity analysis); see [`paper/paper.md`](paper/paper.md) for
scope and limitations.
