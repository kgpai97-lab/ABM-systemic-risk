# Context: Investigating Behaviour Under Liquidation-Dependent Price Impact

This document is a self-contained briefing for a follow-up session whose goal is to
**understand *why* the ABM behaves the way it does once price impact becomes convex in
liquidation size**. It is not a plan; it is the running-start context the next session
needs so it can dive straight into experiments and diagnostics rather than re-discovering
the change.

---

## 1. What was changed in this session

The price-impact module was generalised from strict-linear to a convex form.

**Old formula** ([asset_market.py:103](../bookstaber_abm/market/asset_market.py#L103) pre-change):
```
price_return_m = beta_m * effective_flow_m + noise_m
```

**New formula** ([asset_market.py:99-104](../bookstaber_abm/market/asset_market.py#L99)):
```
beta_eff_m   = beta_m + beta1 * |effective_flow_m|
price_return_m = beta_eff_m * effective_flow_m + noise_m
              = (beta_m + beta1 * |f_m|) * f_m + noise_m
```

Properties:
- **Convex in `|f|`**: marginal impact rises with size. With `beta=0`, doubling `|f|`
  quadruples `|price_return|` (since the formula becomes `beta1 · f · |f|`).
- **Sign-preserving**: positive flow → price up, negative → price down.
- **Per-asset, local**: each asset's `β_eff` is driven by *its own* `|effective_flow|`,
  not by system-wide aggregate. Cross-asset spillovers still come only through
  agents' rebalancing, not through the impact kernel itself.
- **`beta1 = 0` recovers the linear model byte-for-byte** (verified by
  `TestNonlinearPriceImpact::test_recovers_linear_when_beta1_zero`).

### Files changed
- [bookstaber_abm/config.py](../bookstaber_abm/config.py#L20): added `beta1: float = 0.0`.
- [bookstaber_abm/market/asset_market.py](../bookstaber_abm/market/asset_market.py):
  stored `self.beta1`, updated impact formula and docstrings.
- [bookstaber_abm/tests/test_mechanics.py](../bookstaber_abm/tests/test_mechanics.py#L153):
  added `TestNonlinearPriceImpact` (linear-recovery, convexity, sign-preservation).

### Why we chose this form (per user)
Out of {power-law, linear-in-flow, regime/threshold, smooth-logistic}, the user picked
**linear-in-flow** because:
- two interpretable parameters (`β0`, `β1`) rather than an exponent;
- clean degeneracy to the linear baseline at `β1 = 0`;
- units of `β1` are commensurate with `β0` (both are "price return per unit flow"
  modulo the `|f|` factor), so calibration intuition transfers.

Trigger variable is **per-asset net forced flow** (not system-total, not cumulative).

Integration is **replace-entirely** — no `cfg.beta_impact_fn` flag, no dual code path.

---

## 2. Units and calibration intuition

Whether `f` is shares or a fraction depends on `cfg.normalise_beta`
([asset_market.py:98-102](../bookstaber_abm/market/asset_market.py#L98)):

| `normalise_beta` | `effective_flow` is | `|f|` typical range | β₁ scale |
|---|---|---|---|
| `True` (default) | net flow / shares outstanding | `[0, ~0.1]` for big fire sales | `β1` should be on the order of `β0` or larger to bite |
| `False` | raw shares | `[0, 100s–1000s]` | `β1` should be *much smaller* than `β0` (`β0/|f_typical|`-ish) |

**Calibration mental model:** `β0` sets the impact of a small, "typical" trade; `β1 · |f|`
sets how much the impact coefficient *itself* grows once the trade gets large.
A reasonable starting point under `normalise_beta=True` is `β0 ≈ 0.02–0.05`, `β1 ≈
β0 … 2·β0` — meaning that when a single-step liquidation reaches the full free-float of
the asset, marginal impact has roughly doubled or tripled.

The `BASE` config in `batch_run.py` uses `beta=1.0` (default) with `normalise_beta=True`.
Inspect what `shares_outstanding` resolves to in practice
([engine.py:83-89](../bookstaber_abm/simulation/engine.py#L83)) before picking `β1`.

---

## 3. Conversational decisions worth carrying forward

- **No flag, no dual path.** If the next session is tempted to reintroduce a switch
  ("linear vs nonlinear mode"), check first — the user explicitly rejected this.
- **β₁ default is `0.0`.** Existing tests, batch runs, and `outputs/` artefacts are
  bit-identical until someone sets `β1` deliberately. Treat any drift in default-config
  outputs as a bug.
- **Per-asset local impact only.** The user did not pick "total system liquidation" or
  "cumulative." If contagion through impact-kernel coupling becomes a research question,
  that's a *new* change, not an extension of this one.
- **Negative `β1` is allowed.** No validation assertion. Concave impact is a legitimate
  experiment (e.g., resilience studies) — don't add a `β1 >= 0` check without asking.

---

## 4. What the next session should actually investigate

The point of this change is not just "more impact" — it's that **the marginal impact of a
liquidation depends on the size of that same liquidation**, which creates a positive
feedback loop with the fire-sale machinery. The questions worth answering are about
*where* that feedback shows up.

### 4.1 Mechanical predictions to verify
For each, check whether the simulation actually does this, and if not, why:

1. **Peak drawdown grows faster than linearly in `β1`** at fixed `β0` and shock size.
   A simple sweep `β1 ∈ {0, 0.5·β0, β0, 2·β0, 4·β0}` with everything else fixed should
   show a convex `peak_drop` vs `β1` curve. If it's roughly linear, the feedback is being
   absorbed somewhere (likely the `hf_max_liq_frac` / `bd_max_liq_frac` rate limits at
   [CLAUDE.md "Leverage hierarchy enforced"]).

2. **Default cascades cluster earlier in the timeline.** Compare time-to-first-default
   and time-to-last-default. Under nonlinear impact, the first large block of forced
   sales hits a steeper β_eff → bigger price drop → next round of breaches is triggered
   immediately, compressing the cascade.

3. **Multi-step deleveraging is *more* destructive than single-step** when impact is
   convex. Why: a single big sale at high `|f|` pays high `β_eff` *for that one trade*
   but is finished; spreading the same total over many small sales at low `|f|` pays
   low `β_eff` each step. Under linear impact total impact is order-independent; under
   convex impact, **breaking up the sale is strictly better for the seller**. Check
   whether `hf_max_liq_frac < 1.0` therefore *reduces* systemic damage under
   non-zero `β1` more than it does under `β1 = 0`. This is the most interesting and
   non-obvious prediction.

4. **Shock-asset concentration matters more.** Two-phase fire sale rule
   ([hedge_fund.py](../bookstaber_abm/agents/hedge_fund.py)) concentrates HF selling on
   `shock_asset` first. Under linear impact, this concentration is neutral relative to
   proportional selling (total impact across assets is the same). Under convex impact,
   concentrating flow into one asset is *strictly worse* than spreading it — `|f|` is
   higher in the shock asset, so `β_eff` is higher there. Expect a wider gap between
   `shock_asset` drawdown and other-asset drawdowns than under linear impact.

### 4.2 Diagnostics to add (or use)
- `flow_history` is already recorded in `AssetMarket`
  ([asset_market.py:108](../bookstaber_abm/market/asset_market.py#L108)). For each step
  log the *implied* `β_eff` per asset and the *contribution* of the convex term
  (`β1·|f|·f`) versus the linear term (`β0·f`). Plot the ratio over time — when does the
  convex term dominate?
- Compare `outputs/runs/` summary panels with `β1=0` vs `β1>0` at the same seed sweep.
  The 15-panel summary in `batch_run.py` is the right substrate.
- The phase-space sweep in [run.py:83-94](../bookstaber_abm/run.py#L83) currently varies
  `(shock_size, β)`. Consider a `(β, β1)` phase space at fixed shock, to isolate where
  in parameter space the system tips from "absorbs the shock" to "cascades."

### 4.3 Failure modes / things to watch
- **Price flooring at 0** ([asset_market.py:107](../bookstaber_abm/market/asset_market.py#L107)).
  Convex impact can produce price returns < −1 if `β1·|f|²` gets large; the floor masks
  this and can make the system look "stable" when it has actually broken the model. Add
  a warning when any computed `price_return < -1` before the floor.
- **`shares_outstanding` is set once** in the engine and not updated. If `normalise_beta`
  divides by a stale denominator that no longer reflects the post-liquidation float, β₁'s
  effective scale will drift. Verify this isn't material.
- **Noise interaction**: `β_eff` only includes the deterministic flow, not the noise
  shock. So `noise_std` enters additively, but its *consequences* for next-step impact
  go through whatever liquidation it triggers — be careful interpreting `noise_std`
  sweeps under non-zero `β1`.

---

## 5. Quick repro / smoke commands

```bash
# tests (existing 44 + 3 new in TestNonlinearPriceImpact)
python -m pytest bookstaber_abm/tests/test_mechanics.py

# baseline reproducibility (β1 defaults to 0 → should match pre-change outputs)
python -m bookstaber_abm.run

# nonlinear behavioural pair — same seed, same shock, β1 toggled
python -c "
from dataclasses import replace
from bookstaber_abm.config import SimConfig
from bookstaber_abm.simulation.engine import Simulation
base = SimConfig(beta=0.02, shock_size=-0.30, seed=7)
lin  = Simulation(replace(base, beta1=0.0)).run()
nl   = Simulation(replace(base, beta1=0.10)).run()
print('linear  defaults:', sum(1 for h in lin.history[-1]['hedge_funds'].values() if not h['active']))
print('nonlin  defaults:', sum(1 for h in nl.history[-1]['hedge_funds'].values() if not h['active']))
"
```

---

## 6. Pointers into the codebase (load order for a fresh session)

1. [bookstaber_abm/market/asset_market.py](../bookstaber_abm/market/asset_market.py) —
   the entire change lives here in ~5 lines. Read first.
2. [bookstaber_abm/simulation/engine.py:144-217](../bookstaber_abm/simulation/engine.py#L144) —
   the two `update_prices()` call sites (default liquidation in step 2, main forced flow
   in step 9). Understand which one will exhibit the convex feedback most strongly
   (answer: step 9 in the first few periods after the shock).
3. [bookstaber_abm/agents/hedge_fund.py](../bookstaber_abm/agents/hedge_fund.py) — the
   two-phase fire-sale logic (concentrate on `shock_asset` first, then proportional).
   This is what couples to the convex impact in a non-trivial way.
4. [CLAUDE.md](../CLAUDE.md) "Periodic Event Loop" — 12-step ordering. Crucial for
   reasoning about *when* in a period the convex feedback bites and what state is fed
   into it.
