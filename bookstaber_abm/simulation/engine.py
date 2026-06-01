"""
simulation/engine.py
--------------------
The simulation engine — orchestrates the canonical 11-step within-period
sequence described in the implementation guide.

Step order (per period t):
  1.  Apply exogenous shock (if t == shock_step)
  2.  Compute net forced order flow from PREVIOUS period's pending liquidations
  3.  Update asset prices P_m(t) via market impact
  4.  Mark all portfolios to market (compute current assets, leverage)
  5.  Compute orders for ALL agents simultaneously (pure, no mutation)
  6.  Aggregate net forced order flow Q_Dpi for next price update
  7.  Update collateral values CA_k for bank/dealers
  8.  Cash providers compute loans for each bank/dealer
  9.  Distribute funding: CP → Finance desk → Prime broker → HFs
 10.  Apply orders & update all agent state
 11.  Check for defaults; queue defaulted agents for liquidation
 12.  Record state snapshot

Design principle: steps 5–6 use the SAME price snapshot for all agents.
No agent can observe another agent's orders before submitting its own.
"""

from __future__ import annotations
import numpy as np
from typing import Any

from bookstaber_abm.config import SimConfig
from bookstaber_abm.agents.hedge_fund import HedgeFund, HFOrders
from bookstaber_abm.agents.bank_dealer import BankDealer, BDOrders
from bookstaber_abm.agents.cash_provider import CashProvider
from bookstaber_abm.market.asset_market import AssetMarket


class Simulation:
    """
    Main simulation class.

    Usage
    -----
        cfg = SimConfig(shock_size=-0.15, n_steps=200)
        sim = Simulation(cfg)
        results = sim.run()
    """

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.t: int = 0

        # --- Build market ---
        self.market = AssetMarket(cfg, self.rng)

        # --- Build agents ---
        self.cash_providers: list[CashProvider] = [
            CashProvider(f"CP_{i}", cfg)
            for i in range(cfg.n_cash_providers)
        ]

        self.bank_dealers: list[BankDealer] = [
            BankDealer(f"BD_{k}", cfg, self.rng, bd_index=k)
            for k in range(cfg.n_bank_dealers)
        ]

        # Generate allocations: use hetero if specified, else crowding model
        if cfg.hf_allocations_hetero:
            allocs = [cfg.get_hf_allocation(n) for n in range(cfg.n_hedge_funds)]
        else:
            allocs = cfg.make_crowded_allocations(self.rng)

        self.hedge_funds: list[HedgeFund] = [
            HedgeFund(f"HF_{n}", cfg, self.rng, allocation=allocs[n], hf_index=n)
            for n in range(cfg.n_hedge_funds)
        ]

        # --- Wire up the network ---
        self._assign_prime_brokerage()
        self._register_cp_relationships()
        self._register_derivatives_network()

        # Compute shares outstanding per asset from initial holdings (used for beta normalisation)
        if cfg.normalise_beta:
            shares = np.zeros(cfg.n_assets)
            for hf in self.hedge_funds:
                shares += hf.holdings
            for bd in self.bank_dealers:
                shares += bd.holdings
            self.market.shares_outstanding = np.maximum(shares, 1.0)

        # Pending liquidations: HFs that defaulted this step liquidate next step
        self._pending_liquidations: list[HedgeFund] = []

        # History log
        self.history: list[dict] = []

    # ------------------------------------------------------------------ #
    # Network setup                                                        #
    # ------------------------------------------------------------------ #

    def _assign_prime_brokerage(self) -> None:
        """Assign each HF to one or more BDs based on cfg.get_hf_funding_weights.

        Default: round-robin (HF_i funded entirely by BD[i % n_bd]). With
        hf_bd_funding_weights set, each HF gets a share-of-funding from every
        BD with non-zero weight; collateral splits the same way.

        Sets:
            self._hf_to_bd[hf_id] = first BD with non-zero weight (back-compat
                                    for any code that reads this).
            self._hf_bd_weights[hf_id] = {bd_id: weight} dict, weights sum to 1.
        """
        self._hf_to_bd: dict[str, BankDealer] = {}
        self._hf_bd_weights: dict[str, dict[str, float]] = {}
        for i, hf in enumerate(self.hedge_funds):
            weights = self.cfg.get_hf_funding_weights(i)
            self._hf_bd_weights[hf.id] = {}
            primary = None
            for k, bd in enumerate(self.bank_dealers):
                w = float(weights[k])
                if w > 0:
                    bd.register_hedge_fund(hf.id, weight=w)
                    self._hf_bd_weights[hf.id][bd.id] = w
                    if primary is None:
                        primary = bd
            self._hf_to_bd[hf.id] = primary

    def _register_cp_relationships(self) -> None:
        """Each CP lends to every BD."""
        for cp in self.cash_providers:
            for bd in self.bank_dealers:
                cp.register_counterparty(bd.id)

    def _register_derivatives_network(self) -> None:
        """Each BD registers bilateral derivatives with every other BD."""
        if not self.cfg.enable_derivatives_desk:
            return
        for bd_a in self.bank_dealers:
            for bd_b in self.bank_dealers:
                if bd_a.id != bd_b.id:
                    bd_a.register_derivatives_counterparty(bd_b.id)

    def _reconcile_derivatives(self) -> None:
        # Each DerivativesDesk walks its own view of every bilateral contract,
        # so the two sides drift independently and can settle with the same sign
        # — violating BD_i.exposures[BD_j] = -BD_j.exposures[BD_i] and zeroing out
        # the credit loss on default. Force the antisymmetric mean across every
        # surviving pair after each MTM walk.
        bds = self.bank_dealers
        for i, bd_i in enumerate(bds):
            if not bd_i.active or bd_i.derivatives is None:
                continue
            for bd_j in bds[i + 1:]:
                if not bd_j.active or bd_j.derivatives is None:
                    continue
                exp_i = bd_i.derivatives.exposures.get(bd_j.id)
                exp_j = bd_j.derivatives.exposures.get(bd_i.id)
                if exp_i is None or exp_j is None:
                    continue
                avg = (exp_i - exp_j) / 2.0
                bd_i.derivatives.exposures[bd_j.id] = avg
                bd_j.derivatives.exposures[bd_i.id] = -avg

    # ------------------------------------------------------------------ #
    # Main run loop                                                        #
    # ------------------------------------------------------------------ #

    def run(self) -> list[dict]:
        """Run the full simulation.  Returns list of per-step state dicts."""
        for t in range(self.cfg.n_steps):
            self.t = t
            self._step()

        return self.history

    def _step(self) -> None:
        """Execute one simulation period."""
        prices_prev = self.market.prices.copy()

        # ---- Step 1: Exogenous shock ----------------------------------------
        if self.t == self.cfg.shock_step:
            self.market.apply_shock(self.cfg.shock_asset, self.cfg.shock_size)

        # ---- Step 2-3: Process pending liquidations + update prices ----------
        # Liquidations from last step's defaults hit the market first
        pending_flow = np.zeros(self.cfg.n_assets)
        still_liquidating: list = []
        for hf in self._pending_liquidations:
            liq_qty = hf.apply_default_liquidation()
            pending_flow += liq_qty
            # Re-queue if any holdings remain — rate-limit handles multi-step drain.
            if np.any(hf.holdings > 0):
                still_liquidating.append(hf)
        self._pending_liquidations = still_liquidating

        # Contagion counterfactual: default-liquidation flow does not move prices.
        # (Holdings still drain; only the price-impact channel is severed.)
        if self.cfg.suppress_contagion:
            pending_flow = np.zeros(self.cfg.n_assets)

        # Update prices using any pending liquidation flow
        # (normal step: no forced flow yet, just noise + pending liquidations)
        prices_current = self.market.update_prices(pending_flow)

        # ---- Steps 4-6: Funding chain (BEFORE orders) -----------------------
        # Collateral and loans are assessed at prices_current (before this
        # step's forced sales).  This lets HFs respond to funding squeezes in
        # the same step — the cascade unfolds over multiple periods as each
        # step's reduced collateral tightens the next step's funding.

        # 4: Collateral values per BD at current prices
        hf_collaterals = self._compute_hf_collaterals(prices_current)
        bd_collaterals = self._compute_bd_collaterals(prices_current, hf_collaterals)

        # 5: Cash providers compute loans
        bd_funding_received: dict[str, float] = {bd.id: 0.0 for bd in self.bank_dealers}
        for cp in self.cash_providers:
            for bd in self.bank_dealers:
                if bd.active:
                    loan = cp.compute_loan(bd.id, bd_collaterals[bd.id])
                    bd_funding_received[bd.id] += loan

        # 6: BD distributes to HFs — funding_available passed to compute_orders
        hf_funding_received: dict[str, float] = {hf.id: 0.0 for hf in self.hedge_funds}
        hf_map_pre = {hf.id: hf for hf in self.hedge_funds}
        for bd in self.bank_dealers:
            if not bd.active:
                continue
            hf_needs = {
                hf_id: hf_map_pre[hf_id].funding
                for hf_id in bd._hf_ids
                if hf_id in hf_map_pre
            }
            allocated = bd.distribute_funding(bd_funding_received[bd.id], hf_needs, prices_current)
            for hf_id, amount in allocated.items():
                # Accumulate across BDs — with diversified funding the same HF
                # appears on multiple BDs and the total is the sum.
                hf_funding_received[hf_id] += amount

        # Contagion counterfactual: freeze the funding chain so no funding-squeeze
        # cascade fires. Every active agent receives exactly its stated need, so
        # funding is never the binding constraint and only the primary leverage
        # hit (from the exogenous shock) can trigger a forced sale.
        if self.cfg.suppress_contagion:
            for bd in self.bank_dealers:
                if bd.active:
                    bd_funding_received[bd.id] = bd.funding
            for hf in self.hedge_funds:
                if hf.active:
                    hf_funding_received[hf.id] = hf.funding

        # ---- Steps 7-8: Compute orders (simultaneous, pure) -----------------
        # All agents see the same prices_current snapshot and their own
        # funding_available.  Funding-squeeze sales are treated as forced
        # (price-impacting) the same as leverage-breach forced sales.

        hf_orders: dict[str, HFOrders] = {
            hf.id: hf.compute_orders(
                prices_current,
                funding_available=hf_funding_received.get(hf.id),
            )
            for hf in self.hedge_funds
            if hf.active
        }

        bd_orders: dict[str, BDOrders] = {
            bd.id: bd.compute_orders(
                prices_current,
                funding_available=bd_funding_received.get(bd.id),
            )
            for bd in self.bank_dealers
            if bd.active
        }

        # ---- Step 9: Aggregate forced order flow ----------------------------
        net_forced = np.zeros(self.cfg.n_assets)
        for orders in hf_orders.values():
            net_forced += orders.forced
        for orders in bd_orders.values():
            net_forced += orders.forced

        # Contagion counterfactual: forced sales do not move prices. Only the
        # exogenous shock + noise drive the price path; the fire-sale spillover
        # channel is severed. Agents still apply their orders (holdings change),
        # but the secondary price cascade is suppressed.
        impact_flow = np.zeros(self.cfg.n_assets) if self.cfg.suppress_contagion else net_forced

        # Apply price impact from this period's forced sales
        prices_post_impact = self.market.update_prices(impact_flow)

        # ---- Step 10: Apply orders, update state ----------------------------
        hf_map = {hf.id: hf for hf in self.hedge_funds}
        for hf_id, orders in hf_orders.items():
            hf = hf_map[hf_id]
            hf.apply_orders(
                orders,
                new_prices=prices_post_impact,
                old_prices=prices_prev,
                funding_received=hf_funding_received.get(hf_id, 0.0),
            )

        bd_map = {bd.id: bd for bd in self.bank_dealers}
        for bd_id, orders in bd_orders.items():
            bd = bd_map[bd_id]
            bd.apply_orders(
                orders,
                new_prices=prices_post_impact,
                old_prices=prices_prev,
                funding_received=bd_funding_received.get(bd_id, 0.0),
            )

        # ---- Step 11: Default detection — HFs ---------------------------------
        for hf in self.hedge_funds:
            if hf.active and hf.capital <= 0:
                hf.active = False
                hf.defaulted_at = self.t
                self._pending_liquidations.append(hf)

        # ---- Step 11b: BD defaults + derivatives desk -----------------------
        # Collect newly defaulted BDs so derivatives desks can crystallise losses
        newly_defaulted_bd_ids: set[str] = set()
        for bd in self.bank_dealers:
            if bd.active and bd.capital <= 0:
                bd.active = False
                bd.defaulted_at = self.t
                newly_defaulted_bd_ids.add(bd.id)

        # Step derivatives desks — deduct counterparty losses from capital
        if self.cfg.enable_derivatives_desk:
            for bd in self.bank_dealers:
                if bd.active:
                    bd.step_derivatives(newly_defaulted_bd_ids)
            # Enforce zero-sum bilateral consistency across surviving BDs.
            # Without this each BD walks its view of every contract independently
            # and they routinely disagree on sign — making notional a dead lever.
            self._reconcile_derivatives()
            # Re-check BD defaults after derivatives losses
            for bd in self.bank_dealers:
                if bd.active and bd.capital <= 0:
                    bd.active = False
                    bd.defaulted_at = self.t

        # ---- Step 11c: Treasury desk update ---------------------------------
        # Update LiqRatio, CW, and haircuts; check liquidity defaults
        liquidity_defaulted_ids: set[str] = set()
        for bd in self.bank_dealers:
            if not bd.active:
                continue
            liq_debit_this_step = (
                bd_orders[bd.id].liq_debit_needed if bd.id in bd_orders else 0.0
            )
            ftd = bd_funding_received.get(bd.id, 0.0)
            liquidity_defaulted = bd.step_treasury(ftd, liq_debit_this_step)
            if liquidity_defaulted and bd.active:
                bd.active = False
                bd.defaulted_at = self.t
                liquidity_defaulted_ids.add(bd.id)

            # Propagate BD's LiqRatio to haircut updates at each CP (Eq. 22),
            # and mirror BD's CW (Eq. 21) so the CP can gate L_Max next step (Eq. 6).
            for cp in self.cash_providers:
                cp.update_haircut_from_creditworthiness(
                    bd.id, bd.liq_ratio, bd.liq_ratio_min
                )
                cp.update_creditworthiness(bd.id, bd.cw)

        # ---- Step 11d: Derivatives crystallise losses on liquidity defaults --
        # Liquidity defaults happen after step_derivatives in 11b, so derivative
        # counterparties never see them. Fire derivatives a second time against
        # the liquidity-defaulted set, then re-check solvency.
        if self.cfg.enable_derivatives_desk and liquidity_defaulted_ids:
            for bd in self.bank_dealers:
                if bd.active:
                    bd.step_derivatives(liquidity_defaulted_ids)
            self._reconcile_derivatives()
            for bd in self.bank_dealers:
                if bd.active and bd.capital <= 0:
                    bd.active = False
                    bd.defaulted_at = self.t

        # ---- Step 12: Record snapshot ---------------------------------------
        self._record(prices_post_impact, net_forced, hf_orders, bd_orders)

    # ------------------------------------------------------------------ #
    # Collateral helpers                                                   #
    # ------------------------------------------------------------------ #

    def _compute_hf_collaterals(self, prices: np.ndarray) -> dict[str, float]:
        """
        CA_n(t) = F_n / (1 − HC)  where F_n = max(0, A_n − Cap_n).

        Haircut-adjusted so CP loan = CA_n * (1-HC) = F_n, exactly covering
        the HF's funding need.  Without this adjustment a structural 10% gap
        appears at every step, causing spurious pre-shock funding-squeeze sales.
        """
        result = {}
        for hf in self.hedge_funds:
            if not hf.active:
                continue
            funding_need = max(0.0, hf.total_assets(prices) - hf.capital)
            weights = self._hf_bd_weights.get(hf.id, {})
            if weights and self.cash_providers:
                cp = self.cash_providers[0]
                # Each BD posts collateral haircut-adjusted by its own CP haircut.
                # HF's total collateral = Σ_k w_k × need_k / (1 - hc_k).
                collateral = 0.0
                for bd_id, w in weights.items():
                    hc = cp.haircut(bd_id)
                    collateral += w * funding_need / max(1.0 - hc, 1e-9)
            else:
                collateral = funding_need
            result[hf.id] = collateral
        return result

    def _compute_bd_collaterals(
        self,
        prices: np.ndarray,
        hf_collaterals: dict[str, float],
    ) -> dict[str, float]:
        """CA^FD_k — collateral each BD can post to cash providers."""
        result = {}
        for bd in self.bank_dealers:
            if bd.active:
                # Use average haircut from first CP as proxy
                hc = self.cash_providers[0].haircut(bd.id) if self.cash_providers else 0.10
                result[bd.id] = bd.collateral_available(prices, hf_collaterals, hc)
            else:
                result[bd.id] = 0.0
        return result

    # ------------------------------------------------------------------ #
    # Logging                                                              #
    # ------------------------------------------------------------------ #

    def _record(
        self,
        prices: np.ndarray,
        net_forced: np.ndarray,
        hf_orders: dict[str, HFOrders],
        bd_orders: dict[str, BDOrders],
    ) -> None:
        n_fire_sales = sum(1 for o in hf_orders.values() if o.in_fire_sale)
        n_fire_sales += sum(1 for o in bd_orders.values() if o.in_fire_sale)
        n_active_hf = sum(1 for hf in self.hedge_funds if hf.active)
        n_defaults = sum(1 for hf in self.hedge_funds if not hf.active)

        # Portfolio overlap: average pairwise cosine similarity between HF allocations
        active_hfs = [hf for hf in self.hedge_funds if hf.active]
        overlap = 0.0
        if len(active_hfs) > 1:
            pairs = 0
            for i in range(len(active_hfs)):
                for j in range(i+1, len(active_hfs)):
                    a = active_hfs[i].allocation
                    b = active_hfs[j].allocation
                    denom = (np.linalg.norm(a) * np.linalg.norm(b))
                    if denom > 0:
                        overlap += float(np.dot(a, b) / denom)
                        pairs += 1
            if pairs > 0:
                overlap /= pairs

        # Derivatives exposure
        total_deriv_exposure = 0.0
        total_deriv_losses = 0.0
        if self.cfg.enable_derivatives_desk:
            for bd in self.bank_dealers:
                if bd.derivatives is not None:
                    total_deriv_exposure += bd.derivatives.total_positive_exposure()
                    total_deriv_losses += bd.derivatives.loss_this_step

        record = {
            "t": self.t,
            "prices": prices.tolist(),
            "net_forced_flow": net_forced.tolist(),
            "n_fire_sales": n_fire_sales,
            "n_active_hf": n_active_hf,
            "n_defaults": n_defaults,
            "n_bd_defaults": sum(1 for bd in self.bank_dealers if not bd.active),
            "hf_active": [hf.active for hf in self.hedge_funds],
            "bd_active": [bd.active for bd in self.bank_dealers],
            "hf_defaulted_at": [getattr(hf, "defaulted_at", None) for hf in self.hedge_funds],
            "bd_defaulted_at": [getattr(bd, "defaulted_at", None) for bd in self.bank_dealers],
            "hf_capitals": [round(hf.capital, 4) for hf in self.hedge_funds],
            "hf_leverages": [round(hf.current_leverage(prices), 4) for hf in self.hedge_funds],
            "hf_fundings": [round(hf.funding, 4) for hf in self.hedge_funds],
            "hf_holdings": [hf.holdings.tolist() for hf in self.hedge_funds],
            "hf_in_fire_sale": [
                hf_orders[hf.id].in_fire_sale if hf.id in hf_orders else False
                for hf in self.hedge_funds
            ],
            "hf_forced_flows": [
                hf_orders[hf.id].forced.tolist() if hf.id in hf_orders
                else [0.0] * self.cfg.n_assets
                for hf in self.hedge_funds
            ],
            "bd_capitals": [round(bd.capital, 4) for bd in self.bank_dealers],
            "bd_holdings": [bd.holdings.tolist() for bd in self.bank_dealers],
            "bd_leverages": [round(bd.current_leverage(prices), 4) for bd in self.bank_dealers],
            "bd_fundings_from_cp": [round(bd.funding_from_cp, 4) for bd in self.bank_dealers],
            "bd_in_fire_sale": [
                bd_orders[bd.id].in_fire_sale if bd.id in bd_orders else False
                for bd in self.bank_dealers
            ],
            "bd_forced_flows": [
                bd_orders[bd.id].forced.tolist() if bd.id in bd_orders
                else [0.0] * self.cfg.n_assets
                for bd in self.bank_dealers
            ],
            "bd_liq_ratios": [round(bd.liq_ratio, 4) for bd in self.bank_dealers],
            "bd_cw": [round(bd.cw, 4) for bd in self.bank_dealers],
            "bd_liq_debits": [round(bd.liq_debit, 4) for bd in self.bank_dealers],
            "bd_liq_reserves": [round(bd.liquidity_reserve, 4) for bd in self.bank_dealers],
            "haircuts": [
                round(self.cash_providers[0].haircut(bd.id), 4) if self.cash_providers else 0.10
                for bd in self.bank_dealers
            ],
            "cp_loans": [
                {bd.id: round(cp.last_loans.get(bd.id, 0.0), 4) for bd in self.bank_dealers}
                for cp in self.cash_providers
            ],
            "total_forced_flow": float(np.abs(net_forced).sum()),
            "portfolio_overlap": round(overlap, 4),
            "deriv_exposure": round(total_deriv_exposure, 4),
            "deriv_losses": round(total_deriv_losses, 4),
            "shock_active": self.t == self.cfg.shock_step,
        }
        self.history.append(record)

    # ------------------------------------------------------------------ #
    # Conservation check (call as assertion in tests)                     #
    # ------------------------------------------------------------------ #

    def check_conservation(self, prices: np.ndarray) -> dict:
        """
        Sanity checks:
        - Total system capital > 0
        - No active agent has negative holdings

        Returns dict of check results.
        """
        total_cap = sum(hf.capital for hf in self.hedge_funds if hf.active)
        total_cap += sum(bd.capital for bd in self.bank_dealers if bd.active)
        negative_holdings = any(
            (hf.holdings < -1e-9).any()
            for hf in self.hedge_funds
            if hf.active
        )
        return {
            "total_capital": total_cap,
            "negative_holdings": negative_holdings,
            "all_prices_non_negative": bool((prices >= 0).all()),
        }
