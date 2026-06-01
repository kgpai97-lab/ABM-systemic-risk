"""
agents/bank_dealer.py
---------------------
Implements the BankDealer agent from Bookstaber et al. §3.3.

The bank/dealer is a composite agent containing three sub-desks:
  - Finance desk  (§3.3.2): raises funding from cash providers
  - Prime broker  (§3.3.1): intermediates funding to hedge funds
  - Trading desk  (§3.3.3): holds its own portfolio, similar to HedgeFund

The key intermediation chain:
    CashProvider → [Finance Desk] → [Prime Broker] → HedgeFund

Collateral flows in the opposite direction:
    HedgeFund.holdings → PrimeBroker.collateral → FinanceDesk.collateral → CashProvider
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bookstaber_abm.config import SimConfig
    from bookstaber_abm.agents.hedge_fund import HFOrders

from bookstaber_abm.agents.derivatives_desk import DerivativesDesk


@dataclass
class BDOrders:
    """Order bundle for the bank/dealer's trading desk."""
    forced: np.ndarray
    normal: np.ndarray
    in_fire_sale: bool
    liq_debit_needed: float = 0.0  # dollar shortfall when QMax clipped forced sales

    @property
    def total(self) -> np.ndarray:
        return self.forced + self.normal


class BankDealer:
    """
    Bank/dealer with Finance, PrimeBroker, and Trading desks.

    Simplifying assumption from the paper (§3.3.1):
        The prime broker passes funding through with no additional haircut.
        CA^PB_k = sum of all serviced hedge funds' collateral.
    """

    def __init__(
        self,
        agent_id: str,
        cfg: SimConfig,
        rng: np.random.Generator,
        bd_index: int = 0,
    ):
        self.id = agent_id
        self.cfg = cfg
        self.rng = rng
        self.M = cfg.n_assets

        # Leverage parameters (trading desk)
        self.lev_target: float = cfg.bd_lev_target
        self.lev_buffer: float = cfg.bd_lev_buffer
        self.lev_max: float    = cfg.bd_lev_max
        self.liq_rate: float   = cfg.bd_liq_rate

        # Trading desk portfolio
        self.capital: float = cfg.bd_initial_capital
        initial_prices = np.full(self.M, cfg.initial_price)
        # Per-BD allocation: hetero override if configured, else equal-weight.
        self.allocation: np.ndarray = cfg.get_bd_allocation(bd_index)
        target_assets = self.capital * self.lev_target
        self.holdings: np.ndarray = (self.allocation * target_assets / initial_prices).astype(float)
        self.funding: float = float(np.dot(initial_prices, self.holdings)) - self.capital

        # Liquidity reserve (buffer before forced liquidation)
        self.liquidity_reserve: float = self.capital * self.liq_rate

        # Treasury desk state (§3.3.5)
        self.liq_debit: float = 0.0                          # drawn from reserve this step
        self.liq_ratio: float = cfg.bd_liq_ratio_target      # LiqRatio_k (Eq. 19)
        self.liq_ratio_target: float = cfg.bd_liq_ratio_target  # LiqRatioTarget (Eq. 20)
        self.liq_ratio_min: float = cfg.bd_liq_ratio_min     # LiqRatioMin_k
        self.cw: float = 100.0                               # CW_k creditworthiness (Eq. 21)
        self.liquidity_defaulted: bool = False               # True if liq_debit >= reserve

        # Serviced hedge funds (registered at sim init)
        self._hf_ids: list[str] = []
        # Per-HF funding share routed through THIS BD (sums across BDs to 1
        # per HF). Default value 1.0 used when register_hedge_fund() is called
        # without a weight — preserves 1-to-1 round-robin behaviour.
        self._hf_weights: dict[str, float] = {}

        # Funding intermediation state (updated each step)
        self.funding_from_cp: float = 0.0     # F^FD received from cash providers
        self.funding_to_hfs: dict[str, float] = {}  # F^PB distributed per HF

        # Order history
        self.last_orders: BDOrders = BDOrders(
            forced=np.zeros(self.M),
            normal=np.zeros(self.M),
            in_fire_sale=False,
        )
        self._prev_orders: np.ndarray = np.zeros(self.M)
        self._prev_price_delta: np.ndarray = np.zeros(self.M)

        # Derivatives desk (optional — enabled by cfg.enable_derivatives_desk)
        self.derivatives: DerivativesDesk | None = (
            DerivativesDesk(agent_id, cfg, rng)
            if cfg.enable_derivatives_desk else None
        )
        self.defaulted_at: int | None = None
        self.active: bool = True

    # ------------------------------------------------------------------ #
    # Network registration                                                 #
    # ------------------------------------------------------------------ #

    def register_hedge_fund(self, hf_id: str, weight: float = 1.0) -> None:
        """Register a hedge fund as a prime brokerage client.

        weight: fraction of this HF's funding need / collateral routed through
        THIS BD. Default 1.0 preserves 1-to-1 round-robin assignment.
        """
        self._hf_ids.append(hf_id)
        self._hf_weights[hf_id] = float(weight)
        self.funding_to_hfs[hf_id] = 0.0

    def register_derivatives_counterparty(self, bd_id: str) -> None:
        """Wire up a bilateral derivatives position with another BD."""
        if self.derivatives is not None:
            self.derivatives.register_counterparty(bd_id)

    # ------------------------------------------------------------------ #
    # Collateral aggregation (Finance desk)                               #
    # ------------------------------------------------------------------ #

    def collateral_available(
        self,
        prices: np.ndarray,
        hf_collaterals: dict[str, float],
        haircut: float,
    ) -> float:
        """
        CA^FD_k(t) — total collateral the finance desk can post to cash providers.

        = (trading desk assets − own capital) / (1 − haircut)
          + sum of serviced HF collateral (passed through from prime broker)

        Parameters
        ----------
        prices          : current asset prices
        hf_collaterals  : {hf_id: CA_n(t)} for each serviced HF
        haircut         : HC_{c,k} set by the cash provider
        """
        td_assets = float(np.dot(prices, self.holdings))
        td_funding_need = max(0.0, td_assets - self.capital)
        if haircut < 1.0:
            td_collateral = td_funding_need / (1.0 - haircut)
        else:
            td_collateral = 0.0

        # Only the fraction of each HF's collateral routed through this BD
        # (set by _hf_weights, default 1.0) is posted by this BD's finance desk.
        pb_collateral = sum(
            self._hf_weights.get(hf_id, 1.0) * hf_collaterals.get(hf_id, 0.0)
            for hf_id in self._hf_ids
        )
        return td_collateral + pb_collateral

    # ------------------------------------------------------------------ #
    # Funding distribution (Prime broker)                                 #
    # ------------------------------------------------------------------ #

    def distribute_funding(
        self,
        total_received: float,
        hf_funding_needs: dict[str, float],
        prices: np.ndarray,
    ) -> dict[str, float]:
        """
        Distribute funding from cash providers down to hedge funds.

        If total received < total needed, fund proportionally (haircut all HFs equally).
        Returns {hf_id: funding_allocated}.
        """
        self.funding_from_cp = total_received

        # Only this BD's share of each HF's need is serviced from this BD's
        # remaining funding. With round-robin (weight=1.0) this reduces to the
        # original behaviour; with diversified weights, the BD provides only
        # its weighted slice and the rest comes from other BDs.
        weighted_needs = {
            hf_id: self._hf_weights.get(hf_id, 1.0) * hf_funding_needs.get(hf_id, 0.0)
            for hf_id in self._hf_ids
        }
        total_hf_need = sum(weighted_needs.values())
        td_need = max(0.0, float(np.dot(prices, self.holdings)) - self.capital)

        # Allocate trading desk funding first, remainder goes to HFs
        td_allocated = min(td_need, total_received)
        remaining = max(0.0, total_received - td_allocated)

        allocated: dict[str, float] = {}
        if total_hf_need > 0:
            for hf_id in self._hf_ids:
                share = (weighted_needs[hf_id] / total_hf_need) * remaining
                allocated[hf_id] = share
        else:
            for hf_id in self._hf_ids:
                allocated[hf_id] = 0.0

        self.funding_to_hfs = allocated
        return allocated

    # ------------------------------------------------------------------ #
    # Trading desk order logic (mirrors HedgeFund)                        #
    # ------------------------------------------------------------------ #

    def compute_orders(
        self,
        prices: np.ndarray,
        funding_available: float | None = None,
    ) -> BDOrders:
        """
        Pure — reads state, returns orders.  Does NOT mutate self.

        Parameters
        ----------
        prices            : current asset prices (before this step's impact).
        funding_available : dollar funding the cash providers committed to this BD
                            this step (F^FD). None → no funding constraint applied
                            (backward-compat). If less than the trading desk's
                            funding need by more than `hf_funding_squeeze_threshold`,
                            an EDS_k funding-squeeze forced sale is triggered
                            (price-impacting, mirroring HF logic at hedge_fund.py:173).
        """
        if not self.active:
            return BDOrders(
                forced=np.zeros(self.M),
                normal=np.zeros(self.M),
                in_fire_sale=False,
            )

        assets = float(np.dot(prices, self.holdings))
        lev = assets / self.capital if self.capital > 0 else float("inf")

        # ---- Determine the binding target -----------------------------------
        # Two constraints may force sales; take the more restrictive one.
        target_assets = float("inf")
        is_forced = False

        # (1) Leverage breach → must deleverage to Lev^Buffer
        if lev >= self.lev_max:
            target_assets = min(target_assets, self.capital * self.lev_buffer)
            is_forced = True

        # (2) Funding squeeze (EDS_k) → must reduce portfolio to what CP will fund.
        # Trading-desk funding need = max(0, assets - capital). Distribute_funding
        # gives TD priority, so funding_available < td_need is the genuine squeeze.
        if funding_available is not None and self.capital > 0:
            max_supportable = self.capital + max(0.0, funding_available)
            tol = self.cfg.hf_funding_squeeze_threshold * assets
            if max_supportable < assets - tol:
                target_assets = min(target_assets, max_supportable)
                is_forced = True

        if is_forced:
            target_assets = min(target_assets, assets)   # never force a buy
            delta_assets = target_assets - assets         # negative → sell
            shock = self.cfg.shock_asset
            alpha = self.cfg.fire_sale_shock_concentration
            total = self.holdings.sum()
            prop_weights = self.holdings / total if total > 0 else self._current_weights(prices)
            if self.holdings[shock] > 0:
                shock_weights = np.zeros(self.M)
                shock_weights[shock] = 1.0
                weights = alpha * shock_weights + (1.0 - alpha) * prop_weights
            else:
                weights = prop_weights
            forced_dollar = delta_assets * weights
            intended_qty = np.where(prices > 0, forced_dollar / prices, 0.0)
            intended_qty = np.maximum(intended_qty, -self.holdings)

            # Respect max liquidation threshold Q^Max_k
            max_qty = self.cfg.bd_max_liq_frac * np.maximum(self.holdings, 0.0)
            forced_qty = np.maximum(intended_qty, -max_qty)

            # Shortfall from QMax clipping → drawn from liquidity reserve (§3.3.4)
            unsold_qty = forced_qty - intended_qty   # ≥ 0 (unsold shares)
            liq_debit_needed = float(np.dot(np.maximum(unsold_qty, 0.0), prices))

            return BDOrders(
                forced=forced_qty,
                normal=np.zeros(self.M),
                in_fire_sale=True,
                liq_debit_needed=liq_debit_needed,
            )

        else:
            # Normal rebalancing toward lev_target, capped at 10% of capital
            # per asset per step (same rule as HF) to prevent rapid collateral
            # depletion that would falsely trigger the CP stress-haircut.
            target_assets = self.capital * self.lev_target
            delta_assets = target_assets - assets
            weights = np.full(self.M, 1.0 / self.M)
            normal_qty = np.where(prices > 0, (delta_assets * weights) / prices, 0.0)
            max_normal = self.capital * 1.0
            normal_qty = np.clip(
                normal_qty,
                -max_normal / np.maximum(prices, 1e-9),
                max_normal / np.maximum(prices, 1e-9),
            )

            return BDOrders(forced=np.zeros(self.M), normal=normal_qty, in_fire_sale=False)

    def apply_orders(
        self,
        orders: BDOrders,
        new_prices: np.ndarray,
        old_prices: np.ndarray,
        funding_received: float,
    ) -> None:
        """Mutate state after market clears."""
        if not self.active:
            return

        self.holdings = np.maximum(self.holdings + orders.total, 0.0)

        price_delta = old_prices - self._prev_price_delta
        slip = float(np.dot(self._prev_orders, price_delta))

        # Incremental P&L update — same logic as HedgeFund
        holdings_pre_trade = self.holdings - orders.total
        price_pnl = float(np.dot(holdings_pre_trade, new_prices - old_prices))
        self.capital = self.capital + price_pnl - slip
        self.funding = float(np.dot(new_prices, self.holdings)) - self.capital

        self._prev_orders = orders.forced.copy()
        self._prev_price_delta = old_prices.copy()
        self.last_orders = orders

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def total_assets(self, prices: np.ndarray) -> float:
        return float(np.dot(prices, self.holdings))

    def current_leverage(self, prices: np.ndarray) -> float:
        return self.total_assets(prices) / self.capital if self.capital > 0 else float("inf")

    def _current_weights(self, prices: np.ndarray) -> np.ndarray:
        values = prices * self.holdings
        total = values.sum()
        return values / total if total > 0 else np.full(self.M, 1.0 / self.M)

    def step_treasury(self, ftd: float, liq_debit_this_step: float) -> bool:
        """
        Update treasury desk state for one period (§3.3.4–3.3.5).

        Implements Eq. 16 and Eq. 18:
          - LiqReserve_k(t) = LiqRate_k * Cap_k(t)              (Eq. 16, refreshed each step)
          - Cap_k(t) includes ... -LiqDebit_k(t-1)              (Eq. 18, lagged)

        Parameters
        ----------
        ftd               : total funding received from cash providers this step
        liq_debit_this_step : dollar shortfall from QMax clipping (liq_debit_needed)

        Returns
        -------
        True if a liquidity default occurred this step (LiqDebit(t) >= LiqReserve(t)).
        """
        # Eq. 18 (lagged): previous period's LiqDebit reduces capital this step.
        # self.liq_debit still holds last step's value at entry.
        self.capital -= self.liq_debit

        # Eq. 16: reserve is freshly carved from capital each period.
        self.liquidity_reserve = max(0.0, self.liq_rate * self.capital)

        # Record this period's debit (consumed next step per Eq. 18 lag).
        self.liq_debit = liq_debit_this_step

        # Consume this period's debit from this period's reserve.
        post_reserve = self.liquidity_reserve - liq_debit_this_step

        # Eq. 19 — LiqRatio_k(t) = LiqReserve_k / FTD_k (post-consumption residual)
        residual_reserve = max(0.0, post_reserve)
        if ftd > 0:
            self.liq_ratio = residual_reserve / ftd
        else:
            self.liq_ratio = self.liq_ratio_target  # no funding → ratio undefined, keep target

        # Eq. 21 — CW_k(t) = max(0, 100 − φ^CW · max(0, LiqRatioMin − LiqRatio)).
        # Stateless: CW rebuilds to 100 once LiqRatio recovers above the floor.
        deficit = max(0.0, self.liq_ratio_min - self.liq_ratio)
        self.cw = max(0.0, 100.0 - self.cfg.phi_cw * deficit)

        self.liquidity_reserve = residual_reserve

        # Liquidity default: this period's debit exceeded this period's fresh reserve.
        # Equivalent to paper's LiqDebit(t) >= LiqReserve(t) now that Eq. 16 is restored.
        if post_reserve < 0.0 and not self.liquidity_defaulted:
            self.liquidity_defaulted = True
            return True
        return False

    def step_derivatives(self, defaulted_bd_ids: set) -> float:
        """Advance derivatives desk. Returns capital loss from counterparty defaults."""
        if self.derivatives is None:
            return 0.0
        loss = self.derivatives.step(defaulted_bd_ids)
        self.capital -= loss
        if self.capital <= 0 and self.active:
            self.active = False
        return loss

    def snapshot(self, prices: np.ndarray) -> dict:
        snap = {
            "id": self.id,
            "active": self.active,
            "capital": round(self.capital, 6),
            "total_assets": round(self.total_assets(prices), 6),
            "leverage": round(self.current_leverage(prices), 4),
            "funding_from_cp": round(self.funding_from_cp, 4),
            "in_fire_sale": self.last_orders.in_fire_sale,
            "forced_sales": self.last_orders.forced.tolist(),
        }
        if self.derivatives is not None:
            snap["derivatives"] = self.derivatives.snapshot()
        return snap

    def __repr__(self) -> str:
        status = "active" if self.active else "DEFAULTED"
        return f"BankDealer(id={self.id!r}, capital={self.capital:.2f}, status={status})"
