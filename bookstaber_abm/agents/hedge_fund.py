"""
agents/hedge_fund.py
--------------------
Implements the HedgeFund / TradingDesk agent from Bookstaber et al. §3.3.3.

Key design choices
------------------
- All portfolio quantities are numpy arrays of length M (one entry per asset).
- State is NEVER mutated during the order-computation phase.
  compute_orders() is pure (reads state, returns orders).
  apply_orders() is the only method that writes to state.
- forced_sales and normal_orders are tracked separately so the simulation
  can route only forced_sales through the price-impact equation.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bookstaber_abm.config import SimConfig


@dataclass
class HFOrders:
    """
    Immutable order bundle returned by compute_orders().
    Separates forced (price-impacting) from normal (non-impacting) orders.
    """
    forced: np.ndarray    # Q_Dpi — fire-sale quantities (≤ 0, sells only)
    normal: np.ndarray    # routine rebalancing orders (can be + or -)
    in_fire_sale: bool

    @property
    def total(self) -> np.ndarray:
        return self.forced + self.normal


class HedgeFund:
    """
    Single hedge fund agent.

    Balance sheet identity (end of each step):
        A_n(t)   = sum_m [ P_m(t) * Q_{n,m}(t) ]   # total assets
        F_n(t)   = A_n(t) - Cap_n(t)                # funding needed
        Lev(t)   = A_n(t) / Cap_n(t)                # current leverage

    Leverage hierarchy (must hold):
        Lev^Target <= Lev^Buffer <= Lev^Max
    """

    def __init__(
        self,
        agent_id: str,
        cfg: SimConfig,
        rng: np.random.Generator,
        allocation: np.ndarray | None = None,
        hf_index: int = 0,
    ):
        self.id = agent_id
        self.cfg = cfg
        self.rng = rng
        self.M = cfg.n_assets
        self.hf_index = hf_index

        # --- Leverage parameters ---
        self.lev_target: float = cfg.hf_lev_target
        self.lev_buffer: float = cfg.hf_lev_buffer
        self.lev_max: float    = cfg.hf_lev_max
        self.max_liq_frac: float = cfg.get_hf_max_liq_frac(hf_index)

        # --- Asset allocation vector (sums to 1) ---
        # Each HF can have a distinct allocation; default is equal-weight.
        self.allocation: np.ndarray = (
            allocation.copy() if allocation is not None
            else cfg.allocation_vec.copy()
        )

        # --- State variables (all written only by apply_*) ---
        self.capital: float = cfg.hf_initial_capital

        # Initial portfolio: buy assets at target leverage with initial capital
        # Q_{n,m}(0) = allocation_m * Cap(0) * Lev^Target / P_m(0)
        initial_prices = np.full(self.M, cfg.initial_price)
        target_assets = self.capital * self.lev_target
        self.holdings: np.ndarray = (                      # Q_{n,m}
            self.allocation * target_assets / initial_prices
        ).astype(float)

        # Funding = total asset value − own capital
        self.funding: float = np.dot(initial_prices, self.holdings) - self.capital

        # Order history (set each step, read by simulation for logging)
        self.last_orders: HFOrders = HFOrders(
            forced=np.zeros(self.M),
            normal=np.zeros(self.M),
            in_fire_sale=False,
        )

        # Slippage from last period's orders
        self._prev_orders: np.ndarray = np.zeros(self.M)
        self._prev_price_delta: np.ndarray = np.zeros(self.M)

        # Status
        self.active: bool = True
        self.defaulted_at: int | None = None

    # ------------------------------------------------------------------ #
    # Read-only properties                                                 #
    # ------------------------------------------------------------------ #

    def total_assets(self, prices: np.ndarray) -> float:
        """A_n(t) = Σ P_m * Q_{n,m}"""
        return float(np.dot(prices, self.holdings))

    def current_leverage(self, prices: np.ndarray) -> float:
        """Lev^Current = A_n / Cap_n.  Returns inf if capital ≤ 0."""
        cap = self.capital
        if cap <= 0:
            return float("inf")
        return self.total_assets(prices) / cap

    def slippage(self, price_delta: np.ndarray) -> float:
        """
        S_n(t) = Σ_m  Q_Dn,m(t) * (P_m(t-1) - P_m(t-2))
        Uses previous period's orders and the price move between t-2 and t-1.
        """
        return float(np.dot(self._prev_orders, price_delta))

    # ------------------------------------------------------------------ #
    # Pure computation — no state mutation                                 #
    # ------------------------------------------------------------------ #

    def compute_orders(
        self,
        prices: np.ndarray,
        funding_available: float | None = None,
    ) -> HFOrders:
        """
        Determine this period's orders given current prices and available funding.

        Parameters
        ----------
        prices            : current asset prices (before this step's impact)
        funding_available : dollar funding offered by the prime broker this step.
                            None → no funding constraint applied (backward-compat).
                            If less than the portfolio's funding need, a funding-squeeze
                            forced sale is triggered (price-impacting, §3.3.1).

        Returns an HFOrders bundle — does NOT modify self.
        """
        if not self.active:
            return HFOrders(
                forced=np.zeros(self.M),
                normal=np.zeros(self.M),
                in_fire_sale=False,
            )

        assets = self.total_assets(prices)
        lev = assets / self.capital if self.capital > 0 else float("inf")

        # ---- Determine the binding target -----------------------------------
        # Two constraints may force sales; take the more restrictive one.
        target_assets = float("inf")
        is_forced = False

        # (1) Leverage breach → must deleverage to Lev^Buffer
        if lev >= self.lev_max:
            target_assets = min(target_assets, self.capital * self.lev_buffer)
            is_forced = True

        # (2) Funding squeeze → must reduce portfolio to what BD will fund.
        # Only triggers when shortfall exceeds threshold (avoids haircut-rounding
        # oscillations and noise-driven micro-squeezes before any real stress).
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
            forced_qty = np.where(prices > 0, forced_dollar / prices, 0.0)
            forced_qty = np.maximum(forced_qty, -self.holdings)
            # Cap forced sales to max_liq_frac of current holdings per step (per-HF override)
            max_qty = self.max_liq_frac * np.maximum(self.holdings, 0.0)
            forced_qty = np.maximum(forced_qty, -max_qty)
            return HFOrders(
                forced=forced_qty,
                normal=np.zeros(self.M),
                in_fire_sale=True,
            )

        else:
            # ---- Normal rebalancing ----------------------------------------
            target_assets = self.capital * self.lev_target
            delta_assets = target_assets - assets
            if delta_assets < 0:
                total = self.holdings.sum()
                sell_weights = self.holdings / total if total > 0 else self.allocation
                normal_dollar = delta_assets * sell_weights
            else:
                normal_dollar = delta_assets * self.allocation
            normal_qty = np.where(prices > 0, normal_dollar / prices, 0.0)
            max_normal = self.capital * 1.0
            normal_qty = np.clip(
                normal_qty,
                -max_normal / np.maximum(prices, 1e-9),
                max_normal / np.maximum(prices, 1e-9),
            )
            return HFOrders(
                forced=np.zeros(self.M),
                normal=normal_qty,
                in_fire_sale=False,
            )

    # ------------------------------------------------------------------ #
    # State mutation — called only after ALL agents have computed orders   #
    # ------------------------------------------------------------------ #

    def apply_orders(
        self,
        orders: HFOrders,
        new_prices: np.ndarray,
        old_prices: np.ndarray,
        funding_received: float,
    ) -> None:
        """
        Update state after orders are executed and new prices are known.

        Parameters
        ----------
        orders          : the HFOrders returned by compute_orders()
        new_prices      : P_m(t) after market impact is applied
        old_prices      : P_m(t-1) — used for slippage calculation
        funding_received: F_n(t) from the prime broker this period
        """
        if not self.active:
            return

        # Update holdings: Q_{n,m}(t) = Q_{n,m}(t-1) + QD_{n,m}(t)
        self.holdings = self.holdings + orders.total
        self.holdings = np.maximum(self.holdings, 0.0)  # no short positions

        # Slippage: use this period's orders and the price delta t-2 → t-1
        price_delta = old_prices - self._prev_price_delta  # P(t-1) - P(t-2)
        slip = self.slippage(price_delta)

        # Capital update: incremental mark-to-market P&L on pre-trade holdings.
        # cap(t) = cap(t-1) + Σ_m Q_{n,m}(t-1) * (P_m(t) - P_m(t-1)) - slippage
        # This avoids drift from imprecise funding pass-through.
        holdings_pre_trade = self.holdings - orders.total   # holdings entering t
        price_pnl = float(np.dot(holdings_pre_trade, new_prices - old_prices))
        self.capital = self.capital + price_pnl - slip
        self.funding = float(np.dot(new_prices, self.holdings)) - self.capital

        # Store forced orders only for slippage (S_n = Σ Q_D * ΔP per paper §3.3.3)
        self._prev_orders = orders.forced.copy()
        self._prev_price_delta = old_prices.copy()

        # Store orders for logging
        self.last_orders = orders

    def apply_default_liquidation(self) -> np.ndarray:
        """
        Called the period AFTER default is detected. Returns the forced sale
        quantities (negative = sells) so the simulation can route them through
        the price impact model.

        Rate-limited by self.max_liq_frac: each call sells at most that
        fraction of *original* holdings per asset. Engine re-queues this HF
        each step until holdings are fully drained.

        Returns zero-vector once holdings are exhausted (idempotent).
        """
        if not np.any(self.holdings > 0):
            self.capital = 0.0
            return np.zeros(self.M)

        # Initialise the original-holdings reference on first call.
        if not hasattr(self, "_default_initial_holdings"):
            self._default_initial_holdings = self.holdings.copy()

        cap = self.max_liq_frac * self._default_initial_holdings
        sell_qty = np.minimum(self.holdings, cap)
        self.holdings = self.holdings - sell_qty
        self.capital = 0.0
        return -sell_qty

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _current_weights(self, prices: np.ndarray) -> np.ndarray:
        """
        Current portfolio weights by market value.
        Falls back to target allocation if portfolio is empty.
        """
        values = prices * self.holdings
        total = values.sum()
        if total <= 0:
            return self.allocation.copy()
        return values / total

    def snapshot(self, prices: np.ndarray) -> dict:
        """Return a serialisable state dict for logging."""
        return {
            "id": self.id,
            "active": self.active,
            "capital": round(self.capital, 6),
            "total_assets": round(self.total_assets(prices), 6),
            "leverage": round(self.current_leverage(prices), 4),
            "funding": round(self.funding, 6),
            "holdings": self.holdings.tolist(),
            "in_fire_sale": self.last_orders.in_fire_sale,
            "forced_sales": self.last_orders.forced.tolist(),
        }

    def __repr__(self) -> str:
        status = "active" if self.active else "DEFAULTED"
        return f"HedgeFund(id={self.id!r}, capital={self.capital:.2f}, status={status})"
