"""
market/asset_market.py
----------------------
Implements the asset market from Bookstaber et al. §3.1.

Price impact model (liquidation-dependent, convex generalisation of Kyle/Greenwood):
    β_eff_m(t) = β0_m + β1 * |f_m(t)|
    PR_m(t)    = β_eff_m(t) * f_m(t) + ε_m(t)
    P_m(t+1)   = max(0,  P_m(t) * (1 + PR_m(t)))

where:
    f_m         = effective forced order flow in asset m (normalised by shares
                  outstanding when cfg.normalise_beta=True). Negative = net selling.
    ε_m         ~ N(0, noise_std)
    β0_m        = base price impact coefficient for asset m (cfg.beta)
    β1          = marginal impact growth per unit |f| (cfg.beta1); β1=0 recovers
                  the original linear model.

Key invariant: price impact is applied ONCE per period using AGGREGATED
net order flow — not once per agent.  See implementation_guide for why.
"""

from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bookstaber_abm.config import SimConfig


class AssetMarket:
    """
    Stateless price-formation module.

    Holds the current price vector and applies the linear impact model.
    All methods that change prices return the new prices AND update
    self.prices in place — the simulation reads self.prices as the
    canonical price source each step.
    """

    def __init__(self, cfg: SimConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.M = cfg.n_assets

        # P_m(t) — canonical price vector
        self.prices: np.ndarray = np.full(self.M, cfg.initial_price, dtype=float)

        # Per-asset β vector (base coefficient β0)
        self.beta: np.ndarray = cfg.beta_vec.copy()
        # Scalar β1: marginal impact growth per unit |effective_flow|
        self.beta1: float = float(cfg.beta1)

        # Total shares outstanding per asset — set by engine after agents are built.
        # When set and cfg.normalise_beta=True, flow is divided by this before
        # applying beta, making beta dimensionless (price return per unit fraction sold).
        self.shares_outstanding: np.ndarray | None = None

        # Price history for logging (list of snapshots)
        self.price_history: list[np.ndarray] = [self.prices.copy()]

        # Order flow history (for analysis)
        self.flow_history: list[np.ndarray] = [np.zeros(self.M)]

    # ------------------------------------------------------------------ #
    # Shock injection                                                      #
    # ------------------------------------------------------------------ #

    def apply_shock(self, asset_idx: int, shock_size: float) -> np.ndarray:
        """
        Inject an exogenous price shock at t = shock_step.

        Parameters
        ----------
        asset_idx  : which asset to shock (0-indexed)
        shock_size : fractional price change, e.g. -0.10 for −10%

        Returns updated price vector.
        """
        shock_vec = np.zeros(self.M)
        shock_vec[asset_idx] = shock_size
        self.prices = np.maximum(0.0, self.prices * (1.0 + shock_vec))
        return self.prices.copy()

    # ------------------------------------------------------------------ #
    # Price update                                                         #
    # ------------------------------------------------------------------ #

    def update_prices(self, net_forced_flow: np.ndarray) -> np.ndarray:
        """
        Apply one period's price update given net forced order flow.

        PR_m(t)  = (β0_m + β1 * |f_m|) * f_m + ε_m
        P_m(t+1) = max(0, P_m(t) * (1 + PR_m(t)))

        Parameters
        ----------
        net_forced_flow : length-M array of Q_Dpi summed across all agents.
                          Negative values = net selling → prices fall.

        Returns new price vector P_m(t+1).
        """
        noise = self.rng.normal(0.0, self.cfg.noise_std, size=self.M)
        if self.cfg.normalise_beta and self.shares_outstanding is not None:
            denom = np.maximum(self.shares_outstanding, 1.0)
            effective_flow = net_forced_flow / denom
        else:
            effective_flow = net_forced_flow
        beta_eff = self.beta + self.beta1 * np.abs(effective_flow)
        price_return = beta_eff * effective_flow + noise

        new_prices = np.maximum(0.0, self.prices * (1.0 + price_return))

        # Record flow and update state
        self.flow_history.append(net_forced_flow.copy())
        self.prices = new_prices
        self.price_history.append(self.prices.copy())

        return self.prices.copy()

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def price_returns(self, lookback: int = 1) -> np.ndarray:
        """
        Compute realised returns over the last `lookback` periods.
        Returns array of shape (lookback, M).
        """
        hist = np.array(self.price_history)
        if len(hist) < lookback + 1:
            return np.zeros((lookback, self.M))
        p_now  = hist[-1]
        p_then = hist[-1 - lookback]
        return np.where(p_then > 0, (p_now - p_then) / p_then, 0.0)

    def snapshot(self) -> dict:
        return {
            "prices": self.prices.tolist(),
            "price_returns": self.price_returns(1).tolist(),
            "last_flow": self.flow_history[-1].tolist() if self.flow_history else [],
        }
