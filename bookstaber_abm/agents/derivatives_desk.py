"""
agents/derivatives_desk.py
--------------------------
Derivatives desk — §3.3.4 of Bookstaber et al.

The paper describes the derivatives desk as representing counterparty
credit exposure: each bank/dealer holds bilateral derivative contracts
with other bank/dealers, creating a network of credit exposures that
propagates losses when a counterparty defaults.

Model
-----
Each BD k holds a notional derivatives position against every other BD.
When BD j defaults, BD k takes a mark-to-market loss equal to:

    Loss_{k,j} = Exposure_{k,j} * (1 - recovery_rate)

where Exposure_{k,j} = max(0, MTM_{k,j}) is the positive mark-to-market
(the "in the money" side — only positive exposures become credit losses).

The MTM of each contract drifts randomly each period (simulating
market moves on the underlying) and is written down on default.

This creates a second contagion channel beyond the fire-sale / portfolio
overlap channel: a BD default hits other BDs' capital directly, which
can impair their own lending / intermediation capacity.
"""
from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bookstaber_abm.config import SimConfig


class DerivativesDesk:
    """
    Tracks bilateral derivative exposures for one bank/dealer.

    Attributes
    ----------
    exposures : dict[str, float]
        Current mark-to-market exposure to each counterparty BD.
        Positive = we are owed money (credit risk).
        Negative = we owe money (no credit risk to us).
    """

    def __init__(self, owner_id: str, cfg: SimConfig, rng: np.random.Generator):
        self.owner_id = owner_id
        self.cfg = cfg
        self.rng = rng

        # {counterparty_bd_id: mtm_value}
        self.exposures: dict[str, float] = {}
        self.realized_losses: float = 0.0
        self.loss_this_step: float = 0.0

    def register_counterparty(self, bd_id: str) -> None:
        """Set up an initial bilateral position at notional/2 (roughly at-the-money)."""
        # Initial MTM is zero (at-the-money contract at inception)
        self.exposures[bd_id] = 0.0

    def step(self, defaulted_bd_ids: set[str]) -> float:
        """
        Advance one period:
          1. Random walk each MTM exposure (±1% daily vol as a default)
          2. Crystallise losses on defaulted counterparties
          3. Return total capital loss to be deducted from owner's capital

        Parameters
        ----------
        defaulted_bd_ids : set of BD ids that defaulted this period

        Returns
        -------
        Capital loss this period (non-negative float).
        """
        self.loss_this_step = 0.0

        # 1. MTM random walk — OU process mean-reverting to 0
        for cpty_id in list(self.exposures.keys()):
            if cpty_id not in defaulted_bd_ids:
                # dMTM = -0.05*MTM + σ*N(0,1)  (mild mean reversion)
                shock = self.rng.normal(0, self.cfg.bd_derivatives_notional * 0.01)
                self.exposures[cpty_id] = self.exposures[cpty_id] * 0.95 + shock

        # 2. Crystallise losses on defaults
        for cpty_id in defaulted_bd_ids:
            if cpty_id in self.exposures:
                mtm = self.exposures[cpty_id]
                # Only positive exposures become losses (we were owed money)
                loss = max(0.0, mtm) * (1.0 - self.cfg.bd_derivatives_recovery)
                self.loss_this_step += loss
                self.realized_losses += loss
                # Zero out the exposure — contract is gone
                self.exposures[cpty_id] = 0.0

        return self.loss_this_step

    def total_positive_exposure(self) -> float:
        """Sum of all positive MTM exposures = gross credit risk."""
        return sum(max(0.0, v) for v in self.exposures.values())

    def net_exposure(self) -> float:
        """Net MTM across all counterparties."""
        return sum(self.exposures.values())

    def snapshot(self) -> dict:
        return {
            "owner": self.owner_id,
            "exposures": {k: round(v, 4) for k, v in self.exposures.items()},
            "positive_exposure": round(self.total_positive_exposure(), 4),
            "loss_this_step": round(self.loss_this_step, 4),
            "realized_losses": round(self.realized_losses, 4),
        }
