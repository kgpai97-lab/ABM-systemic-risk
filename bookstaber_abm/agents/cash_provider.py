"""
agents/cash_provider.py
-----------------------
CashProvider agent — §3.2 of Bookstaber et al.

Haircut motion is governed entirely by Eq. 22 (LiqRatio deficit) via
``update_haircut_from_creditworthiness``. Loan sizing follows Eq. 5–6
with the ``L_Max`` cap scaled by the borrower's creditworthiness
``CW_k / 100`` — the paper's "the rating determines... how much
funding is provided" lever.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bookstaber_abm.config import SimConfig


class CashProvider:
    def __init__(self, agent_id: str, cfg: SimConfig):
        self.id = agent_id
        self.cfg = cfg

        self._haircuts: dict[str, float] = {}
        self._max_loans: dict[str, float] = {}
        self._creditworthiness: dict[str, float] = {}
        self._cw_smoothed: dict[str, float] = {}
        self.last_loans: dict[str, float] = {}

    def register_counterparty(self, bd_id: str) -> None:
        self._haircuts[bd_id] = self.cfg.cp_haircut_normal
        self._max_loans[bd_id] = self.cfg.cp_max_loan
        self._creditworthiness[bd_id] = 100.0
        self._cw_smoothed[bd_id] = 100.0

    def compute_loan(self, bd_id: str, collateral_value: float) -> float:
        hc = self._haircuts[bd_id]
        # Use the EMA-smoothed CW to gate L_max — damps the CW↔LiqRatio 2-cycle
        # that arises because FTD sits in the LiqRatio denominator.
        cw = self._cw_smoothed.get(bd_id, 100.0)
        cw_factor = max(0.0, min(1.0, cw / 100.0))

        l_target = collateral_value * (1.0 - hc)        # Eq. 5
        l_max = self._max_loans[bd_id] * cw_factor       # Eq. 6 with CW gate
        loan = min(l_max, max(0.0, l_target))
        self.last_loans[bd_id] = loan
        return loan

    def haircut(self, bd_id: str) -> float:
        return self._haircuts.get(bd_id, self.cfg.cp_haircut_normal)

    def creditworthiness(self, bd_id: str) -> float:
        return self._creditworthiness.get(bd_id, 100.0)

    def update_haircut_from_creditworthiness(
        self, bd_id: str, liq_ratio: float, liq_ratio_min: float
    ) -> None:
        """
        Eq. 22 — raise haircut when BD's LiqRatio falls below LiqRatioMin.

        HC_k(t) = min(HC_stressed, HC_k(t-1) + phi_hc * max(0, LiqRatioMin - LiqRatio_k))
        """
        deficit = max(0.0, liq_ratio_min - liq_ratio)
        new_hc = self._haircuts.get(bd_id, self.cfg.cp_haircut_normal) + self.cfg.phi_hc * deficit
        self._haircuts[bd_id] = min(self.cfg.cp_haircut_stressed, new_hc)

    def update_creditworthiness(self, bd_id: str, cw: float) -> None:
        """Mirror the BD's current creditworthiness (Eq. 21) into the CP's view
        and advance the EMA-smoothed value used by the loan gate."""
        self._creditworthiness[bd_id] = max(0.0, min(100.0, cw))
        alpha = self.cfg.cp_cw_smoothing_alpha
        prev = self._cw_smoothed.get(bd_id, 100.0)
        self._cw_smoothed[bd_id] = alpha * self._creditworthiness[bd_id] + (1.0 - alpha) * prev

    def cw_smoothed(self, bd_id: str) -> float:
        return self._cw_smoothed.get(bd_id, 100.0)

    def snapshot(self) -> dict:
        return {
            "id": self.id,
            "loans": {k: round(v, 4) for k, v in self.last_loans.items()},
            "haircuts": {k: round(v, 4) for k, v in self._haircuts.items()},
            "creditworthiness": {k: round(v, 4) for k, v in self._creditworthiness.items()},
            "creditworthiness_smoothed": {k: round(v, 4) for k, v in self._cw_smoothed.items()},
        }

    def __repr__(self) -> str:
        return f"CashProvider(id={self.id!r})"
