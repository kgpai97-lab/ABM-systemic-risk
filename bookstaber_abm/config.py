"""
config.py — Central parameter store for the Bookstaber-Paddrik-Tivnan ABM.
"""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class SimConfig:
    # Structure
    n_assets: int = 3
    n_hedge_funds: int = 5
    n_bank_dealers: int = 2
    n_cash_providers: int = 2
    n_steps: int = 200
    seed: int = 42

    # Asset market
    beta: float = 1.0          # price impact per unit of *normalised* forced flow (base, β0)
    beta1: float = 0.0         # marginal growth of impact per unit |effective_flow|; 0.0 recovers linear model
    noise_std: float = 0.005
    initial_price: float = 100.0
    normalise_beta: bool = True  # divide net flow by shares_outstanding before applying beta

    # Hedge fund
    hf_initial_capital: float = 10000.0
    # Minimum funding shortfall (as fraction of current assets) that triggers a
    # funding-squeeze forced sale.  Eliminates oscillation from haircut rounding
    # and small noise-driven drifts; set to 0 to trigger on any shortfall.
    hf_funding_squeeze_threshold: float = 0.02
    hf_lev_target: float = 3.0
    hf_lev_buffer: float = 7.0
    hf_lev_max: float = 10.0
    hf_allocation: list = field(default_factory=list)
    # Per-HF allocations for heterogeneous portfolios (list of N lists summing to 1)
    hf_allocations_hetero: list = field(default_factory=list)
    # Per-HF funding weights across BDs (list of N lists of length n_bank_dealers,
    # each summing to 1). Default empty → 1-to-1 round-robin (HF_i funded entirely
    # by BD[i % n_bd]). Setting this lets each HF diversify funding across BDs,
    # which decouples one BD's failure from any single HF.
    hf_bd_funding_weights: list = field(default_factory=list)
    # crowding ∈ [0,1]: 0=diversified, 1=identical portfolios
    crowding: float = 0.5

    # Bank / dealer
    bd_initial_capital: float = 10000.0
    bd_lev_target: float = 8.0
    bd_lev_buffer: float = 14.0
    bd_lev_max: float = 20.0
    bd_liq_rate: float = 0.3
    # Per-BD allocations for asymmetric BD portfolios (list of K lists summing to 1).
    # If empty, BDs use equal-weight allocation. Breaks the bit-identical BD trajectory
    # problem documented in CLAUDE.md.
    bd_allocations_hetero: list = field(default_factory=list)
    bd_max_liq_frac: float = 1.0   # fraction of holdings BD can force-sell per step (1.0 = no cap)
    hf_max_liq_frac: float = 1.0   # fraction of holdings HF can force-sell per step (1.0 = no cap)
    # Per-HF override of hf_max_liq_frac (length n_hedge_funds). Empty → all HFs use scalar.
    # Use to attenuate a specific HF's forced flow (e.g. HF0's huge sale after a deep shock)
    # so that price-impact cascade does not run away in 1–2 steps.
    hf_max_liq_frac_per_hf: list = field(default_factory=list)
    # Fire-sale weight allocation. 1.0 = legacy two-phase (all shock_asset, then proportional once depleted).
    # 0.0 = pure proportional-by-current-holdings from step 1. 0.5 = blend.
    fire_sale_shock_concentration: float = 0.5
    # Treasury desk / creditworthiness (§3.3.5)
    bd_liq_ratio_min: float = 0.20     # LiqRatioMin — below this CW and HC deteriorate
    bd_liq_ratio_target: float = 0.25  # LiqRatioTarget
    phi_cw: float = 100.0              # Eq. 21: CW drop per unit of LiqRatio deficit
    phi_hc: float = 0.10               # Eq. 22: HC increase per unit of LiqRatio deficit
    # Derivatives desk
    enable_derivatives_desk: bool = True
    bd_derivatives_notional: float = 20000.0
    bd_derivatives_recovery: float = 0.40

    # Cash provider
    cp_haircut_normal: float = 0.10     # initial haircut, lower clamp
    cp_haircut_stressed: float = 0.25   # upper clamp for Eq. 22 increments
    cp_max_loan: float = 500000.0       # gated by CW/100 in compute_loan (Eq. 6)
    cp_cw_smoothing_alpha: float = 0.10  # EMA factor on CW used by the loan gate

    # Shock
    shock_step: int = 50
    shock_asset: int = 0
    shock_size: float = -0.10

    # Contagion-decomposition counterfactual (experiments/contagion_decomposition.py).
    # When True, the post-shock ENDOGENOUS response is suppressed: forced sales and
    # default liquidations do not move prices (only the exogenous shock + noise do),
    # and the funding chain is frozen at its pre-shock level so no funding-squeeze
    # cascade fires. Agents still mark-to-market the primary shock. The RNG/noise
    # path is preserved identically to a normal run, so a suppress-vs-normal pair at
    # the same seed isolates the contagion effect as (normal - suppressed).
    suppress_contagion: bool = False

    # Monte Carlo
    mc_runs: int = 20

    def __post_init__(self):
        assert self.hf_lev_target < self.hf_lev_buffer < self.hf_lev_max, \
            "HF leverage hierarchy violated: Target < Buffer < Max required"
        assert self.bd_lev_target < self.bd_lev_buffer < self.bd_lev_max, \
            "BD leverage hierarchy violated"
        assert -1.0 < self.shock_size < 0.0 or self.shock_size == 0.0, \
            "shock_size must be in (-1, 0]"
        assert 0.0 <= self.crowding <= 1.0, "crowding must be in [0, 1]"
        assert 0.0 < self.cp_cw_smoothing_alpha <= 1.0, "cp_cw_smoothing_alpha must be in (0, 1]"

        if not self.hf_allocation:
            self.hf_allocation = [1.0 / self.n_assets] * self.n_assets
        assert abs(sum(self.hf_allocation) - 1.0) < 1e-9
        assert len(self.hf_allocation) == self.n_assets

        if self.hf_allocations_hetero:
            assert len(self.hf_allocations_hetero) == self.n_hedge_funds
            for i, a in enumerate(self.hf_allocations_hetero):
                assert abs(sum(a) - 1.0) < 1e-9, f"hetero alloc {i} must sum to 1"
                assert len(a) == self.n_assets

        if self.bd_allocations_hetero:
            assert len(self.bd_allocations_hetero) == self.n_bank_dealers
            for k, a in enumerate(self.bd_allocations_hetero):
                assert abs(sum(a) - 1.0) < 1e-9, f"BD hetero alloc {k} must sum to 1"
                assert len(a) == self.n_assets

        if self.hf_max_liq_frac_per_hf:
            assert len(self.hf_max_liq_frac_per_hf) == self.n_hedge_funds, (
                f"hf_max_liq_frac_per_hf must have length {self.n_hedge_funds}")
            for i, v in enumerate(self.hf_max_liq_frac_per_hf):
                assert 0.0 < v <= 1.0, f"hf_max_liq_frac_per_hf[{i}] must be in (0, 1]"

        if self.hf_bd_funding_weights:
            assert len(self.hf_bd_funding_weights) == self.n_hedge_funds
            for i, w in enumerate(self.hf_bd_funding_weights):
                assert len(w) == self.n_bank_dealers, (
                    f"hf_bd_funding_weights[{i}] must have length {self.n_bank_dealers}")
                assert abs(sum(w) - 1.0) < 1e-9, (
                    f"hf_bd_funding_weights[{i}] must sum to 1")
                assert all(x >= 0 for x in w), (
                    f"hf_bd_funding_weights[{i}] must be non-negative")

    @property
    def beta_vec(self) -> np.ndarray:
        return np.full(self.n_assets, self.beta)

    @property
    def allocation_vec(self) -> np.ndarray:
        return np.array(self.hf_allocation)

    def get_hf_allocation(self, hf_index: int) -> np.ndarray:
        """Return allocation for HF n, respecting hetero overrides."""
        if self.hf_allocations_hetero:
            return np.array(self.hf_allocations_hetero[hf_index])
        return self.allocation_vec.copy()

    def get_hf_max_liq_frac(self, hf_index: int) -> float:
        """Per-HF liquidation cap; falls back to scalar hf_max_liq_frac."""
        if self.hf_max_liq_frac_per_hf:
            return float(self.hf_max_liq_frac_per_hf[hf_index])
        return float(self.hf_max_liq_frac)

    def get_hf_funding_weights(self, hf_index: int) -> np.ndarray:
        """Per-BD funding share for HF n. Default = one-hot at (hf_index % n_bd).
        Setting hf_bd_funding_weights in config overrides to diversified shares.
        """
        if self.hf_bd_funding_weights:
            return np.array(self.hf_bd_funding_weights[hf_index])
        w = np.zeros(self.n_bank_dealers)
        w[hf_index % self.n_bank_dealers] = 1.0
        return w

    def get_bd_allocation(self, bd_index: int) -> np.ndarray:
        """Return allocation for BD k, respecting hetero overrides; else equal-weight."""
        if self.bd_allocations_hetero:
            return np.array(self.bd_allocations_hetero[bd_index])
        return np.full(self.n_assets, 1.0 / self.n_assets)

    def make_crowded_allocations(self, rng: np.random.Generator) -> list:
        """
        Generate N allocation vectors with controlled portfolio overlap.
        crowding=0 → independent random weights per HF
        crowding=1 → all HFs hold the equal-weight benchmark
        """
        benchmark = np.full(self.n_assets, 1.0 / self.n_assets)
        result = []
        for _ in range(self.n_hedge_funds):
            idio = rng.dirichlet(np.ones(self.n_assets))
            blended = (1.0 - self.crowding) * idio + self.crowding * benchmark
            blended /= blended.sum()
            result.append(blended)
        return result
