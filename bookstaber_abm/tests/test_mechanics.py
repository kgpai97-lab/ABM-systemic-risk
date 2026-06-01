"""
tests/test_mechanics.py
Run with: python -m unittest discover -s bookstaber_abm/tests -v
      or: python bookstaber_abm/tests/test_mechanics.py
"""

import unittest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from bookstaber_abm.config import SimConfig
from bookstaber_abm.agents.hedge_fund import HedgeFund
from bookstaber_abm.agents.cash_provider import CashProvider
from bookstaber_abm.market.asset_market import AssetMarket
from bookstaber_abm.simulation.engine import Simulation


def base_cfg(**kw):
    defaults = dict(n_assets=2, n_hedge_funds=3, n_bank_dealers=1,
                    n_cash_providers=1, n_steps=10, shock_size=0.0, seed=0)
    defaults.update(kw)
    return SimConfig(**defaults)


class TestConfig(unittest.TestCase):
    def test_leverage_hierarchy_enforced(self):
        with self.assertRaises(AssertionError):
            SimConfig(hf_lev_target=10.0, hf_lev_buffer=5.0, hf_lev_max=20.0)

    def test_allocation_must_sum_to_one(self):
        with self.assertRaises(AssertionError):
            SimConfig(n_assets=2, hf_allocation=[0.3, 0.3])

    def test_beta_vec_length(self):
        cfg = base_cfg()
        self.assertEqual(len(cfg.beta_vec), cfg.n_assets)


class TestHedgeFundInit(unittest.TestCase):
    def setUp(self):
        self.cfg = base_cfg()
        self.rng = np.random.default_rng(0)
        self.hf  = HedgeFund("HF_test", self.cfg, self.rng)
        self.prices = np.full(self.cfg.n_assets, self.cfg.initial_price)

    def test_initial_leverage_near_target(self):
        lev = self.hf.current_leverage(self.prices)
        self.assertAlmostEqual(lev, self.cfg.hf_lev_target, delta=0.01)

    def test_holdings_non_negative(self):
        self.assertTrue((self.hf.holdings >= 0).all())

    def test_holdings_length(self):
        self.assertEqual(len(self.hf.holdings), self.cfg.n_assets)

    def test_allocation_weights_applied(self):
        values  = self.prices * self.hf.holdings
        weights = values / values.sum()
        np.testing.assert_allclose(weights, self.cfg.allocation_vec, atol=1e-9)


class TestLeverageBreach(unittest.TestCase):
    def setUp(self):
        self.cfg    = base_cfg()
        self.rng    = np.random.default_rng(0)
        self.prices = np.full(self.cfg.n_assets, self.cfg.initial_price)

    def test_no_breach_no_forced_sales(self):
        hf = HedgeFund("HF_ok", self.cfg, self.rng)
        orders = hf.compute_orders(self.prices)
        self.assertFalse(orders.in_fire_sale)
        self.assertTrue((orders.forced == 0).all())

    def test_breach_triggers_fire_sale(self):
        hf = HedgeFund("HF_breach", self.cfg, self.rng)
        hf.capital = 1.0
        orders = hf.compute_orders(self.prices)
        self.assertTrue(orders.in_fire_sale)
        self.assertTrue((orders.forced <= 0).all())
        self.assertLess(orders.forced.sum(), 0)

    def test_forced_sale_targets_buffer(self):
        # Fire sales are now concentrated on shock_asset first (multi-step deleveraging).
        hf = HedgeFund("HF_buf", self.cfg, self.rng)
        hf.capital = 1.0
        orders = hf.compute_orders(self.prices)
        # While shock_asset still has holdings, only that asset should be sold
        shock = self.cfg.shock_asset
        if hf.holdings[shock] > 0:
            for m in range(self.cfg.n_assets):
                if m != shock:
                    self.assertEqual(orders.forced[m], 0.0)
        # Simulate repeated steps until leverage reaches buffer (at most 1000 iterations)
        for _ in range(1000):
            if hf.current_leverage(self.prices) <= self.cfg.hf_lev_buffer + 0.01:
                break
            orders = hf.compute_orders(self.prices)
            hf.holdings = np.maximum(hf.holdings + orders.total, 0.0)
        self.assertLessEqual(hf.current_leverage(self.prices), self.cfg.hf_lev_buffer + 0.01)

    def test_forced_sales_cannot_exceed_holdings(self):
        hf = HedgeFund("HF_noshort", self.cfg, self.rng)
        hf.capital = 0.01
        orders = hf.compute_orders(self.prices)
        self.assertTrue((orders.forced >= -hf.holdings - 1e-9).all())

    def test_compute_orders_is_pure(self):
        hf = HedgeFund("HF_pure", self.cfg, self.rng)
        hf.capital = 1.0
        h_before = hf.holdings.copy()
        cap_before = hf.capital
        hf.compute_orders(self.prices)
        hf.compute_orders(self.prices)
        np.testing.assert_array_equal(hf.holdings, h_before)
        self.assertEqual(hf.capital, cap_before)


class TestPriceImpact(unittest.TestCase):
    def setUp(self):
        self.cfg = base_cfg(noise_std=0.0)
        self.rng = np.random.default_rng(0)
        self.mkt = AssetMarket(self.cfg, self.rng)

    def test_negative_flow_reduces_price(self):
        p_before = self.mkt.prices.copy()
        flow     = np.full(self.cfg.n_assets, -10.0)
        p_after  = self.mkt.update_prices(flow)
        self.assertTrue((p_after <= p_before).all())

    def test_price_floor_at_zero(self):
        flow    = np.full(self.cfg.n_assets, -1e9)
        p_after = self.mkt.update_prices(flow)
        self.assertTrue((p_after >= 0).all())

    def test_price_impact_scales_with_beta(self):
        flow    = np.full(2, -5.0)
        cfg_lo  = SimConfig(n_assets=2, beta=0.01, noise_std=0.0, seed=1)
        cfg_hi  = SimConfig(n_assets=2, beta=0.10, noise_std=0.0, seed=1)
        mkt_lo  = AssetMarket(cfg_lo, np.random.default_rng(1))
        mkt_hi  = AssetMarket(cfg_hi, np.random.default_rng(1))
        p_lo    = mkt_lo.update_prices(flow)
        p_hi    = mkt_hi.update_prices(flow)
        self.assertTrue((p_hi < p_lo).all())

    def test_shock_applies_correctly(self):
        p_before = self.mkt.prices.copy()
        self.mkt.apply_shock(0, -0.10)
        self.assertAlmostEqual(self.mkt.prices[0], p_before[0] * 0.90, places=9)
        self.assertEqual(self.mkt.prices[1], p_before[1])


class TestNonlinearPriceImpact(unittest.TestCase):
    """β_eff = β0 + β1 * |f| → impact is convex in flow size."""

    def _mkt(self, beta, beta1):
        cfg = SimConfig(n_assets=2, beta=beta, beta1=beta1,
                        noise_std=0.0, normalise_beta=False, seed=1)
        return AssetMarket(cfg, np.random.default_rng(1)), cfg

    def test_recovers_linear_when_beta1_zero(self):
        flow = np.array([-5.0, -2.0])
        mkt, cfg = self._mkt(beta=0.01, beta1=0.0)
        p0 = mkt.prices.copy()
        p1 = mkt.update_prices(flow)
        expected = p0 * (1.0 + cfg.beta * flow)
        np.testing.assert_allclose(p1, expected, atol=1e-12)

    def test_convexity_in_flow(self):
        # With β0=0, return is purely β1 * f * |f|; doubling |f| → 4x return magnitude.
        mkt_a, _ = self._mkt(beta=0.0, beta1=0.001)
        mkt_b, _ = self._mkt(beta=0.0, beta1=0.001)
        f1 = np.array([-3.0, -3.0])
        f2 = 2.0 * f1
        p_a = mkt_a.update_prices(f1)
        p_b = mkt_b.update_prices(f2)
        r_a = (p_a / 100.0) - 1.0   # initial_price = 100
        r_b = (p_b / 100.0) - 1.0
        # |r_b| should exceed 2 * |r_a| (in fact ≈ 4x) — strictly convex.
        self.assertTrue(np.all(np.abs(r_b) > 2.0 * np.abs(r_a) + 1e-9))

    def test_sign_preserved_with_beta1(self):
        mkt, _ = self._mkt(beta=0.01, beta1=0.01)
        p0 = mkt.prices.copy()
        p1 = mkt.update_prices(np.array([4.0, -4.0]))
        self.assertGreater(p1[0], p0[0])   # buying → price up
        self.assertLess(p1[1], p0[1])      # selling → price down


class TestCashProvider(unittest.TestCase):
    def setUp(self):
        self.cfg = base_cfg()
        self.cp  = CashProvider("CP_test", self.cfg)
        self.cp.register_counterparty("BD_0")

    def test_loan_respects_max_cap(self):
        loan = self.cp.compute_loan("BD_0", collateral_value=1e9)
        self.assertLessEqual(loan, self.cfg.cp_max_loan + 1e-9)

    def test_loan_scales_with_collateral(self):
        l1 = self.cp.compute_loan("BD_0", 100.0)
        l2 = self.cp.compute_loan("BD_0", 1000.0)
        self.assertGreater(l2, l1)

    def test_loan_zero_on_zero_collateral(self):
        loan = self.cp.compute_loan("BD_0", 0.0)
        self.assertEqual(loan, 0.0)

    def test_haircut_does_not_move_without_liq_ratio_deficit(self):
        """Eq. 22 is the sole haircut writer; compute_loan must not change HC."""
        self.cp.compute_loan("BD_0", 1000.0)
        hc1 = self.cp.haircut("BD_0")
        self.cp.compute_loan("BD_0", 100.0)
        hc2 = self.cp.haircut("BD_0")
        self.assertAlmostEqual(hc1, hc2, places=12)

    def test_loan_uses_initial_haircut(self):
        """Eq. 5: L_target = collateral * (1 - HC). HC starts at cp_haircut_normal."""
        loan = self.cp.compute_loan("BD_0", 500.0)
        expected = 500.0 * (1.0 - self.cfg.cp_haircut_normal)
        self.assertAlmostEqual(loan, expected, places=6)


class TestSimulationValidation(unittest.TestCase):
    def test_no_shock_stable(self):
        cfg = SimConfig(n_assets=2, n_hedge_funds=3, n_bank_dealers=1,
                        n_cash_providers=1, n_steps=100, shock_size=0.0,
                        noise_std=0.001, hf_lev_target=2.0,
                        hf_lev_max=8.0, hf_lev_buffer=6.0, seed=42)
        sim     = Simulation(cfg)
        history = sim.run()
        self.assertEqual(history[-1]["n_defaults"], 0)
        final_prices = np.array(history[-1]["prices"])
        pct = np.abs(final_prices - cfg.initial_price) / cfg.initial_price
        self.assertTrue((pct < 0.20).all())

    def test_large_shock_triggers_fire_sales(self):
        cfg = SimConfig(n_assets=2, n_hedge_funds=5, n_bank_dealers=2,
                        n_cash_providers=2, n_steps=100, shock_step=30,
                        shock_size=-0.15, noise_std=0.001,
                        hf_lev_target=6.0, hf_lev_buffer=8.0,
                        hf_lev_max=10.0, beta=0.05, seed=7)
        sim     = Simulation(cfg)
        history = sim.run()
        max_fire = max(r["n_fire_sales"] for r in history)
        self.assertGreater(max_fire, 0)

    def test_no_negative_prices(self):
        cfg = SimConfig(n_assets=3, n_hedge_funds=5, n_bank_dealers=2,
                        n_cash_providers=2, n_steps=100, shock_step=20,
                        shock_size=-0.50, noise_std=0.002,
                        hf_lev_target=8.0, hf_lev_buffer=9.0,
                        hf_lev_max=10.0, beta=0.10, seed=99)
        sim     = Simulation(cfg)
        history = sim.run()
        for r in history:
            self.assertTrue((np.array(r["prices"]) >= 0).all())

    def test_compute_orders_does_not_mutate_holdings(self):
        cfg     = SimConfig(n_assets=2, seed=0)
        hf      = HedgeFund("HF_mut", cfg, np.random.default_rng(0))
        prices  = np.full(cfg.n_assets, cfg.initial_price)
        hf.capital = 1.0
        h_before = hf.holdings.copy()
        hf.compute_orders(prices)
        hf.compute_orders(prices)
        np.testing.assert_array_equal(hf.holdings, h_before)

    def test_defaulted_agent_count_never_rises(self):
        cfg = SimConfig(n_assets=2, n_hedge_funds=2, n_bank_dealers=1,
                        n_cash_providers=1, n_steps=60, shock_step=5,
                        shock_size=-0.40, noise_std=0.0,
                        hf_lev_target=8.0, hf_lev_buffer=9.0,
                        hf_lev_max=10.0, beta=0.08, seed=3)
        sim     = Simulation(cfg)
        history = sim.run()
        default_steps = [r["t"] for r in history if r["n_defaults"] > 0]
        if not default_steps:
            return  # no defaults — test not applicable
        first  = default_steps[0]
        counts = [r["n_active_hf"] for r in history if r["t"] >= first]
        for i in range(1, len(counts)):
            self.assertLessEqual(counts[i], counts[i-1])




class TestHeterogeneousAllocations(unittest.TestCase):
    def test_crowded_allocations_count(self):
        cfg = SimConfig(n_assets=3, n_hedge_funds=4, crowding=0.5, seed=1)
        rng = np.random.default_rng(1)
        allocs = cfg.make_crowded_allocations(rng)
        self.assertEqual(len(allocs), 4)
        for a in allocs:
            self.assertAlmostEqual(a.sum(), 1.0, places=9)
            self.assertTrue((a >= 0).all())

    def test_fully_crowded_gives_equal_weight(self):
        """crowding=1 must produce the equal-weight benchmark for all HFs."""
        cfg = SimConfig(n_assets=3, n_hedge_funds=5, crowding=1.0, seed=0)
        rng = np.random.default_rng(0)
        allocs = cfg.make_crowded_allocations(rng)
        benchmark = np.full(3, 1/3)
        for a in allocs:
            np.testing.assert_allclose(a, benchmark, atol=1e-9)

    def test_hetero_allocs_applied_per_hf(self):
        """Each HF receives its own allocation from hf_allocations_hetero."""
        allocs = [[0.7, 0.3], [0.2, 0.8], [0.5, 0.5]]
        cfg = SimConfig(n_assets=2, n_hedge_funds=3, hf_allocations_hetero=allocs,
                        n_bank_dealers=1, n_cash_providers=1, n_steps=1,
                        shock_size=0.0, seed=0)
        sim = Simulation(cfg)
        for i, hf in enumerate(sim.hedge_funds):
            np.testing.assert_allclose(hf.allocation, allocs[i], atol=1e-9)

    def test_diversified_allocs_differ(self):
        """crowding=0 should produce meaningfully different allocations per HF."""
        cfg = SimConfig(n_assets=4, n_hedge_funds=5, crowding=0.0, seed=42)
        rng = np.random.default_rng(42)
        allocs = cfg.make_crowded_allocations(rng)
        # At least one pair should differ by more than 5%
        found_diff = False
        for i in range(len(allocs)):
            for j in range(i+1, len(allocs)):
                if np.abs(allocs[i] - allocs[j]).max() > 0.05:
                    found_diff = True
        self.assertTrue(found_diff, "Expected diverse allocations with crowding=0")


class TestDerivativesDeskContagion(unittest.TestCase):
    def test_bd_default_causes_counterparty_loss(self):
        """When a BD defaults, its derivatives counterparties take a capital hit."""
        from bookstaber_abm.agents.derivatives_desk import DerivativesDesk
        cfg = SimConfig(
            n_assets=2, n_bank_dealers=2, enable_derivatives_desk=True,
            bd_derivatives_notional=1000.0, bd_derivatives_recovery=0.0,
            seed=0,
        )
        rng = np.random.default_rng(0)
        desk = DerivativesDesk("BD_0", cfg, rng)
        desk.register_counterparty("BD_1")
        # Manually set a large positive exposure
        desk.exposures["BD_1"] = 500.0
        loss = desk.step(defaulted_bd_ids={"BD_1"})
        self.assertAlmostEqual(loss, 500.0, places=4)
        self.assertEqual(desk.exposures["BD_1"], 0.0)

    def test_negative_exposure_no_loss_on_default(self):
        """Negative MTM (we owe) means no credit loss when counterparty defaults."""
        from bookstaber_abm.agents.derivatives_desk import DerivativesDesk
        cfg = SimConfig(n_assets=2, enable_derivatives_desk=True,
                        bd_derivatives_recovery=0.0, seed=0)
        rng = np.random.default_rng(0)
        desk = DerivativesDesk("BD_0", cfg, rng)
        desk.register_counterparty("BD_1")
        desk.exposures["BD_1"] = -300.0   # we owe them — no risk to us
        loss = desk.step(defaulted_bd_ids={"BD_1"})
        self.assertEqual(loss, 0.0)

    def test_recovery_rate_reduces_loss(self):
        """Higher recovery rate should reduce crystallised loss."""
        from bookstaber_abm.agents.derivatives_desk import DerivativesDesk
        cfg_0 = SimConfig(n_assets=2, enable_derivatives_desk=True,
                          bd_derivatives_recovery=0.0, seed=0)
        cfg_4 = SimConfig(n_assets=2, enable_derivatives_desk=True,
                          bd_derivatives_recovery=0.4, seed=0)
        rng = np.random.default_rng(0)

        desk_0 = DerivativesDesk("BD_0", cfg_0, rng)
        desk_0.register_counterparty("BD_1")
        desk_0.exposures["BD_1"] = 1000.0

        desk_4 = DerivativesDesk("BD_0", cfg_4, rng)
        desk_4.register_counterparty("BD_1")
        desk_4.exposures["BD_1"] = 1000.0

        loss_0 = desk_0.step({"BD_1"})
        loss_4 = desk_4.step({"BD_1"})
        self.assertGreater(loss_0, loss_4)
        self.assertAlmostEqual(loss_0, 1000.0, places=4)
        self.assertAlmostEqual(loss_4, 600.0, places=4)

    def test_simulation_runs_with_derivatives_enabled(self):
        """Full simulation should complete without error when derivatives are on."""
        cfg = SimConfig(
            n_assets=2, n_hedge_funds=3, n_bank_dealers=2, n_cash_providers=1,
            n_steps=30, shock_size=0.0, enable_derivatives_desk=True,
            bd_derivatives_notional=50.0, seed=5,
        )
        sim = Simulation(cfg)
        history = sim.run()
        self.assertEqual(len(history), 30)
        # All prices should remain non-negative
        for r in history:
            self.assertTrue((np.array(r["prices"]) >= 0).all())


class TestEq22HaircutRamp(unittest.TestCase):
    """Eq. 22 — haircut ratchets up while LiqRatio < LiqRatioMin, never decays."""

    def _make_cp(self, **kw):
        defaults = dict(
            n_assets=2, cp_haircut_normal=0.10, cp_haircut_stressed=0.25,
            cp_max_loan=1e9, phi_hc=5.0,
            bd_liq_ratio_min=0.20, bd_liq_ratio_target=0.25, seed=0,
        )
        defaults.update(kw)
        cfg = SimConfig(**defaults)
        cp = CashProvider("CP_eq22", cfg)
        cp.register_counterparty("BD_0")
        return cfg, cp

    def test_haircut_monotonic_under_persistent_deficit(self):
        cfg, cp = self._make_cp()
        hc_history = [cp.haircut("BD_0")]
        for _ in range(5):
            # LiqRatio well below LiqRatioMin → positive deficit each call.
            cp.update_haircut_from_creditworthiness(
                "BD_0", liq_ratio=0.05, liq_ratio_min=cfg.bd_liq_ratio_min,
            )
            hc_history.append(cp.haircut("BD_0"))
        # Strictly increasing until clamped at cp_haircut_stressed.
        for prev, curr in zip(hc_history, hc_history[1:]):
            self.assertGreaterEqual(curr, prev - 1e-12)
        self.assertGreater(hc_history[-1], hc_history[0])

    def test_haircut_clamped_at_stressed(self):
        cfg, cp = self._make_cp(phi_hc=100.0)  # giant phi to blow through cap
        for _ in range(20):
            cp.update_haircut_from_creditworthiness(
                "BD_0", liq_ratio=0.0, liq_ratio_min=cfg.bd_liq_ratio_min,
            )
        self.assertLessEqual(cp.haircut("BD_0"), cfg.cp_haircut_stressed + 1e-12)
        self.assertAlmostEqual(cp.haircut("BD_0"), cfg.cp_haircut_stressed, places=12)

    def test_haircut_does_not_decay_when_deficit_clears(self):
        """Eq. 22 has no decrement term — once ratcheted up, HC stays."""
        cfg, cp = self._make_cp()
        cp.update_haircut_from_creditworthiness(
            "BD_0", liq_ratio=0.05, liq_ratio_min=cfg.bd_liq_ratio_min,
        )
        hc_stressed = cp.haircut("BD_0")
        self.assertGreater(hc_stressed, cfg.cp_haircut_normal)
        # Now recovery — LiqRatio above min, deficit = 0.
        for _ in range(10):
            cp.update_haircut_from_creditworthiness(
                "BD_0", liq_ratio=0.50, liq_ratio_min=cfg.bd_liq_ratio_min,
            )
        self.assertAlmostEqual(cp.haircut("BD_0"), hc_stressed, places=12)


class TestCWGatesLoanAmount(unittest.TestCase):
    """Eq. 6 + paper §3.3.5 — CW modulates L_Max."""

    def _make_cp(self, cp_max_loan=1000.0):
        # alpha=1.0 disables EMA smoothing so the unit assertions about Eq. 6
        # apply immediately. The smoother itself is tested in TestCWSmoothing.
        cfg = SimConfig(n_assets=2, cp_max_loan=cp_max_loan,
                        cp_cw_smoothing_alpha=1.0, seed=0)
        cp = CashProvider("CP_cw", cfg)
        cp.register_counterparty("BD_0")
        return cfg, cp

    def test_full_cw_yields_haircut_implied_target(self):
        cfg, cp = self._make_cp(cp_max_loan=1e9)  # cap non-binding
        loan = cp.compute_loan("BD_0", collateral_value=500.0)
        self.assertAlmostEqual(loan, 500.0 * (1.0 - cfg.cp_haircut_normal), places=6)

    def test_half_cw_halves_loan_cap(self):
        cfg, cp = self._make_cp(cp_max_loan=400.0)
        cp.update_creditworthiness("BD_0", 50.0)  # CW/100 = 0.5
        # Target would be 500*(1-0.1)=450, exceeds 0.5*400=200 → loan = 200.
        loan = cp.compute_loan("BD_0", collateral_value=500.0)
        self.assertAlmostEqual(loan, 200.0, places=6)

    def test_zero_cw_blocks_lending(self):
        cfg, cp = self._make_cp()
        cp.update_creditworthiness("BD_0", 0.0)
        loan = cp.compute_loan("BD_0", collateral_value=500.0)
        self.assertEqual(loan, 0.0)

    def test_cw_clamped_to_unit_interval(self):
        cfg, cp = self._make_cp()
        cp.update_creditworthiness("BD_0", 5000.0)  # absurd input → clamp to 100
        self.assertEqual(cp.creditworthiness("BD_0"), 100.0)
        cp.update_creditworthiness("BD_0", -5.0)
        self.assertEqual(cp.creditworthiness("BD_0"), 0.0)


class TestCWSmoothing(unittest.TestCase):
    """EMA-smoothed CW used by the loan gate damps the CW↔LiqRatio 2-cycle."""

    def _make_cp(self, alpha):
        cfg = SimConfig(n_assets=2, cp_max_loan=1e9,
                        cp_cw_smoothing_alpha=alpha, seed=0)
        cp = CashProvider("CP_sm", cfg)
        cp.register_counterparty("BD_0")
        return cfg, cp

    def test_alpha_half_first_step(self):
        cfg, cp = self._make_cp(alpha=0.5)
        cp.update_creditworthiness("BD_0", 0.0)
        self.assertAlmostEqual(cp.cw_smoothed("BD_0"), 50.0, places=6)

    def test_alpha_half_second_step(self):
        cfg, cp = self._make_cp(alpha=0.5)
        cp.update_creditworthiness("BD_0", 0.0)
        cp.update_creditworthiness("BD_0", 0.0)
        self.assertAlmostEqual(cp.cw_smoothed("BD_0"), 25.0, places=6)

    def test_alpha_one_disables_smoothing(self):
        cfg, cp = self._make_cp(alpha=1.0)
        cp.update_creditworthiness("BD_0", 30.0)
        self.assertAlmostEqual(cp.cw_smoothed("BD_0"), 30.0, places=6)

    def test_geometric_convergence(self):
        """Repeated CW=0 inputs converge geometrically toward 0."""
        cfg, cp = self._make_cp(alpha=0.15)
        for _ in range(50):
            cp.update_creditworthiness("BD_0", 0.0)
        # (1-0.15)^50 ≈ 0.00029 of the initial 100
        self.assertLess(cp.cw_smoothed("BD_0"), 0.1)

    def test_smoothing_damps_oscillation_in_simulation(self):
        """Regression: post-shock smoothed CW does not exhibit a period-2 flip-flop."""
        import numpy as np
        from bookstaber_abm.config import SimConfig
        from bookstaber_abm.simulation.engine import Simulation
        cfg = SimConfig(
            n_assets=6, n_hedge_funds=4, n_bank_dealers=2, n_cash_providers=1,
            n_steps=50, shock_step=30, shock_asset=0, shock_size=-0.20,
            beta=0.1, beta1=0.02, normalise_beta=True, noise_std=0.005,
            hf_max_liq_frac=0.05, bd_max_liq_frac=0.05,
            hf_lev_target=5.0, hf_lev_buffer=5.3, hf_lev_max=5.5,
            bd_lev_target=8, bd_lev_buffer=10, bd_lev_max=13,
            bd_liq_ratio_min=0.025, bd_liq_ratio_target=0.04, phi_cw=4000.0,
            crowding=0.5, hf_funding_squeeze_threshold=1.10,
            cp_max_loan=100000.0, enable_derivatives_desk=False,
            cp_cw_smoothing_alpha=0.15, seed=0,
        )
        history = Simulation(cfg).run()
        # Use FTD as the observable — it's the loan-gate output and the
        # quantity the user perceives as "oscillating"; smoothing should
        # collapse its period-2 amplitude.
        ftd = np.array([r["bd_fundings_from_cp"] for r in history])[:, 0]
        post = ftd[31:45]
        period2_amp = float(np.abs(np.diff(post)).mean())
        self.assertLess(period2_amp, 5000.0,
            f"FTD still oscillating with amplitude {period2_amp:.0f} despite smoothing")


class TestBDFundingSqueeze(unittest.TestCase):
    """
    Paper §3.3 — when CP funding falls short of BD trading-desk need, the BD
    must EDS_k force-sell (mirrors HF funding-squeeze logic in hedge_fund.py).
    """

    def _make_bd(self, **cfg_kw):
        from bookstaber_abm.agents.bank_dealer import BankDealer
        defaults = dict(
            n_assets=3, n_hedge_funds=1, n_bank_dealers=1, n_cash_providers=1,
            n_steps=5, shock_size=0.0, seed=0,
            shock_asset=0,
            bd_lev_target=12, bd_lev_buffer=18, bd_lev_max=20,
            bd_max_liq_frac=1.0,
            hf_funding_squeeze_threshold=0.05,
        )
        defaults.update(cfg_kw)
        cfg = SimConfig(**defaults)
        rng = np.random.default_rng(0)
        bd = BankDealer("BD_test", cfg, rng)
        return cfg, bd

    def test_no_squeeze_when_funding_meets_need(self):
        """Ample funding → no fire sale, normal rebalancing."""
        cfg, bd = self._make_bd()
        prices = np.full(cfg.n_assets, 100.0)
        td_need = bd.total_assets(prices) - bd.capital
        orders = bd.compute_orders(prices, funding_available=td_need * 2.0)
        self.assertFalse(orders.in_fire_sale)
        self.assertTrue(np.allclose(orders.forced, 0.0))

    def test_no_squeeze_when_funding_available_is_none(self):
        """Backward compat: None means no funding constraint applied."""
        cfg, bd = self._make_bd()
        prices = np.full(cfg.n_assets, 100.0)
        orders = bd.compute_orders(prices, funding_available=None)
        self.assertFalse(orders.in_fire_sale)

    def test_squeeze_triggers_fire_sale_targeting_shock_asset(self):
        """funding_available << td_need → EDS_k fire sale on shock asset first."""
        cfg, bd = self._make_bd()
        prices = np.full(cfg.n_assets, 100.0)
        assets_before = bd.total_assets(prices)
        # Starve: CP commits zero. max_supportable = capital, target = capital.
        orders = bd.compute_orders(prices, funding_available=0.0)
        self.assertTrue(orders.in_fire_sale)
        # All selling concentrated on shock_asset (cfg.shock_asset=0).
        self.assertLess(orders.forced[cfg.shock_asset], 0.0)
        for m in range(cfg.n_assets):
            if m != cfg.shock_asset:
                self.assertAlmostEqual(orders.forced[m], 0.0, places=6)
        # Dollar magnitude of forced sale should be roughly (assets - capital),
        # clipped by holdings in the shock asset.
        forced_dollar = -float(orders.forced[cfg.shock_asset]) * prices[cfg.shock_asset]
        shock_holdings_dollar = bd.holdings[cfg.shock_asset] * prices[cfg.shock_asset]
        expected = min(assets_before - bd.capital, shock_holdings_dollar)
        self.assertAlmostEqual(forced_dollar, expected, delta=1.0)

    def test_squeeze_falls_back_to_proportional_when_shock_holdings_zero(self):
        """If BD has no shock-asset holdings, fall back to pro-rata sales."""
        cfg, bd = self._make_bd()
        prices = np.full(cfg.n_assets, 100.0)
        # Zero out shock asset; redistribute its holdings into other assets.
        original_total = bd.holdings.sum()
        bd.holdings[cfg.shock_asset] = 0.0
        # Renormalize so total assets are preserved (keeps leverage realistic).
        other = np.arange(cfg.n_assets) != cfg.shock_asset
        bd.holdings[other] *= original_total / bd.holdings[other].sum()

        orders = bd.compute_orders(prices, funding_available=0.0)
        self.assertTrue(orders.in_fire_sale)
        self.assertAlmostEqual(orders.forced[cfg.shock_asset], 0.0, places=6)
        # All other assets sold (negative qty)
        for m in range(cfg.n_assets):
            if m != cfg.shock_asset:
                self.assertLess(orders.forced[m], 0.0)

    def test_squeeze_threshold_avoids_micro_squeeze(self):
        """Small shortfall within threshold should NOT trigger a fire sale."""
        cfg, bd = self._make_bd(hf_funding_squeeze_threshold=0.20)
        prices = np.full(cfg.n_assets, 100.0)
        assets = bd.total_assets(prices)
        td_need = assets - bd.capital
        # Shortfall is 5% of assets — below the 20% threshold.
        funding_short_by_5pct = td_need - 0.05 * assets
        orders = bd.compute_orders(prices, funding_available=funding_short_by_5pct)
        self.assertFalse(orders.in_fire_sale)

    def test_leverage_breach_and_squeeze_take_tighter_target(self):
        """When both fire — leverage breach + funding squeeze — pick the tighter."""
        cfg, bd = self._make_bd()
        prices = np.full(cfg.n_assets, 100.0)
        # Induce a leverage breach by halving capital (assets unchanged → lev doubles).
        bd.capital *= 0.5
        self.assertGreaterEqual(bd.current_leverage(prices), cfg.bd_lev_max)
        # Funding squeeze targets capital; leverage breach targets capital * lev_buffer.
        # Tighter = funding squeeze (smaller target_assets).
        orders = bd.compute_orders(prices, funding_available=0.0)
        self.assertTrue(orders.in_fire_sale)
        sold_dollar = -float(np.dot(orders.forced, prices))
        lev_breach_only = bd.total_assets(prices) - bd.capital * cfg.bd_lev_buffer
        self.assertGreaterEqual(sold_dollar, lev_breach_only - 1.0)


class TestLiquidityReserve(unittest.TestCase):
    """
    Paper §3.3.4 — verify Eq. 16 (reserve = liq_rate * capital, refreshed each step)
    and Eq. 18 (previous-period LiqDebit reduces capital).
    """

    def _make_bd(self, **cfg_kw):
        from bookstaber_abm.agents.bank_dealer import BankDealer
        defaults = dict(
            n_assets=2, n_hedge_funds=1, n_bank_dealers=1, n_cash_providers=1,
            n_steps=5, shock_size=0.0, seed=0,
        )
        defaults.update(cfg_kw)
        cfg = SimConfig(**defaults)
        rng = np.random.default_rng(0)
        bd = BankDealer("BD_test", cfg, rng)
        return cfg, bd

    def test_initial_reserve_matches_liq_rate_times_capital(self):
        """t=0: LiqReserve = liq_rate * capital."""
        cfg, bd = self._make_bd()
        self.assertAlmostEqual(bd.liquidity_reserve, bd.capital * cfg.bd_liq_rate, places=6)

    def test_liq_debit_reduces_capital_with_one_step_lag(self):
        """Eq. 18: this period's debit reduces capital at the next treasury step."""
        cfg, bd = self._make_bd()
        cap_before = bd.capital

        # Step t: take a debit
        debit = 100.0
        defaulted = bd.step_treasury(ftd=1000.0, liq_debit_this_step=debit)
        self.assertFalse(defaulted)
        # Capital not yet reduced (lagged)
        self.assertAlmostEqual(bd.capital, cap_before, places=6)
        self.assertAlmostEqual(bd.liq_debit, debit, places=6)

        # Step t+1: zero new debit; previous step's debit hits capital now
        bd.step_treasury(ftd=1000.0, liq_debit_this_step=0.0)
        self.assertAlmostEqual(bd.capital, cap_before - debit, places=6)

    def test_reserve_refreshed_each_step_from_current_capital(self):
        """Eq. 16: reserve at step t reflects capital at step t, not initial."""
        cfg, bd = self._make_bd()
        cap0 = bd.capital

        # Simulate a P&L bump (e.g., price gains) by directly raising capital.
        bd.capital = cap0 * 1.5

        bd.step_treasury(ftd=1000.0, liq_debit_this_step=0.0)
        self.assertAlmostEqual(
            bd.liquidity_reserve, bd.capital * cfg.bd_liq_rate, places=6,
            msg="Reserve should refresh from current capital (Eq. 16)",
        )
        # Now drop capital and verify reserve shrinks
        bd.capital = cap0 * 0.5
        bd.step_treasury(ftd=1000.0, liq_debit_this_step=0.0)
        self.assertAlmostEqual(
            bd.liquidity_reserve, bd.capital * cfg.bd_liq_rate, places=6,
        )

    def test_reserve_rebuilds_after_stress_when_capital_recovers(self):
        """A drawn-down reserve should rebuild when capital recovers (was broken pre-fix)."""
        cfg, bd = self._make_bd()
        cap0 = bd.capital
        reserve0 = bd.liquidity_reserve

        # Stress: take a debit close to but below the reserve
        debit = 0.5 * reserve0
        bd.step_treasury(ftd=1000.0, liq_debit_this_step=debit)
        residual = bd.liquidity_reserve
        self.assertLess(residual, reserve0)

        # Next step settles the debit against capital (Eq. 18 lag) and refreshes the reserve
        bd.step_treasury(ftd=1000.0, liq_debit_this_step=0.0)
        self.assertAlmostEqual(bd.capital, cap0 - debit, places=6)
        self.assertAlmostEqual(bd.liquidity_reserve, bd.capital * cfg.bd_liq_rate, places=6)
        # Reserve should be larger than the post-consumption residual from the previous step
        # (because capital, though slightly reduced, is still ample relative to the residual).
        self.assertGreater(bd.liquidity_reserve, residual)

    def test_cw_rebounds_when_liq_ratio_recovers(self):
        """Eq. 21 stateless form: CW returns to 100 once LiqRatio climbs back above LiqRatioMin."""
        cfg, bd = self._make_bd()
        # Pick ftd so the post-consumption residual sits below liq_ratio_min.
        # liq_ratio = (reserve - debit) / ftd; force this < liq_ratio_min.
        reserve = bd.liquidity_reserve
        ftd = reserve / max(cfg.bd_liq_ratio_min * 0.5, 1e-9)  # residual/ftd ≈ 0.5 * min
        debit = 0.0
        bd.step_treasury(ftd=ftd, liq_debit_this_step=debit)
        self.assertLess(bd.liq_ratio, cfg.bd_liq_ratio_min)
        self.assertLess(bd.cw, 100.0)
        cw_stressed = bd.cw

        # Recovery: small ftd → LiqRatio shoots above min → CW must rebound to 100.
        bd.step_treasury(ftd=1.0, liq_debit_this_step=0.0)
        self.assertGreater(bd.liq_ratio, cfg.bd_liq_ratio_min)
        self.assertAlmostEqual(bd.cw, 100.0, places=6)
        self.assertGreater(bd.cw, cw_stressed)

    def test_single_period_debit_exceeds_reserve_triggers_liquidity_default(self):
        """Paper's instantaneous condition: LiqDebit(t) >= LiqReserve(t) → default."""
        cfg, bd = self._make_bd()
        reserve = bd.capital * cfg.bd_liq_rate
        massive_debit = reserve * 2.0

        defaulted = bd.step_treasury(ftd=1000.0, liq_debit_this_step=massive_debit)
        self.assertTrue(defaulted)
        self.assertTrue(bd.liquidity_defaulted)
        self.assertEqual(bd.liquidity_reserve, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
