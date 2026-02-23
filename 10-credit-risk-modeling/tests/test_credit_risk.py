"""
Unit Tests -- Credit Risk Modeling
====================================
Tests Merton model, hazard rates, Vasicek formula, CDS pricing,
CreditMetrics migration, and Credit VaR Monte Carlo.

Author: Jose Orlando Bobadilla Fuentes, CQF
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from scipy.stats import norm

from src.models.merton import MertonModel
from src.models.reduced_form import HazardRateModel
from src.models.vasicek import VasicekPortfolioModel
from src.models.cds_pricing import CDSPricer
from src.models.creditmetrics import CreditMetricsEngine
from src.credit_var import CreditVaREngine


# -----------------------------------------------------------------------
# Merton Model Tests
# -----------------------------------------------------------------------
class TestMertonModel:

    def test_equity_plus_debt_equals_assets(self):
        """V = E + B must hold (balance sheet identity)."""
        m = MertonModel(risk_free_rate=0.05)
        V, sigma_V, D, T = 150.0, 0.25, 100.0, 1.0
        E = m.price_equity(V, sigma_V, D, T)
        B = m.price_debt(V, sigma_V, D, T)
        np.testing.assert_allclose(E + B, V, rtol=1e-10)

    def test_pd_increases_with_leverage(self):
        """Higher leverage -> higher default probability."""
        m = MertonModel(risk_free_rate=0.05)
        pd_low = m.default_probability(200, 0.25, 100, 1.0)["risk_neutral"]
        pd_high = m.default_probability(120, 0.25, 100, 1.0)["risk_neutral"]
        assert pd_high > pd_low

    def test_pd_increases_with_volatility(self):
        """Higher asset volatility -> higher PD."""
        m = MertonModel(risk_free_rate=0.05)
        pd_low = m.default_probability(150, 0.15, 100, 1.0)["risk_neutral"]
        pd_high = m.default_probability(150, 0.50, 100, 1.0)["risk_neutral"]
        assert pd_high > pd_low

    def test_calibration_roundtrip(self):
        """Calibrate from equity, verify equity reprices correctly."""
        m = MertonModel(risk_free_rate=0.05)
        res = m.calibrate_from_equity(E_market=50, sigma_E=0.40, D=80, T=1.0)
        np.testing.assert_allclose(res.equity_value, 50.0, rtol=0.01)

    def test_credit_spread_positive(self):
        """Credit spread must be non-negative."""
        m = MertonModel(risk_free_rate=0.05)
        spread = m.credit_spread(150, 0.25, 100, 1.0)
        assert spread >= 0

    def test_dd_positive_for_low_leverage(self):
        """Distance to default should be positive when V >> D."""
        m = MertonModel(risk_free_rate=0.05)
        dd = m.distance_to_default(200, 0.20, 50, 1.0, mu=0.08)
        assert dd > 0


# -----------------------------------------------------------------------
# Reduced-Form Tests
# -----------------------------------------------------------------------
class TestHazardRate:

    def test_constant_hazard_from_spread(self):
        """lambda = spread / (1-R) for constant hazard."""
        h = HazardRateModel(recovery_rate=0.40)
        lam = h.constant_hazard(0.0120)
        np.testing.assert_allclose(lam, 0.02, rtol=1e-10)

    def test_survival_decreases_with_time(self):
        """Q(0,T) should decrease as T increases."""
        h = HazardRateModel()
        q1 = h.survival_probability(0.02, 1.0)
        q5 = h.survival_probability(0.02, 5.0)
        assert q5 < q1

    def test_bootstrap_hazard_rates_positive(self):
        """Bootstrapped hazard rates should be positive."""
        h = HazardRateModel(recovery_rate=0.40)
        tenors = np.array([1.0, 3.0, 5.0])
        spreads = np.array([0.0080, 0.0120, 0.0150])
        rf = np.array([0.04, 0.043, 0.045])
        curve = h.bootstrap_hazard_rates(spreads, tenors, rf)
        assert np.all(curve.hazard_rates > 0)

    def test_survival_curve_monotone(self):
        """Survival probabilities must be monotonically decreasing."""
        h = HazardRateModel(recovery_rate=0.40)
        tenors = np.array([1.0, 2.0, 3.0, 5.0])
        spreads = np.array([0.0080, 0.0100, 0.0115, 0.0130])
        rf = np.array([0.04, 0.042, 0.043, 0.045])
        curve = h.bootstrap_hazard_rates(spreads, tenors, rf)
        diffs = np.diff(curve.survival_probs)
        assert np.all(diffs <= 0)


# -----------------------------------------------------------------------
# Vasicek Tests
# -----------------------------------------------------------------------
class TestVasicek:

    def test_expected_loss(self):
        """EL = PD * LGD."""
        v = VasicekPortfolioModel(pd=0.02, lgd=0.45, rho=0.20)
        np.testing.assert_allclose(v.expected_loss(), 0.009, rtol=1e-10)

    def test_var_exceeds_el(self):
        """VaR at any confidence level must exceed expected loss."""
        v = VasicekPortfolioModel(pd=0.02, lgd=0.45, rho=0.20)
        var = v.loss_quantile(0.999)
        assert var > v.expected_loss()

    def test_higher_rho_higher_var(self):
        """Higher correlation -> higher tail risk."""
        v1 = VasicekPortfolioModel(pd=0.02, lgd=0.45, rho=0.10)
        v2 = VasicekPortfolioModel(pd=0.02, lgd=0.45, rho=0.40)
        assert v2.loss_quantile(0.999) > v1.loss_quantile(0.999)

    def test_cdf_monotone(self):
        """Loss CDF must be monotonically increasing."""
        v = VasicekPortfolioModel(pd=0.02, lgd=0.45, rho=0.20)
        x_grid = np.linspace(0.001, 0.44, 100)
        cdf_vals = [v.loss_distribution_cdf(x) for x in x_grid]
        assert all(cdf_vals[i] <= cdf_vals[i+1] for i in range(len(cdf_vals)-1))

    def test_basel_correlation_decreasing(self):
        """Basel correlation decreases with PD for corporates."""
        rho_low_pd = VasicekPortfolioModel.basel_correlation(0.001)
        rho_high_pd = VasicekPortfolioModel.basel_correlation(0.10)
        assert rho_low_pd > rho_high_pd


# -----------------------------------------------------------------------
# CDS Pricing Tests
# -----------------------------------------------------------------------
class TestCDS:

    def test_fair_spread_positive(self):
        """Fair CDS spread must be positive."""
        cds = CDSPricer(recovery_rate=0.40)
        Q = np.array([0.98, 0.96, 0.93, 0.88, 0.84])
        rf = np.array([0.04, 0.042, 0.043, 0.045, 0.046])
        tenors = np.array([1.0, 2.0, 3.0, 5.0, 7.0])
        spread = cds.fair_spread(Q, rf, tenors)
        assert spread > 0

    def test_mtm_zero_at_fair_spread(self):
        """MTM should be ~0 when contract spread = fair spread."""
        cds = CDSPricer(recovery_rate=0.40)
        Q = np.array([0.98, 0.96, 0.93])
        rf = np.array([0.04, 0.042, 0.043])
        tenors = np.array([1.0, 2.0, 3.0])
        fair_s = cds.fair_spread(Q, rf, tenors)
        val = cds.mark_to_market(fair_s, Q, rf, tenors, 1e7)
        assert abs(val.mtm_value) < 1000  # Near zero


# -----------------------------------------------------------------------
# CreditMetrics Tests
# -----------------------------------------------------------------------
class TestCreditMetrics:

    def test_simulation_runs(self):
        """CreditMetrics simulation should complete without error."""
        cm = CreditMetricsEngine(seed=42)
        n = 20
        exp = np.full(n, 1e6)
        ratings = np.full(n, 3, dtype=int)  # BBB
        lgds = np.full(n, 0.45)
        corr = cm.build_correlation_matrix(n, 0.25, 0.10, [0]*n)
        result = cm.simulate(exp, ratings, lgds, corr, n_simulations=1000)
        assert result.expected_loss >= 0
        assert len(result.portfolio_losses) == 1000

    def test_correlation_matrix_positive_definite(self):
        """Correlation matrix must be positive semi-definite."""
        cm = CreditMetricsEngine(seed=42)
        corr = cm.build_correlation_matrix(50, 0.30, 0.10, list(range(5))*10)
        eigvals = np.linalg.eigvalsh(corr)
        assert np.all(eigvals >= -1e-10)


# -----------------------------------------------------------------------
# Credit VaR Tests
# -----------------------------------------------------------------------
class TestCreditVaR:

    def test_var_hierarchy(self):
        """VaR_95 < VaR_99 < VaR_999."""
        engine = CreditVaREngine(seed=42)
        n = 50
        res = engine.simulate(
            exposures=np.full(n, 1e6),
            pds=np.full(n, 0.03),
            lgds=np.full(n, 0.45),
            rhos=np.full(n, 0.20),
            n_simulations=20_000,
        )
        assert res.credit_var_95 <= res.credit_var_99 <= res.credit_var_999

    def test_stress_increases_loss(self):
        """Stressed scenario should produce higher losses than base."""
        engine = CreditVaREngine(seed=42)
        n = 30
        exp = np.full(n, 1e6)
        pds = np.full(n, 0.02)
        lgds = np.full(n, 0.45)
        rhos = np.full(n, 0.20)
        stress = {"recession": {"pd_mult": 3.0, "lgd_add": 0.10}}
        results = engine.stress_test(exp, pds, lgds, rhos, stress, 10_000)
        assert results["recession"].expected_loss > results["base"].expected_loss

    def test_concentration_hhi(self):
        """Equal exposures should have HHI = 1/N."""
        engine = CreditVaREngine(seed=42)
        n = 100
        conc = engine.concentration_risk(
            np.full(n, 1e6), np.full(n, 0.02), np.full(n, 0.45)
        )
        np.testing.assert_allclose(conc["hhi"], 1.0/n, rtol=1e-10)
        np.testing.assert_allclose(conc["effective_n"], n, rtol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
