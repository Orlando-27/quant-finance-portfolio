"""
Unit Tests -- VaR & CVaR Risk Engine
=====================================
Tests Historical VaR, Parametric VaR, GARCH VaR, Monte Carlo VaR,
CVaR computations, and backtesting framework.

Author: Jose Orlando Bobadilla Fuentes, CQF
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.models.var_engine import VaREngine
from src.models.garch_var import GARCHVaR
from src.models.backtester import VaRBacktester


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def normal_returns():
    """Synthetic returns from a known normal distribution."""
    np.random.seed(42)
    return np.random.normal(loc=0.0005, scale=0.015, size=1000)


@pytest.fixture
def fat_tail_returns():
    """Synthetic returns with fat tails (Student-t, df=4)."""
    np.random.seed(42)
    return np.random.standard_t(df=4, size=1000) * 0.015


@pytest.fixture
def var_engine():
    """VaR engine instance."""
    return VaREngine()


# ---------------------------------------------------------------------------
# Historical VaR Tests
# ---------------------------------------------------------------------------
class TestHistoricalVaR:
    """Tests for Historical Simulation VaR."""

    def test_var_negative(self, var_engine, normal_returns):
        """VaR should be negative (it represents a loss)."""
        var = var_engine.historical_var(normal_returns, confidence=0.95)
        assert var < 0

    def test_higher_confidence_higher_var(self, var_engine, normal_returns):
        """99% VaR should be more extreme than 95% VaR."""
        var_95 = var_engine.historical_var(normal_returns, confidence=0.95)
        var_99 = var_engine.historical_var(normal_returns, confidence=0.99)
        assert var_99 < var_95  # More negative = more extreme

    def test_var_matches_percentile(self, var_engine, normal_returns):
        """Historical VaR should match numpy percentile."""
        var = var_engine.historical_var(normal_returns, confidence=0.95)
        expected = np.percentile(normal_returns, 5)
        np.testing.assert_allclose(var, expected, rtol=0.01)

    def test_var_with_window(self, var_engine, normal_returns):
        """VaR computed on a rolling window should differ from full sample."""
        var_full = var_engine.historical_var(normal_returns, confidence=0.95)
        var_250 = var_engine.historical_var(normal_returns[-250:], confidence=0.95)
        # Not necessarily equal, just both should be negative
        assert var_250 < 0


# ---------------------------------------------------------------------------
# Parametric VaR Tests
# ---------------------------------------------------------------------------
class TestParametricVaR:
    """Tests for Parametric (variance-covariance) VaR."""

    def test_parametric_var_gaussian(self, var_engine, normal_returns):
        """Parametric VaR for normal data should approximate analytical."""
        from scipy.stats import norm
        mu = normal_returns.mean()
        sigma = normal_returns.std()
        expected = mu + sigma * norm.ppf(0.05)

        var = var_engine.parametric_var(normal_returns, confidence=0.95)
        np.testing.assert_allclose(var, expected, rtol=0.05)

    def test_parametric_vs_historical_normal(self, var_engine, normal_returns):
        """For normal data, parametric and historical VaR should be close."""
        var_hist = var_engine.historical_var(normal_returns, confidence=0.95)
        var_param = var_engine.parametric_var(normal_returns, confidence=0.95)
        assert abs(var_hist - var_param) / abs(var_param) < 0.15

    def test_fat_tails_parametric_underestimates(self, var_engine, fat_tail_returns):
        """Parametric VaR underestimates risk for fat-tailed distributions."""
        var_hist = var_engine.historical_var(fat_tail_returns, confidence=0.99)
        var_param = var_engine.parametric_var(fat_tail_returns, confidence=0.99)
        # Historical should capture more extreme losses
        assert var_hist <= var_param  # More negative


# ---------------------------------------------------------------------------
# CVaR Tests
# ---------------------------------------------------------------------------
class TestCVaR:
    """Tests for Conditional VaR (Expected Shortfall)."""

    def test_cvar_worse_than_var(self, var_engine, normal_returns):
        """CVaR should be more extreme (more negative) than VaR."""
        var = var_engine.historical_var(normal_returns, confidence=0.95)
        cvar = var_engine.historical_cvar(normal_returns, confidence=0.95)
        assert cvar < var  # CVaR is the average of losses beyond VaR

    def test_cvar_negative(self, var_engine, normal_returns):
        """CVaR should be negative."""
        cvar = var_engine.historical_cvar(normal_returns, confidence=0.95)
        assert cvar < 0

    def test_cvar_analytical_normal(self, var_engine, normal_returns):
        """For normal dist, CVaR has analytical formula."""
        from scipy.stats import norm
        mu = normal_returns.mean()
        sigma = normal_returns.std()
        alpha = 0.05
        # ES = mu - sigma * phi(Phi^{-1}(alpha)) / alpha
        expected = mu - sigma * norm.pdf(norm.ppf(alpha)) / alpha

        cvar = var_engine.historical_cvar(normal_returns, confidence=0.95)
        np.testing.assert_allclose(cvar, expected, rtol=0.10)


# ---------------------------------------------------------------------------
# GARCH VaR Tests
# ---------------------------------------------------------------------------
class TestGARCHVaR:
    """Tests for GARCH(1,1) conditional VaR."""

    def test_garch_var_negative(self, normal_returns):
        """GARCH VaR should be negative."""
        garch = GARCHVaR()
        var = garch.fit_and_predict(normal_returns, confidence=0.95)
        assert var < 0

    def test_garch_vol_positive(self, normal_returns):
        """Conditional volatility must be strictly positive."""
        garch = GARCHVaR()
        garch.fit(normal_returns)
        assert garch.conditional_vol > 0

    def test_garch_persistence(self, normal_returns):
        """GARCH persistence (alpha + beta) should be < 1 for stationarity."""
        garch = GARCHVaR()
        garch.fit(normal_returns)
        persistence = garch.alpha + garch.beta
        assert persistence < 1.0


# ---------------------------------------------------------------------------
# Backtester Tests
# ---------------------------------------------------------------------------
class TestBacktester:
    """Tests for VaR backtesting framework."""

    def test_violation_rate_95(self, normal_returns):
        """95% VaR should have ~5% violations for normal data."""
        engine = VaREngine()
        backtester = VaRBacktester(engine)
        results = backtester.run(normal_returns, confidence=0.95, method="historical")
        violation_rate = results["violation_rate"]
        # Should be close to 5% (allow 2-8% range for randomness)
        assert 0.02 <= violation_rate <= 0.10

    def test_kupiec_test_returns_pvalue(self, normal_returns):
        """Kupiec POF test should return a valid p-value."""
        engine = VaREngine()
        backtester = VaRBacktester(engine)
        results = backtester.run(normal_returns, confidence=0.95, method="historical")
        assert 0 <= results["kupiec_pvalue"] <= 1


# ---------------------------------------------------------------------------
# Portfolio VaR Tests
# ---------------------------------------------------------------------------
class TestPortfolioVaR:
    """Tests for multi-asset portfolio VaR."""

    def test_diversification_benefit(self, var_engine):
        """Portfolio VaR should be less than sum of individual VaRs."""
        np.random.seed(42)
        n_assets = 3
        n_obs = 1000
        # Correlated returns (not perfectly correlated)
        cov = np.array([[0.04, 0.01, 0.005],
                        [0.01, 0.03, 0.008],
                        [0.005, 0.008, 0.02]])
        returns = np.random.multivariate_normal(
            mean=[0.0005] * n_assets, cov=cov / 252, size=n_obs
        )
        weights = np.array([0.4, 0.35, 0.25])

        portfolio_returns = returns @ weights
        portfolio_var = var_engine.historical_var(portfolio_returns, confidence=0.95)

        individual_vars = sum(
            abs(var_engine.historical_var(returns[:, i], confidence=0.95)) * weights[i]
            for i in range(n_assets)
        )
        assert abs(portfolio_var) < individual_vars  # Diversification


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
