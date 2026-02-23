"""
Unit Tests -- Stochastic Processes Simulator
=============================================
Tests GBM, Ornstein-Uhlenbeck, CIR, and Heston processes for
correct statistical properties, boundary conditions, and convergence.

Author: Jose Orlando Bobadilla Fuentes, CQF
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.models.processes import (
    GeometricBrownianMotion,
    OrnsteinUhlenbeck,
    CoxIngersollRoss,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def gbm():
    return GeometricBrownianMotion(mu=0.08, sigma=0.20, seed=42)


@pytest.fixture
def ou():
    return OrnsteinUhlenbeck(theta=5.0, mu=0.03, sigma=0.02, seed=42)


@pytest.fixture
def cir():
    return CoxIngersollRoss(kappa=3.0, theta=0.05, sigma=0.10, seed=42)


# ---------------------------------------------------------------------------
# GBM Tests
# ---------------------------------------------------------------------------
class TestGBM:
    """Tests for Geometric Brownian Motion."""

    def test_positive_paths(self, gbm):
        """GBM paths must be strictly positive."""
        paths = gbm.simulate(S0=100, T=1.0, n_paths=5000, n_steps=252)
        assert np.all(paths > 0)

    def test_initial_condition(self, gbm):
        """All paths start at S0."""
        paths = gbm.simulate(S0=50.0, T=1.0, n_paths=1000, n_steps=100)
        np.testing.assert_allclose(paths[:, 0], 50.0)

    def test_expected_value(self, gbm):
        """E[S_T] = S0 * exp(mu * T) under physical measure."""
        S0, T = 100.0, 1.0
        paths = gbm.simulate(S0=S0, T=T, n_paths=100_000, n_steps=252)
        expected = S0 * np.exp(gbm.mu * T)
        empirical = paths[:, -1].mean()
        assert abs(empirical - expected) / expected < 0.02

    def test_log_returns_normality(self, gbm):
        """Log returns of GBM should be approximately normal."""
        from scipy.stats import shapiro
        paths = gbm.simulate(S0=100, T=1.0, n_paths=1000, n_steps=252)
        log_returns = np.log(paths[:, -1] / paths[:, 0])
        # Shapiro-Wilk on subsample
        _, p_value = shapiro(log_returns[:500])
        assert p_value > 0.01  # Should not reject normality

    def test_variance_scales_with_time(self, gbm):
        """Var[log(S_T/S_0)] = sigma^2 * T."""
        paths = gbm.simulate(S0=100, T=2.0, n_paths=50_000, n_steps=504)
        log_returns = np.log(paths[:, -1] / paths[:, 0])
        expected_var = gbm.sigma**2 * 2.0
        empirical_var = log_returns.var()
        np.testing.assert_allclose(empirical_var, expected_var, rtol=0.05)


# ---------------------------------------------------------------------------
# Ornstein-Uhlenbeck Tests
# ---------------------------------------------------------------------------
class TestOU:
    """Tests for Ornstein-Uhlenbeck (Vasicek) process."""

    def test_mean_reversion(self, ou):
        """Terminal distribution should center near long-run mean."""
        paths = ou.simulate(X0=0.10, T=5.0, n_paths=50_000, n_steps=1000)
        terminal_mean = paths[:, -1].mean()
        np.testing.assert_allclose(terminal_mean, ou.mu, atol=0.002)

    def test_stationary_variance(self, ou):
        """Stationary variance = sigma^2 / (2 * theta)."""
        paths = ou.simulate(X0=ou.mu, T=10.0, n_paths=50_000, n_steps=2000)
        terminal_var = paths[:, -1].var()
        expected_var = ou.sigma**2 / (2 * ou.theta)
        np.testing.assert_allclose(terminal_var, expected_var, rtol=0.10)

    def test_convergence_from_above(self, ou):
        """Starting above mu, process should drift down."""
        paths = ou.simulate(X0=0.10, T=2.0, n_paths=10_000, n_steps=500)
        assert paths[:, -1].mean() < 0.10  # Closer to mu=0.03

    def test_convergence_from_below(self, ou):
        """Starting below mu, process should drift up."""
        paths = ou.simulate(X0=0.001, T=2.0, n_paths=10_000, n_steps=500)
        assert paths[:, -1].mean() > 0.001


# ---------------------------------------------------------------------------
# CIR Tests
# ---------------------------------------------------------------------------
class TestCIR:
    """Tests for Cox-Ingersoll-Ross process."""

    def test_feller_condition_positive(self, cir):
        """If 2*kappa*theta > sigma^2 (Feller), paths stay positive."""
        feller = 2 * cir.kappa * cir.theta > cir.sigma**2
        assert feller, "Test parameters should satisfy Feller condition"
        paths = cir.simulate(X0=0.05, T=5.0, n_paths=10_000, n_steps=1000)
        assert np.all(paths >= 0)

    def test_mean_reversion_cir(self, cir):
        """Terminal mean should approach long-run theta."""
        paths = cir.simulate(X0=0.10, T=10.0, n_paths=50_000, n_steps=2000)
        terminal_mean = paths[:, -1].mean()
        np.testing.assert_allclose(terminal_mean, cir.theta, atol=0.005)

    def test_cir_nonnegative(self, cir):
        """CIR process should remain non-negative."""
        paths = cir.simulate(X0=0.01, T=5.0, n_paths=20_000, n_steps=2000)
        assert np.all(paths >= 0)

    def test_initial_condition_cir(self, cir):
        """All CIR paths start at X0."""
        paths = cir.simulate(X0=0.045, T=1.0, n_paths=1000, n_steps=100)
        np.testing.assert_allclose(paths[:, 0], 0.045)


# ---------------------------------------------------------------------------
# Cross-Process Tests
# ---------------------------------------------------------------------------
class TestCrossProcess:
    """General tests applicable to all processes."""

    def test_reproducibility_gbm(self):
        """Same seed -> identical GBM paths."""
        g1 = GeometricBrownianMotion(mu=0.08, sigma=0.20, seed=99)
        g2 = GeometricBrownianMotion(mu=0.08, sigma=0.20, seed=99)
        p1 = g1.simulate(S0=100, T=1.0, n_paths=50, n_steps=100)
        p2 = g2.simulate(S0=100, T=1.0, n_paths=50, n_steps=100)
        np.testing.assert_array_equal(p1, p2)

    def test_reproducibility_ou(self):
        """Same seed -> identical OU paths."""
        o1 = OrnsteinUhlenbeck(theta=5.0, mu=0.03, sigma=0.02, seed=99)
        o2 = OrnsteinUhlenbeck(theta=5.0, mu=0.03, sigma=0.02, seed=99)
        p1 = o1.simulate(X0=0.05, T=1.0, n_paths=50, n_steps=100)
        p2 = o2.simulate(X0=0.05, T=1.0, n_paths=50, n_steps=100)
        np.testing.assert_array_equal(p1, p2)

    def test_path_shape_gbm(self, gbm):
        """Output shape should be (n_paths, n_steps+1)."""
        paths = gbm.simulate(S0=100, T=1.0, n_paths=200, n_steps=50)
        assert paths.shape == (200, 51)

    def test_path_shape_ou(self, ou):
        """Output shape should be (n_paths, n_steps+1)."""
        paths = ou.simulate(X0=0.05, T=1.0, n_paths=200, n_steps=50)
        assert paths.shape == (200, 51)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
