"""
Unit Tests -- Portfolio Optimizer Black-Litterman
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.markowitz import MarkowitzOptimizer
from src.models.black_litterman import BlackLittermanModel
from src.models.mean_cvar import MeanCVaROptimizer
from src.models.risk_parity import RiskParityOptimizer


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 4
    names = ["SPY", "TLT", "GLD", "QQQ"]
    mu = np.array([0.10, 0.04, 0.06, 0.12])
    A = np.random.randn(n, n) * 0.1
    Sigma = A @ A.T + 0.01 * np.eye(n)
    return mu, Sigma, names


@pytest.fixture
def sample_returns():
    np.random.seed(42)
    n_days, n_assets = 500, 4
    names = ["SPY", "TLT", "GLD", "QQQ"]
    mu = np.array([0.10, 0.04, 0.06, 0.12]) / 252
    vols = np.array([0.16, 0.07, 0.15, 0.20]) / np.sqrt(252)
    returns = np.random.multivariate_normal(mu, np.diag(vols**2), size=n_days)
    return returns, names


# ============================================================================
# TEST: MARKOWITZ
# ============================================================================

class TestMarkowitz:

    def test_min_variance_weights_sum(self, sample_data):
        mu, Sigma, names = sample_data
        opt = MarkowitzOptimizer(mu, Sigma, asset_names=names)
        result = opt.min_variance()
        assert abs(result.weights.sum() - 1.0) < 1e-6

    def test_min_variance_non_negative(self, sample_data):
        mu, Sigma, names = sample_data
        opt = MarkowitzOptimizer(mu, Sigma, asset_names=names)
        result = opt.min_variance()
        assert np.all(result.weights >= -1e-6)

    def test_max_sharpe(self, sample_data):
        mu, Sigma, names = sample_data
        opt = MarkowitzOptimizer(mu, Sigma, asset_names=names)
        result = opt.max_sharpe(risk_free_rate=0.03)
        assert result.sharpe_ratio > 0
        assert abs(result.weights.sum() - 1.0) < 1e-6

    def test_efficient_frontier(self, sample_data):
        mu, Sigma, names = sample_data
        opt = MarkowitzOptimizer(mu, Sigma, asset_names=names)
        frontier = opt.efficient_frontier(n_points=10, risk_free_rate=0.0)
        assert len(frontier) == 10


# ============================================================================
# TEST: BLACK-LITTERMAN
# ============================================================================

class TestBlackLitterman:

    def test_implied_returns(self, sample_data):
        mu, Sigma, names = sample_data
        w_mkt = np.array([0.4, 0.2, 0.1, 0.3])
        bl = BlackLittermanModel(Sigma, market_cap_weights=w_mkt, asset_names=names)
        pi = bl.implied_returns()
        assert len(pi) == 4
        assert not np.any(np.isnan(pi))

    def test_posterior_no_views(self, sample_data):
        mu, Sigma, names = sample_data
        w_mkt = np.array([0.4, 0.2, 0.1, 0.3])
        bl = BlackLittermanModel(Sigma, market_cap_weights=w_mkt, asset_names=names)
        mu_bl, Sigma_bl = bl.posterior()
        assert len(mu_bl) == 4
        assert Sigma_bl.shape == (4, 4)

    def test_posterior_with_absolute_view(self, sample_data):
        mu, Sigma, names = sample_data
        w_mkt = np.array([0.4, 0.2, 0.1, 0.3])
        bl = BlackLittermanModel(Sigma, market_cap_weights=w_mkt, asset_names=names)
        bl.add_absolute_view(asset_idx=0, return_view=0.12, confidence=0.8)
        mu_bl, Sigma_bl = bl.posterior()
        assert len(mu_bl) == 4
        assert Sigma_bl.shape == (4, 4)

    def test_posterior_with_relative_view(self, sample_data):
        mu, Sigma, names = sample_data
        w_mkt = np.array([0.4, 0.2, 0.1, 0.3])
        bl = BlackLittermanModel(Sigma, market_cap_weights=w_mkt, asset_names=names)
        bl.add_relative_view(long_idx=3, short_idx=1, spread=0.05, confidence=0.6)
        mu_bl, Sigma_bl = bl.posterior()
        assert len(mu_bl) == 4


# ============================================================================
# TEST: MEAN-CVaR
# ============================================================================

class TestMeanCVaR:

    def test_optimization(self, sample_returns):
        returns, names = sample_returns
        opt = MeanCVaROptimizer(returns, confidence_level=0.95, asset_names=names)
        result = opt.optimize(target_return=0.0003)
        assert abs(result.weights.sum() - 1.0) < 1e-4
        assert result.cvar > 0

    def test_weights_bounded(self, sample_returns):
        returns, names = sample_returns
        opt = MeanCVaROptimizer(returns, asset_names=names)
        result = opt.optimize(target_return=0.0003)
        assert np.all(result.weights >= -1e-6)
        assert np.all(result.weights <= 1.0 + 1e-6)


# ============================================================================
# TEST: RISK PARITY
# ============================================================================

class TestRiskParity:

    def test_equal_risk_contribution(self, sample_data):
        mu, Sigma, names = sample_data
        rp = RiskParityOptimizer(Sigma, asset_names=names)
        result = rp.optimize()
        assert abs(result.weights.sum() - 1.0) < 1e-4
        rc = result.risk_contributions
        rc_norm = rc / rc.sum()
        target = 1.0 / len(names)
        assert np.allclose(rc_norm, target, atol=0.05)

    def test_weights_positive(self, sample_data):
        mu, Sigma, names = sample_data
        rp = RiskParityOptimizer(Sigma, asset_names=names)
        result = rp.optimize()
        assert np.all(result.weights > 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
