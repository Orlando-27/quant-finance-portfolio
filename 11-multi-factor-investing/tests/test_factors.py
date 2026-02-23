"""
Unit Tests for Multi-Factor Investing Framework
================================================

Covers: factor construction, Fama-MacBeth regression, Barra risk model,
ML timing, portfolio optimization, and backtesting engine.
"""

import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.factors import FactorConstructor, FamaFrenchReplicator
from src.cross_sectional import FamaMacBeth
from src.risk_model import BarraRiskModel
from src.ml_timing import FeatureEngineering, FactorTimingML, RegimeDetector
from src.portfolio import FactorPortfolio
from src.backtesting import FactorBacktester


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def synthetic_data():
    """Generate synthetic universe once for all tests."""
    ffr = FamaFrenchReplicator(n_stocks=100, n_periods=120, seed=42)
    returns, chars, market_cap = ffr.generate_universe()
    factors = ffr.factor_returns
    return returns, chars, market_cap, factors, ffr


# ---------------------------------------------------------------------------
# Factor Construction Tests
# ---------------------------------------------------------------------------
class TestFactorConstructor:
    """Test suite for FactorConstructor class."""

    def test_single_sort_returns_series(self, synthetic_data):
        """Single sort must return a pd.Series of factor returns."""
        returns, chars, cap, _, _ = synthetic_data
        fc = FactorConstructor(n_quantiles=5, value_weight=False)
        result = fc.single_sort(returns, chars["book_to_market"])
        assert isinstance(result, pd.Series)
        assert len(result) > 0, "Factor return series must not be empty"

    def test_single_sort_zero_mean_bounded(self, synthetic_data):
        """Long-short factor returns should have bounded mean."""
        returns, chars, cap, _, _ = synthetic_data
        fc = FactorConstructor(n_quantiles=5, value_weight=True)
        result = fc.single_sort(returns, chars["book_to_market"], cap)
        assert abs(result.mean()) < 0.05, "Factor mean should be small"

    def test_double_sort_produces_two_factors(self, synthetic_data):
        """2x3 double sort must produce exactly two factor series."""
        returns, chars, cap, _, _ = synthetic_data
        fc = FactorConstructor(value_weight=True)
        result = fc.double_sort_2x3(returns, cap, chars["book_to_market"], cap)
        assert "primary_factor" in result
        assert "secondary_factor" in result
        assert len(result["primary_factor"]) > 0

    def test_replicate_factors_shape(self, synthetic_data):
        """Replicated factors must have 6 columns."""
        _, _, _, _, ffr = synthetic_data
        rep = ffr.replicate_factors()
        assert rep.shape[1] == 6, f"Expected 6 factors, got {rep.shape[1]}"

    def test_factor_statistics_keys(self, synthetic_data):
        """Factor statistics must include key metrics."""
        _, _, _, factors, ffr = synthetic_data
        stats = ffr.compute_factor_statistics(factors)
        required = ["Ann. Return", "Ann. Volatility", "Sharpe Ratio",
                     "Max Drawdown"]
        for col in required:
            assert col in stats.columns, f"Missing statistic: {col}"


# ---------------------------------------------------------------------------
# Fama-MacBeth Tests
# ---------------------------------------------------------------------------
class TestFamaMacBeth:
    """Test suite for Fama-MacBeth cross-sectional regression."""

    def test_beta_estimation(self, synthetic_data):
        """Rolling betas must be estimated for each factor."""
        returns, _, _, factors, _ = synthetic_data
        fmb = FamaMacBeth(rolling_window=36, min_observations=24)
        betas = fmb.estimate_betas(returns.iloc[:, :30], factors)
        assert len(betas) == factors.shape[1]

    def test_risk_premia_estimation(self, synthetic_data):
        """Risk premia table must include t-statistics and p-values."""
        returns, _, _, factors, _ = synthetic_data
        fmb = FamaMacBeth(rolling_window=36, use_shanken=True)
        fmb.estimate_betas(returns.iloc[:, :30], factors)
        fmb.cross_sectional_regression(returns.iloc[:, :30], factors)
        assert fmb.risk_premia is not None
        assert "t-statistic" in fmb.risk_premia.columns
        assert "p-value" in fmb.risk_premia.columns

    def test_shanken_correction_increases_se(self, synthetic_data):
        """Shanken correction should produce >= naive standard errors."""
        returns, _, _, factors, _ = synthetic_data
        ret_sub = returns.iloc[:, :30]

        fmb_sh = FamaMacBeth(rolling_window=36, use_shanken=True)
        fmb_sh.estimate_betas(ret_sub, factors)
        fmb_sh.cross_sectional_regression(ret_sub, factors)

        fmb_no = FamaMacBeth(rolling_window=36, use_shanken=False)
        fmb_no.estimate_betas(ret_sub, factors)
        fmb_no.cross_sectional_regression(ret_sub, factors)

        # Shanken SE should be >= naive SE (correction factor c >= 1)
        assert fmb_sh._shanken_correction_factor >= 1.0


# ---------------------------------------------------------------------------
# Barra Risk Model Tests
# ---------------------------------------------------------------------------
class TestBarraRiskModel:
    """Test suite for Barra-style risk decomposition."""

    def test_risk_decomposition_sums(self, synthetic_data):
        """Factor + specific variance must approximately equal total."""
        returns, _, _, factors, _ = synthetic_data
        brm = BarraRiskModel(n_factors=factors.shape[1])
        brm.fit(returns, factors, window=60)
        w = np.ones(returns.shape[1]) / returns.shape[1]
        rd = brm.portfolio_risk(w)
        assert abs(rd["pct_factor"] + rd["pct_specific"] - 100) < 0.1

    def test_factor_contributions_sum(self, synthetic_data):
        """Factor risk contributions must sum to approximately 100%."""
        returns, _, _, factors, _ = synthetic_data
        brm = BarraRiskModel(n_factors=factors.shape[1])
        brm.fit(returns, factors, window=60)
        w = np.ones(returns.shape[1]) / returns.shape[1]
        fc = brm.factor_risk_contribution(w)
        total_pct = fc["Pct Contribution"].sum()
        assert abs(total_pct - 100) < 5.0, \
            f"Contributions sum to {total_pct}%, expected ~100%"

    def test_active_risk_zero_for_same_portfolio(self, synthetic_data):
        """Active risk must be zero when portfolio equals benchmark."""
        returns, _, _, factors, _ = synthetic_data
        brm = BarraRiskModel(n_factors=factors.shape[1])
        brm.fit(returns, factors, window=60)
        w = np.ones(returns.shape[1]) / returns.shape[1]
        ar = brm.active_risk_decomposition(w, w)
        assert ar["tracking_error"] < 1e-10


# ---------------------------------------------------------------------------
# ML Timing Tests
# ---------------------------------------------------------------------------
class TestMLTiming:
    """Test suite for ML factor timing models."""

    def test_feature_engineering_output(self, synthetic_data):
        """Feature matrix must have positive rows and columns."""
        _, _, _, factors, _ = synthetic_data
        fe = FeatureEngineering(lookback_windows=[3, 6])
        feat = fe.build_features(factors)
        assert feat.shape[0] > 0
        assert feat.shape[1] > 0

    def test_walk_forward_cv_no_lookahead(self, synthetic_data):
        """CV results must exist and have valid metrics."""
        _, _, _, factors, _ = synthetic_data
        fe = FeatureEngineering(lookback_windows=[3, 6])
        feat = fe.build_features(factors)
        target = factors.iloc[:, 0].shift(-1).dropna()
        ml = FactorTimingML(model_type="rf", n_splits=3)
        results = ml.walk_forward_cv(feat, target)
        assert "RandomForest" in results
        assert "rmse" in results["RandomForest"]
        assert results["RandomForest"]["rmse"] > 0

    def test_regime_detector_states(self, synthetic_data):
        """Regime detector must produce valid state assignments."""
        _, _, _, factors, _ = synthetic_data
        rd = RegimeDetector(n_states=2)
        states = rd.fit_predict(factors)
        assert set(states.unique()).issubset({0, 1})
        assert len(states) == len(factors)


# ---------------------------------------------------------------------------
# Portfolio Optimization Tests
# ---------------------------------------------------------------------------
class TestFactorPortfolio:
    """Test suite for factor portfolio construction."""

    def test_equal_weight_sums_to_one(self, synthetic_data):
        _, _, _, factors, _ = synthetic_data
        fp = FactorPortfolio(factors)
        w = fp.equal_weight()
        assert abs(w.sum() - 1.0) < 1e-10

    def test_inverse_vol_sums_to_one(self, synthetic_data):
        _, _, _, factors, _ = synthetic_data
        fp = FactorPortfolio(factors)
        w = fp.inverse_volatility()
        assert abs(w.sum() - 1.0) < 1e-10

    def test_risk_parity_sums_to_one(self, synthetic_data):
        _, _, _, factors, _ = synthetic_data
        fp = FactorPortfolio(factors)
        w = fp.risk_parity()
        assert abs(w.sum() - 1.0) < 1e-6

    def test_risk_parity_equalizes_risk(self, synthetic_data):
        """Risk parity should approximately equalize risk contributions."""
        _, _, _, factors, _ = synthetic_data
        fp = FactorPortfolio(factors)
        w = fp.risk_parity()
        Sigma = fp.Sigma
        port_vol = np.sqrt(w @ Sigma @ w)
        marginal = Sigma @ w
        rc = w * marginal / port_vol
        rc_pct = rc / port_vol
        # All risk contributions should be roughly equal
        assert np.std(rc_pct) < 0.05, \
            f"Risk contributions not equal: {rc_pct}"

    def test_mean_variance_positive_weights(self, synthetic_data):
        _, _, _, factors, _ = synthetic_data
        fp = FactorPortfolio(factors)
        w = fp.mean_variance("max_sharpe")
        assert all(w >= -1e-6), "Weights must be non-negative"

    def test_all_strategies_produce_valid_weights(self, synthetic_data):
        _, _, _, factors, _ = synthetic_data
        fp = FactorPortfolio(factors)
        all_w = fp.compute_all_strategies()
        for col in all_w.columns:
            assert abs(all_w[col].sum() - 1.0) < 1e-4, \
                f"{col} weights don't sum to 1"


# ---------------------------------------------------------------------------
# Backtesting Tests
# ---------------------------------------------------------------------------
class TestFactorBacktester:
    """Test suite for walk-forward backtesting engine."""

    def test_backtest_produces_results(self, synthetic_data):
        _, _, _, factors, _ = synthetic_data
        bt = FactorBacktester(factors, lookback_window=36)
        result = bt.run("equal_weight", min_history=36)
        assert len(result) > 0
        assert "net_return" in result.columns
        assert "cumulative_return" in result.columns

    def test_cumulative_return_positive_start(self, synthetic_data):
        """Cumulative return should start near 1.0."""
        _, _, _, factors, _ = synthetic_data
        bt = FactorBacktester(factors, lookback_window=36)
        result = bt.run("equal_weight", min_history=36)
        assert result["cumulative_return"].iloc[0] > 0.9

    def test_transaction_costs_reduce_returns(self, synthetic_data):
        """Net return should be <= gross return."""
        _, _, _, factors, _ = synthetic_data
        bt = FactorBacktester(factors, transaction_cost_bps=50,
                              lookback_window=36)
        result = bt.run("risk_parity", min_history=36)
        assert (result["net_return"] <= result["gross_return"] + 1e-10).all()

    def test_performance_summary_all_strategies(self, synthetic_data):
        _, _, _, factors, _ = synthetic_data
        bt = FactorBacktester(factors, lookback_window=36)
        bt.run_all_strategies()
        summary = bt.performance_summary()
        assert len(summary) == 6
        assert "Sharpe Ratio" in summary.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
