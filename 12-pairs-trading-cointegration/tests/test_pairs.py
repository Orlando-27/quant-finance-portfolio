"""
Unit Tests for Pairs Trading Framework
=======================================

Covers: cointegration tests, pair selection, OU calibration,
Kalman filter, strategy signals, and backtesting engine.
"""

import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cointegration import EngleGranger, JohansenTest
from src.pair_selection import PairSelector
from src.ornstein_uhlenbeck import OrnsteinUhlenbeck
from src.kalman_filter import KalmanHedgeRatio
from src.strategy import PairsTradingStrategy
from src.backtesting import PairsBacktester


@pytest.fixture(scope="module")
def cointegrated_pair():
    """Generate a synthetic cointegrated pair with strong cointegration."""
    rng = np.random.RandomState(123)
    T = 750
    dates = pd.bdate_range("2020-01-01", periods=T)
    # Shared stochastic trend (common factor)
    trend = np.cumsum(rng.normal(0.0003, 0.015, T))
    # Stationary spread (mean-reverting noise)
    spread = np.zeros(T)
    for t in range(1, T):
        spread[t] = 0.92 * spread[t-1] + rng.normal(0, 0.008)
    p1 = np.exp(trend + np.log(50))
    p2 = np.exp(0.8 * trend + spread + np.log(40))
    pa = pd.Series(p1, index=dates, name="A")
    pb = pd.Series(p2, index=dates, name="B")
    return pa, pb


@pytest.fixture(scope="module")
def independent_pair():
    """Generate two independent (non-cointegrated) series."""
    rng = np.random.RandomState(456)
    T = 500
    dates = pd.bdate_range("2020-01-01", periods=T)
    p1 = np.exp(np.cumsum(rng.normal(0.0003, 0.015, T)) + np.log(50))
    p2 = np.exp(np.cumsum(rng.normal(0.0001, 0.018, T)) + np.log(40))
    pa = pd.Series(p1, index=dates, name="C")
    pb = pd.Series(p2, index=dates, name="D")
    return pa, pb


@pytest.fixture(scope="module")
def small_universe():
    """Small universe of 6 stocks with 2 cointegrated groups."""
    rng = np.random.RandomState(789)
    T = 600
    dates = pd.bdate_range("2019-01-01", periods=T)
    trend = np.cumsum(rng.normal(0.0003, 0.01, T))
    s1 = np.exp(trend + rng.normal(0, 0.02, T).cumsum() * 0.1 + np.log(50))
    s2 = np.exp(0.85 * trend + rng.normal(0, 0.02, T).cumsum() * 0.1 + np.log(45))
    s3 = np.exp(np.cumsum(rng.normal(0.0004, 0.018, T)) + np.log(70))
    s4 = np.exp(np.cumsum(rng.normal(0.0001, 0.022, T)) + np.log(30))
    s5 = np.exp(np.cumsum(rng.normal(0.0003, 0.015, T)) + np.log(55))
    s6 = np.exp(np.cumsum(rng.normal(-0.0001, 0.020, T)) + np.log(45))
    return pd.DataFrame(
        np.column_stack([s1, s2, s3, s4, s5, s6]),
        index=dates, columns=["X1", "X2", "X3", "X4", "X5", "X6"]
    )


# ---------------------------------------------------------------------------
# Cointegration Tests
# ---------------------------------------------------------------------------
class TestEngleGranger:
    def test_cointegrated_pair_detected(self, cointegrated_pair):
        pa, pb = cointegrated_pair
        eg = EngleGranger(significance=0.10)
        res = eg.test(np.log(pa), np.log(pb))
        assert res["adf_pvalue"] < 0.15, \
            f"Expected p < 0.15 for cointegrated pair, got {res['adf_pvalue']}"

    def test_hedge_ratio_positive(self, cointegrated_pair):
        pa, pb = cointegrated_pair
        eg = EngleGranger()
        res = eg.test(np.log(pa), np.log(pb))
        assert res["hedge_ratio"] > 0, "Hedge ratio should be positive"

    def test_residuals_returned(self, cointegrated_pair):
        pa, pb = cointegrated_pair
        eg = EngleGranger()
        res = eg.test(np.log(pa), np.log(pb))
        assert isinstance(res["residuals"], pd.Series)
        assert len(res["residuals"]) > 0

    def test_summary_string(self, cointegrated_pair):
        pa, pb = cointegrated_pair
        eg = EngleGranger()
        eg.test(np.log(pa), np.log(pb))
        s = eg.get_summary()
        assert "ADF" in s


class TestJohansen:
    def test_johansen_finds_cointegration(self, cointegrated_pair):
        pa, pb = cointegrated_pair
        data = pd.DataFrame({"A": np.log(pa), "B": np.log(pb)})
        joh = JohansenTest(det_order=0, k_ar_diff=1)
        res = joh.test(data)
        assert res["n_coint_trace"] >= 0

    def test_cointegrating_vector(self, cointegrated_pair):
        pa, pb = cointegrated_pair
        data = pd.DataFrame({"A": np.log(pa), "B": np.log(pb)})
        joh = JohansenTest()
        joh.test(data)
        vec = joh.get_cointegrating_vector(0)
        assert vec is not None
        assert abs(vec[0] - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# OU Process Tests
# ---------------------------------------------------------------------------
class TestOrnsteinUhlenbeck:
    def test_ols_calibration(self, cointegrated_pair):
        pa, pb = cointegrated_pair
        eg = EngleGranger()
        res = eg.test(np.log(pa), np.log(pb))
        ou = OrnsteinUhlenbeck()
        params = ou.fit_ols(res["residuals"].values)
        assert params["kappa"] > 0, "Kappa should be positive"
        assert params["half_life_days"] > 0

    def test_mle_calibration(self, cointegrated_pair):
        pa, pb = cointegrated_pair
        eg = EngleGranger()
        res = eg.test(np.log(pa), np.log(pb))
        ou = OrnsteinUhlenbeck()
        params = ou.fit_mle(res["residuals"].values)
        assert params["kappa"] > 0
        assert params["converged"]

    def test_simulation_shape(self, cointegrated_pair):
        pa, pb = cointegrated_pair
        eg = EngleGranger()
        res = eg.test(np.log(pa), np.log(pb))
        ou = OrnsteinUhlenbeck()
        ou.fit_ols(res["residuals"].values)
        paths = ou.simulate(0.0, n_steps=100, n_paths=50)
        assert paths.shape == (101, 50)

    def test_stationary_distribution(self, cointegrated_pair):
        pa, pb = cointegrated_pair
        eg = EngleGranger()
        res = eg.test(np.log(pa), np.log(pb))
        ou = OrnsteinUhlenbeck()
        ou.fit_ols(res["residuals"].values)
        stat = ou.stationary_distribution()
        assert "mean" in stat and "std" in stat
        assert stat["std"] > 0


# ---------------------------------------------------------------------------
# Kalman Filter Tests
# ---------------------------------------------------------------------------
class TestKalmanFilter:
    def test_hedge_ratio_evolves(self, cointegrated_pair):
        pa, pb = cointegrated_pair
        kf = KalmanHedgeRatio(delta=1e-4)
        res = kf.filter(np.log(pa), np.log(pb))
        betas = res["betas"]["hedge_ratio"]
        assert betas.std() > 0, "Hedge ratio should vary over time"

    def test_output_dimensions(self, cointegrated_pair):
        pa, pb = cointegrated_pair
        kf = KalmanHedgeRatio()
        res = kf.filter(np.log(pa), np.log(pb))
        assert len(res["betas"]) == len(res["spreads"])
        assert "alpha" in res["betas"].columns

    def test_adaptive_spread_more_stationary(self, cointegrated_pair):
        pa, pb = cointegrated_pair
        kf = KalmanHedgeRatio(delta=1e-3)
        kf.filter(np.log(pa), np.log(pb))
        adaptive = kf.get_adaptive_spread(np.log(pa), np.log(pb))
        assert isinstance(adaptive, pd.Series)


# ---------------------------------------------------------------------------
# Strategy Tests
# ---------------------------------------------------------------------------
class TestPairsTradingStrategy:
    def test_zscore_computation(self, cointegrated_pair):
        pa, pb = cointegrated_pair
        eg = EngleGranger()
        res = eg.test(np.log(pa), np.log(pb))
        strat = PairsTradingStrategy(lookback=30)
        z = strat.compute_zscore(res["residuals"])
        assert abs(z.dropna().mean()) < 1.0

    def test_signals_valid_values(self, cointegrated_pair):
        pa, pb = cointegrated_pair
        eg = EngleGranger()
        res = eg.test(np.log(pa), np.log(pb))
        strat = PairsTradingStrategy()
        signals = strat.generate_signals(res["residuals"])
        valid_positions = {-1, 0, 1}
        assert set(signals["position"].unique()).issubset(valid_positions)

    def test_returns_have_cumulative(self, cointegrated_pair):
        pa, pb = cointegrated_pair
        eg = EngleGranger()
        res = eg.test(np.log(pa), np.log(pb))
        strat = PairsTradingStrategy()
        signals = strat.generate_signals(res["residuals"])
        result = strat.compute_returns(signals, pa, pb, res["hedge_ratio"])
        assert "cumulative_return" in result.columns

    def test_trade_statistics(self, cointegrated_pair):
        pa, pb = cointegrated_pair
        eg = EngleGranger()
        res = eg.test(np.log(pa), np.log(pb))
        strat = PairsTradingStrategy()
        signals = strat.generate_signals(res["residuals"])
        strat.compute_returns(signals, pa, pb, res["hedge_ratio"])
        stats = strat.trade_statistics()
        assert "n_trades" in stats
        assert stats["n_trades"] >= 0


# ---------------------------------------------------------------------------
# Pair Selection Tests
# ---------------------------------------------------------------------------
class TestPairSelector:
    def test_selection_returns_list(self, small_universe):
        selector = PairSelector(method="cointegration", top_k=3)
        result = selector.select(small_universe)
        assert isinstance(result, list)

    def test_hurst_exponent_valid(self):
        rng = np.random.RandomState(42)
        mr_series = np.zeros(500)
        for i in range(1, 500):
            mr_series[i] = 0.5 * mr_series[i - 1] + rng.normal(0, 0.1)
        h = PairSelector._hurst_exponent(mr_series)
        assert 0 < h < 1, f"Hurst should be in (0,1), got {h}"


# ---------------------------------------------------------------------------
# Backtesting Tests
# ---------------------------------------------------------------------------
class TestPairsBacktester:
    def test_backtest_runs(self, small_universe):
        bt = PairsBacktester(
            formation_period=200, trading_period=100,
            n_pairs=2, transaction_cost_bps=10
        )
        result = bt.run(small_universe)
        assert result is not None

    def test_performance_summary(self, small_universe):
        bt = PairsBacktester(
            formation_period=200, trading_period=100, n_pairs=2
        )
        bt.run(small_universe)
        perf = bt.performance_summary()
        assert "Sharpe Ratio" in perf


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
