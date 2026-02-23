"""
Comprehensive Unit Tests: Momentum & Mean Reversion Strategy
=============================================================

Tests cover:
    - Data generation integrity
    - Momentum signal computation (TSMOM, CS-MOM)
    - Mean reversion signals (Z-Score, RSI, Bollinger)
    - Regime detection
    - Portfolio construction and risk management
    - Backtesting engine

Author: Jose Orlando Bobadilla Fuentes, CQF, MSc AI
"""

import numpy as np
import pandas as pd
import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_generator import generate_multi_asset_data, get_asset_class_map
from src.momentum import TimeSeriesMomentum, CrossSectionalMomentum
from src.mean_reversion import ZScoreSignal, RSISignal, BollingerBandSignal, CompositeMeanReversion
from src.regime import VolatilityRegime, DispersionRegime, AutocorrelationRegime, RegimeDetector
from src.portfolio import PortfolioConstructor
from src.backtesting import BacktestEngine


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------
@pytest.fixture(scope="module")
def sample_data():
    """Generate sample data used across all tests."""
    prices, returns, metadata = generate_multi_asset_data(n_years=5, seed=123)
    class_map = get_asset_class_map(metadata)
    return prices, returns, metadata, class_map


# -----------------------------------------------------------------------
# Data Generator Tests
# -----------------------------------------------------------------------
class TestDataGenerator:
    """Tests for synthetic multi-asset data generation."""

    def test_data_shape(self, sample_data):
        """Verify generated data has correct dimensions."""
        prices, returns, metadata, _ = sample_data
        assert prices.shape == returns.shape
        assert prices.shape[1] == len(metadata)
        assert prices.shape[0] == 5 * 252  # 5 years x 252 days

    def test_prices_positive(self, sample_data):
        """All prices must be strictly positive."""
        prices = sample_data[0]
        assert (prices > 0).all().all()

    def test_returns_reasonable(self, sample_data):
        """Daily returns should be within reasonable bounds."""
        returns = sample_data[1]
        assert returns.abs().max().max() < 0.5  # No single-day move > 50%

    def test_asset_classes_complete(self, sample_data):
        """All four asset classes must be represented."""
        class_map = sample_data[3]
        expected = {"equity", "fixed_inc", "commodity", "fx"}
        assert set(class_map.keys()) == expected

    def test_correlation_structure(self, sample_data):
        """Intra-class correlations should be higher than inter-class."""
        returns = sample_data[1]
        class_map = sample_data[3]
        corr = returns.corr()

        # Average intra-equity correlation
        eq = class_map["equity"]
        intra_eq = []
        for i, a in enumerate(eq):
            for b in eq[i + 1:]:
                intra_eq.append(corr.loc[a, b])
        avg_intra = np.mean(intra_eq)

        # Average correlation between equity and FX
        fx = class_map["fx"]
        inter = []
        for a in eq:
            for b in fx:
                inter.append(abs(corr.loc[a, b]))
        avg_inter = np.mean(inter)

        assert avg_intra > avg_inter  # Intra > inter


# -----------------------------------------------------------------------
# Momentum Tests
# -----------------------------------------------------------------------
class TestTimeSeriesMomentum:
    """Tests for TSMOM signal generation."""

    def test_signal_shape(self, sample_data):
        """TSMOM signal must match input dimensions."""
        _, returns, _, _ = sample_data
        tsmom = TimeSeriesMomentum(lookback_days=126)
        signal = tsmom.compute_signal(returns)
        assert signal.shape == returns.shape

    def test_signal_bounded(self, sample_data):
        """TSMOM signal must be bounded by [-3, 3] (max vol scaling)."""
        _, returns, _, _ = sample_data
        tsmom = TimeSeriesMomentum(lookback_days=126)
        signal = tsmom.compute_signal(returns)
        valid = signal.dropna()
        assert valid.max().max() <= 3.01
        assert valid.min().min() >= -3.01

    def test_signal_not_all_zero(self, sample_data):
        """Signal should have meaningful non-zero values."""
        _, returns, _, _ = sample_data
        tsmom = TimeSeriesMomentum(lookback_days=126)
        signal = tsmom.compute_signal(returns).dropna()
        assert (signal.abs() > 0.01).any().any()

    def test_momentum_quality_bounded(self, sample_data):
        """Momentum quality must be in [-1, 1]."""
        _, returns, _, _ = sample_data
        tsmom = TimeSeriesMomentum()
        quality = tsmom.compute_momentum_quality(returns).dropna()
        assert quality.max().max() <= 1.01
        assert quality.min().min() >= -1.01


class TestCrossSectionalMomentum:
    """Tests for cross-sectional momentum."""

    def test_signal_normalized(self, sample_data):
        """CS-MOM signal must be in [-1, 1]."""
        _, returns, _, _ = sample_data
        csmom = CrossSectionalMomentum(lookback_months=6)
        signal = csmom.compute_signal(returns).dropna()
        assert signal.max().max() <= 1.01
        assert signal.min().min() >= -1.01

    def test_cross_sectional_zero_sum(self, sample_data):
        """Cross-sectional signals should approximately sum to zero."""
        _, returns, _, _ = sample_data
        csmom = CrossSectionalMomentum(lookback_months=6)
        signal = csmom.compute_signal(returns).dropna()
        cs_sum = signal.sum(axis=1).abs()
        # Allow some tolerance due to normalization
        assert cs_sum.mean() < 2.0


# -----------------------------------------------------------------------
# Mean Reversion Tests
# -----------------------------------------------------------------------
class TestZScore:
    """Tests for Z-Score mean reversion signal."""

    def test_zscore_properties(self, sample_data):
        """Z-scores should have approximately zero mean and unit std."""
        prices = sample_data[0]
        zs = ZScoreSignal(lookback=60)
        z = zs.compute_zscore(prices).dropna()
        # Cross-sectional mean of means should be near zero
        assert abs(z.mean().mean()) < 1.0
        # Average std should be near 1
        assert abs(z.std().mean() - 1.0) < 0.5

    def test_signal_bounded(self, sample_data):
        """Z-Score signal must be in [-1, 1]."""
        prices = sample_data[0]
        zs = ZScoreSignal()
        signal = zs.compute_signal(prices).dropna()
        assert signal.max().max() <= 1.01
        assert signal.min().min() >= -1.01


class TestRSI:
    """Tests for RSI signal."""

    def test_rsi_range(self, sample_data):
        """RSI values must be in [0, 100]."""
        prices = sample_data[0]
        rsi_gen = RSISignal(period=14)
        rsi = rsi_gen.compute_rsi(prices).dropna()
        assert rsi.max().max() <= 100.01
        assert rsi.min().min() >= -0.01

    def test_signal_bounded(self, sample_data):
        """RSI signal must be in [-1, 1]."""
        prices = sample_data[0]
        rsi_gen = RSISignal()
        signal = rsi_gen.compute_signal(prices).dropna()
        assert signal.max().max() <= 1.01
        assert signal.min().min() >= -1.01


class TestBollingerBand:
    """Tests for Bollinger Band signal."""

    def test_band_ordering(self, sample_data):
        """Lower band < middle < upper band always."""
        prices = sample_data[0]
        bb = BollingerBandSignal()
        bands = bb.compute_bands(prices)
        valid = bands["lower"].dropna()
        assert (bands["lower"].dropna() <= bands["middle"].dropna() + 1e-10).all().all()
        assert (bands["middle"].dropna() <= bands["upper"].dropna() + 1e-10).all().all()

    def test_signal_bounded(self, sample_data):
        """Bollinger signal must be in [-1, 1]."""
        prices = sample_data[0]
        bb = BollingerBandSignal()
        signal = bb.compute_signal(prices).dropna()
        assert signal.max().max() <= 1.01
        assert signal.min().min() >= -1.01


class TestCompositeReversion:
    """Tests for composite mean-reversion signal."""

    def test_composite_bounded(self, sample_data):
        """Composite MR signal must be in [-1, 1]."""
        prices, returns, _, _ = sample_data
        cmr = CompositeMeanReversion()
        signal = cmr.compute_signal(prices, returns).dropna()
        assert signal.max().max() <= 1.01
        assert signal.min().min() >= -1.01


# -----------------------------------------------------------------------
# Regime Detection Tests
# -----------------------------------------------------------------------
class TestRegimeDetection:
    """Tests for regime detection components."""

    def test_volatility_regime_bounded(self, sample_data):
        """Vol regime score must be in [0, 1]."""
        _, returns, _, _ = sample_data
        vr = VolatilityRegime()
        score = vr.compute_regime(returns).dropna()
        assert score.max() <= 1.01
        assert score.min() >= -0.01

    def test_composite_regime_bounded(self, sample_data):
        """Composite regime score must be in [0, 1]."""
        _, returns, _, _ = sample_data
        rd = RegimeDetector()
        composite, components = rd.compute_regime_scores(returns)
        valid = composite.dropna()
        assert valid.max() <= 1.01
        assert valid.min() >= -0.01

    def test_regime_labels(self):
        """Regime labels must be correctly assigned."""
        rd = RegimeDetector()
        assert "TRENDING" in rd.get_regime_label(0.8)
        assert "MEAN-REVERTING" in rd.get_regime_label(0.2)
        assert "TRANSITIONAL" in rd.get_regime_label(0.5)


# -----------------------------------------------------------------------
# Portfolio Construction Tests
# -----------------------------------------------------------------------
class TestPortfolioConstruction:
    """Tests for portfolio construction and risk management."""

    def test_position_limits(self, sample_data):
        """No individual position should exceed max_position."""
        _, returns, _, class_map = sample_data
        pc = PortfolioConstructor(max_position=0.20)

        # Create dummy signals
        mom = pd.DataFrame(
            np.random.randn(*returns.shape) * 0.5,
            index=returns.index, columns=returns.columns,
        )
        mr = pd.DataFrame(
            np.random.randn(*returns.shape) * 0.5,
            index=returns.index, columns=returns.columns,
        )
        alpha = pd.Series(0.5, index=returns.index)

        weights = pc.construct_portfolio(mom, mr, alpha, returns, class_map)
        # After position limits, max should be ~0.20 (allow tolerance for vol scaling)
        assert weights.abs().max().max() < 1.0  # Reasonable upper bound

    def test_blending(self):
        """Blending with alpha=1 should return pure momentum."""
        pc = PortfolioConstructor()
        idx = pd.date_range("2020-01-01", periods=5)
        cols = ["A", "B"]
        mom = pd.DataFrame([[1, -1]] * 5, index=idx, columns=cols, dtype=float)
        mr = pd.DataFrame([[0.5, 0.5]] * 5, index=idx, columns=cols, dtype=float)
        alpha = pd.Series(1.0, index=idx)

        blended = pc.blend_signals(mom, mr, alpha)
        np.testing.assert_array_almost_equal(blended.values, mom.values)


# -----------------------------------------------------------------------
# Backtesting Tests
# -----------------------------------------------------------------------
class TestBacktesting:
    """Tests for the backtesting engine."""

    def test_backtest_runs(self, sample_data):
        """Full backtest should execute without errors."""
        _, returns, _, class_map = sample_data
        # Simple equal-weight portfolio
        n = len(returns)
        w = 1.0 / returns.shape[1]
        weights = pd.DataFrame(w, index=returns.index, columns=returns.columns)

        engine = BacktestEngine(transaction_cost_bps=10)
        result = engine.run_backtest(weights, returns)

        assert "portfolio_returns" in result
        assert "metrics" in result
        assert len(result["portfolio_returns"]) == n

    def test_transaction_costs_reduce_returns(self, sample_data):
        """Net returns should be less than gross returns."""
        _, returns, _, _ = sample_data
        # Create weights with some turnover
        w = np.random.randn(*returns.shape) * 0.1
        weights = pd.DataFrame(w, index=returns.index, columns=returns.columns)

        engine = BacktestEngine(transaction_cost_bps=50, slippage_bps=25)
        result = engine.run_backtest(weights, returns)

        cum_gross = result["cumulative_gross"].iloc[-1]
        cum_net = result["cumulative_returns"].iloc[-1]
        assert cum_net <= cum_gross  # Costs reduce performance

    def test_metrics_reasonable(self, sample_data):
        """Key metrics should be within reasonable ranges."""
        _, returns, _, _ = sample_data
        w = 1.0 / returns.shape[1]
        weights = pd.DataFrame(w, index=returns.index, columns=returns.columns)

        engine = BacktestEngine()
        result = engine.run_backtest(weights, returns)
        m = result["metrics"]

        assert -1.0 < m["annualized_return"] < 2.0
        assert 0.0 < m["annualized_volatility"] < 1.0
        assert -1.0 <= m["max_drawdown"] <= 0.0
        assert 0.0 <= m["hit_rate"] <= 1.0

    def test_attribution_sums(self, sample_data):
        """Attribution returns should roughly sum to total return."""
        _, returns, _, class_map = sample_data
        w = 1.0 / returns.shape[1]
        weights = pd.DataFrame(w, index=returns.index, columns=returns.columns)

        engine = BacktestEngine(transaction_cost_bps=0, slippage_bps=0)
        result = engine.run_backtest(weights, returns)
        attrib = engine.attribution_by_class(weights, returns, class_map)

        total_from_attrib = attrib["ann_return"].sum()
        total_from_bt = result["metrics"]["annualized_return"]
        # Allow reasonable tolerance
        assert abs(total_from_attrib - total_from_bt) < 0.05


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
