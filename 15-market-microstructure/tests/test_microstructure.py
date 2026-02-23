"""
Unit Tests — Market Microstructure Analysis
=============================================
22 tests covering all core modules.

Test coverage:
    - SpreadModels:         Roll spread, Corwin-Schultz, quoted spread,
                            effective spread decomposition
    - OrderFlowAnalyzer:    tick rule, OFI, VPIN, ACF
    - IlliquidityModels:    Amihud ILLIQ, Kyle lambda, composite score
    - AlmgrenChrissModel:   trajectory, efficient frontier, TWAP
    - SyntheticTickGenerator: data generation, aggregation
    - Helpers:              VWAP, TWAP, volume profile, NW t-stat

Run with:
    pytest tests/ -v --tb=short
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Resolve src path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from models.spread_models import SpreadModels
from models.order_flow    import OrderFlowAnalyzer
from models.illiquidity   import IlliquidityModels
from models.market_impact import AlmgrenChrissModel, AlmgrenChrissParams
from utils.data_loader    import SyntheticTickGenerator
from utils.helpers        import (
    compute_vwap, compute_twap, intraday_volume_profile,
    newey_west_tstat, price_impact_regression,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def price_series() -> pd.Series:
    """Synthetic daily price series with realistic drift and vol."""
    np.random.seed(0)
    n      = 500
    ret    = np.random.normal(5e-4, 0.012, n)
    prices = 100.0 * np.exp(np.cumsum(ret))
    dates  = pd.date_range("2020-01-02", periods=n, freq="B")
    return pd.Series(prices, index=dates, name="Close")


@pytest.fixture(scope="module")
def ohlcv_df(price_series) -> pd.DataFrame:
    """OHLCV DataFrame derived from price series."""
    np.random.seed(1)
    n = len(price_series)
    high   = price_series * (1 + np.abs(np.random.normal(0, 0.005, n)))
    low    = price_series * (1 - np.abs(np.random.normal(0, 0.005, n)))
    volume = np.random.lognormal(mean=15, sigma=0.6, size=n)
    return pd.DataFrame({
        "Close" : price_series.values,
        "High"  : high.values,
        "Low"   : low.values,
        "Volume": volume,
    }, index=price_series.index)


@pytest.fixture(scope="module")
def tick_data() -> pd.DataFrame:
    """Synthetic intraday tick data (5-minute bars)."""
    gen   = SyntheticTickGenerator(S0=100.0, n_ticks=2_000, seed=7)
    ticks = gen.generate()
    return gen.aggregate_to_bars(ticks, freq="5T")


# =============================================================================
# SpreadModels tests (6 tests)
# =============================================================================
class TestSpreadModels:

    def test_roll_spread_returns_series(self, ohlcv_df):
        """Roll spread returns a pandas Series of correct length."""
        result = SpreadModels.roll_spread(
            pd.Series(ohlcv_df["Close"].values,
                      index=ohlcv_df.index), window=60
        )
        assert isinstance(result, pd.Series)
        assert len(result) == len(ohlcv_df)

    def test_roll_spread_nonnegative(self, ohlcv_df):
        """All identified Roll spread estimates are non-negative."""
        result = SpreadModels.roll_spread(
            pd.Series(ohlcv_df["Close"].values, index=ohlcv_df.index), window=60
        ).dropna()
        assert (result >= 0).all(), "Roll spread should be non-negative"

    def test_corwin_schultz_nonnegative(self, ohlcv_df):
        """Corwin-Schultz spread is clipped to [0, ∞)."""
        cs = SpreadModels.corwin_schultz_spread(
            pd.Series(ohlcv_df["High"].values,  index=ohlcv_df.index),
            pd.Series(ohlcv_df["Low"].values,   index=ohlcv_df.index),
        ).dropna()
        assert (cs >= 0).all()

    def test_corwin_schultz_positive_spread_when_volatile(self):
        """CS spread is positive for high-volatility prices."""
        idx    = pd.date_range("2024-01-01", periods=100, freq="B")
        np.random.seed(5)
        prices = 100 * np.exp(np.cumsum(np.random.normal(0, 0.02, 100)))
        high   = pd.Series(prices * 1.01, index=idx)
        low    = pd.Series(prices * 0.99, index=idx)
        cs     = SpreadModels.corwin_schultz_spread(high, low).dropna()
        assert cs.mean() > 0

    def test_quoted_spread_positive(self, ohlcv_df):
        """Quoted OHLC spread is strictly positive."""
        qs = SpreadModels.quoted_spread_ohlc(
            pd.Series(ohlcv_df["High"].values,  index=ohlcv_df.index),
            pd.Series(ohlcv_df["Low"].values,   index=ohlcv_df.index),
            pd.Series(ohlcv_df["Close"].values, index=ohlcv_df.index),
        )
        assert (qs > 0).all()

    def test_spread_comparison_columns(self, ohlcv_df):
        """spread_comparison returns DataFrame with three expected columns."""
        df = SpreadModels.spread_comparison(
            pd.Series(ohlcv_df["Close"].values, index=ohlcv_df.index),
            pd.Series(ohlcv_df["High"].values,  index=ohlcv_df.index),
            pd.Series(ohlcv_df["Low"].values,   index=ohlcv_df.index),
        )
        assert isinstance(df, pd.DataFrame)
        assert df.shape[1] == 3


# =============================================================================
# OrderFlowAnalyzer tests (5 tests)
# =============================================================================
class TestOrderFlowAnalyzer:

    def test_tick_rule_values(self, price_series):
        """Tick rule returns only +1 or -1."""
        direction = OrderFlowAnalyzer.tick_rule(price_series)
        assert set(direction.dropna().unique()).issubset({1.0, -1.0})

    def test_tick_rule_length(self, price_series):
        """Tick rule output length equals input length."""
        direction = OrderFlowAnalyzer.tick_rule(price_series)
        assert len(direction) == len(price_series)

    def test_ofi_bounded(self, ohlcv_df):
        """OFI is bounded in [-1, +1]."""
        ofi = OrderFlowAnalyzer.order_flow_imbalance(
            pd.Series(ohlcv_df["Close"].values, index=ohlcv_df.index),
            pd.Series(ohlcv_df["Volume"].values, index=ohlcv_df.index),
            window=20,
        ).dropna()
        assert ofi.between(-1.001, 1.001).all()

    def test_vpin_bounded(self, ohlcv_df):
        """VPIN is bounded in [0, 1]."""
        vpin = OrderFlowAnalyzer.vpin(
            pd.Series(ohlcv_df["Close"].values, index=ohlcv_df.index),
            pd.Series(ohlcv_df["Volume"].values, index=ohlcv_df.index),
            n_buckets=30,
        ).dropna()
        assert vpin.between(-0.001, 1.001).all()

    def test_ofi_acf_length(self, ohlcv_df):
        """OFI ACF has max_lag entries."""
        ofi = OrderFlowAnalyzer.order_flow_imbalance(
            pd.Series(ohlcv_df["Close"].values, index=ohlcv_df.index),
            pd.Series(ohlcv_df["Volume"].values, index=ohlcv_df.index),
        )
        acf = OrderFlowAnalyzer.ofi_autocorrelation(ofi, max_lag=20)
        assert len(acf) == 20


# =============================================================================
# IlliquidityModels tests (5 tests)
# =============================================================================
class TestIlliquidityModels:

    def test_amihud_nonnegative(self, ohlcv_df):
        """Amihud ILLIQ is non-negative."""
        returns    = pd.Series(ohlcv_df["Close"].values, index=ohlcv_df.index).pct_change()
        dollar_vol = (pd.Series(ohlcv_df["Close"].values, index=ohlcv_df.index)
                      * pd.Series(ohlcv_df["Volume"].values, index=ohlcv_df.index))
        illiq = IlliquidityModels.amihud_illiq(returns, dollar_vol).dropna()
        assert (illiq >= 0).all()

    def test_amihud_increases_with_lower_volume(self):
        """Amihud ILLIQ is higher for thinly traded assets."""
        idx   = pd.date_range("2020-01-02", periods=300, freq="B")
        np.random.seed(2)
        ret   = pd.Series(np.random.normal(0, 0.01, 300), index=idx)
        high_vol = pd.Series(np.ones(300) * 1e9, index=idx)  # very liquid
        low_vol  = pd.Series(np.ones(300) * 1e4, index=idx)  # illiquid
        illiq_h = IlliquidityModels.amihud_illiq(ret, high_vol).dropna().mean()
        illiq_l = IlliquidityModels.amihud_illiq(ret, low_vol).dropna().mean()
        assert illiq_l > illiq_h

    def test_kyle_lambda_returns_series(self, ohlcv_df):
        """Kyle lambda returns a Series of correct length."""
        result = IlliquidityModels.kyle_lambda(
            pd.Series(ohlcv_df["Close"].values, index=ohlcv_df.index),
            pd.Series(ohlcv_df["Volume"].values, index=ohlcv_df.index),
            window=20,
        )
        assert isinstance(result, pd.Series)
        assert len(result) == len(ohlcv_df)

    def test_composite_score_mean_zero(self, ohlcv_df):
        """Composite liquidity score has near-zero mean (Z-score)."""
        returns = pd.Series(ohlcv_df["Close"].values, index=ohlcv_df.index).pct_change()
        dv      = (pd.Series(ohlcv_df["Close"].values, index=ohlcv_df.index)
                   * pd.Series(ohlcv_df["Volume"].values, index=ohlcv_df.index))
        amihud  = IlliquidityModels.amihud_illiq(returns, dv)
        kyle    = IlliquidityModels.kyle_lambda(
            pd.Series(ohlcv_df["Close"].values, index=ohlcv_df.index),
            pd.Series(ohlcv_df["Volume"].values, index=ohlcv_df.index),
        )
        turn    = IlliquidityModels.turnover_liquidity(
            pd.Series(ohlcv_df["Volume"].values, index=ohlcv_df.index)
        )
        score = IlliquidityModels.composite_liquidity_score(amihud, kyle, turn).dropna()
        assert abs(score.mean()) < 1.0   # Z-score should be centred near zero

    def test_turnover_nonnegative(self, ohlcv_df):
        """Turnover ratio is non-negative."""
        turn = IlliquidityModels.turnover_liquidity(
            pd.Series(ohlcv_df["Volume"].values, index=ohlcv_df.index)
        ).dropna()
        assert (turn >= 0).all()


# =============================================================================
# AlmgrenChrissModel tests (4 tests)
# =============================================================================
class TestAlmgrenChrissModel:

    def setup_method(self):
        self.model = AlmgrenChrissModel(
            AlmgrenChrissParams(X=100_000, T=1.0, N=10, sigma=0.015,
                                eta=0.01, gamma=1e-6, S0=100.0)
        )

    def test_inventory_decreasing(self):
        """Optimal inventory is non-increasing over time."""
        traj = self.model.optimal_trajectory(lam=1e-5)
        assert all(np.diff(traj["inventory"]) <= 1e-6), \
            "Inventory should be non-increasing"

    def test_inventory_starts_at_X(self):
        """Trajectory starts with full position."""
        traj = self.model.optimal_trajectory(lam=1e-5)
        assert abs(traj["inventory"][0] - self.model.p.X) < 1e-3

    def test_trade_sizes_sum_to_X(self):
        """Total shares traded equals initial position."""
        traj = self.model.optimal_trajectory(lam=1e-5)
        assert abs(np.sum(traj["trade_size"]) - self.model.p.X) < 1.0

    def test_efficient_frontier_shape(self):
        """Efficient frontier DataFrame has correct shape."""
        df = self.model.efficient_frontier(n_points=50)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 50
        assert "E_cost" in df.columns and "std_cost" in df.columns

    def test_twap_is_uniform(self):
        """TWAP trajectory produces equal trade sizes."""
        traj  = self.model.twap_trajectory()
        sizes = traj["trade_size"]
        assert np.std(sizes) / np.mean(sizes) < 0.05, \
            "TWAP should have uniform trade sizes"


# =============================================================================
# SyntheticTickGenerator tests (2 tests)
# =============================================================================
class TestSyntheticTickGenerator:

    def test_tick_data_shape(self):
        """Generator produces correct number of ticks."""
        gen   = SyntheticTickGenerator(n_ticks=1_000, seed=0)
        ticks = gen.generate()
        assert len(ticks) == 1_000
        assert "price" in ticks.columns and "volume" in ticks.columns

    def test_aggregation_ohlcv(self, tick_data):
        """Aggregated bars have required columns and positive OHLCV."""
        assert all(c in tick_data.columns for c in ["open", "high", "low", "close", "volume"])
        assert (tick_data["volume"] >= 0).all()
        assert (tick_data["high"] >= tick_data["low"]).all()


# =============================================================================
# Helpers tests (3 tests)
# =============================================================================
class TestHelpers:

    def test_vwap_between_low_high(self, tick_data):
        """Cumulative VWAP stays within bar low-high range."""
        vwap = compute_vwap(tick_data["close"], tick_data["volume"])
        assert (vwap >= tick_data["low"].min() * 0.99).all()
        assert (vwap <= tick_data["high"].max() * 1.01).all()

    def test_twap_between_low_high(self, tick_data):
        """TWAP stays within observed price range."""
        twap = compute_twap(tick_data["close"])
        mn   = tick_data["close"].min()
        mx   = tick_data["close"].max()
        assert (twap >= mn * 0.99).all()
        assert (twap <= mx * 1.01).all()

    def test_newey_west_returns_tuple(self):
        """newey_west_tstat returns (float, float)."""
        np.random.seed(9)
        x = pd.Series(np.random.normal(0.01, 0.1, 200))
        t, p = newey_west_tstat(x, lags=5)
        assert isinstance(t, float)
        assert 0.0 <= p <= 1.0
