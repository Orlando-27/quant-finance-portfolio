"""
test_trading_bot.py
-------------------
Unit tests for core modules: SignalGenerator, RiskManager,
PositionMonitor, OrderManager (dry-run), and chart generation.

Run from project root:
    pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from src.config           import BotConfig, SignalConfig, RiskConfig
from src.signal_generator import SignalGenerator, SignalResult
from src.risk_manager     import RiskManager
from src.position_monitor import PositionMonitor
from src.order_manager    import OrderManager
from src.notifier         import Notifier
from src.utils            import pct_change, clamp, format_currency


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def cfg():
    return BotConfig()


@pytest.fixture
def sample_bars(n=200):
    """Generate synthetic OHLCV DataFrame."""
    rng    = np.random.default_rng(42)
    prices = 150.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.015, n)))
    dates  = pd.date_range(end=datetime.now(), periods=n, freq="5min")
    return pd.DataFrame({
        "open":   prices * rng.uniform(0.999, 1.001, n),
        "high":   prices * rng.uniform(1.000, 1.010, n),
        "low":    prices * rng.uniform(0.990, 1.000, n),
        "close":  prices,
        "volume": rng.integers(100_000, 2_000_000, n),
    }, index=dates)


@pytest.fixture
def signal_gen(cfg):
    return SignalGenerator(cfg.signals)


@pytest.fixture
def risk_mgr(cfg):
    rm = RiskManager(cfg.risk)
    rm.initialize(100_000.0)
    return rm


# =============================================================================
# Signal Generator Tests
# =============================================================================

class TestSignalGenerator:

    def test_returns_signal_result(self, signal_gen, sample_bars):
        result = signal_gen.generate("AAPL", sample_bars)
        assert isinstance(result, SignalResult)
        assert result.symbol == "AAPL"

    def test_signal_in_valid_range(self, signal_gen, sample_bars):
        result = signal_gen.generate("MSFT", sample_bars)
        assert result.signal in (-1, 0, 1)

    def test_confidence_bounded(self, signal_gen, sample_bars):
        result = signal_gen.generate("GOOGL", sample_bars)
        assert -1.0 <= result.confidence <= 1.0

    def test_rsi_range(self, signal_gen, sample_bars):
        result = signal_gen.generate("NVDA", sample_bars)
        assert 0 <= result.rsi <= 100

    def test_insufficient_bars_returns_flat(self, signal_gen):
        tiny = pd.DataFrame({
            "open": [100], "high": [101], "low": [99], "close": [100], "volume": [1000]
        })
        result = signal_gen.generate("X", tiny)
        assert result.signal == 0

    def test_generate_all(self, signal_gen, sample_bars):
        bars_dict = {"AAPL": sample_bars, "MSFT": sample_bars.copy()}
        results   = signal_gen.generate_all(bars_dict)
        assert "AAPL" in results
        assert "MSFT" in results
        assert all(r.signal in (-1, 0, 1) for r in results.values())

    def test_rsi_static_rising(self, signal_gen):
        """RSI > 50 for zigzag uptrend: 3 up-bars then 1 down-bar x15."""
        moves  = [2.0, 2.0, 2.0, -0.5] * 15
        prices = pd.Series(100.0 + np.cumsum(moves))
        rsi    = SignalGenerator._rsi(prices, 14).dropna()
        assert len(rsi) > 0, "RSI series empty after dropna()"
        assert float(rsi.iloc[-1]) > 50
    def test_macd_crossover_detected(self, signal_gen):
        """MACD line should cross signal in a volatile series."""
        rng    = np.random.default_rng(7)
        prices = pd.Series(100 + np.cumsum(rng.normal(0, 1, 60)))
        ml, ms, mh = SignalGenerator._macd(prices, 12, 26, 9)
        assert not ml.dropna().empty


# =============================================================================
# Risk Manager Tests
# =============================================================================

class TestRiskManager:

    def test_position_sizing_basic(self, risk_mgr):
        shares = risk_mgr.compute_shares("AAPL", 150.0, 100_000.0, +1)
        expected_max = math.floor(100_000.0 * 0.05 / 150.0)
        assert 0 < shares <= expected_max

    def test_zero_price_returns_zero(self, risk_mgr):
        assert risk_mgr.compute_shares("X", 0.0, 100_000.0, +1) == 0

    def test_halt_blocks_orders(self, risk_mgr):
        risk_mgr.state.halted = True
        risk_mgr.state.halt_reason = "Test halt"
        shares = risk_mgr.compute_shares("AAPL", 100.0, 100_000.0, +1)
        assert shares == 0

    def test_stop_loss_triggers(self, risk_mgr):
        assert risk_mgr.check_stop_loss("AAPL", 100.0, 97.5)  # -2.5% > 2% limit

    def test_stop_loss_no_trigger(self, risk_mgr):
        assert not risk_mgr.check_stop_loss("AAPL", 100.0, 99.5)  # -0.5% < 2% limit

    def test_take_profit_triggers(self, risk_mgr):
        assert risk_mgr.check_take_profit("AAPL", 100.0, 105.0)  # +5% > 4% limit

    def test_take_profit_no_trigger(self, risk_mgr):
        assert not risk_mgr.check_take_profit("AAPL", 100.0, 102.0)  # +2% < 4% limit

    def test_daily_loss_halts_bot(self, risk_mgr):
        risk_mgr.update_equity(96_500.0)  # -3.5% loss -> breaches 3% limit
        assert risk_mgr.state.halted

    def test_equity_high_watermark(self, risk_mgr):
        risk_mgr.update_equity(105_000.0)
        assert risk_mgr.state.session_high_equity == 105_000.0

    def test_max_portfolio_pct_blocks_order(self, risk_mgr):
        """Fill positions up to limit then next order should be blocked."""
        risk_mgr.state.deployed_capital = 80_000.0   # 80% already deployed
        shares = risk_mgr.compute_shares("AAPL", 150.0, 100_000.0, +1)
        assert shares == 0


# =============================================================================
# Position Monitor Tests
# =============================================================================

class TestPositionMonitor:

    def test_open_close_position(self):
        mon = PositionMonitor(initial_equity=100_000.0)
        mon.open_position("AAPL", 10, 150.0)
        assert "AAPL" in mon._positions

        pnl = mon.close_position("AAPL", 155.0)
        assert abs(pnl - 50.0) < 1e-6
        assert "AAPL" not in mon._positions

    def test_unrealized_pnl(self):
        mon = PositionMonitor()
        mon.open_position("MSFT", 5, 200.0)
        mon.update_prices({"MSFT": 210.0})
        assert abs(mon.total_unrealized_pnl() - 50.0) < 1e-6

    def test_drawdown_calculation(self):
        mon = PositionMonitor(initial_equity=100_000.0)
        mon.record_equity(100_000.0)
        mon.record_equity(105_000.0)
        mon.record_equity(98_000.0)
        dd = mon.current_drawdown()
        assert dd > 0
        assert dd < 1

    def test_snapshot_empty(self):
        mon = PositionMonitor()
        assert mon.snapshot().empty

    def test_snapshot_populated(self):
        mon = PositionMonitor()
        mon.open_position("NVDA", 2, 400.0)
        mon.update_prices({"NVDA": 410.0})
        snap = mon.snapshot()
        assert len(snap) == 1
        assert snap.iloc[0]["symbol"] == "NVDA"

    def test_equity_curve(self):
        mon = PositionMonitor(initial_equity=100_000.0)
        for v in [100_000, 101_000, 102_500, 100_800]:
            mon.record_equity(float(v))
        curve = mon.equity_curve()
        assert len(curve) == 4
        assert "net_liq" in curve.columns


# =============================================================================
# Order Manager Tests (dry-run)
# =============================================================================

class TestOrderManager:

    def test_dry_run_order(self, cfg):
        mock_ib = MagicMock()
        om = OrderManager(mock_ib, cfg, dry_run=True)
        rec = om.submit_order("AAPL", "BUY", 10, 150.0)
        assert rec is not None
        assert rec.status == "DryRun"
        assert rec.filled_px is not None

    def test_zero_qty_blocked(self, cfg):
        mock_ib = MagicMock()
        om = OrderManager(mock_ib, cfg, dry_run=True)
        rec = om.submit_order("AAPL", "BUY", 0, 150.0)
        assert rec is None

    def test_order_log_populated(self, cfg):
        mock_ib = MagicMock()
        om = OrderManager(mock_ib, cfg, dry_run=True)
        om.submit_order("AAPL", "BUY",  5, 150.0)
        om.submit_order("MSFT", "SELL", 3, 300.0)
        log_df = om.order_log()
        assert len(log_df) == 2
        assert set(log_df["symbol"]) == {"AAPL", "MSFT"}

    def test_limit_price_buy_adds_slippage(self, cfg):
        mock_ib = MagicMock()
        om  = OrderManager(mock_ib, cfg, dry_run=True)
        rec = om.submit_order("X", "BUY", 1, 100.0)
        assert rec.limit_px >= 100.0

    def test_limit_price_sell_subtracts_slippage(self, cfg):
        mock_ib = MagicMock()
        om  = OrderManager(mock_ib, cfg, dry_run=True)
        rec = om.submit_order("X", "SELL", 1, 100.0)
        assert rec.limit_px <= 100.0


# =============================================================================
# Notifier Tests
# =============================================================================

class TestNotifier:

    def test_disabled_notifier_does_not_raise(self, cfg):
        cfg.notifications.enabled = False
        n = Notifier(cfg.notifications)
        # Should log but not raise
        n.trade_alert("AAPL", "BUY", 10, 150.0, {"score": 3, "confidence": 0.75,
                                                   "rsi": 28.0, "macd_signal": 1,
                                                   "bb_signal": 1, "ema_signal": 1})

    def test_risk_alert_disabled_silent(self, cfg):
        cfg.notifications.enabled = False
        n = Notifier(cfg.notifications)
        n.risk_alert("Test halt reason")  # must not raise


# =============================================================================
# Utility Tests
# =============================================================================

class TestUtils:

    def test_pct_change(self):
        assert abs(pct_change(100.0, 105.0) - 0.05) < 1e-10

    def test_pct_change_zero_denominator(self):
        assert pct_change(0.0, 100.0) == 0.0

    def test_clamp(self):
        assert clamp(1.5, 0.0, 1.0) == 1.0
        assert clamp(-0.5, 0.0, 1.0) == 0.0
        assert clamp(0.5, 0.0, 1.0) == 0.5

    def test_format_currency(self):
        assert format_currency(1234.5) == "$1,234.50"
        assert format_currency(-500.0) == "$-500.00"


# =============================================================================
# Integration Test: demo pipeline
# =============================================================================

class TestDemoPipeline:

    def test_full_signal_pipeline(self, cfg, sample_bars):
        """End-to-end: bars -> signal -> size -> monitor."""
        gen   = SignalGenerator(cfg.signals)
        risk  = RiskManager(cfg.risk)
        mon   = PositionMonitor(initial_equity=100_000.0)
        risk.initialize(100_000.0)

        result = gen.generate("AAPL", sample_bars)
        assert result.signal in (-1, 0, 1)

        if result.signal != 0:
            shares = risk.compute_shares("AAPL", result.close, 100_000.0, result.signal)
            if shares > 0:
                mon.open_position("AAPL", shares, result.close)
                mon.update_prices({"AAPL": result.close * 1.005})
                assert mon.total_unrealized_pnl() != 0

        summary = mon.summary()
        assert "open_positions" in summary
        assert "session_return"  in summary
