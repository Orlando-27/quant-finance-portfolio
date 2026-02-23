# =============================================================================
# tests/test_risk.py | Project 18 | Jose Orlando Bobadilla Fuentes | CQF
# Unit tests: risk metrics, attribution, alerts
# =============================================================================
import pytest
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.analytics.risk import (
    portfolio_returns, var_historical, cvar_historical,
    var_parametric, max_drawdown, sharpe_ratio, sortino_ratio,
    rolling_volatility, drawdown_series, full_metrics,
)
from src.analytics.attribution import brinson_attribution, summary_attribution
from src.alerts.manager import AlertManager

np.random.seed(42)
N = 252
TICKERS = ["A", "B", "C"]
RETS = pd.DataFrame(
    np.random.normal(0.0004, 0.015, (N, 3)), columns=TICKERS)
WEIGHTS = [0.5, 0.3, 0.2]
PORT = portfolio_returns(RETS, WEIGHTS)

class TestPortfolioReturns:
    def test_shape(self):
        assert len(PORT) == N

    def test_weighted_correctly(self):
        manual = RETS.dot(np.array(WEIGHTS))
        pd.testing.assert_series_equal(PORT, manual.rename("Portfolio"), check_names=False)

    def test_weights_normalized(self):
        port2 = portfolio_returns(RETS, [5, 3, 2])
        pd.testing.assert_series_equal(PORT, port2, check_names=False)

class TestVaR:
    def test_var_positive(self):
        assert var_historical(PORT) > 0

    def test_var99_geq_var95(self):
        assert var_historical(PORT, 0.01) >= var_historical(PORT, 0.05)

    def test_cvar_geq_var(self):
        assert cvar_historical(PORT) >= var_historical(PORT)

    def test_var_parametric_positive(self):
        assert var_parametric(PORT) > 0

    def test_var_level(self):
        v = var_historical(PORT)
        assert 0 < v < 0.20

class TestDrawdown:
    def test_max_drawdown_negative(self):
        assert max_drawdown(PORT) <= 0

    def test_drawdown_series_max_zero(self):
        dd = drawdown_series(PORT)
        assert dd.max() <= 1e-10

    def test_drawdown_series_length(self):
        assert len(drawdown_series(PORT)) == N

class TestRatios:
    def test_sharpe_finite(self):
        assert np.isfinite(sharpe_ratio(PORT))

    def test_sortino_finite(self):
        assert np.isfinite(sortino_ratio(PORT))

    def test_rolling_vol_length(self):
        rv = rolling_volatility(PORT, 21)
        assert len(rv) == N

    def test_rolling_vol_positive(self):
        rv = rolling_volatility(PORT, 21).dropna()
        assert (rv > 0).all()

class TestFullMetrics:
    def test_keys_present(self):
        m = full_metrics(PORT)
        for k in ["Ann. Return", "Sharpe Ratio", "VaR 95% (1d)", "Max Drawdown"]:
            assert k in m

    def test_positive_days_in_range(self):
        m = full_metrics(PORT)
        assert 0 <= m["Positive Days %"] <= 100

class TestAttribution:
    def test_output_shape(self):
        pw = np.array(WEIGHTS)
        bw = np.ones(3) / 3
        pr = np.array([0.001, 0.002, -0.001])
        br = np.array([0.0008, 0.0015, -0.0005])
        df = brinson_attribution(pw, bw, pr, br, TICKERS)
        assert df.shape == (3, 9)

    def test_active_ret_equals_sum(self):
        pw = np.array(WEIGHTS)
        bw = np.ones(3) / 3
        pr = np.array([0.001, 0.002, -0.001])
        br = np.array([0.0008, 0.0015, -0.0005])
        df = brinson_attribution(pw, bw, pr, br, TICKERS)
        for _, row in df.iterrows():
            total = row["Allocation"] + row["Selection"] + row["Interaction"]
            assert abs(total - row["Active Ret"]) < 1e-8

class TestAlerts:
    def test_no_alerts_normal(self):
        m = {"VaR 95% (1d)": 0.5, "Max Drawdown": -2.0, "Ann. Volatility": 10.0}
        mgr = AlertManager()
        alerts = mgr.check(m)
        assert len(alerts) == 0

    def test_critical_alert_high_var(self):
        m = {"VaR 95% (1d)": 5.0, "Max Drawdown": -5.0, "Ann. Volatility": 10.0}
        mgr = AlertManager()
        alerts = mgr.check(m)
        assert any(a.severity == "CRITICAL" for a in alerts)

    def test_summary_counts(self):
        m = {"VaR 95% (1d)": 5.0, "Max Drawdown": -20.0, "Ann. Volatility": 30.0}
        mgr = AlertManager()
        mgr.check(m)
        s = mgr.summary()
        assert s["total"] == s["critical"] + s["warning"] + s["info"]
