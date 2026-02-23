# =============================================================================
# app.py | Project 18 | Jose Orlando Bobadilla Fuentes | CQF | MSc AI
# Real-Time Portfolio Analytics Dashboard — Streamlit multi-page app
# Usage: streamlit run app.py
# =============================================================================
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import numpy as np
import pandas as pd

from config.settings import (DEFAULT_TICKERS, DEFAULT_WEIGHTS,
    DEFAULT_BENCHMARK, DEFAULT_PERIOD, PERIODS, ALERT_THRESHOLDS)
from src.data.fetcher import fetch_prices, fetch_returns, fetch_benchmark
from src.analytics.risk import portfolio_returns, full_metrics, var_historical, rolling_volatility, drawdown_series
from src.analytics.attribution import brinson_attribution, summary_attribution
from src.alerts.manager import AlertManager
from src.visualization.charts import (
    cumulative_return_chart, drawdown_chart, rolling_vol_chart,
    var_histogram_chart, weight_donut_chart, correlation_heatmap,
    attribution_waterfall, risk_return_scatter,
)

st.set_page_config(page_title="Portfolio Analytics", layout="wide",
    initial_sidebar_state="expanded")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Portfolio Analytics")
    st.markdown("**Jose Orlando Bobadilla Fuentes | CQF**")
    st.markdown("---")

    tickers_input = st.text_input("Tickers (comma-separated)",
        value=", ".join(DEFAULT_TICKERS))
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    period_label = st.selectbox("Period", list(PERIODS.keys()), index=3)
    period = PERIODS[period_label]

    benchmark = st.text_input("Benchmark", value=DEFAULT_BENCHMARK).upper()

    st.markdown("**Weights (%)**")
    n = len(tickers)
    default_w = DEFAULT_WEIGHTS[:n] if n <= len(DEFAULT_WEIGHTS) else [1/n]*n
    raw_w = [st.number_input(t, min_value=0.0, max_value=100.0,
        value=round(default_w[i]*100, 1), step=0.1, key=f"w{i}")
        for i, t in enumerate(tickers)]
    total_w = sum(raw_w)
    weights = [w/total_w for w in raw_w] if total_w > 0 else [1/n]*n
    st.caption(f"Sum: {total_w:.1f}% (auto-normalised)")

    run = st.button("Refresh Data", type="primary", use_container_width=True)

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("Real-Time Portfolio Analytics Dashboard")
st.caption("Senior Quantitative Portfolio Manager | Colombian Pension Fund — Investment Division")
st.markdown("---")

@st.cache_data(ttl=300)
def load_data(tickers, benchmark, period):
    rets  = fetch_returns(tickers, period)
    bench = fetch_benchmark(benchmark, period)
    return rets, bench

with st.spinner("Loading market data..."):
    try:
        returns, bench_rets = load_data(tuple(tickers), benchmark, period)
        port_rets = portfolio_returns(returns, weights)
        metrics   = full_metrics(port_rets)
        alerts    = AlertManager().check(metrics)
    except Exception as e:
        st.error(f"Data error: {e}")
        st.stop()

# ── Alerts ────────────────────────────────────────────────────────────────────
if alerts:
    for a in alerts:
        if a.severity == "CRITICAL":
            st.error(f"[{a.severity}] {a.metric}: {a.message}")
        else:
            st.warning(f"[{a.severity}] {a.metric}: {a.message}")

# ── KPI Metrics ───────────────────────────────────────────────────────────────
cols = st.columns(6)
kpis = ["Ann. Return", "Ann. Volatility", "Sharpe Ratio",
        "Max Drawdown", "VaR 95% (1d)", "CVaR 95% (1d)"]
units = ["%", "%", "x", "%", "%", "%"]
for col, k, u in zip(cols, kpis, units):
    v = metrics.get(k, 0)
    delta_color = "inverse" if k in ["Max Drawdown","VaR 95% (1d)","CVaR 95% (1d)","Ann. Volatility"] else "normal"
    col.metric(k, f"{v}{u}")

st.markdown("---")

# ── Charts row 1 ──────────────────────────────────────────────────────────────
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(cumulative_return_chart(port_rets, bench_rets), use_container_width=True)
with c2:
    st.plotly_chart(drawdown_chart(port_rets), use_container_width=True)

c3, c4 = st.columns(2)
with c3:
    var95 = abs(metrics.get("VaR 95% (1d)", 0))
    var99 = abs(metrics.get("VaR 99% (1d)", 0))
    st.plotly_chart(var_histogram_chart(port_rets, var95, var99), use_container_width=True)
with c4:
    st.plotly_chart(rolling_vol_chart(port_rets), use_container_width=True)

# ── Charts row 2 ──────────────────────────────────────────────────────────────
c5, c6 = st.columns(2)
with c5:
    st.plotly_chart(weight_donut_chart(tickers, weights), use_container_width=True)
with c6:
    aligned = returns.reindex(columns=tickers).dropna()
    st.plotly_chart(correlation_heatmap(aligned), use_container_width=True)

# ── Attribution ───────────────────────────────────────────────────────────────
st.markdown("### Performance Attribution (BHB)")
bench_w = np.ones(len(tickers)) / len(tickers)
aligned_rets = returns.reindex(columns=tickers).dropna()
if len(aligned_rets) > 0:
    port_asset_ret  = aligned_rets.mean().values
    bench_asset_ret = aligned_rets.mean().values
    attr_df = brinson_attribution(np.array(weights), bench_w,
        port_asset_ret, bench_asset_ret, tickers)
    attr_sum = summary_attribution(attr_df)
    c7, c8 = st.columns([3, 2])
    with c7:
        st.dataframe(attr_df, use_container_width=True, hide_index=True)
    with c8:
        st.plotly_chart(attribution_waterfall(attr_sum), use_container_width=True)

# ── Risk-Return ───────────────────────────────────────────────────────────────
st.plotly_chart(risk_return_scatter(aligned_rets, weights), use_container_width=True)

# ── Full Metrics Table ────────────────────────────────────────────────────────
st.markdown("### Full Risk Metrics")
m_df = pd.DataFrame(metrics.items(), columns=["Metric", "Value"])
st.dataframe(m_df, use_container_width=True, hide_index=True)
