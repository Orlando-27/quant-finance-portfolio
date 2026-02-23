# =============================================================================
# src/visualization/charts.py | Project 18 | Jose Orlando Bobadilla Fuentes | CQF
# Dark-theme Plotly chart factory — watermark on all charts
# =============================================================================
from __future__ import annotations
import copy
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, List
from config.settings import COLORS, CHART_LAYOUT

def _base(title: str, h: int = 440) -> go.Figure:
    L = copy.deepcopy(CHART_LAYOUT)
    L["height"] = h
    L["title"]  = dict(text=title, font=dict(size=14, color=COLORS["text"]), x=0.02)
    return go.Figure(layout=go.Layout(**L))

def cumulative_return_chart(port_rets: pd.Series, bench_rets: pd.Series = None) -> go.Figure:
    fig = _base("Cumulative Return")
    cum = (1 + port_rets).cumprod() - 1
    fig.add_trace(go.Scatter(x=cum.index, y=cum * 100, name="Portfolio",
        line=dict(color=COLORS["accent"], width=2),
        fill="tozeroy", fillcolor="rgba(0,212,255,0.07)"))
    if bench_rets is not None:
        cb = (1 + bench_rets).cumprod() - 1
        fig.add_trace(go.Scatter(x=cb.index, y=cb * 100, name="Benchmark",
            line=dict(color=COLORS["yellow"], width=1.5, dash="dash")))
    fig.update_yaxes(title="Return (%)")
    return fig

def drawdown_chart(port_rets: pd.Series) -> go.Figure:
    fig = _base("Drawdown")
    cum = (1 + port_rets).cumprod()
    dd  = ((cum - cum.cummax()) / cum.cummax()) * 100
    fig.add_trace(go.Scatter(x=dd.index, y=dd, name="Drawdown",
        line=dict(color=COLORS["red"], width=1.5),
        fill="tozeroy", fillcolor="rgba(255,75,75,0.12)"))
    fig.update_yaxes(title="Drawdown (%)")
    return fig

def rolling_vol_chart(port_rets: pd.Series, window: int = 21) -> go.Figure:
    fig = _base("Rolling Volatility (21d)")
    rv  = port_rets.rolling(window).std() * (252 ** 0.5) * 100
    fig.add_trace(go.Scatter(x=rv.index, y=rv, name=f"Vol {window}d",
        line=dict(color=COLORS["purple"], width=2)))
    fig.update_yaxes(title="Ann. Volatility (%)")
    return fig

def var_histogram_chart(port_rets: pd.Series, var95: float, var99: float) -> go.Figure:
    fig = _base("Return Distribution & VaR")
    r = port_rets.dropna() * 100
    fig.add_trace(go.Histogram(x=r, nbinsx=60, name="Daily Returns",
        marker_color=COLORS["accent"], opacity=0.7))
    fig.add_vline(x=-var95, line=dict(color=COLORS["yellow"], dash="dash", width=2),
        annotation_text="VaR 95%", annotation_font_color=COLORS["yellow"])
    fig.add_vline(x=-var99, line=dict(color=COLORS["red"], dash="dot", width=2),
        annotation_text="VaR 99%", annotation_font_color=COLORS["red"])
    fig.update_xaxes(title="Daily Return (%)")
    return fig

def weight_donut_chart(tickers: List[str], weights: List[float]) -> go.Figure:
    fig = _base("Portfolio Weights", h=380)
    colors = [COLORS["accent"], COLORS["accent2"], COLORS["green"],
              COLORS["yellow"], COLORS["purple"]]
    fig.add_trace(go.Pie(labels=tickers, values=weights,
        hole=0.55, marker=dict(colors=colors[:len(tickers)]),
        textinfo="label+percent", textfont=dict(size=11)))
    return fig

def correlation_heatmap(returns: pd.DataFrame) -> go.Figure:
    fig = _base("Correlation Matrix", h=400)
    corr = returns.corr().round(2)
    fig.add_trace(go.Heatmap(z=corr.values, x=corr.columns.tolist(),
        y=corr.index.tolist(), colorscale="RdBu_r", zmid=0,
        text=corr.values, texttemplate="%{text}",
        colorbar=dict(thickness=12, len=0.8)))
    return fig

def attribution_waterfall(attr_summary: Dict) -> go.Figure:
    fig = _base("Performance Attribution (BHB)")
    cats   = ["Allocation", "Selection", "Interaction", "Active Return"]
    vals   = [attr_summary.get("Total Allocation", 0),
               attr_summary.get("Total Selection", 0),
               attr_summary.get("Total Interaction", 0),
               attr_summary.get("Total Active Ret", 0)]
    colors = [COLORS["green"] if v >= 0 else COLORS["red"] for v in vals]
    fig.add_trace(go.Bar(x=cats, y=vals, marker_color=colors, name="Effect"))
    fig.update_yaxes(title="Effect (%)")
    return fig

def risk_return_scatter(returns: pd.DataFrame, weights: List[float], rf: float = 0.0525) -> go.Figure:
    fig = _base("Risk-Return Map")
    ann_ret = returns.mean() * 252 * 100
    ann_vol = returns.std() * (252 ** 0.5) * 100
    for i, (ticker, vol, ret) in enumerate(zip(returns.columns, ann_vol, ann_ret)):
        fig.add_trace(go.Scatter(x=[vol], y=[ret], mode="markers+text",
            name=ticker, text=[ticker], textposition="top center",
            marker=dict(size=12)))
    xs = ann_vol.values
    if len(xs) > 0 and xs.max() > 0:
        cml_x = [0, float(xs.max()) * 1.2]
        slope = (ann_ret.mean() - rf * 100) / ann_vol.mean() if ann_vol.mean() > 0 else 1
        cml_y = [rf * 100, rf * 100 + slope * cml_x[1]]
        fig.add_trace(go.Scatter(x=cml_x, y=cml_y, mode="lines", name="CML",
            line=dict(color=COLORS["yellow"], dash="dash", width=1.5)))
    fig.update_xaxes(title="Ann. Volatility (%)")
    fig.update_yaxes(title="Ann. Return (%)")
    return fig
