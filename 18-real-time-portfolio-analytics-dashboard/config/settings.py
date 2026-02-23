# =============================================================================
# config/settings.py | Project 18 | Jose Orlando Bobadilla Fuentes | CQF
# Centralized configuration: colors, defaults, thresholds, watermark
# =============================================================================
from __future__ import annotations
from typing import Dict, List

WATERMARK = dict(
    text="Jose Orlando Bobadilla Fuentes | CQF | Senior Quant PM",
    xref="paper", yref="paper", x=0.99, y=0.01,
    xanchor="right", yanchor="bottom",
    font=dict(size=9, color="rgba(150,150,150,0.5)"), showarrow=False,
)
COLORS = {
    "bg":       "#0E1117", "bg2":      "#161B22", "bg3":      "#1C2128",
    "accent":   "#00D4FF", "accent2":  "#FF6B35", "green":    "#00FF88",
    "red":      "#FF4B4B", "yellow":   "#FFD700", "purple":   "#B48EFF",
    "grid":     "rgba(255,255,255,0.06)",
    "text":     "#E6EDF3", "text_muted": "#8B949E",
}
CHART_LAYOUT = dict(
    paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["bg2"],
    font=dict(family="monospace", color=COLORS["text"], size=11),
    xaxis=dict(gridcolor=COLORS["grid"], zeroline=False),
    yaxis=dict(gridcolor=COLORS["grid"], zeroline=False),
    margin=dict(l=60, r=30, t=60, b=50),
    legend=dict(bgcolor="rgba(22,27,34,0.85)", bordercolor=COLORS["bg3"], borderwidth=1),
    annotations=[WATERMARK],
)
DEFAULT_TICKERS   : List[str]   = ["AAPL", "MSFT", "GOOGL", "AMZN", "BRK-B"]
DEFAULT_WEIGHTS   : List[float] = [0.25, 0.25, 0.20, 0.15, 0.15]
DEFAULT_BENCHMARK : str         = "SPY"
DEFAULT_PERIOD    : str         = "1y"
RISK_FREE_RATE    : float       = 0.0525
TRADING_DAYS      : int         = 252
ALERT_THRESHOLDS  : Dict = {
    "var_95_pct": 0.02, "max_drawdown_pct": 0.10,
    "single_day_loss": 0.03, "volatility_ann": 0.25,
}
PERIODS : Dict = {
    "1 Month": "1mo", "3 Months": "3mo", "6 Months": "6mo",
    "1 Year": "1y",   "2 Years": "2y",   "5 Years": "5y",
}
