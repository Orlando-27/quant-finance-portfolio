# =============================================================================
# src/analytics/risk.py | Project 18 | Jose Orlando Bobadilla Fuentes | CQF
# Risk metrics: VaR, CVaR, Sharpe, Sortino, max drawdown, rolling vol
# =============================================================================
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional
from config.settings import RISK_FREE_RATE, TRADING_DAYS

def portfolio_returns(returns: pd.DataFrame, weights: List[float]) -> pd.Series:
    """Compute weighted portfolio log returns."""
    w = np.array(weights)
    w = w / w.sum()
    return returns.fillna(0).dot(w).rename("Portfolio")

def var_historical(rets: pd.Series, alpha: float = 0.05) -> float:
    """Historical VaR at confidence level (1-alpha)."""
    return float(-np.percentile(rets.dropna(), alpha * 100))

def cvar_historical(rets: pd.Series, alpha: float = 0.05) -> float:
    """Historical CVaR (Expected Shortfall)."""
    v = var_historical(rets, alpha)
    tail = rets[rets <= -v]
    return float(-tail.mean()) if len(tail) > 0 else v

def var_parametric(rets: pd.Series, alpha: float = 0.05) -> float:
    """Parametric (Gaussian) VaR."""
    mu, sigma = rets.mean(), rets.std()
    return float(-(mu + stats.norm.ppf(alpha) * sigma))

def max_drawdown(rets: pd.Series) -> float:
    """Maximum drawdown from cumulative returns."""
    cum = (1 + rets).cumprod()
    roll_max = cum.cummax()
    dd = (cum - roll_max) / roll_max
    return float(dd.min())

def sharpe_ratio(rets: pd.Series, rf: float = RISK_FREE_RATE) -> float:
    """Annualised Sharpe ratio."""
    excess = rets.mean() * TRADING_DAYS - rf
    vol = rets.std() * np.sqrt(TRADING_DAYS)
    return float(excess / vol) if vol > 0 else 0.0

def sortino_ratio(rets: pd.Series, rf: float = RISK_FREE_RATE) -> float:
    """Annualised Sortino ratio."""
    excess = rets.mean() * TRADING_DAYS - rf
    down = rets[rets < 0].std() * np.sqrt(TRADING_DAYS)
    return float(excess / down) if down > 0 else 0.0

def calmar_ratio(rets: pd.Series) -> float:
    """Calmar ratio: annualised return / max drawdown."""
    ann_ret = rets.mean() * TRADING_DAYS
    mdd = abs(max_drawdown(rets))
    return float(ann_ret / mdd) if mdd > 0 else 0.0

def rolling_volatility(rets: pd.Series, window: int = 21) -> pd.Series:
    """Rolling annualised volatility."""
    return rets.rolling(window).std() * np.sqrt(TRADING_DAYS)

def drawdown_series(rets: pd.Series) -> pd.Series:
    """Full drawdown time series."""
    cum = (1 + rets).cumprod()
    roll_max = cum.cummax()
    return (cum - roll_max) / roll_max

def full_metrics(rets: pd.Series) -> Dict:
    """All risk metrics in one call."""
    return {
        "Ann. Return":      round(rets.mean() * TRADING_DAYS * 100, 2),
        "Ann. Volatility":  round(rets.std() * np.sqrt(TRADING_DAYS) * 100, 2),
        "Sharpe Ratio":     round(sharpe_ratio(rets), 3),
        "Sortino Ratio":    round(sortino_ratio(rets), 3),
        "Calmar Ratio":     round(calmar_ratio(rets), 3),
        "Max Drawdown":     round(max_drawdown(rets) * 100, 2),
        "VaR 95% (1d)":     round(var_historical(rets) * 100, 3),
        "CVaR 95% (1d)":    round(cvar_historical(rets) * 100, 3),
        "VaR 99% (1d)":     round(var_historical(rets, 0.01) * 100, 3),
        "Skewness":         round(float(rets.skew()), 3),
        "Kurtosis":         round(float(rets.kurtosis()), 3),
        "Positive Days %":  round((rets > 0).mean() * 100, 1),
    }
