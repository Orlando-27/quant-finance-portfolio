"""
Portfolio Performance Metrics
===============================

Comprehensive risk-adjusted performance measures:
    Sharpe, Sortino, Max Drawdown, Calmar, Information Ratio,
    VaR, CVaR, Win Rate, Gain/Loss Ratio.

Author: Jose Orlando Bobadilla Fuentes | CQF
"""

import numpy as np
import pandas as pd
from typing import Optional


def annualized_return(returns, periods=252):
    return np.prod(1 + returns) ** (periods / len(returns)) - 1

def annualized_volatility(returns, periods=252):
    return np.std(returns, ddof=1) * np.sqrt(periods)

def sharpe_ratio(returns, rf=0.0, periods=252):
    ar = annualized_return(returns, periods)
    av = annualized_volatility(returns, periods)
    return (ar - rf) / av if av > 0 else 0.0

def sortino_ratio(returns, rf=0.0, periods=252):
    ar = annualized_return(returns, periods)
    target = rf / periods
    downside = returns[returns < target] - target
    if len(downside) == 0:
        return np.inf
    dd = np.sqrt(np.mean(downside ** 2)) * np.sqrt(periods)
    return (ar - rf) / dd if dd > 0 else 0.0

def max_drawdown(returns):
    cum = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cum)
    dd = cum / peak - 1
    trough = np.argmin(dd)
    peak_idx = np.argmax(cum[:trough + 1])
    return {"max_drawdown": float(dd[trough]), "peak_idx": int(peak_idx),
            "trough_idx": int(trough), "duration": int(trough - peak_idx)}

def calmar_ratio(returns, periods=252):
    ar = annualized_return(returns, periods)
    mdd = abs(max_drawdown(returns)["max_drawdown"])
    return ar / mdd if mdd > 0 else 0.0

def compute_all_metrics(returns, rf=0.0, periods=252, benchmark=None):
    """Compute full performance metrics suite."""
    mdd = max_drawdown(returns)
    m = {
        "Annualized Return": annualized_return(returns, periods),
        "Annualized Volatility": annualized_volatility(returns, periods),
        "Sharpe Ratio": sharpe_ratio(returns, rf, periods),
        "Sortino Ratio": sortino_ratio(returns, rf, periods),
        "Calmar Ratio": calmar_ratio(returns, periods),
        "Max Drawdown": mdd["max_drawdown"],
        "Drawdown Duration": mdd["duration"],
        "Skewness": float(pd.Series(returns).skew()),
        "Kurtosis": float(pd.Series(returns).kurtosis()),
        "VaR 95%": float(np.percentile(returns, 5)),
        "CVaR 95%": float(np.mean(returns[returns <= np.percentile(returns, 5)])),
        "Win Rate": float(np.mean(returns > 0)),
    }
    if benchmark is not None:
        excess = returns - benchmark[:len(returns)]
        te = np.std(excess, ddof=1) * np.sqrt(periods)
        m["Tracking Error"] = te
        m["Information Ratio"] = np.mean(excess) * periods / te if te > 0 else 0
    return pd.Series(m)
