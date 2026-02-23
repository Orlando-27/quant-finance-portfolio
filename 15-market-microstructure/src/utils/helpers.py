"""
Statistical & Utility Helpers
================================
Common functions for rolling statistics, signal processing, and
performance metrics used across the microstructure modules.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def newey_west_tstat(
    x: pd.Series,
    lags: int = 5,
) -> tuple[float, float]:
    """
    Compute the Newey-West (1987) heteroscedasticity and autocorrelation
    consistent (HAC) t-statistic for H0: mean(x) = 0.

    Returns
    -------
    (t_stat, p_value)
    """
    n    = len(x)
    mu   = x.mean()
    # Sandwich variance estimate
    gamma_0 = np.var(x, ddof=1)
    kernel  = gamma_0
    for lag in range(1, lags + 1):
        cov = np.cov(x[lag:].values, x[:-lag].values)[0, 1]
        w   = 1.0 - lag / (lags + 1.0)
        kernel += 2.0 * w * cov
    se = np.sqrt(kernel / n)
    t  = mu / se if se > 0 else np.nan
    p  = 2.0 * (1.0 - stats.t.cdf(abs(t), df=n - 1)) if not np.isnan(t) else np.nan
    return float(t), float(p)


def rolling_sharpe(
    returns: pd.Series,
    window : int = 252,
    rf     : float = 0.0,
    annualize: bool = True,
) -> pd.Series:
    """Rolling Sharpe ratio."""
    excess = returns - rf / 252.0
    mu  = excess.rolling(window).mean()
    sig = excess.rolling(window).std()
    sr  = mu / sig.replace(0, np.nan)
    if annualize:
        sr *= np.sqrt(252)
    sr.name = "rolling_sharpe"
    return sr


def price_impact_regression(
    returns   : pd.Series,
    signed_vol: pd.Series,
) -> dict:
    """
    OLS regression of price changes on signed volume:
        r_t = α + λ * sv_t + ε_t

    Returns slope (λ), R², t-stat, p-value.
    """
    clean = pd.concat([returns, signed_vol], axis=1).dropna()
    slope, intercept, r, p, se = stats.linregress(
        clean.iloc[:, 1], clean.iloc[:, 0]
    )
    return {
        "lambda"    : slope,
        "intercept" : intercept,
        "r_squared" : r**2,
        "t_stat"    : slope / se if se > 0 else np.nan,
        "p_value"   : p,
    }


def compute_vwap(
    prices: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """Cumulative VWAP from the start of the series."""
    cum_pv  = (prices * volume).cumsum()
    cum_vol = volume.cumsum()
    vwap    = cum_pv / cum_vol.replace(0, np.nan)
    vwap.name = "vwap"
    return vwap


def compute_twap(prices: pd.Series) -> pd.Series:
    """Running TWAP (equal-time-weighted average price)."""
    n    = np.arange(1, len(prices) + 1)
    twap = pd.Series(
        np.cumsum(prices.values) / n,
        index=prices.index,
        name="twap",
    )
    return twap


def intraday_volume_profile(
    bars: pd.DataFrame,
    volume_col: str = "volume",
) -> pd.DataFrame:
    """
    Compute the intraday volume profile (fraction of daily volume
    at each bar), averaged across all trading days.
    """
    bars = bars.copy()
    bars["time_of_day"] = bars.index.time
    profile = bars.groupby("time_of_day")[volume_col].mean()
    profile = profile / profile.sum()
    profile.name = "volume_pct"
    return profile.reset_index()
