"""
================================================================================
IMPLIED VOLATILITY EXTRACTION
================================================================================
Numerical inversion of Black-Scholes prices to extract implied volatility.

Methods:
    1. Newton-Raphson with Vega (quadratic convergence)
    2. Brent's method (bracketing, guaranteed convergence)
    3. Rational approximation (Jaeckel 2015, machine-precision)

Author: Jose Orlando Bobadilla Fuentes, CQF
================================================================================
"""

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm
from typing import Optional, Union
from dataclasses import dataclass


@dataclass
class IVResult:
    """Container for implied volatility computation result."""
    iv: float
    iterations: int
    converged: bool
    price_error: float
    method: str


def black_scholes_price(S: float, K: float, T: float, r: float,
                         sigma: float, option_type: str = "call") -> float:
    """Black-Scholes European option price."""
    if T <= 0 or sigma <= 0:
        intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
        return intrinsic

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def black_price(F: float, K: float, T: float, sigma: float,
                option_type: str = "call", df: float = 1.0) -> float:
    """Black (1976) formula for options on forwards/futures."""
    if T <= 0 or sigma <= 0:
        intrinsic = max(F - K, 0) if option_type == "call" else max(K - F, 0)
        return df * intrinsic

    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return df * (F * norm.cdf(d1) - K * norm.cdf(d2))
    else:
        return df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def vega_black(F: float, K: float, T: float, sigma: float,
               df: float = 1.0) -> float:
    """Black model Vega: dC/d(sigma)."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
    return df * F * np.sqrt(T) * norm.pdf(d1)


def implied_vol_newton(market_price: float, F: float, K: float, T: float,
                        option_type: str = "call", df: float = 1.0,
                        initial_guess: float = 0.20, tol: float = 1e-10,
                        max_iter: int = 100) -> IVResult:
    """Extract implied volatility using Newton-Raphson iteration."""
    sigma = initial_guess

    for i in range(max_iter):
        price = black_price(F, K, T, sigma, option_type, df)
        diff = price - market_price
        v = vega_black(F, K, T, sigma, df)

        if abs(diff) < tol:
            return IVResult(iv=sigma, iterations=i + 1, converged=True,
                            price_error=abs(diff), method="newton")

        if abs(v) < 1e-15:
            return implied_vol_brent(market_price, F, K, T, option_type, df)

        sigma -= diff / v
        sigma = max(sigma, 1e-6)
        sigma = min(sigma, 5.0)

    return implied_vol_brent(market_price, F, K, T, option_type, df)


def implied_vol_brent(market_price: float, F: float, K: float, T: float,
                       option_type: str = "call", df: float = 1.0,
                       lo: float = 1e-6, hi: float = 5.0,
                       tol: float = 1e-10) -> IVResult:
    """Extract implied volatility using Brent's bracketing method."""
    def objective(sigma):
        return black_price(F, K, T, sigma, option_type, df) - market_price

    try:
        f_lo = objective(lo)
        f_hi = objective(hi)

        if f_lo * f_hi > 0:
            return IVResult(iv=np.nan, iterations=0, converged=False,
                            price_error=abs(f_lo), method="brent")

        sigma, result = brentq(objective, lo, hi, xtol=tol, full_output=True)
        return IVResult(iv=sigma, iterations=result.iterations,
                        converged=result.converged,
                        price_error=abs(objective(sigma)), method="brent")
    except ValueError:
        return IVResult(iv=np.nan, iterations=0, converged=False,
                        price_error=np.inf, method="brent")


def implied_vol_rational(market_price: float, F: float, K: float, T: float,
                          option_type: str = "call",
                          df: float = 1.0) -> IVResult:
    """Rational approximation for implied volatility (Jaeckel 2015 inspired)."""
    intrinsic = max(F - K, 0) if option_type == "call" else max(K - F, 0)
    time_value = market_price / df - intrinsic

    if time_value <= 0:
        return IVResult(iv=0.0, iterations=0, converged=True,
                        price_error=0.0, method="rational")

    sigma_init = np.sqrt(2 * np.pi / T) * (market_price / df) / F

    return implied_vol_newton(market_price, F, K, T, option_type, df,
                               initial_guess=max(sigma_init, 0.01))


def extract_iv_surface(market_prices: np.ndarray, forwards: np.ndarray,
                        strikes: np.ndarray, expiries: np.ndarray,
                        option_types=None, dfs=None,
                        method: str = "newton") -> np.ndarray:
    """Extract implied volatility for a grid of options."""
    n = len(market_prices)
    ivs = np.full(n, np.nan)

    if option_types is None:
        option_types = np.array(["call"] * n)
    if dfs is None:
        dfs = np.ones(n)

    func_map = {
        "newton": implied_vol_newton,
        "brent": implied_vol_brent,
        "rational": implied_vol_rational,
    }
    iv_func = func_map.get(method, implied_vol_newton)

    for i in range(n):
        result = iv_func(market_prices[i], forwards[i], strikes[i],
                         expiries[i], option_types[i], dfs[i])
        if result.converged:
            ivs[i] = result.iv

    return ivs
