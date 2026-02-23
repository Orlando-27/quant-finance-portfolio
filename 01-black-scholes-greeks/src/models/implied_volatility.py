"""
Implied Volatility Solver
==========================

Numerical methods for extracting implied volatility from observed
market option prices by inverting the Black-Scholes formula.

Methods:
    1. Newton-Raphson with Vega as analytical derivative (quadratic convergence)
    2. Brent's method as robust fallback (guaranteed convergence)
    3. Brenner-Subrahmanyam (1988) approximation for initial guess

Author: Jose Orlando Bobadilla Fuentes | CQF

References:
    Jaeckel, P. (2015). Let's Be Rational. Wilmott Magazine.
    Brenner, M., & Subrahmanyam, M.G. (1988). A Simple Formula to Compute
    the Implied Standard Deviation. FAJ.
"""

import numpy as np
from scipy.optimize import brentq
from typing import Optional

from .black_scholes import BlackScholesEngine, OptionParameters, OptionType


class ImpliedVolatilitySolver:
    """
    Numerical solver for Black-Scholes implied volatility.

    Attempts Newton-Raphson first for speed, falling back to Brent's
    method if convergence is not achieved.

    Usage:
        >>> solver = ImpliedVolatilitySolver()
        >>> iv = solver.solve(market_price=10.45, S=100, K=100, T=1.0,
        ...                   r=0.05, q=0.0, option_type=OptionType.CALL)
    """

    def __init__(self, tol: float = 1e-8, max_iter: int = 100,
                 vol_bounds: tuple = (0.001, 5.0)):
        self.engine = BlackScholesEngine()
        self.tol = tol
        self.max_iter = max_iter
        self.vol_bounds = vol_bounds

    def _brenner_guess(self, price: float, S: float, K: float,
                       T: float, r: float, q: float) -> float:
        """Brenner-Subrahmanyam (1988) initial guess for ATM options."""
        forward = S * np.exp((r - q) * T)
        return price * np.sqrt(2.0 * np.pi / T) / forward

    def _newton_raphson(self, market_price, S, K, T, r, q, option_type,
                        initial_guess=None):
        """
        Newton-Raphson solver: sigma_{n+1} = sigma_n - [BS(sigma_n) - mkt] / Vega
        Quadratic convergence near the solution.
        """
        sigma = initial_guess or self._brenner_guess(market_price, S, K, T, r, q)
        sigma = max(self.vol_bounds[0], min(sigma, self.vol_bounds[1]))

        for _ in range(self.max_iter):
            params = OptionParameters(S=S, K=K, T=T, r=r, sigma=sigma, q=q)
            bs_price = self.engine.price(params, option_type)
            vega = self.engine.vega(params) * 100.0  # raw vega

            if abs(vega) < 1e-12:
                return None

            diff = bs_price - market_price
            if abs(diff) < self.tol:
                return sigma

            sigma -= diff / vega
            if sigma <= self.vol_bounds[0] or sigma >= self.vol_bounds[1]:
                return None

        return None

    def _brent_solver(self, market_price, S, K, T, r, q, option_type):
        """Brent's method fallback: guaranteed convergence within brackets."""
        def obj(sigma):
            params = OptionParameters(S=S, K=K, T=T, r=r, sigma=sigma, q=q)
            return self.engine.price(params, option_type) - market_price
        try:
            return brentq(obj, self.vol_bounds[0], self.vol_bounds[1],
                          xtol=self.tol, maxiter=self.max_iter)
        except ValueError:
            return None

    def solve(self, market_price, S, K, T, r, q, option_type,
              initial_guess=None):
        """
        Extract implied volatility from market price.
        Strategy: Newton-Raphson first, Brent's fallback.
        """
        if option_type == OptionType.CALL:
            intrinsic = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0)
        else:
            intrinsic = max(K * np.exp(-r * T) - S * np.exp(-q * T), 0)

        if market_price < intrinsic - self.tol:
            return None

        iv = self._newton_raphson(market_price, S, K, T, r, q, option_type,
                                  initial_guess)
        return iv if iv is not None else self._brent_solver(
            market_price, S, K, T, r, q, option_type)

    def solve_surface(self, market_prices, S, strikes, expirations, r, q,
                      option_type):
        """
        Construct implied volatility surface from a grid of market prices.
        Returns 2D array (n_expirations x n_strikes), NaN where solver fails.
        """
        n_T, n_K = len(expirations), len(strikes)
        iv_surface = np.full((n_T, n_K), np.nan)
        for i, T in enumerate(expirations):
            for j, K in enumerate(strikes):
                price = market_prices[i, j]
                if np.isnan(price) or price <= 0:
                    continue
                iv = self.solve(price, S, K, T, r, q, option_type)
                if iv is not None:
                    iv_surface[i, j] = iv
        return iv_surface
