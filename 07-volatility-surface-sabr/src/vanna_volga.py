"""
================================================================================
VANNA-VOLGA METHOD
================================================================================
Model-free implied volatility interpolation using three liquid instruments
(25-delta put, ATM, 25-delta call). Standard in FX options markets.

Based on: Castagna & Mercurio (2007).

Author: Jose Orlando Bobadilla Fuentes, CQF
================================================================================
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from dataclasses import dataclass

from src.implied_vol import black_price, vega_black


@dataclass
class VannaVolgaQuotes:
    """Three-pillar market quotes for Vanna-Volga."""
    K_put: float
    K_atm: float
    K_call: float
    vol_put: float
    vol_atm: float
    vol_call: float
    forward: float
    expiry: float
    df: float = 1.0


class VannaVolga:
    """Vanna-Volga implied volatility interpolation."""

    def __init__(self, quotes: VannaVolgaQuotes):
        self.q = quotes
        self._pillar_K = np.array([quotes.K_put, quotes.K_atm, quotes.K_call])
        self._pillar_vol = np.array([quotes.vol_put, quotes.vol_atm, quotes.vol_call])

    @staticmethod
    def _d1(F: float, K: float, T: float, sigma: float) -> float:
        return (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))

    def _vega(self, K: float, sigma: float) -> float:
        d1 = self._d1(self.q.forward, K, self.q.expiry, sigma)
        return self.q.df * self.q.forward * np.sqrt(self.q.expiry) * norm.pdf(d1)

    def _vanna(self, K: float, sigma: float) -> float:
        d1 = self._d1(self.q.forward, K, self.q.expiry, sigma)
        d2 = d1 - sigma * np.sqrt(self.q.expiry)
        vega = self._vega(K, sigma)
        return -d2 / (sigma * np.sqrt(self.q.expiry) + 1e-15) * vega / self.q.forward

    def _volga(self, K: float, sigma: float) -> float:
        d1 = self._d1(self.q.forward, K, self.q.expiry, sigma)
        d2 = d1 - sigma * np.sqrt(self.q.expiry)
        vega = self._vega(K, sigma)
        return vega * d1 * d2 / (sigma + 1e-15)

    def implied_vol(self, K: float) -> float:
        """Compute Vanna-Volga implied volatility at strike K."""
        F, T, df = self.q.forward, self.q.expiry, self.q.df
        sigma_atm = self.q.vol_atm

        y = np.log(K / F)
        y_pillars = np.log(self._pillar_K / F)

        costs = np.array([
            black_price(F, Ki, T, vi, "call", df)
            - black_price(F, Ki, T, sigma_atm, "call", df)
            for Ki, vi in zip(self._pillar_K, self._pillar_vol)
        ])

        y1, y2, y3 = y_pillars
        denom = (y1 - y2) * (y1 - y3) * (y2 - y3)
        if abs(denom) < 1e-15:
            return sigma_atm

        x1 = y * (y - y2) * (y - y3) / ((y1 - y2) * (y1 - y3) * y1 + 1e-15) * y1
        x2 = y * (y - y1) * (y - y3) / ((y2 - y1) * (y2 - y3) * y2 + 1e-15) * y2
        x3 = y * (y - y1) * (y - y2) / ((y3 - y1) * (y3 - y2) * y3 + 1e-15) * y3

        overhedge = x1 * costs[0] + x2 * costs[1] + x3 * costs[2]

        bs_price = black_price(F, K, T, sigma_atm, "call", df)
        target_price = bs_price + overhedge

        intrinsic = max(F - K, 0) * df
        if target_price <= intrinsic + 1e-10:
            return sigma_atm

        try:
            def obj(sigma):
                return black_price(F, K, T, sigma, "call", df) - target_price
            iv = brentq(obj, 1e-4, 3.0, xtol=1e-10)
            return iv
        except ValueError:
            return sigma_atm

    def smile(self, strikes: np.ndarray) -> np.ndarray:
        """Compute Vanna-Volga smile across strikes."""
        return np.array([self.implied_vol(K) for K in strikes])

    def fit_quality(self) -> dict:
        """Check fit at the three pillar points."""
        model_vols = self.smile(self._pillar_K)
        errors = model_vols - self._pillar_vol
        return {
            "pillar_strikes": self._pillar_K,
            "market_vols": self._pillar_vol,
            "model_vols": model_vols,
            "errors": errors,
            "max_error": np.max(np.abs(errors)),
        }
