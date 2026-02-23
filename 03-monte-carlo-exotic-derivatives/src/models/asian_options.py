"""
Asian Option Pricing via Monte Carlo
=======================================

Types: Arithmetic average (no closed-form), Geometric average (closed-form).
Variance reduction via geometric Asian as control variate.

Author: Jose Orlando Bobadilla Fuentes | CQF
"""

import numpy as np
from scipy.stats import norm
from dataclasses import dataclass
from typing import Optional
from .path_generator import GBMPathGenerator, SimulationConfig


@dataclass
class AsianOptionResult:
    price: float
    std_error: float
    ci_lower: float
    ci_upper: float
    n_paths: int
    method: str


class AsianOptionPricer:
    """
    Monte Carlo pricer for Asian options with variance reduction.

    Usage:
        >>> pricer = AsianOptionPricer(config, K=100)
        >>> result = pricer.price_with_control_variate()
    """

    def __init__(self, config: SimulationConfig, K: float, n_averaging=None):
        self.config = config
        self.K = K
        self.n_avg = n_averaging or config.n_steps
        self.generator = GBMPathGenerator(config)

    def _geometric_closed_form(self) -> float:
        """Closed-form geometric Asian call (log-normal property)."""
        S0, r, sigma, T, q = (
            self.config.S0, self.config.r, self.config.sigma,
            self.config.T, self.config.q)
        n = self.n_avg
        sigma_G = sigma * np.sqrt((2*n + 1) / (6*(n + 1)))
        mu_G = 0.5 * (r - q - 0.5*sigma**2 + sigma_G**2)
        d1 = (np.log(S0/self.K) + (mu_G + 0.5*sigma_G**2)*T) / (sigma_G*np.sqrt(T))
        d2 = d1 - sigma_G * np.sqrt(T)
        return max(np.exp(-r*T) * (S0*np.exp(mu_G*T)*norm.cdf(d1)
                                    - self.K*norm.cdf(d2)), 0.0)

    def _get_averaging_prices(self, paths):
        step = max(1, paths.shape[1] // self.n_avg)
        return paths[:, step::step]

    def price_arithmetic_call(self) -> AsianOptionResult:
        """Standard MC: payoff = max(A_arith - K, 0)."""
        paths = self.generator.generate()
        A = np.mean(self._get_averaging_prices(paths), axis=1)
        payoffs = np.exp(-self.config.r * self.config.T) * np.maximum(A - self.K, 0)
        p = np.mean(payoffs)
        se = np.std(payoffs, ddof=1) / np.sqrt(self.config.n_paths)
        return AsianOptionResult(p, se, p-1.96*se, p+1.96*se,
                                 self.config.n_paths, "Standard MC")

    def price_with_control_variate(self) -> AsianOptionResult:
        """
        Control variate using geometric Asian as control.
        V_cv = V_arith - beta * (V_geo_mc - V_geo_exact)
        Typically reduces SE by 80-95%.
        """
        paths = self.generator.generate()
        avg = self._get_averaging_prices(paths)

        A_arith = np.mean(avg, axis=1)
        A_geo = np.exp(np.mean(np.log(avg), axis=1))

        disc = np.exp(-self.config.r * self.config.T)
        d_arith = disc * np.maximum(A_arith - self.K, 0)
        d_geo = disc * np.maximum(A_geo - self.K, 0)
        geo_exact = self._geometric_closed_form()

        cov = np.cov(d_arith, d_geo)
        beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 1.0
        cv = d_arith - beta * (d_geo - geo_exact)

        p = np.mean(cv)
        se = np.std(cv, ddof=1) / np.sqrt(self.config.n_paths)
        se_std = np.std(d_arith, ddof=1) / np.sqrt(self.config.n_paths)
        vr = (se_std / se)**2 if se > 0 else np.inf

        return AsianOptionResult(p, se, p-1.96*se, p+1.96*se,
                                 self.config.n_paths,
                                 f"Control Variate (VR: {vr:.1f}x)")

    def price_with_antithetic(self) -> AsianOptionResult:
        """Antithetic variates: average payoffs from (Z, -Z) pairs."""
        p_orig, p_anti = self.generator.generate_antithetic()
        disc = np.exp(-self.config.r * self.config.T)

        def payoff(paths):
            A = np.mean(self._get_averaging_prices(paths), axis=1)
            return disc * np.maximum(A - self.K, 0)

        paired = (payoff(p_orig) + payoff(p_anti)) / 2.0
        p = np.mean(paired)
        se = np.std(paired, ddof=1) / np.sqrt(self.config.n_paths)
        return AsianOptionResult(p, se, p-1.96*se, p+1.96*se,
                                 self.config.n_paths*2, "Antithetic Variates")
