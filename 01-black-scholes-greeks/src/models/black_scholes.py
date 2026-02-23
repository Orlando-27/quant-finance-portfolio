"""
Black-Scholes-Merton Options Pricing Engine
============================================

This module implements the core Black-Scholes-Merton (BSM) framework for
pricing European options. The implementation follows the original derivation
by Black & Scholes (1973) and the dividend extension by Merton (1973).

Mathematical Framework:
    Under the risk-neutral measure Q, the underlying asset price S(t) follows:

        dS = (r - q) * S * dt + sigma * S * dW^Q

    where r is the risk-free rate, q is the continuous dividend yield,
    sigma is the volatility, and W^Q is a standard Brownian motion under Q.

Author: Jose Orlando Bobadilla Fuentes | CQF
"""

import numpy as np
from scipy.stats import norm
from dataclasses import dataclass
from enum import Enum
from typing import Union


class OptionType(Enum):
    """Enumeration of option types."""
    CALL = "call"
    PUT = "put"


@dataclass
class OptionParameters:
    """
    Container for option pricing parameters.

    Attributes:
        S: Current spot price of the underlying asset
        K: Strike price of the option
        T: Time to expiration in years (T > 0)
        r: Annualized risk-free interest rate (continuous compounding)
        sigma: Annualized volatility of the underlying asset
        q: Continuous dividend yield (default: 0.0)

    Example:
        >>> params = OptionParameters(S=100, K=105, T=0.25, r=0.05, sigma=0.20)
    """
    S: float
    K: float
    T: float
    r: float
    sigma: float
    q: float = 0.0

    def __post_init__(self):
        """Validate input parameters after initialization."""
        if self.S <= 0:
            raise ValueError(f"Spot price must be positive, got {self.S}")
        if self.K <= 0:
            raise ValueError(f"Strike price must be positive, got {self.K}")
        if self.T <= 0:
            raise ValueError(f"Time to expiration must be positive, got {self.T}")
        if self.sigma <= 0:
            raise ValueError(f"Volatility must be positive, got {self.sigma}")


class BlackScholesEngine:
    """
    Production-grade Black-Scholes-Merton pricing engine.

    Computes European option prices and all first-order and second-order
    Greeks using closed-form analytical solutions.

    Supports:
        - European calls and puts with continuous dividend yield
        - Full Greeks suite (Delta, Gamma, Vega, Theta, Rho, Vanna, Volga)
        - Vectorized computation for surfaces and parameter sweeps
        - Put-call parity verification

    Usage:
        >>> engine = BlackScholesEngine()
        >>> params = OptionParameters(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        >>> price = engine.price(params, OptionType.CALL)
        >>> greeks = engine.compute_all_greeks(params, OptionType.CALL)
    """

    def __init__(self):
        """Initialize the Black-Scholes pricing engine."""
        self._norm_cdf = norm.cdf
        self._norm_pdf = norm.pdf

    def _compute_d1_d2(
        self,
        S: Union[float, np.ndarray],
        K: Union[float, np.ndarray],
        T: Union[float, np.ndarray],
        r: float,
        sigma: Union[float, np.ndarray],
        q: float = 0.0
    ) -> tuple:
        """
        Compute the d1 and d2 parameters of the Black-Scholes formula.

        Mathematical definition:
            d1 = [ln(S/K) + (r - q + 0.5 * sigma^2) * T] / (sigma * sqrt(T))
            d2 = d1 - sigma * sqrt(T)

        Parameters:
            S: Spot price (scalar or array for vectorized computation)
            K: Strike price (scalar or array)
            T: Time to expiration in years
            r: Risk-free rate
            sigma: Volatility (scalar or array)
            q: Continuous dividend yield

        Returns:
            Tuple of (d1, d2) values
        """
        sqrt_T = np.sqrt(T)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        return d1, d2

    def price(self, params: OptionParameters, option_type: OptionType) -> float:
        """
        Compute the Black-Scholes price of a European option.

        For a European call:
            C = S * exp(-q*T) * N(d1) - K * exp(-r*T) * N(d2)

        For a European put (via put-call parity):
            P = K * exp(-r*T) * N(-d2) - S * exp(-q*T) * N(-d1)

        Parameters:
            params: OptionParameters dataclass with pricing inputs
            option_type: OptionType.CALL or OptionType.PUT

        Returns:
            Option price as a float
        """
        d1, d2 = self._compute_d1_d2(
            params.S, params.K, params.T, params.r, params.sigma, params.q
        )
        discount = np.exp(-params.r * params.T)
        fwd_discount = np.exp(-params.q * params.T)

        if option_type == OptionType.CALL:
            return (params.S * fwd_discount * self._norm_cdf(d1)
                    - params.K * discount * self._norm_cdf(d2))
        else:
            return (params.K * discount * self._norm_cdf(-d2)
                    - params.S * fwd_discount * self._norm_cdf(-d1))

    def price_vectorized(
        self, S: np.ndarray, K: np.ndarray, T: np.ndarray,
        r: float, sigma: np.ndarray, q: float, option_type: OptionType
    ) -> np.ndarray:
        """
        Vectorized pricing for parameter sweeps and surface generation.
        Leverages NumPy broadcasting for implied vol surface construction.
        """
        d1, d2 = self._compute_d1_d2(S, K, T, r, sigma, q)
        disc = np.exp(-r * T)
        fwd = np.exp(-q * T)
        if option_type == OptionType.CALL:
            return S * fwd * self._norm_cdf(d1) - K * disc * self._norm_cdf(d2)
        else:
            return K * disc * self._norm_cdf(-d2) - S * fwd * self._norm_cdf(-d1)

    def delta(self, params: OptionParameters, option_type: OptionType) -> float:
        """
        Delta: first derivative of option price with respect to spot.
        Measures sensitivity to $1 change in underlying. Also the hedge ratio.

        Call Delta = exp(-q*T) * N(d1)        [range: 0 to 1]
        Put Delta  = -exp(-q*T) * N(-d1)      [range: -1 to 0]
        """
        d1, _ = self._compute_d1_d2(
            params.S, params.K, params.T, params.r, params.sigma, params.q
        )
        fwd = np.exp(-params.q * params.T)
        if option_type == OptionType.CALL:
            return fwd * self._norm_cdf(d1)
        else:
            return -fwd * self._norm_cdf(-d1)

    def gamma(self, params: OptionParameters) -> float:
        """
        Gamma: second derivative of price w.r.t. spot. Identical for calls/puts.
        Gamma = exp(-q*T) * n(d1) / (S * sigma * sqrt(T))
        """
        d1, _ = self._compute_d1_d2(
            params.S, params.K, params.T, params.r, params.sigma, params.q
        )
        fwd = np.exp(-params.q * params.T)
        return fwd * self._norm_pdf(d1) / (params.S * params.sigma * np.sqrt(params.T))

    def vega(self, params: OptionParameters) -> float:
        """
        Vega: sensitivity to 1% change in volatility. Identical for calls/puts.
        Vega = S * exp(-q*T) * n(d1) * sqrt(T) / 100
        """
        d1, _ = self._compute_d1_d2(
            params.S, params.K, params.T, params.r, params.sigma, params.q
        )
        fwd = np.exp(-params.q * params.T)
        return params.S * fwd * self._norm_pdf(d1) * np.sqrt(params.T) / 100.0

    def theta(self, params: OptionParameters, option_type: OptionType) -> float:
        """
        Theta: time decay per calendar day.
        Returned as daily theta (annual theta / 365).
        """
        d1, d2 = self._compute_d1_d2(
            params.S, params.K, params.T, params.r, params.sigma, params.q
        )
        fwd = np.exp(-params.q * params.T)
        disc = np.exp(-params.r * params.T)
        sqrt_T = np.sqrt(params.T)
        term1 = -(params.S * params.sigma * fwd * self._norm_pdf(d1)) / (2.0 * sqrt_T)

        if option_type == OptionType.CALL:
            theta_ann = (term1
                         - params.r * params.K * disc * self._norm_cdf(d2)
                         + params.q * params.S * fwd * self._norm_cdf(d1))
        else:
            theta_ann = (term1
                         + params.r * params.K * disc * self._norm_cdf(-d2)
                         - params.q * params.S * fwd * self._norm_cdf(-d1))
        return theta_ann / 365.0

    def rho(self, params: OptionParameters, option_type: OptionType) -> float:
        """
        Rho: sensitivity to 1% change in risk-free rate.
        Call Rho = K * T * exp(-r*T) * N(d2) / 100
        """
        _, d2 = self._compute_d1_d2(
            params.S, params.K, params.T, params.r, params.sigma, params.q
        )
        disc = np.exp(-params.r * params.T)
        if option_type == OptionType.CALL:
            return params.K * params.T * disc * self._norm_cdf(d2) / 100.0
        else:
            return -params.K * params.T * disc * self._norm_cdf(-d2) / 100.0

    def vanna(self, params: OptionParameters) -> float:
        """Vanna: cross-derivative d2C/(dS dsigma). How Delta changes with vol."""
        d1, d2 = self._compute_d1_d2(
            params.S, params.K, params.T, params.r, params.sigma, params.q
        )
        fwd = np.exp(-params.q * params.T)
        return -fwd * self._norm_pdf(d1) * d2 / params.sigma

    def volga(self, params: OptionParameters) -> float:
        """Volga (Vomma): d2C/dsigma2. Convexity of price w.r.t. volatility."""
        d1, d2 = self._compute_d1_d2(
            params.S, params.K, params.T, params.r, params.sigma, params.q
        )
        fwd = np.exp(-params.q * params.T)
        vega_raw = params.S * fwd * self._norm_pdf(d1) * np.sqrt(params.T)
        return vega_raw * d1 * d2 / params.sigma

    def compute_all_greeks(self, params: OptionParameters, option_type: OptionType) -> dict:
        """
        Compute all Greeks in a single pass for efficiency.
        Avoids redundant computation of d1, d2.

        Returns:
            Dictionary: delta, gamma, vega, theta, rho, vanna, volga
        """
        d1, d2 = self._compute_d1_d2(
            params.S, params.K, params.T, params.r, params.sigma, params.q
        )
        fwd = np.exp(-params.q * params.T)
        disc = np.exp(-params.r * params.T)
        sqrt_T = np.sqrt(params.T)
        n_d1 = self._norm_pdf(d1)

        if option_type == OptionType.CALL:
            delta = fwd * self._norm_cdf(d1)
        else:
            delta = -fwd * self._norm_cdf(-d1)

        gamma = fwd * n_d1 / (params.S * params.sigma * sqrt_T)
        vega = params.S * fwd * n_d1 * sqrt_T / 100.0

        term1 = -(params.S * params.sigma * fwd * n_d1) / (2.0 * sqrt_T)
        if option_type == OptionType.CALL:
            theta = (term1 - params.r * params.K * disc * self._norm_cdf(d2)
                     + params.q * params.S * fwd * self._norm_cdf(d1)) / 365.0
            rho = params.K * params.T * disc * self._norm_cdf(d2) / 100.0
        else:
            theta = (term1 + params.r * params.K * disc * self._norm_cdf(-d2)
                     - params.q * params.S * fwd * self._norm_cdf(-d1)) / 365.0
            rho = -params.K * params.T * disc * self._norm_cdf(-d2) / 100.0

        vanna = -fwd * n_d1 * d2 / params.sigma
        vega_raw = params.S * fwd * n_d1 * sqrt_T
        volga = vega_raw * d1 * d2 / params.sigma

        return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta,
                "rho": rho, "vanna": vanna, "volga": volga}

    def put_call_parity_check(self, params: OptionParameters) -> dict:
        """
        Verify put-call parity: C - P = S*exp(-qT) - K*exp(-rT).
        Fundamental no-arbitrage relationship.
        """
        call = self.price(params, OptionType.CALL)
        put = self.price(params, OptionType.PUT)
        theoretical = (params.S * np.exp(-params.q * params.T)
                       - params.K * np.exp(-params.r * params.T))
        actual = call - put
        return {
            "call_price": call, "put_price": put,
            "theoretical_C_minus_P": theoretical, "actual_C_minus_P": actual,
            "parity_error": abs(actual - theoretical),
            "parity_holds": abs(actual - theoretical) < 1e-10,
        }
