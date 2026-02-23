"""
Cox-Ross-Rubinstein Binomial Tree for American Options
========================================================

CRR lattice model for pricing American-style options with early exercise.

Tree parameters:
    u = exp(sigma * sqrt(dt))       -- up factor
    d = 1/u                         -- down factor
    p = (exp((r-q)*dt) - d) / (u - d)  -- risk-neutral probability

Convergence: O(1/N). Richardson extrapolation accelerates to O(1/N^2).

Author: Jose Orlando Bobadilla Fuentes | CQF

References:
    Cox, J. C., Ross, S. A., & Rubinstein, M. (1979). Option Pricing:
    A Simplified Approach. Journal of Financial Economics, 7(3), 229-263.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from .black_scholes import OptionType


@dataclass
class BinomialTreeResult:
    """Container for binomial tree pricing results."""
    price: float
    delta: float
    gamma: float
    theta: float
    early_exercise_boundary: Optional[np.ndarray] = None


class BinomialTreePricer:
    """
    CRR binomial tree pricer for American and European options.

    Usage:
        >>> pricer = BinomialTreePricer(n_steps=500)
        >>> result = pricer.price(S=100, K=100, T=1.0, r=0.05, sigma=0.20,
        ...                       q=0.02, option_type=OptionType.PUT, american=True)
    """

    def __init__(self, n_steps: int = 200):
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")
        self.n_steps = n_steps

    def _price_core(self, S, K, T, r, sigma, q, option_type, american, N):
        """Core pricing logic: forward tree build, backward induction."""
        if T <= 0 or N < 1:
            if option_type == OptionType.CALL:
                return max(S - K, 0.0)
            return max(K - S, 0.0)

        dt = T / N
        u = np.exp(sigma * np.sqrt(dt))
        d = 1.0 / u
        p = (np.exp((r - q) * dt) - d) / (u - d)
        disc = np.exp(-r * dt)

        # Terminal payoffs
        j = np.arange(N + 1)
        ST = S * (u ** (N - j)) * (d ** j)
        if option_type == OptionType.CALL:
            V = np.maximum(ST - K, 0.0)
        else:
            V = np.maximum(K - ST, 0.0)

        # Backward induction with early exercise
        for i in range(N - 1, -1, -1):
            V = disc * (p * V[:i+1] + (1 - p) * V[1:i+2])
            if american:
                j_arr = np.arange(i + 1)
                S_ij = S * (u ** (i - j_arr)) * (d ** j_arr)
                if option_type == OptionType.CALL:
                    V = np.maximum(V, np.maximum(S_ij - K, 0.0))
                else:
                    V = np.maximum(V, np.maximum(K - S_ij, 0.0))
        return V[0]

    def price(self, S, K, T, r, sigma, q=0.0,
              option_type=OptionType.CALL, american=True):
        """
        Price an option using the CRR binomial tree.

        Phase 1 (Forward): S(i,j) = S * u^j * d^(i-j)
        Phase 2 (Backward): V = disc * [p*V_up + (1-p)*V_down]
            American: V = max(V_continuation, exercise_value)
        """
        price_val = self._price_core(S, K, T, r, sigma, q, option_type,
                                     american, self.n_steps)

        # Extract Greeks via shifted trees
        dt = T / self.n_steps
        dS = S * 0.01
        f_up = self._price_core(S + dS, K, T, r, sigma, q, option_type,
                                american, self.n_steps)
        f_dn = self._price_core(S - dS, K, T, r, sigma, q, option_type,
                                american, self.n_steps)
        delta = (f_up - f_dn) / (2 * dS)
        gamma = (f_up - 2 * price_val + f_dn) / (dS ** 2)

        f_dt = self._price_core(S, K, T - dt, r, sigma, q, option_type,
                                american, max(self.n_steps - 1, 1))
        theta = (f_dt - price_val) / dt / 365.0

        return BinomialTreeResult(price=price_val, delta=delta,
                                  gamma=gamma, theta=theta)

    def price_with_richardson(self, S, K, T, r, sigma, q=0.0,
                              option_type=OptionType.PUT, american=True):
        """
        Richardson extrapolation: P_extrap = 2*P(N) - P(N/2).
        Eliminates leading-order error, improving O(1/N) to O(1/N^2).
        """
        p_N = self._price_core(S, K, T, r, sigma, q, option_type,
                               american, self.n_steps)
        p_half = self._price_core(S, K, T, r, sigma, q, option_type,
                                  american, self.n_steps // 2)
        return 2.0 * p_N - p_half
