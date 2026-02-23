"""
Ornstein-Uhlenbeck Process Calibration & Simulation
====================================================

The OU process is the continuous-time analog of a stationary AR(1) process
and is the canonical model for mean-reverting spreads in pairs trading.

    dS_t = kappa * (theta - S_t) * dt + sigma * dW_t

Calibration methods:
    1. OLS on discretized AR(1): S_t = c + phi * S_{t-1} + eps
       kappa = -ln(phi)/dt, theta = c/(1-phi), sigma from residual variance.
    2. Maximum Likelihood: closed-form MLE for OU parameters.

Applications:
    - Half-life estimation for trade holding period
    - Optimal entry/exit threshold computation
    - Monte Carlo simulation for strategy stress testing
    - Expected P&L calculation conditional on mean reversion

References:
    Uhlenbeck & Ornstein (1930), Vasicek (1977), Elliott et al. (2005)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from scipy.optimize import minimize_scalar


class OrnsteinUhlenbeck:
    """
    Ornstein-Uhlenbeck process calibration, analysis, and simulation.

    Parameters
    ----------
    dt : float
        Time step in years (default 1/252 for daily data).
    """

    def __init__(self, dt: float = 1.0 / 252):
        self.dt = dt
        self.kappa = None    # speed of mean reversion
        self.theta = None    # long-run mean
        self.sigma = None    # volatility
        self.half_life = None
        self._fitted = False

    def fit_ols(self, spread: np.ndarray) -> Dict:
        """
        Calibrate OU parameters via OLS on the discretized AR(1) model.

        The discrete-time representation:
            S_t = c + phi * S_{t-1} + eps_t

        maps to continuous-time OU parameters:
            phi = exp(-kappa * dt)
            c = theta * (1 - phi)
            sigma_eps = sigma * sqrt((1 - phi^2) / (2 * kappa))

        Parameters
        ----------
        spread : np.ndarray
            Time series of spread values.

        Returns
        -------
        dict
            kappa, theta, sigma, half_life, phi, ols_r_squared.
        """
        y = spread[1:]
        x = spread[:-1]
        X = np.column_stack([np.ones(len(x)), x])

        coef = np.linalg.lstsq(X, y, rcond=None)[0]
        c, phi = coef[0], coef[1]
        resid = y - X @ coef
        sigma_eps = resid.std()

        # Map to OU parameters
        if 0 < phi < 1:
            kappa = -np.log(phi) / self.dt
            theta = c / (1 - phi)
            sigma = sigma_eps * np.sqrt(
                2 * kappa / (1 - phi ** 2)
            )
            half_life = np.log(2) / kappa
        else:
            kappa, theta, sigma, half_life = 0.0, spread.mean(), spread.std(), 999.0

        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.half_life = half_life
        self._fitted = True

        # R-squared
        ss_res = (resid ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        return {
            "kappa": kappa, "theta": theta, "sigma": sigma,
            "half_life": half_life, "half_life_days": half_life * 252,
            "phi": phi, "ols_r_squared": r2,
        }

    def fit_mle(self, spread: np.ndarray) -> Dict:
        """
        Calibrate OU parameters via Maximum Likelihood Estimation.

        The transition density of the OU process is Gaussian:
            S_t | S_{t-1} ~ N(mu_cond, var_cond)

        where:
            mu_cond = theta + (S_{t-1} - theta) * exp(-kappa * dt)
            var_cond = sigma^2 / (2*kappa) * (1 - exp(-2*kappa*dt))

        The log-likelihood is the sum of Gaussian log-densities.

        Parameters
        ----------
        spread : np.ndarray
            Time series of spread values.

        Returns
        -------
        dict
            kappa, theta, sigma, half_life, log_likelihood.
        """
        n = len(spread)

        def neg_log_lik(params):
            kappa, theta, sigma = params
            if kappa <= 0 or sigma <= 0:
                return 1e12
            exp_k = np.exp(-kappa * self.dt)
            var_cond = (sigma ** 2) / (2 * kappa) * (1 - np.exp(-2 * kappa * self.dt))
            if var_cond <= 0:
                return 1e12
            mu_cond = theta + (spread[:-1] - theta) * exp_k
            resid = spread[1:] - mu_cond
            ll = -0.5 * (n - 1) * np.log(2 * np.pi * var_cond) \
                 - 0.5 * np.sum(resid ** 2) / var_cond
            return -ll

        # Initialize from OLS
        ols_res = self.fit_ols(spread)
        x0 = [max(ols_res["kappa"], 0.01), ols_res["theta"],
               max(ols_res["sigma"], 0.01)]

        from scipy.optimize import minimize
        result = minimize(neg_log_lik, x0, method="Nelder-Mead",
                          options={"maxiter": 5000, "xatol": 1e-8})

        kappa, theta, sigma = result.x
        self.kappa = max(kappa, 1e-6)
        self.theta = theta
        self.sigma = max(sigma, 1e-6)
        self.half_life = np.log(2) / self.kappa
        self._fitted = True

        return {
            "kappa": self.kappa, "theta": self.theta, "sigma": self.sigma,
            "half_life": self.half_life,
            "half_life_days": self.half_life * 252,
            "log_likelihood": -result.fun,
            "converged": result.success,
        }

    def simulate(self, S0: float, n_steps: int, n_paths: int = 1000,
                 seed: int = 42) -> np.ndarray:
        """
        Monte Carlo simulation of OU process paths.

        Uses the exact transition density (not Euler discretization)
        for accurate simulation at any time step.

        Parameters
        ----------
        S0 : float
            Initial spread value.
        n_steps : int
            Number of time steps to simulate.
        n_paths : int
            Number of Monte Carlo paths.
        seed : int
            Random seed.

        Returns
        -------
        np.ndarray
            (n_steps+1 x n_paths) matrix of simulated paths.
        """
        if not self._fitted:
            raise RuntimeError("Fit the model first.")

        rng = np.random.RandomState(seed)
        exp_k = np.exp(-self.kappa * self.dt)
        var_cond = (self.sigma ** 2) / (2 * self.kappa) * (1 - exp_k ** 2)
        std_cond = np.sqrt(max(var_cond, 1e-12))

        paths = np.zeros((n_steps + 1, n_paths))
        paths[0] = S0

        for t in range(1, n_steps + 1):
            mu = self.theta + (paths[t - 1] - self.theta) * exp_k
            paths[t] = mu + std_cond * rng.randn(n_paths)

        return paths

    def expected_hitting_time(self, S0: float, target: float) -> float:
        """
        Expected time to hit target level from S0 under OU dynamics.

        For the OU process, the expected first passage time from S0 to
        theta (the mean) is approximately:
            E[tau] ~ half_life * |S0 - theta| / |target - theta|

        This is an approximation; exact formulas involve confluent
        hypergeometric functions.

        Parameters
        ----------
        S0 : float
            Current spread level.
        target : float
            Target spread level.

        Returns
        -------
        float
            Expected hitting time in the same units as dt.
        """
        if not self._fitted or self.kappa <= 0:
            return np.inf
        return abs(S0 - target) / (self.kappa * abs(self.theta - S0 + 1e-10))

    def stationary_distribution(self) -> Dict:
        """
        Return the stationary (long-run) distribution of the OU process.

        The stationary distribution is Gaussian:
            S ~ N(theta, sigma^2 / (2 * kappa))
        """
        if not self._fitted:
            raise RuntimeError("Fit the model first.")
        var_stat = self.sigma ** 2 / (2 * self.kappa) if self.kappa > 0 else np.inf
        return {
            "mean": self.theta,
            "variance": var_stat,
            "std": np.sqrt(var_stat),
        }
