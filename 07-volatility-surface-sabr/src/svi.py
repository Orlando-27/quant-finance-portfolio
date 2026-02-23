"""
================================================================================
SVI: STOCHASTIC VOLATILITY INSPIRED PARAMETRIZATION
================================================================================
Gatheral (2004) parametric model for the total implied variance surface.

Raw SVI:  w(k) = a + b * (rho*(k-m) + sqrt((k-m)^2 + sigma^2))

Includes Gatheral-Jacquier (2014) arbitrage-free conditions and
quasi-explicit calibration.

Author: Jose Orlando Bobadilla Fuentes, CQF
================================================================================
"""

import numpy as np
from scipy.optimize import minimize, least_squares
from typing import Optional, Dict
from dataclasses import dataclass


@dataclass
class SVIParams:
    """Raw SVI parameters."""
    a: float
    b: float
    rho: float
    m: float
    sigma: float


@dataclass
class SVIFitResult:
    """SVI calibration result."""
    params: SVIParams
    rmse: float
    max_error: float
    arbitrage_free: bool
    model_variance: np.ndarray
    market_variance: np.ndarray


class SVIModel:
    """SVI implied variance parametrization with arbitrage checks."""

    def __init__(self):
        self._params: Optional[SVIParams] = None
        self._log_moneyness: Optional[np.ndarray] = None

    @property
    def params(self) -> Optional[SVIParams]:
        return self._params

    @staticmethod
    def svi_raw(k: np.ndarray, a: float, b: float, rho: float,
                m: float, sigma: float) -> np.ndarray:
        """Raw SVI total variance function."""
        km = k - m
        return a + b * (rho * km + np.sqrt(km ** 2 + sigma ** 2))

    def total_variance(self, k: np.ndarray) -> np.ndarray:
        """Evaluate fitted SVI at given log-moneyness values."""
        if self._params is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        p = self._params
        return self.svi_raw(k, p.a, p.b, p.rho, p.m, p.sigma)

    def implied_vol(self, k: np.ndarray, T: float) -> np.ndarray:
        """Convert total variance to implied volatility for given expiry."""
        w = self.total_variance(k)
        return np.sqrt(np.maximum(w / T, 0))

    def fit(self, k: np.ndarray, w_market: np.ndarray,
            weights: Optional[np.ndarray] = None,
            method: str = "quasi_explicit") -> SVIFitResult:
        """Calibrate raw SVI to market total variance."""
        k = np.asarray(k, dtype=float)
        w_market = np.asarray(w_market, dtype=float)
        self._log_moneyness = k

        if weights is None:
            weights = np.ones_like(k)
        weights = weights / weights.sum()

        if method == "quasi_explicit":
            params = self._quasi_explicit_fit(k, w_market, weights)
        else:
            params = self._least_squares_fit(k, w_market, weights)

        self._params = params
        model_w = self.svi_raw(k, params.a, params.b, params.rho,
                                params.m, params.sigma)
        errors = model_w - w_market
        rmse = np.sqrt(np.mean(errors ** 2))
        max_err = np.max(np.abs(errors))
        arb_free = self._check_butterfly_arbitrage(k)

        return SVIFitResult(
            params=params, rmse=rmse, max_error=max_err,
            arbitrage_free=arb_free, model_variance=model_w,
            market_variance=w_market
        )

    def _quasi_explicit_fit(self, k: np.ndarray, w: np.ndarray,
                             weights: np.ndarray) -> SVIParams:
        """Quasi-explicit SVI calibration (Zeliade 2012 approach)."""
        best_obj = np.inf
        best_params = None

        m_grid = np.linspace(k.min() - 0.05, k.max() + 0.05, 30)
        s_grid = np.linspace(0.01, 0.5, 20)

        for m_try in m_grid:
            for s_try in s_grid:
                km = k - m_try
                sqrt_term = np.sqrt(km ** 2 + s_try ** 2)

                A = np.column_stack([np.ones_like(k), km, sqrt_term])
                W = np.diag(weights)
                try:
                    theta = np.linalg.lstsq(W @ A, W @ w, rcond=None)[0]
                except np.linalg.LinAlgError:
                    continue

                a_fit = theta[0]
                b_rho = theta[1]
                b_fit = theta[2]

                if b_fit <= 0:
                    continue

                rho_fit = b_rho / b_fit
                if abs(rho_fit) >= 1:
                    continue

                model_w = self.svi_raw(k, a_fit, b_fit, rho_fit, m_try, s_try)
                obj = np.sum(weights * (model_w - w) ** 2)

                if obj < best_obj:
                    best_obj = obj
                    best_params = SVIParams(a=a_fit, b=b_fit, rho=rho_fit,
                                            m=m_try, sigma=s_try)

        if best_params is None:
            return self._least_squares_fit(k, w, weights)

        return self._polish(k, w, weights, best_params)

    def _least_squares_fit(self, k: np.ndarray, w: np.ndarray,
                            weights: np.ndarray) -> SVIParams:
        """Direct nonlinear least-squares fit."""
        w_atm = np.interp(0, k, w)

        def residuals(params):
            a, b, rho, m, sigma = params
            model = self.svi_raw(k, a, b, rho, m, sigma)
            return np.sqrt(weights) * (model - w)

        x0 = [w_atm, 0.1, -0.3, 0.0, 0.1]
        bounds = ([0, 1e-6, -0.999, k.min() - 1, 1e-4],
                  [2 * w.max(), 5.0, 0.999, k.max() + 1, 2.0])

        result = least_squares(residuals, x0, bounds=bounds, method="trf")
        return SVIParams(*result.x)

    def _polish(self, k: np.ndarray, w: np.ndarray,
                weights: np.ndarray, initial: SVIParams) -> SVIParams:
        """Local optimization to polish initial estimate."""
        def objective(params):
            a, b, rho, m, sigma = params
            model = self.svi_raw(k, a, b, rho, m, sigma)
            return np.sum(weights * (model - w) ** 2)

        x0 = [initial.a, initial.b, initial.rho, initial.m, initial.sigma]
        bounds = [(None, None), (1e-6, 5.0), (-0.999, 0.999),
                  (k.min() - 1, k.max() + 1), (1e-4, 2.0)]

        result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
        return SVIParams(*result.x)

    def _check_butterfly_arbitrage(self, k: np.ndarray) -> bool:
        """Check Gatheral-Jacquier (2014) sufficient conditions."""
        if self._params is None:
            return False

        p = self._params
        k_fine = np.linspace(k.min() - 0.1, k.max() + 0.1, 500)
        w = self.svi_raw(k_fine, p.a, p.b, p.rho, p.m, p.sigma)

        if np.any(w <= 0):
            return False

        dk = k_fine[1] - k_fine[0]
        w_prime = np.gradient(w, dk)
        w_double_prime = np.gradient(w_prime, dk)

        g = (1 - k_fine * w_prime / (2 * w)) ** 2 \
            - w_prime ** 2 / 4 * (1 / w + 0.25) \
            + w_double_prime / 2

        return bool(np.all(g >= -1e-8))

    def check_arbitrage(self) -> Dict[str, bool]:
        """Run all arbitrage diagnostics."""
        if self._params is None or self._log_moneyness is None:
            raise RuntimeError("Fit model first.")

        k = self._log_moneyness
        w = self.total_variance(k)

        return {
            "butterfly_free": self._check_butterfly_arbitrage(k),
            "positive_variance": bool(np.all(w > 0)),
            "positive_slope_right_wing": bool(self._params.b > 0),
            "valid_rho": bool(abs(self._params.rho) < 1),
        }
