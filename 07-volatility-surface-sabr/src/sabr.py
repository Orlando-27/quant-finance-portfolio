"""
================================================================================
SABR MODEL: HAGAN & OBLOJ APPROXIMATIONS WITH CALIBRATION
================================================================================
Industry-standard stochastic volatility model for smile interpolation.

Implements:
    - Hagan et al. (2002) original approximation
    - Obloj (2008) corrected formula
    - Least-squares calibration with parameter bounds
    - ATM volatility and risk reversal / butterfly decomposition

Author: Jose Orlando Bobadilla Fuentes, CQF
================================================================================
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class SABRParams:
    """SABR model parameters."""
    alpha: float
    beta: float
    rho: float
    nu: float


@dataclass
class SABRCalibrationResult:
    """Result container for SABR calibration."""
    params: SABRParams
    rmse: float
    max_error: float
    vega_weighted_rmse: float
    model_vols: np.ndarray
    market_vols: np.ndarray
    strikes: np.ndarray
    converged: bool


class SABRModel:
    """
    SABR stochastic volatility model.

    Parameters
    ----------
    beta : float
        CEV exponent. beta=1 lognormal, beta=0.5 CIR, beta=0 normal.
    formula : str
        "hagan" (original) or "obloj" (corrected).
    """

    def __init__(self, beta: float = 0.5, formula: str = "hagan"):
        self.beta = beta
        self.formula = formula.lower()
        self._params: Optional[SABRParams] = None

    @property
    def params(self) -> Optional[SABRParams]:
        return self._params

    def _hagan_vol(self, K: float, F: float, T: float,
                    alpha: float, rho: float, nu: float) -> float:
        """Hagan et al. (2002) SABR implied Black volatility."""
        beta = self.beta

        if abs(F - K) < 1e-12:
            FK_mid = F
            logFK = 0.0
        else:
            FK_mid = (F * K) ** ((1 - beta) / 2)
            logFK = np.log(F / K)

        if abs(logFK) < 1e-12:
            term1 = alpha / (F ** (1 - beta))
            correction = 1 + (
                ((1 - beta) ** 2 / 24) * (alpha ** 2) / (F ** (2 * (1 - beta)))
                + 0.25 * rho * beta * nu * alpha / (F ** (1 - beta))
                + (2 - 3 * rho ** 2) / 24 * nu ** 2
            ) * T
            return term1 * correction

        z = (nu / alpha) * FK_mid * logFK
        x_z = np.log((np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho) / (1 - rho))

        if abs(x_z) < 1e-12:
            zeta_over_xz = 1.0
        else:
            zeta_over_xz = z / x_z

        prefix = alpha / (
            FK_mid * (
                1 + (1 - beta) ** 2 / 24 * logFK ** 2
                + (1 - beta) ** 4 / 1920 * logFK ** 4
            )
        )

        correction = 1 + (
            (1 - beta) ** 2 / 24 * alpha ** 2 / (F * K) ** (1 - beta)
            + 0.25 * rho * beta * nu * alpha / ((F * K) ** ((1 - beta) / 2))
            + (2 - 3 * rho ** 2) / 24 * nu ** 2
        ) * T

        return prefix * zeta_over_xz * correction

    def _obloj_vol(self, K: float, F: float, T: float,
                    alpha: float, rho: float, nu: float) -> float:
        """Obloj (2008) corrected SABR formula."""
        beta = self.beta

        if abs(F - K) < 1e-12:
            return self._hagan_vol(K, F, T, alpha, rho, nu)

        logFK = np.log(F / K)
        FK_beta = (F * K) ** ((1 - beta) / 2)

        if abs(1 - beta) < 1e-10:
            I1 = 0.0
        else:
            I1 = (F ** (1 - beta) - K ** (1 - beta)) / ((1 - beta) * logFK)

        z = (nu / alpha) * I1 * logFK
        x_z = np.log((np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho) / (1 - rho))

        if abs(x_z) < 1e-12:
            zeta_over_xz = 1.0
        else:
            zeta_over_xz = z / x_z

        prefix = alpha * logFK / (
            (F ** (1 - beta) - K ** (1 - beta)) / (1 - beta)
            if abs(1 - beta) > 1e-10 else logFK
        )

        correction = 1 + (
            (1 - beta) ** 2 / 24 * alpha ** 2 / (F * K) ** (1 - beta)
            + 0.25 * rho * beta * nu * alpha / FK_beta
            + (2 - 3 * rho ** 2) / 24 * nu ** 2
        ) * T

        return prefix * zeta_over_xz * correction

    def implied_vol(self, strikes: np.ndarray, forward: float, expiry: float,
                     alpha: Optional[float] = None, rho: Optional[float] = None,
                     nu: Optional[float] = None) -> np.ndarray:
        """Compute SABR implied volatility for given strikes."""
        if alpha is None:
            if self._params is None:
                raise RuntimeError("Model not calibrated. Call calibrate() first.")
            alpha = self._params.alpha
            rho = self._params.rho
            nu = self._params.nu

        vol_func = self._obloj_vol if self.formula == "obloj" else self._hagan_vol
        strikes = np.atleast_1d(strikes).astype(float)
        vols = np.array([
            vol_func(K, forward, expiry, alpha, rho, nu) for K in strikes
        ])
        return np.maximum(vols, 1e-8)

    def atm_vol(self, forward: float, expiry: float) -> float:
        """Compute ATM implied volatility."""
        return float(self.implied_vol(np.array([forward]), forward, expiry)[0])

    def calibrate(self, forward: float, strikes: np.ndarray,
                   market_vols: np.ndarray, expiry: float,
                   weights: Optional[np.ndarray] = None,
                   method: str = "local",
                   alpha_init: float = 0.2,
                   rho_init: float = -0.3,
                   nu_init: float = 0.4) -> SABRCalibrationResult:
        """Calibrate SABR parameters (alpha, rho, nu) to market implied vols."""
        strikes = np.asarray(strikes, dtype=float)
        market_vols = np.asarray(market_vols, dtype=float)

        if weights is None:
            moneyness = np.abs(np.log(strikes / forward))
            weights = np.exp(-0.5 * (moneyness / 0.1) ** 2)
            weights /= weights.sum()

        bounds = [(1e-6, 5.0), (-0.999, 0.999), (1e-6, 5.0)]

        def objective(params):
            a, r, n = params
            try:
                model_vols = self.implied_vol(strikes, forward, expiry, a, r, n)
                errors = (model_vols - market_vols) ** 2
                return np.sum(weights * errors)
            except Exception:
                return 1e10

        if method == "global":
            result = differential_evolution(objective, bounds, seed=42,
                                             maxiter=500, tol=1e-12)
        else:
            x0 = [alpha_init, rho_init, nu_init]
            result = minimize(objective, x0, method="L-BFGS-B",
                              bounds=bounds, options={"maxiter": 1000})

        alpha_opt, rho_opt, nu_opt = result.x
        self._params = SABRParams(
            alpha=alpha_opt, beta=self.beta,
            rho=rho_opt, nu=nu_opt
        )

        model_vols = self.implied_vol(strikes, forward, expiry)
        errors = model_vols - market_vols
        rmse = np.sqrt(np.mean(errors ** 2))
        max_err = np.max(np.abs(errors))
        vega_rmse = np.sqrt(np.sum(weights * errors ** 2))

        return SABRCalibrationResult(
            params=self._params, rmse=rmse, max_error=max_err,
            vega_weighted_rmse=vega_rmse, model_vols=model_vols,
            market_vols=market_vols, strikes=strikes,
            converged=result.success
        )

    def risk_reversal(self, forward: float, expiry: float,
                       delta: float = 0.25) -> float:
        """25-delta risk reversal: vol(25d call) - vol(25d put)."""
        atm_v = self.atm_vol(forward, expiry)
        K_call = forward * np.exp(0.5 * atm_v ** 2 * expiry
                                   + atm_v * np.sqrt(expiry) * norm.ppf(1 - delta))
        K_put = forward * np.exp(0.5 * atm_v ** 2 * expiry
                                  + atm_v * np.sqrt(expiry) * norm.ppf(delta))
        vol_call = float(self.implied_vol(np.array([K_call]), forward, expiry)[0])
        vol_put = float(self.implied_vol(np.array([K_put]), forward, expiry)[0])
        return vol_call - vol_put

    def butterfly(self, forward: float, expiry: float,
                   delta: float = 0.25) -> float:
        """25-delta butterfly: 0.5*(vol(25d call) + vol(25d put)) - vol(ATM)."""
        atm_v = self.atm_vol(forward, expiry)
        K_call = forward * np.exp(0.5 * atm_v ** 2 * expiry
                                   + atm_v * np.sqrt(expiry) * norm.ppf(1 - delta))
        K_put = forward * np.exp(0.5 * atm_v ** 2 * expiry
                                  + atm_v * np.sqrt(expiry) * norm.ppf(delta))
        vol_call = float(self.implied_vol(np.array([K_call]), forward, expiry)[0])
        vol_put = float(self.implied_vol(np.array([K_put]), forward, expiry)[0])
        return 0.5 * (vol_call + vol_put) - atm_v
