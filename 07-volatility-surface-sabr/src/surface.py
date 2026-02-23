"""
================================================================================
VOLATILITY SURFACE CONSTRUCTION & INTERPOLATION
================================================================================
Builds a complete implied volatility surface from market data across
multiple expiries. Supports SABR and SVI slice fitting with smooth
interpolation in the time dimension.

Author: Jose Orlando Bobadilla Fuentes, CQF
================================================================================
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

from src.sabr import SABRModel, SABRCalibrationResult
from src.svi import SVIModel, SVIFitResult


@dataclass
class SliceData:
    """Market data for a single expiry slice."""
    expiry: float
    forward: float
    strikes: np.ndarray
    market_vols: np.ndarray
    weights: Optional[np.ndarray] = None


@dataclass
class SurfaceDiagnostics:
    """Diagnostics for the fitted surface."""
    per_slice_rmse: Dict[float, float]
    per_slice_max_error: Dict[float, float]
    per_slice_arbitrage: Dict[float, bool]
    calendar_arbitrage_free: bool
    total_rmse: float


class VolSurface:
    """
    Implied volatility surface with per-slice model fitting and
    smooth interpolation across strikes and maturities.
    """

    def __init__(self, method: str = "sabr", beta: float = 0.5,
                 sabr_formula: str = "hagan"):
        self.method = method.lower()
        self.beta = beta
        self.sabr_formula = sabr_formula
        self._slices: Dict[float, object] = {}
        self._slice_data: Dict[float, SliceData] = {}
        self._expiries: np.ndarray = np.array([])
        self._atm_vols: Dict[float, float] = {}

    @property
    def expiries(self) -> np.ndarray:
        return self._expiries

    def fit(self, market_data: List[SliceData]) -> SurfaceDiagnostics:
        """Fit volatility surface from market data."""
        diagnostics_rmse = {}
        diagnostics_max = {}
        diagnostics_arb = {}
        all_errors = []

        for sd in sorted(market_data, key=lambda x: x.expiry):
            T = sd.expiry
            self._slice_data[T] = sd

            if self.method == "sabr":
                model = SABRModel(beta=self.beta, formula=self.sabr_formula)
                result = model.calibrate(
                    forward=sd.forward, strikes=sd.strikes,
                    market_vols=sd.market_vols, expiry=T,
                    weights=sd.weights, method="local"
                )
                self._slices[T] = model
                diagnostics_rmse[T] = result.rmse
                diagnostics_max[T] = result.max_error
                diagnostics_arb[T] = True
                all_errors.extend((result.model_vols - result.market_vols).tolist())
                self._atm_vols[T] = model.atm_vol(sd.forward, T)

            elif self.method == "svi":
                svi = SVIModel()
                k = np.log(sd.strikes / sd.forward)
                w_market = sd.market_vols ** 2 * T
                result = svi.fit(k, w_market, weights=sd.weights)
                self._slices[T] = svi
                diagnostics_rmse[T] = result.rmse
                diagnostics_max[T] = result.max_error
                diagnostics_arb[T] = result.arbitrage_free
                model_vols = np.sqrt(np.maximum(result.model_variance / T, 0))
                all_errors.extend((model_vols - sd.market_vols).tolist())
                self._atm_vols[T] = float(svi.implied_vol(np.array([0.0]), T)[0])

        self._expiries = np.array(sorted(self._slices.keys()))
        cal_arb = self._check_calendar_arbitrage()
        total_rmse = np.sqrt(np.mean(np.array(all_errors) ** 2))

        return SurfaceDiagnostics(
            per_slice_rmse=diagnostics_rmse,
            per_slice_max_error=diagnostics_max,
            per_slice_arbitrage=diagnostics_arb,
            calendar_arbitrage_free=cal_arb,
            total_rmse=total_rmse,
        )

    def get_vol(self, strike: float, expiry: float,
                forward: Optional[float] = None) -> float:
        """Get implied volatility at any (strike, expiry) point."""
        if expiry in self._slices:
            return self._get_slice_vol(strike, expiry, forward)

        T_lo, T_hi = self._bracket_expiry(expiry)
        if T_lo is None or T_hi is None:
            T_nearest = T_lo if T_lo is not None else T_hi
            return self._get_slice_vol(strike, T_nearest, forward)

        vol_lo = self._get_slice_vol(strike, T_lo, forward)
        vol_hi = self._get_slice_vol(strike, T_hi, forward)

        w_lo = vol_lo ** 2 * T_lo
        w_hi = vol_hi ** 2 * T_hi
        frac = (expiry - T_lo) / (T_hi - T_lo)
        w_interp = w_lo + frac * (w_hi - w_lo)

        return np.sqrt(max(w_interp / expiry, 0))

    def get_vol_grid(self, strikes: np.ndarray,
                      expiries: np.ndarray) -> np.ndarray:
        """Compute implied volatility on a (strikes x expiries) grid."""
        grid = np.zeros((len(strikes), len(expiries)))
        for j, T in enumerate(expiries):
            for i, K in enumerate(strikes):
                grid[i, j] = self.get_vol(K, T)
        return grid

    def _get_slice_vol(self, strike: float, expiry: float,
                        forward: Optional[float] = None) -> float:
        model = self._slices[expiry]
        sd = self._slice_data[expiry]
        F = forward if forward is not None else sd.forward

        if self.method == "sabr":
            return float(model.implied_vol(np.array([strike]), F, expiry)[0])
        else:
            k = np.log(strike / F)
            return float(model.implied_vol(np.array([k]), expiry)[0])

    def _bracket_expiry(self, T: float) -> Tuple[Optional[float], Optional[float]]:
        below = self._expiries[self._expiries <= T]
        above = self._expiries[self._expiries >= T]
        T_lo = float(below[-1]) if len(below) > 0 else None
        T_hi = float(above[0]) if len(above) > 0 else None
        return T_lo, T_hi

    def _check_calendar_arbitrage(self) -> bool:
        """Check no-calendar-spread-arbitrage."""
        if len(self._expiries) < 2:
            return True

        k_test = np.linspace(-0.3, 0.3, 50)

        for i in range(len(self._expiries) - 1):
            T1, T2 = self._expiries[i], self._expiries[i + 1]
            for k in k_test:
                sd1 = self._slice_data[T1]
                sd2 = self._slice_data[T2]
                K1 = sd1.forward * np.exp(k)
                K2 = sd2.forward * np.exp(k)
                v1 = self.get_vol(K1, T1)
                v2 = self.get_vol(K2, T2)
                w1 = v1 ** 2 * T1
                w2 = v2 ** 2 * T2
                if w2 < w1 - 1e-8:
                    return False
        return True

    def atm_term_structure(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return ATM volatility term structure."""
        vols = np.array([self._atm_vols[T] for T in self._expiries])
        return self._expiries.copy(), vols

    def atm_skew_term_structure(self, delta_k: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
        """Return ATM skew term structure."""
        skews = []
        for T in self._expiries:
            sd = self._slice_data[T]
            F = sd.forward
            v_up = self.get_vol(F * np.exp(delta_k), T)
            v_dn = self.get_vol(F * np.exp(-delta_k), T)
            skew = (v_up - v_dn) / (2 * delta_k)
            skews.append(skew)
        return self._expiries.copy(), np.array(skews)
