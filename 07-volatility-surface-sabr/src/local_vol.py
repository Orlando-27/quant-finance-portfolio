"""
================================================================================
DUPIRE LOCAL VOLATILITY
================================================================================
Extracts the local volatility surface sigma_loc(K, T) from the implied
volatility surface using Dupire's formula (1994).

Author: Jose Orlando Bobadilla Fuentes, CQF
================================================================================
"""

import numpy as np
from typing import Optional
from src.surface import VolSurface


class LocalVolSurface:
    """Dupire local volatility surface derived from an implied vol surface."""

    def __init__(self, iv_surface: VolSurface):
        self.iv_surface = iv_surface

    def local_vol(self, strike: float, expiry: float,
                   forward: Optional[float] = None,
                   dk: float = 0.005, dT: float = 1 / 252) -> float:
        """Compute Dupire local volatility at (K, T)."""
        if forward is None:
            T_arr = self.iv_surface.expiries
            idx = np.argmin(np.abs(T_arr - expiry))
            sd = self.iv_surface._slice_data[T_arr[idx]]
            forward = sd.forward

        k = np.log(strike / forward)

        vol_c = self.iv_surface.get_vol(strike, expiry, forward)
        w_c = vol_c ** 2 * expiry

        T_up = expiry + dT
        T_dn = max(expiry - dT, dT)
        vol_T_up = self.iv_surface.get_vol(strike, T_up, forward)
        vol_T_dn = self.iv_surface.get_vol(strike, T_dn, forward)
        w_T_up = vol_T_up ** 2 * T_up
        w_T_dn = vol_T_dn ** 2 * T_dn
        dw_dT = (w_T_up - w_T_dn) / (T_up - T_dn)

        K_up = forward * np.exp(k + dk)
        K_dn = forward * np.exp(k - dk)
        vol_k_up = self.iv_surface.get_vol(K_up, expiry, forward)
        vol_k_dn = self.iv_surface.get_vol(K_dn, expiry, forward)
        w_k_up = vol_k_up ** 2 * expiry
        w_k_dn = vol_k_dn ** 2 * expiry

        dw_dk = (w_k_up - w_k_dn) / (2 * dk)
        d2w_dk2 = (w_k_up - 2 * w_c + w_k_dn) / (dk ** 2)

        numerator = dw_dT
        denominator = (
            1 - k / w_c * dw_dk
            + 0.25 * (-0.25 - 1 / w_c + k ** 2 / w_c ** 2) * dw_dk ** 2
            + 0.5 * d2w_dk2
        ) if w_c > 1e-10 else 1.0

        if denominator <= 0 or numerator <= 0:
            return vol_c

        return np.sqrt(numerator / denominator)

    def local_vol_grid(self, strikes: np.ndarray, expiries: np.ndarray,
                        forward: Optional[float] = None) -> np.ndarray:
        """Compute local vol on a (strikes x expiries) grid."""
        grid = np.zeros((len(strikes), len(expiries)))
        for j, T in enumerate(expiries):
            for i, K in enumerate(strikes):
                grid[i, j] = self.local_vol(K, T, forward)
        return grid

    def compare_to_implied(self, strikes: np.ndarray, expiry: float,
                            forward: Optional[float] = None) -> dict:
        """Compare local vol vs implied vol for a given expiry."""
        iv = np.array([self.iv_surface.get_vol(K, expiry, forward)
                        for K in strikes])
        lv = np.array([self.local_vol(K, expiry, forward)
                        for K in strikes])
        return {
            "strikes": strikes,
            "implied_vol": iv,
            "local_vol": lv,
            "ratio": lv / (iv + 1e-10),
        }
