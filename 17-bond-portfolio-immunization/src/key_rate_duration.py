"""
key_rate_duration.py — Key Rate Duration (KRD) Analysis
=========================================================
Measures the sensitivity of bond prices and portfolio value to
movements at specific points (key rates) on the yield curve.

Theory
------
Given a bond with cash flows CF_t discounted at spot rates s(t),
the key rate duration at tenor τ_k is computed by bumping only
the spot rate at τ_k by ±Δy (typically 1 bps) while holding all
other spot rates fixed, and applying cubic spline interpolation
for cash flows at intermediate maturities.

    KRD_k = -(P_up - P_down) / (2 * Δy * P_base)     [central difference]

The KRD vector sums (approximately) to modified duration:
    Σ_k KRD_k ≈ D_mod

Dollar KRD (KR-DV01):
    KRDV01_k = KRD_k × P × 0.0001

Author  : Jose Orlando Bobadilla Fuentes | CQF | MSc AI
Project : 16 — Bond Portfolio Immunization
"""

import numpy as np
from scipy.interpolate import CubicSpline
from typing import List
from src.bond import Bond


# ==============================================================================
# Key Rate DV01 Engine
# ==============================================================================
class KeyRateDuration:
    """
    Compute key rate durations for a single bond or a portfolio.

    Parameters
    ----------
    key_tenors  : List of key rate tenors in years, e.g. [0.25, 0.5, 1, 2, 5, 10, 30]
    shock_bps   : Bump size in basis points (default 1 bps)
    """

    def __init__(self, key_tenors: List[float] = None,
                 shock_bps: float = 1.0):
        # Standard USD key rates (Fed / CMT convention)
        self.key_tenors = key_tenors or [
            0.0833, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0
        ]
        self.shock      = shock_bps * 1e-4          # convert to decimal
        self.n_keys     = len(self.key_tenors)

    # ── Spot Curve Construction ───────────────────────────────────────────────
    @staticmethod
    def _build_flat_spot_curve(flat_ytm: float,
                               tenors: List[float]) -> np.ndarray:
        """Return flat spot rate curve at given YTM across all tenors."""
        return np.full(len(tenors), flat_ytm)

    def _interpolated_discount(self, cf_times: np.ndarray,
                                spot_tenors: np.ndarray,
                                spot_rates: np.ndarray) -> np.ndarray:
        """
        Interpolate spot rates at cash flow times using cubic spline,
        then compute discount factors e^{-s(t)*t}.

        Parameters
        ----------
        cf_times     : Cash flow maturities (years)
        spot_tenors  : Tenor grid for spot curve
        spot_rates   : Spot rates at each tenor

        Returns
        -------
        discount_factors : array of shape (len(cf_times),)
        """
        # Extend spline domain to cover all cash flow times
        t_ext = np.concatenate([[0.0], spot_tenors])
        r_ext = np.concatenate([[spot_rates[0]], spot_rates])

        cs     = CubicSpline(t_ext, r_ext, bc_type="natural")
        rates  = np.clip(cs(cf_times), 0.0, 1.0)    # prevent extrapolation blow-up
        return np.exp(-rates * cf_times)

    # ── Single Bond KRD ───────────────────────────────────────────────────────
    def compute_bond_krd(self, bond: Bond, flat_ytm: float) -> dict:
        """
        Compute Key Rate Duration vector for a single bond.

        Uses central finite difference: bump spot rate at key tenor k up and
        down by `shock`, revalue bond, compute KRD_k.

        Parameters
        ----------
        bond     : Bond instance
        flat_ytm : Starting flat yield (used as base spot curve)

        Returns
        -------
        dict with:
          krd_vector : array(n_keys) — KRD at each key tenor (in years)
          krdv01     : array(n_keys) — Dollar KRD per 1 bps (USD)
          sum_krd    : total KRD (should ≈ D_mod)
          d_mod      : true modified duration for comparison
          base_price : bond price at flat_ytm
        """
        times, cfs = bond.cash_flows()
        base_spots = self._build_flat_spot_curve(flat_ytm, self.key_tenors)
        base_df    = self._interpolated_discount(times, np.array(self.key_tenors), base_spots)
        base_price = float(np.dot(cfs, base_df))

        krd  = np.zeros(self.n_keys)

        for k in range(self.n_keys):
            # ── Up shock ──────────────────────────────────────────────────────
            spots_up     = base_spots.copy()
            spots_up[k] += self.shock
            df_up        = self._interpolated_discount(times, np.array(self.key_tenors), spots_up)
            p_up         = float(np.dot(cfs, df_up))

            # ── Down shock ────────────────────────────────────────────────────
            spots_dn     = base_spots.copy()
            spots_dn[k] -= self.shock
            df_dn        = self._interpolated_discount(times, np.array(self.key_tenors), spots_dn)
            p_dn         = float(np.dot(cfs, df_dn))

            # ── Central difference KRD ────────────────────────────────────────
            krd[k] = -(p_up - p_dn) / (2 * self.shock * base_price)

        krdv01  = krd * base_price * 1e-4
        d_mod   = bond.modified_duration(flat_ytm)

        return {
            "krd_vector" : krd,
            "krdv01"     : krdv01,
            "sum_krd"    : float(np.sum(krd)),
            "d_mod"      : d_mod,
            "base_price" : base_price,
        }

    # ── Portfolio KRD ─────────────────────────────────────────────────────────
    def compute_portfolio_krd(self, bonds: List[Bond],
                               yields: List[float],
                               weights: np.ndarray) -> dict:
        """
        Aggregate KRD across a bond portfolio.

        Portfolio dollar KRD = Σ_j w_j * KRDV01_j
        Portfolio KRD        = Σ_j (w_j * P_j * KRD_j) / PV_portfolio

        Parameters
        ----------
        bonds   : List of Bond objects
        yields  : List of YTMs corresponding to each bond
        weights : Number of units held for each bond

        Returns
        -------
        dict with portfolio-level krd_vector, krdv01, d_mod
        """
        port_krdv01  = np.zeros(self.n_keys)
        port_pv_krd  = np.zeros(self.n_keys)
        total_pv     = 0.0

        for bond, ytm, w in zip(bonds, yields, weights):
            if w <= 0:
                continue
            res      = self.compute_bond_krd(bond, ytm)
            bond_pv  = res["base_price"] * w

            port_krdv01  += w * res["krdv01"]
            port_pv_krd  += bond_pv * res["krd_vector"]
            total_pv     += bond_pv

        port_krd = port_pv_krd / total_pv if total_pv > 0 else np.zeros(self.n_keys)

        return {
            "krd_vector"  : port_krd,
            "krdv01"      : port_krdv01,
            "sum_krd"     : float(np.sum(port_krd)),
            "portfolio_pv": total_pv,
        }
