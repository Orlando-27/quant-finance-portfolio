"""
Fixed Income & Yield Curve Utilities
======================================
Interpolation methods, duration/DV01 calculations, and
goodness-of-fit metrics for yield curve analysis.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, interp1d


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------
def cubic_spline_curve(
    tenors    : np.ndarray,
    yields    : np.ndarray,
    tau_grid  : np.ndarray,
    bc_type   : str = "not-a-knot",
) -> np.ndarray:
    """
    Cubic spline interpolation of a yield curve.

    Parameters
    ----------
    tenors   : observed maturities (knots).
    yields   : observed yields at knots.
    tau_grid : fine maturity grid for output.
    bc_type  : boundary condition ('not-a-knot', 'natural', 'clamped').

    Returns
    -------
    np.ndarray  Interpolated yields on tau_grid.
    """
    cs = CubicSpline(tenors, yields, bc_type=bc_type)
    return cs(tau_grid)


def linear_interp_curve(
    tenors  : np.ndarray,
    yields  : np.ndarray,
    tau_grid: np.ndarray,
) -> np.ndarray:
    """Piecewise linear interpolation (benchmark)."""
    f = interp1d(tenors, yields, kind="linear",
                 bounds_error=False, fill_value=(yields[0], yields[-1]))
    return f(tau_grid)


# ---------------------------------------------------------------------------
# Duration & DV01
# ---------------------------------------------------------------------------
def modified_duration(
    cf_times  : np.ndarray,
    cf_amounts: np.ndarray,
    ytm       : float,
    freq      : int = 2,
) -> float:
    """
    Compute modified duration for a bond with known cash flows.

    Parameters
    ----------
    cf_times   : array of cash flow dates in years.
    cf_amounts : array of cash flow amounts ($).
    ytm        : yield to maturity (annual, decimal).
    freq       : coupon frequency (2 = semi-annual).

    Returns
    -------
    float  Modified duration (years).
    """
    y_per = ytm / freq
    pv_cf = cf_amounts / (1 + y_per) ** (cf_times * freq)
    price = pv_cf.sum()
    mac_dur = np.sum(cf_times * pv_cf) / price
    return mac_dur / (1 + y_per)


def dv01(price: float, mod_dur: float) -> float:
    """
    DV01 (Dollar Value of a basis point): sensitivity of price
    to a 1bp parallel shift in yield.

        DV01 = Price * ModDur / 10_000
    """
    return price * mod_dur / 10_000.0


def par_yield(
    tenors: np.ndarray,
    spot  : np.ndarray,
    maturity: float,
    freq  : int = 2,
) -> float:
    """
    Compute par yield from spot rates via bootstrapping.

    The par yield c is the coupon rate such that a bond priced
    at par has the given maturity:

        1 = c/freq * Σ P(tᵢ) + P(T)
        → c = (1 - P(T)) / Σ_{i=1}^{n} P(tᵢ) * freq
    """
    # Interpolate spot rate to coupon dates
    coupon_dates = np.arange(1/freq, maturity + 1e-9, 1/freq)
    f = interp1d(tenors, spot, kind="linear",
                 bounds_error=False, fill_value=(spot[0], spot[-1]))
    spot_interp = f(coupon_dates)
    discount    = np.exp(-spot_interp * coupon_dates)
    return (1.0 - discount[-1]) / discount.sum() * freq


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def curve_fit_metrics(
    yields_obs : np.ndarray,
    yields_fit : np.ndarray,
    tenors     : np.ndarray | None = None,
) -> dict:
    """
    Comprehensive goodness-of-fit metrics for yield curve models.

    Returns RMSE (bps), MAE (bps), R², max error (bps), and
    optionally a per-tenor residual table.
    """
    bps    = 10_000.0
    resid  = yields_obs - yields_fit
    rmse   = float(np.sqrt(np.mean(resid**2))) * bps
    mae    = float(np.mean(np.abs(resid)))     * bps
    max_e  = float(np.max(np.abs(resid)))      * bps
    ss_tot = float(np.sum((yields_obs - yields_obs.mean())**2))
    r2     = float(1.0 - np.sum(resid**2) / ss_tot) if ss_tot > 0 else np.nan

    result = {
        "RMSE (bps)"     : round(rmse, 3),
        "MAE (bps)"      : round(mae,  3),
        "Max Error (bps)": round(max_e, 3),
        "R²"             : round(r2,   6),
    }
    if tenors is not None:
        result["per_tenor"] = pd.Series(resid * bps,
                                        index=tenors, name="resid_bps")
    return result


def yield_change_decomposition(
    yields_panel: pd.DataFrame,
) -> pd.DataFrame:
    """
    Decompose yield changes by the traditional L/S/C attribution:

    For each date t:
        Level change   = mean(Δy across all tenors)
        Slope change   = Δy(long end) - Δy(short end)
        Curvature chg  = 2*Δy(mid) - Δy(short) - Δy(long)

    Returns
    -------
    pd.DataFrame  with columns [level_chg, slope_chg, curv_chg].
    """
    dy     = yields_panel.diff().dropna()
    tenors = yields_panel.columns.astype(float).tolist()
    n      = len(tenors)
    short  = tenors[0]
    long_  = tenors[-1]
    mid_   = tenors[n // 2]

    level  = dy.mean(axis=1)
    slope  = dy[long_] - dy[short]
    curv   = 2 * dy[mid_] - dy[short] - dy[long_]

    return pd.DataFrame({
        "level_chg": level,
        "slope_chg": slope,
        "curv_chg" : curv,
    })
