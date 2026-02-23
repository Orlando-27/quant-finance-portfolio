"""
bond.py — Fixed-Coupon Bond Pricing Engine
==========================================
Implements:
  • Bond pricing (present value of cash flows)
  • Yield-to-Maturity (Newton-Raphson iterative solver)
  • Macaulay & Modified Duration
  • Convexity (second-order Taylor term)
  • DV01 / Dollar Duration
  • Full price approximation under yield shock
  • Cash flow schedule generation

Author  : Jose Orlando Bobadilla Fuentes | CQF | MSc AI
Project : 16 — Bond Portfolio Immunization
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ==============================================================================
# Bond Data Class
# ==============================================================================
@dataclass
class Bond:
    """
    Represents a fixed-coupon bond with standard conventions.

    Parameters
    ----------
    face_value   : Par/face amount (default USD 1,000)
    coupon_rate  : Annual coupon rate as decimal (e.g. 0.05 = 5%)
    maturity     : Years to maturity (float, e.g. 5.0)
    frequency    : Coupon payments per year (1=annual, 2=semi-annual)
    isin         : Optional bond identifier
    issuer       : Optional issuer label
    """
    face_value  : float = 1000.0
    coupon_rate : float = 0.05
    maturity    : float = 5.0
    frequency   : int   = 2          # semi-annual by default
    isin        : str   = ""
    issuer      : str   = ""

    # ── Derived ──────────────────────────────────────────────────────────────
    def coupon_payment(self) -> float:
        """Periodic coupon cash flow: (rate × face) / frequency."""
        return self.coupon_rate * self.face_value / self.frequency

    def n_periods(self) -> int:
        """Total number of coupon periods."""
        return int(round(self.maturity * self.frequency))

    def period_length(self) -> float:
        """Length of each coupon period in years."""
        return 1.0 / self.frequency

    # ── Cash Flow Schedule ────────────────────────────────────────────────────
    def cash_flows(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate cash flow schedule.

        Returns
        -------
        times : ndarray of shape (n,) — cash flow times in years
        cfs   : ndarray of shape (n,) — cash flow amounts (USD)
        """
        n   = self.n_periods()
        dt  = self.period_length()
        C   = self.coupon_payment()

        times = np.arange(1, n + 1) * dt                # [dt, 2dt, ..., T]
        cfs   = np.full(n, C)
        cfs[-1] += self.face_value                       # final: coupon + par

        return times, cfs

    # ── Price ─────────────────────────────────────────────────────────────────
    def price(self, ytm: float) -> float:
        """
        Compute full (dirty) price given yield-to-maturity (YTM).

        Uses periodic discounting:
            P = Σ CF_t / (1 + y/m)^(t*m)

        Parameters
        ----------
        ytm : Annual yield-to-maturity as decimal

        Returns
        -------
        Dirty price in same currency as face_value
        """
        times, cfs = self.cash_flows()
        periodic_y = ytm / self.frequency
        discount   = (1 + periodic_y) ** (times * self.frequency)
        return float(np.sum(cfs / discount))

    # ── YTM Solver ────────────────────────────────────────────────────────────
    def ytm(self, market_price: float, tol: float = 1e-10,
            max_iter: int = 200) -> float:
        """
        Solve for Yield-to-Maturity given market price (Newton-Raphson).

        Objective: find y such that price(y) = market_price.

        Parameters
        ----------
        market_price : Observed clean/dirty price
        tol          : Convergence tolerance
        max_iter     : Maximum iterations

        Returns
        -------
        Annual YTM as decimal
        """
        # Initial guess: approximate YTM from coupon and rough discount
        guess = self.coupon_rate

        times, cfs = self.cash_flows()
        m = self.frequency

        for _ in range(max_iter):
            # ── Function: f(y) = price(y) - market_price ─────────────────────
            r  = guess / m
            discount_factors = (1 + r) ** (times * m)
            pv = float(np.sum(cfs / discount_factors))
            f  = pv - market_price

            # ── Derivative: f'(y) = dP/dy ────────────────────────────────────
            # dP/dy = -Σ t * CF_t / (1 + y/m)^(t*m+1)  (chain rule)
            dpdy = float(np.sum(-times * cfs / ((1 + r) ** (times * m + 1))))

            if abs(dpdy) < 1e-15:
                break

            # Newton-Raphson update
            guess_new = guess - f / dpdy

            if abs(guess_new - guess) < tol:
                return guess_new

            guess = guess_new

        return guess   # return best estimate if not fully converged

    # ── Macaulay Duration ─────────────────────────────────────────────────────
    def macaulay_duration(self, ytm: float) -> float:
        """
        Macaulay Duration — weighted average time to cash flows.

            D_mac = Σ [ t * PV(CF_t) ] / P

        Parameters
        ----------
        ytm : Annual yield-to-maturity

        Returns
        -------
        Duration in years
        """
        times, cfs = self.cash_flows()
        m = self.frequency
        r = ytm / m

        pv_cfs = cfs / (1 + r) ** (times * m)
        P      = float(np.sum(pv_cfs))

        return float(np.dot(times, pv_cfs) / P)

    # ── Modified Duration ─────────────────────────────────────────────────────
    def modified_duration(self, ytm: float) -> float:
        """
        Modified Duration — proportional price sensitivity to yield.

            D_mod = D_mac / (1 + y/m)

        Interpretation: 1% increase in yield → D_mod% fall in price.
        """
        d_mac = self.macaulay_duration(ytm)
        return d_mac / (1.0 + ytm / self.frequency)

    # ── DV01 / Dollar Duration ────────────────────────────────────────────────
    def dv01(self, ytm: float) -> float:
        """
        Dollar Value of 1 Basis Point (DV01).

            DV01 = D_mod × P × 0.0001

        Represents the dollar P&L from a 1 bps yield move.
        """
        P     = self.price(ytm)
        d_mod = self.modified_duration(ytm)
        return d_mod * P * 0.0001

    # ── Convexity ─────────────────────────────────────────────────────────────
    def convexity(self, ytm: float) -> float:
        """
        Convexity — second-order price sensitivity (curvature correction).

            C = (1/P) * Σ [ t*(t + 1/m) * PV(CF_t) ] / (1 + y/m)^2

        (Standard textbook formula, periods in years.)
        """
        times, cfs = self.cash_flows()
        m = self.frequency
        r = ytm / m

        pv_cfs = cfs / (1 + r) ** (times * m)
        P      = float(np.sum(pv_cfs))

        # Weight each PV by t*(t + dt)
        dt     = self.period_length()
        convex = float(np.dot(times * (times + dt), pv_cfs))
        return convex / (P * (1 + r) ** 2)

    # ── Price Approximation ───────────────────────────────────────────────────
    def price_change_approx(self, ytm: float, delta_y: float) -> float:
        """
        Taylor-series approximation of price change under yield shock Δy.

            ΔP/P ≈ -D_mod * Δy + 0.5 * C * Δy²

        Returns
        -------
        Approximate new price (not delta).
        """
        P      = self.price(ytm)
        d_mod  = self.modified_duration(ytm)
        conv   = self.convexity(ytm)
        pct_chg = -d_mod * delta_y + 0.5 * conv * delta_y ** 2
        return P * (1 + pct_chg)

    # ── Summary ───────────────────────────────────────────────────────────────
    def summary(self, ytm: float) -> dict:
        """Return full analytics dictionary for this bond at given YTM."""
        P = self.price(ytm)
        return {
            "isin"             : self.isin or "N/A",
            "issuer"           : self.issuer or "N/A",
            "face_value"       : self.face_value,
            "coupon_rate_pct"  : self.coupon_rate * 100,
            "maturity_yrs"     : self.maturity,
            "frequency"        : self.frequency,
            "price"            : round(P, 6),
            "ytm_pct"          : round(ytm * 100, 6),
            "macaulay_dur"     : round(self.macaulay_duration(ytm), 6),
            "modified_dur"     : round(self.modified_duration(ytm), 6),
            "dv01"             : round(self.dv01(ytm), 6),
            "convexity"        : round(self.convexity(ytm), 6),
        }

    def __repr__(self) -> str:
        return (f"Bond(isin={self.isin or 'N/A'}, C={self.coupon_rate*100:.2f}%, "
                f"T={self.maturity}Y, F={self.face_value})")
