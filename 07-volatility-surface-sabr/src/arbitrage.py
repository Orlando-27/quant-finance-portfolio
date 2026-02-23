"""
================================================================================
ARBITRAGE DIAGNOSTICS & CONSTRAINTS
================================================================================
Comprehensive checks for static arbitrage violations in implied volatility
surfaces: butterfly, calendar spread, and call spread constraints.

Author: Jose Orlando Bobadilla Fuentes, CQF
================================================================================
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

from src.implied_vol import black_price
from src.surface import VolSurface


@dataclass
class ArbitrageViolation:
    """Single arbitrage violation record."""
    type: str
    strike: float
    expiry: float
    severity: float
    details: str


@dataclass
class ArbitrageDiagnostics:
    """Complete arbitrage diagnostics report."""
    butterfly_free: bool
    calendar_free: bool
    call_spread_free: bool
    n_violations: int
    violations: List[ArbitrageViolation]
    density_negative_pct: float
    summary: str


class ArbitrageChecker:
    """Comprehensive arbitrage diagnostics for volatility surfaces."""

    def __init__(self, surface: VolSurface):
        self.surface = surface

    def full_check(self, strikes: Optional[np.ndarray] = None,
                    n_strikes: int = 100,
                    forward: float = 100.0,
                    r: float = 0.0) -> ArbitrageDiagnostics:
        """Run all arbitrage checks across the surface."""
        expiries = self.surface.expiries

        if strikes is None:
            strikes = np.linspace(forward * 0.7, forward * 1.3, n_strikes)

        violations = []

        bf_viols = self._check_butterfly(strikes, expiries, forward, r)
        violations.extend(bf_viols)

        cal_viols = self._check_calendar(strikes, expiries, forward)
        violations.extend(cal_viols)

        cs_viols = self._check_call_spread(strikes, expiries, forward, r)
        violations.extend(cs_viols)

        bf_free = not any(v.type == "butterfly" for v in violations)
        cal_free = not any(v.type == "calendar" for v in violations)
        cs_free = not any(v.type == "call_spread" for v in violations)

        neg_pct = self._negative_density_pct(strikes, expiries, forward, r)

        summary_parts = []
        if bf_free:
            summary_parts.append("No butterfly arbitrage detected")
        else:
            n_bf = sum(1 for v in violations if v.type == "butterfly")
            summary_parts.append(f"Butterfly violations: {n_bf}")
        if cal_free:
            summary_parts.append("No calendar arbitrage detected")
        else:
            n_cal = sum(1 for v in violations if v.type == "calendar")
            summary_parts.append(f"Calendar violations: {n_cal}")
        if cs_free:
            summary_parts.append("No call spread violations")
        else:
            n_cs = sum(1 for v in violations if v.type == "call_spread")
            summary_parts.append(f"Call spread violations: {n_cs}")

        summary_parts.append(f"Negative density: {neg_pct:.2f}% of grid")

        return ArbitrageDiagnostics(
            butterfly_free=bf_free, calendar_free=cal_free,
            call_spread_free=cs_free, n_violations=len(violations),
            violations=violations, density_negative_pct=neg_pct,
            summary=" | ".join(summary_parts),
        )

    def _check_butterfly(self, strikes, expiries, forward, r, tol=-1e-6):
        violations = []
        dK = strikes[1] - strikes[0] if len(strikes) > 1 else 1.0

        for T in expiries:
            df = np.exp(-r * T)
            prices = np.array([
                black_price(forward, K, T, self.surface.get_vol(K, T), "call", df)
                for K in strikes
            ])
            d2C = np.diff(prices, n=2) / dK ** 2

            for i, val in enumerate(d2C):
                if val < tol:
                    violations.append(ArbitrageViolation(
                        type="butterfly", strike=strikes[i + 1],
                        expiry=T, severity=abs(val),
                        details=f"d2C/dK2 = {val:.6f} < 0 at K={strikes[i+1]:.2f}, T={T:.4f}"
                    ))
        return violations

    def _check_calendar(self, strikes, expiries, forward, tol=-1e-6):
        violations = []

        for i in range(len(expiries) - 1):
            T1, T2 = expiries[i], expiries[i + 1]
            for K in strikes:
                v1 = self.surface.get_vol(K, T1, forward)
                v2 = self.surface.get_vol(K, T2, forward)
                w1 = v1 ** 2 * T1
                w2 = v2 ** 2 * T2

                if w2 - w1 < tol:
                    violations.append(ArbitrageViolation(
                        type="calendar", strike=K, expiry=T2,
                        severity=abs(w2 - w1),
                        details=f"w(T2={T2:.4f})={w2:.6f} < w(T1={T1:.4f})={w1:.6f}"
                    ))
        return violations

    def _check_call_spread(self, strikes, expiries, forward, r, tol=1e-6):
        violations = []
        dK = strikes[1] - strikes[0] if len(strikes) > 1 else 1.0

        for T in expiries:
            df = np.exp(-r * T)
            prices = np.array([
                black_price(forward, K, T, self.surface.get_vol(K, T), "call", df)
                for K in strikes
            ])
            dCdK = np.diff(prices) / dK

            for i, val in enumerate(dCdK):
                if val > tol:
                    violations.append(ArbitrageViolation(
                        type="call_spread", strike=strikes[i],
                        expiry=T, severity=val,
                        details=f"dC/dK = {val:.6f} > 0 at K={strikes[i]:.2f}"
                    ))
                elif val < -1 - tol:
                    violations.append(ArbitrageViolation(
                        type="call_spread", strike=strikes[i],
                        expiry=T, severity=abs(val + 1),
                        details=f"dC/dK = {val:.6f} < -1 at K={strikes[i]:.2f}"
                    ))
        return violations

    def _negative_density_pct(self, strikes, expiries, forward, r):
        total = 0
        negative = 0
        dK = strikes[1] - strikes[0] if len(strikes) > 1 else 1.0

        for T in expiries:
            df = np.exp(-r * T)
            prices = np.array([
                black_price(forward, K, T, self.surface.get_vol(K, T), "call", df)
                for K in strikes
            ])
            d2C = np.diff(prices, n=2) / dK ** 2
            total += len(d2C)
            negative += np.sum(d2C < -1e-8)

        return 100 * negative / max(total, 1)
