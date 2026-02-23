"""
portfolio.py — Bond Portfolio Aggregation & Scenario Analysis
=============================================================
Implements:
  • Portfolio-level PV, duration, convexity aggregation
  • Parallel yield shift P&L simulation
  • Non-parallel curve scenario analysis (steepener, flattener, butterfly)
  • Surplus analysis: asset PV vs liability PV under rate shocks

Author  : Jose Orlando Bobadilla Fuentes | CQF | MSc AI
Project : 16 — Bond Portfolio Immunization
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict
from src.bond import Bond
from src.immunization import Liability


# ==============================================================================
# Portfolio Scenario Engine
# ==============================================================================
class PortfolioScenarioEngine:
    """
    Runs rate shock scenarios on a bond portfolio and computes
    P&L, surplus changes, and immunization effectiveness.

    Parameters
    ----------
    bonds     : List of Bond objects in the portfolio
    yields    : Current market YTMs for each bond
    weights   : Units held of each bond (from immunization solver)
    liabilities : List of Liability objects (for surplus analysis)
    """

    def __init__(self, bonds: List[Bond], yields: List[float],
                 weights: np.ndarray, liabilities: List[Liability]):
        self.bonds       = bonds
        self.yields      = yields
        self.weights     = weights
        self.liabilities = liabilities

        # Base portfolio PV
        self.base_bond_pvs = np.array([b.price(y) for b, y in zip(bonds, yields)])
        self.base_portfolio_pv = float(np.dot(weights, self.base_bond_pvs))

        # Base liability PV (flat curve = weighted avg yield)
        avg_ytm = float(np.average(yields, weights=self.base_bond_pvs * weights
                                    if np.any(weights > 0) else None))
        self.base_liability_pv = sum(
            L.amount * np.exp(-avg_ytm * L.time) for L in liabilities
        )
        self.base_surplus = self.base_portfolio_pv - self.base_liability_pv

    # ── Parallel Shift Scenario ───────────────────────────────────────────────
    def parallel_shift_analysis(self, shocks_bps: List[float]) -> List[dict]:
        """
        Evaluate portfolio P&L under a range of parallel yield curve shocks.

        For each shock Δy:
          • Reprice all bonds at (yield_i + Δy)
          • Recompute liability PV at (avg_yield + Δy)
          • Compute new portfolio PV, liability PV, surplus

        Parameters
        ----------
        shocks_bps : list of yield shocks in basis points (e.g. [-300, -100, 100, 300])

        Returns
        -------
        List of dicts, one per shock, with portfolio and liability metrics
        """
        results = []

        avg_base_ytm = float(np.average(self.yields,
                              weights=self.base_bond_pvs * self.weights
                              if np.any(self.weights > 0) else None))

        for shock_bps in shocks_bps:
            dy = shock_bps * 1e-4

            # ── Reprice bonds ─────────────────────────────────────────────────
            shocked_prices = np.array([
                b.price(y + dy)
                for b, y in zip(self.bonds, self.yields)
            ])
            port_pv = float(np.dot(self.weights, shocked_prices))

            # ── Reprice liabilities ───────────────────────────────────────────
            liab_ytm = avg_base_ytm + dy
            liab_pv  = sum(
                L.amount * np.exp(-liab_ytm * L.time)
                for L in self.liabilities
            )

            surplus       = port_pv - liab_pv
            port_pnl      = port_pv - self.base_portfolio_pv
            liab_pnl      = liab_pv - self.base_liability_pv
            surplus_change = surplus - self.base_surplus

            # ── Duration approximation for comparison ─────────────────────────
            # Compute portfolio-level D_mod and convexity at base
            if np.any(self.weights > 0):
                d_mods = np.array([b.modified_duration(y) for b, y in zip(self.bonds, self.yields)])
                convs  = np.array([b.convexity(y) for b, y in zip(self.bonds, self.yields)])
                pv_w   = self.base_bond_pvs * self.weights
                port_d = float(np.dot(pv_w, d_mods) / self.base_portfolio_pv)
                port_c = float(np.dot(pv_w, convs) / self.base_portfolio_pv)
                approx_pnl = self.base_portfolio_pv * (
                    -port_d * dy + 0.5 * port_c * dy**2
                )
            else:
                approx_pnl = 0.0

            results.append({
                "shock_bps"      : shock_bps,
                "portfolio_pv"   : port_pv,
                "liability_pv"   : liab_pv,
                "surplus"        : surplus,
                "surplus_change" : surplus_change,
                "portfolio_pnl"  : port_pnl,
                "liability_pnl"  : liab_pnl,
                "approx_pnl"     : approx_pnl,
                "taylor_error_pct": (abs(port_pnl - approx_pnl) / max(abs(port_pnl), 1e-6)) * 100,
            })

        return results

    # ── Non-Parallel Scenarios ────────────────────────────────────────────────
    def curve_scenario_analysis(self, scenarios: List[Dict]) -> List[dict]:
        """
        Non-parallel yield curve scenarios: steepener, flattener, butterfly.

        Each scenario specifies per-bond yield adjustments (matched by maturity bucket).

        Parameters
        ----------
        scenarios : List of dicts with keys:
            'name'       : str
            'short_shock': yield change for bonds with maturity <= 2Y (bps)
            'mid_shock'  : yield change for bonds with 2Y < maturity <= 7Y (bps)
            'long_shock' : yield change for bonds with maturity > 7Y (bps)

        Returns
        -------
        List of dicts with PV, surplus, and bond-level impact
        """
        results = []
        avg_ytm = float(np.average(self.yields,
                         weights=self.base_bond_pvs * self.weights
                         if np.any(self.weights > 0) else None))

        for sc in scenarios:
            shocked_prices = np.zeros(len(self.bonds))
            bond_shocks    = np.zeros(len(self.bonds))

            for i, (bond, ytm) in enumerate(zip(self.bonds, self.yields)):
                if bond.maturity <= 2.0:
                    dy = sc.get("short_shock", 0) * 1e-4
                elif bond.maturity <= 7.0:
                    dy = sc.get("mid_shock", 0) * 1e-4
                else:
                    dy = sc.get("long_shock", 0) * 1e-4

                bond_shocks[i]    = dy * 1e4     # back to bps for display
                shocked_prices[i] = bond.price(ytm + dy)

            port_pv     = float(np.dot(self.weights, shocked_prices))
            avg_shock   = float(np.mean(bond_shocks))
            liab_ytm    = avg_ytm + avg_shock * 1e-4
            liab_pv     = sum(L.amount * np.exp(-liab_ytm * L.time) for L in self.liabilities)
            surplus     = port_pv - liab_pv

            results.append({
                "scenario"      : sc["name"],
                "portfolio_pv"  : port_pv,
                "liability_pv"  : liab_pv,
                "surplus"       : surplus,
                "surplus_change": surplus - self.base_surplus,
                "bond_shocks_bps": bond_shocks,
            })

        return results

    # ── Summary Table ─────────────────────────────────────────────────────────
    def bond_analytics_table(self) -> List[dict]:
        """Return per-bond analytics for the current portfolio."""
        rows = []
        for bond, ytm, w, pv in zip(self.bonds, self.yields,
                                     self.weights, self.base_bond_pvs):
            rows.append({
                "issuer"      : bond.issuer or "Bond",
                "maturity"    : bond.maturity,
                "coupon_pct"  : bond.coupon_rate * 100,
                "ytm_pct"     : ytm * 100,
                "price"       : pv,
                "units"       : w,
                "market_value": pv * w,
                "d_mod"       : bond.modified_duration(ytm),
                "convexity"   : bond.convexity(ytm),
                "dv01"        : bond.dv01(ytm) * w,
                "weight_pct"  : (pv * w / self.base_portfolio_pv * 100
                                 if self.base_portfolio_pv > 0 else 0),
            })
        return rows
