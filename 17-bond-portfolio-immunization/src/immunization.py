"""
immunization.py — Bond Portfolio Immunization Strategies
=========================================================
Implements:
  1. Redington Immunization  — duration & convexity matching via optimization
  2. Cash Flow Matching (Dedication) — LP minimizing cost subject to CF ≥ liabilities
  3. Analytical bullet / barbell / ladder construction helpers

Theory
------
Redington (1952): A bond portfolio is immunized against small parallel rate shifts if
  (a) PV(assets) = PV(liabilities)
  (b) D_mod(assets) = D_mod(liabilities)
  (c) C(assets)   ≥ C(liabilities)

Cash flow matching eliminates reinvestment risk by ensuring bond cash flows
cover each liability at or before its due date. The LP:

    min  Σ_j c_j * x_j          (minimize cost)
    s.t. Σ_j CF_{j,t} * x_j ≥ L_t   ∀ t   (liability coverage)
         x_j ≥ 0                           (no short selling)

Author  : Jose Orlando Bobadilla Fuentes | CQF | MSc AI
Project : 16 — Bond Portfolio Immunization
"""

import numpy as np
import scipy.optimize as opt
from dataclasses import dataclass, field
from typing import List, Optional
from src.bond import Bond


# ==============================================================================
# Liability Definition
# ==============================================================================
@dataclass
class Liability:
    """
    Single liability payment.

    Parameters
    ----------
    time   : Payment time in years
    amount : Payment amount in same currency as bond face values
    """
    time  : float
    amount: float


# ==============================================================================
# Immunization Engine
# ==============================================================================
class ImmunizationEngine:
    """
    Implements Redington immunization and cash flow matching.

    Parameters
    ----------
    bond_universe : List of available Bond instruments
    liabilities   : List of Liability payments to hedge
    yields        : Market YTMs for each bond in universe (same order)
    """

    def __init__(self, bond_universe: List[Bond],
                 liabilities: List[Liability],
                 yields: List[float]):
        self.bonds       = bond_universe
        self.liabilities = sorted(liabilities, key=lambda x: x.time)
        self.yields      = yields
        self.n_bonds     = len(bond_universe)

        # Precompute bond analytics (price, Dmod, convexity) at current yields
        self._bond_prices    = np.array([b.price(y) for b, y in zip(self.bonds, self.yields)])
        self._bond_dmod      = np.array([b.modified_duration(y) for b, y in zip(self.bonds, self.yields)])
        self._bond_convexity = np.array([b.convexity(y) for b, y in zip(self.bonds, self.yields)])

    # ── Portfolio-level analytics ─────────────────────────────────────────────
    def portfolio_pv(self, weights: np.ndarray) -> float:
        """Total PV of weighted bond portfolio (weights = # units of each bond)."""
        return float(np.dot(weights, self._bond_prices))

    def portfolio_duration(self, weights: np.ndarray) -> float:
        """Dollar-weighted modified duration of bond portfolio."""
        pv = self.portfolio_pv(weights)
        if pv == 0:
            return 0.0
        dollar_dur = np.dot(weights, self._bond_prices * self._bond_dmod)
        return float(dollar_dur / pv)

    def portfolio_convexity(self, weights: np.ndarray) -> float:
        """Dollar-weighted convexity of bond portfolio."""
        pv = self.portfolio_pv(weights)
        if pv == 0:
            return 0.0
        dollar_conv = np.dot(weights, self._bond_prices * self._bond_convexity)
        return float(dollar_conv / pv)

    # ── Liability analytics ───────────────────────────────────────────────────
    def _liability_pv(self, flat_ytm: float) -> float:
        """PV of all liabilities discounted at a flat yield."""
        return sum(
            L.amount * np.exp(-flat_ytm * L.time)
            for L in self.liabilities
        )

    def _liability_duration(self, flat_ytm: float) -> float:
        """Modified duration of liability stream at flat yield."""
        pv  = self._liability_pv(flat_ytm)
        dur = sum(
            L.time * L.amount * np.exp(-flat_ytm * L.time)
            for L in self.liabilities
        )
        return dur / pv if pv > 0 else 0.0

    def _liability_convexity(self, flat_ytm: float) -> float:
        """Convexity of liability stream at flat yield."""
        pv   = self._liability_pv(flat_ytm)
        conv = sum(
            L.time**2 * L.amount * np.exp(-flat_ytm * L.time)
            for L in self.liabilities
        )
        return conv / pv if pv > 0 else 0.0

    # ── Redington Immunization ────────────────────────────────────────────────
    def redington_immunization(self, flat_ytm: float,
                                budget: Optional[float] = None) -> dict:
        """
        Find bond portfolio weights satisfying Redington conditions:
          1. PV match   : Σ w_i * P_i = PV(liabilities)
          2. D match    : Σ w_i * P_i * D_i / PV = D_liab
          3. C dominance: Σ w_i * P_i * C_i / PV ≥ C_liab (objective: maximize)

        Uses scipy SLSQP with equality constraints.

        Parameters
        ----------
        flat_ytm : Flat yield for discounting liabilities
        budget   : Optional dollar budget override

        Returns
        -------
        dict with weights, portfolio stats, and immunization report
        """
        pv_liab = self._liability_pv(flat_ytm)
        d_liab  = self._liability_duration(flat_ytm)
        c_liab  = self._liability_convexity(flat_ytm)
        target_pv = budget if budget else pv_liab

        # ── Objective: maximize convexity surplus (minimize -convexity) ───────
        def objective(w):
            conv = np.dot(w, self._bond_prices * self._bond_convexity)
            pv   = np.dot(w, self._bond_prices)
            return -conv / pv if pv > 0 else 0.0

        # ── Constraints ───────────────────────────────────────────────────────
        def con_pv(w):
            """PV of portfolio must equal PV of liabilities."""
            return np.dot(w, self._bond_prices) - target_pv

        def con_dur(w):
            """Portfolio modified duration must match liability duration."""
            pv  = np.dot(w, self._bond_prices)
            dur = np.dot(w, self._bond_prices * self._bond_dmod)
            return (dur / pv if pv > 0 else 0.0) - d_liab

        constraints = [
            {"type": "eq", "fun": con_pv},
            {"type": "eq", "fun": con_dur},
        ]
        bounds = [(0, None)] * self.n_bonds          # no short selling

        # Initial guess: equal weights normalized to budget
        w0 = np.ones(self.n_bonds) / self.n_bonds
        w0 *= (target_pv / np.dot(w0, self._bond_prices))

        result = opt.minimize(
            objective, w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-10, "maxiter": 1000, "disp": False}
        )

        w_opt   = np.maximum(result.x, 0)           # clip numerical noise
        pv_port = self.portfolio_pv(w_opt)
        d_port  = self.portfolio_duration(w_opt)
        c_port  = self.portfolio_convexity(w_opt)

        return {
            "weights"           : w_opt,
            "portfolio_pv"      : pv_port,
            "liability_pv"      : target_pv,
            "pv_match"          : abs(pv_port - target_pv) < 1.0,
            "portfolio_duration": d_port,
            "liability_duration": d_liab,
            "duration_match"    : abs(d_port - d_liab) < 0.01,
            "portfolio_convexity": c_port,
            "liability_convexity": c_liab,
            "convexity_surplus" : c_port - c_liab,
            "immunized"         : (abs(pv_port - target_pv) < 1.0 and
                                   abs(d_port - d_liab) < 0.01 and
                                   c_port >= c_liab),
            "optimizer_success" : result.success,
            "optimizer_message" : result.message,
        }

    # ── Cash Flow Matching (Dedication LP) ────────────────────────────────────
    def cash_flow_matching(self) -> dict:
        """
        Solve the Cash Flow Matching (Dedication) LP:

            min  Σ_j P_j * x_j         (minimize cost)
            s.t. Σ_j CF_{j,t} * x_j ≥ L_t   ∀ t
                 x_j ≥ 0

        Builds a cash flow matrix CF[t, j] from bond cash flow schedules.

        Returns
        -------
        dict with optimal units, total cost, coverage ratios, and status
        """
        liab_times   = [L.time for L in self.liabilities]
        liab_amounts = np.array([L.amount for L in self.liabilities])

        # ── Build CF matrix: rows=liability times, cols=bonds ─────────────────
        cf_matrix = np.zeros((len(self.liabilities), self.n_bonds))

        for j, (bond, ytm) in enumerate(zip(self.bonds, self.yields)):
            t_cfs, cfs = bond.cash_flows()
            for i, L_time in enumerate(liab_times):
                # Accumulate bond cash flows at or before each liability date
                mask = t_cfs <= L_time + 1e-9
                cf_matrix[i, j] = float(np.sum(cfs[mask]))

        # ── scipy.optimize.linprog ────────────────────────────────────────────
        # linprog: min c @ x  s.t. A_ub @ x <= b_ub  (flip sign for >=)
        c      = self._bond_prices                # objective: cost
        A_ub   = -cf_matrix                       # negate for <= form
        b_ub   = -liab_amounts
        bounds = [(0, None)] * self.n_bonds

        result = opt.linprog(c, A_ub=A_ub, b_ub=b_ub,
                             bounds=bounds, method="highs")

        x_opt     = result.x if result.success else np.zeros(self.n_bonds)
        total_cost = float(np.dot(x_opt, self._bond_prices))

        # Coverage ratios per liability date
        coverage = (cf_matrix @ x_opt) / liab_amounts if result.success else np.zeros(len(self.liabilities))

        return {
            "units"            : x_opt,
            "total_cost"       : total_cost,
            "cf_matrix"        : cf_matrix,
            "coverage_ratios"  : coverage,
            "fully_matched"    : result.success and bool(np.all(coverage >= 0.999)),
            "optimizer_success": result.success,
            "optimizer_message": result.message,
        }

    # ── Portfolio Structure Builders ──────────────────────────────────────────
    @staticmethod
    def build_bullet(bond_universe: List[Bond], yields: List[float],
                     target_maturity: float, budget: float) -> np.ndarray:
        """
        Bullet: allocate entire budget to bond closest to target maturity.

        Returns weight vector (# units per bond).
        """
        mats  = np.array([b.maturity for b in bond_universe])
        idx   = int(np.argmin(np.abs(mats - target_maturity)))
        prices = np.array([b.price(y) for b, y in zip(bond_universe, yields)])
        w      = np.zeros(len(bond_universe))
        w[idx] = budget / prices[idx]
        return w

    @staticmethod
    def build_barbell(bond_universe: List[Bond], yields: List[float],
                      budget: float, split: float = 0.5) -> np.ndarray:
        """
        Barbell: split budget between shortest and longest maturity bonds.

        Parameters
        ----------
        split : fraction allocated to short end (default 50/50)
        """
        prices = np.array([b.price(y) for b, y in zip(bond_universe, yields)])
        w      = np.zeros(len(bond_universe))
        short  = 0                              # index of shortest maturity
        long   = len(bond_universe) - 1        # index of longest maturity
        w[short] = split * budget / prices[short]
        w[long]  = (1 - split) * budget / prices[long]
        return w

    @staticmethod
    def build_ladder(bond_universe: List[Bond], yields: List[float],
                     budget: float) -> np.ndarray:
        """
        Ladder: equal-dollar allocation across all bonds in universe.
        """
        prices   = np.array([b.price(y) for b, y in zip(bond_universe, yields)])
        n        = len(bond_universe)
        per_bond = budget / n
        w        = per_bond / prices
        return w
