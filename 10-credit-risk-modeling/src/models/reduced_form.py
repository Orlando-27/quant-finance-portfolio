"""
Reduced-Form Credit Risk Models
================================
Hazard rate models where default is driven by an exogenous Poisson
process. Implements Jarrow-Turnbull (1995) framework with constant
and piecewise-constant intensity, plus hazard rate bootstrapping
from CDS spreads.

References:
    Jarrow, R. & Turnbull, S. (1995). Pricing Derivatives on Financial
    Securities Subject to Credit Risk.
    Duffie, D. & Singleton, K. (1999). Modeling Term Structures of
    Defaultable Bonds.

Author: Jose Orlando Bobadilla Fuentes, CQF
"""

import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict


@dataclass
class SurvivalCurve:
    """Term structure of survival probabilities."""
    tenors: np.ndarray         # In years
    survival_probs: np.ndarray # Q(0, T)
    hazard_rates: np.ndarray   # Piecewise-constant lambda

    def interpolate(self, t: float) -> float:
        """Interpolate survival probability at arbitrary time t."""
        if t <= 0:
            return 1.0
        if t >= self.tenors[-1]:
            return self.survival_probs[-1]
        f = interp1d(np.concatenate([[0], self.tenors]),
                     np.concatenate([[1.0], self.survival_probs]),
                     kind="linear")
        return float(f(t))

    def default_prob(self, t: float) -> float:
        """Cumulative default probability by time t."""
        return 1.0 - self.interpolate(t)

    def forward_default_prob(self, t1: float, t2: float) -> float:
        """Probability of defaulting between t1 and t2, given survival to t1."""
        Q1, Q2 = self.interpolate(t1), self.interpolate(t2)
        return 1.0 - Q2 / Q1 if Q1 > 0 else 1.0


class HazardRateModel:
    """
    Reduced-form model with piecewise-constant hazard rate.

    Under the risk-neutral measure, the default time tau follows:
        P(tau > T) = Q(0, T) = exp(-integral_0^T lambda(s) ds)

    For piecewise-constant lambda on intervals [t_{i-1}, t_i]:
        Q(0, T) = prod_{i=1}^{k} exp(-lambda_i * (t_i - t_{i-1}))

    The risky discount factor combines time value and credit:
        D_risky(0, T) = D_rf(0, T) * [R + (1-R) * Q(0, T)]

    where R is the recovery rate.
    """

    def __init__(self, recovery_rate: float = 0.40):
        """
        Args:
            recovery_rate: Expected recovery as fraction of par (0 to 1).
                           Industry standard: 40% for senior unsecured.
        """
        self.R = recovery_rate

    def constant_hazard(self, spread: float) -> float:
        """
        Compute constant hazard rate from credit spread.

        Under simplifying assumptions:
            lambda = spread / (1 - R)

        This is the first-order approximation widely used in practice.

        Args:
            spread: Credit spread in decimal (e.g., 0.0200 for 200 bps).

        Returns:
            Hazard rate lambda.
        """
        return spread / (1.0 - self.R)

    def survival_probability(self, hazard_rate: float, T: float) -> float:
        """
        Survival probability under constant hazard rate.

            Q(0, T) = exp(-lambda * T)

        Args:
            hazard_rate: Constant intensity lambda.
            T: Time horizon in years.

        Returns:
            Survival probability Q(0, T).
        """
        return np.exp(-hazard_rate * T)

    def risky_bond_price(self, par: float, coupon: float,
                        rf_curve: np.ndarray, tenors: np.ndarray,
                        hazard_rate: float) -> float:
        """
        Price a risky coupon bond using reduced-form model.

        Args:
            par:     Face value.
            coupon:  Annual coupon rate (decimal).
            rf_curve: Risk-free zero rates for each tenor.
            tenors:  Payment times in years.
            hazard_rate: Constant hazard rate.

        Returns:
            Dirty price of the risky bond.
        """
        price = 0.0
        for i, t in enumerate(tenors):
            df_rf = np.exp(-rf_curve[i] * t)
            Q_t = self.survival_probability(hazard_rate, t)
            # Coupon received if survived
            price += coupon * par * df_rf * Q_t
            # Recovery if default in period (t_{i-1}, t_i)
            t_prev = tenors[i - 1] if i > 0 else 0.0
            Q_prev = self.survival_probability(hazard_rate, t_prev)
            price += self.R * par * df_rf * (Q_prev - Q_t)
        # Principal at maturity if survived
        df_T = np.exp(-rf_curve[-1] * tenors[-1])
        Q_T = self.survival_probability(hazard_rate, tenors[-1])
        price += par * df_T * Q_T
        return price

    def bootstrap_hazard_rates(self, cds_spreads: np.ndarray,
                               cds_tenors: np.ndarray,
                               rf_rates: np.ndarray) -> SurvivalCurve:
        """
        Bootstrap piecewise-constant hazard rates from CDS term structure.

        For each CDS maturity, solve for lambda_i such that the fair CDS
        spread matches the market quote, given previously bootstrapped
        hazard rates for shorter maturities.

        Args:
            cds_spreads: Market CDS spreads (decimal) for each tenor.
            cds_tenors:  CDS maturities in years (e.g., [1, 2, 3, 5, 7, 10]).
            rf_rates:    Risk-free zero rates for each tenor.

        Returns:
            SurvivalCurve with bootstrapped hazard rates.
        """
        n = len(cds_tenors)
        lambdas = np.zeros(n)
        surv_probs = np.ones(n)

        for k in range(n):
            t_k = cds_tenors[k]
            s_k = cds_spreads[k]

            def objective(lam_k):
                # Build piecewise hazard rate up to tenor k
                test_lambdas = lambdas.copy()
                test_lambdas[k] = lam_k

                # Compute survival probabilities
                Q = np.ones(k + 1)
                for i in range(k + 1):
                    t_prev = cds_tenors[i - 1] if i > 0 else 0.0
                    dt = cds_tenors[i] - t_prev
                    Q_prev = Q[i - 1] if i > 0 else 1.0
                    Q[i] = Q_prev * np.exp(-test_lambdas[i] * dt)

                # Premium leg: s * sum(delta_i * df_i * Q_i)
                prem_leg = 0.0
                for i in range(k + 1):
                    dt = cds_tenors[i] - (cds_tenors[i-1] if i > 0 else 0.0)
                    df = np.exp(-rf_rates[i] * cds_tenors[i])
                    prem_leg += dt * df * Q[i]
                prem_leg *= s_k

                # Protection leg: (1-R) * sum(df_i * (Q_{i-1} - Q_i))
                prot_leg = 0.0
                for i in range(k + 1):
                    Q_prev = Q[i - 1] if i > 0 else 1.0
                    df = np.exp(-rf_rates[i] * cds_tenors[i])
                    prot_leg += df * (Q_prev - Q[i])
                prot_leg *= (1.0 - self.R)

                return prem_leg - prot_leg

            lambdas[k] = brentq(objective, 1e-6, 5.0)

            # Compute survival probability at tenor k
            if k == 0:
                surv_probs[k] = np.exp(-lambdas[k] * cds_tenors[k])
            else:
                dt = cds_tenors[k] - cds_tenors[k - 1]
                surv_probs[k] = surv_probs[k - 1] * np.exp(-lambdas[k] * dt)

        return SurvivalCurve(
            tenors=cds_tenors.copy(),
            survival_probs=surv_probs.copy(),
            hazard_rates=lambdas.copy(),
        )
