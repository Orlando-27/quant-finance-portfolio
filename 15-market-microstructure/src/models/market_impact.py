"""
Market Impact & Optimal Execution
===================================
Almgren-Chriss (2001) framework for optimal trade scheduling.

Theory:
    A trader wishes to liquidate X shares over T periods.
    The execution cost has two components:
        1. Permanent impact (linear in total volume):
               g(v) = γ * v
        2. Temporary impact (concave in instantaneous rate):
               h(v) = η * v^β  (linear case: β=1)

    The efficient frontier minimizes:
        E[Cost] + λ * Var[Cost]

    The optimal strategy is:
        x(t) = x₀ * sinh(κ(T-t)) / sinh(κT)
        v(t) = dx/dt

    where κ = sqrt(λ * σ² / η) and σ is price volatility.

References:
    Almgren, R. & Chriss, N. (2001). Journal of Risk, 3(2), 5-39.
    Almgren, R. (2003). Applied Mathematical Finance, 10(1), 1-18.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class AlmgrenChrissParams:
    """Parameters for the Almgren-Chriss execution model."""
    X     : float = 1_000_000   # Total shares to liquidate
    T     : float = 1.0         # Execution horizon (days)
    N     : int   = 10          # Number of periods
    sigma : float = 0.015       # Daily price volatility (fraction)
    eta   : float = 0.01        # Temporary impact coefficient (linear)
    gamma : float = 1e-6        # Permanent impact coefficient
    epsilon: float = 0.0        # Fixed cost per trade (half-spread)
    S0    : float = 100.0       # Initial stock price


class AlmgrenChrissModel:
    """
    Almgren-Chriss (2001) Optimal Execution Framework.

    Computes the efficient frontier of execution strategies,
    balancing expected cost vs. execution risk (variance).
    """

    def __init__(self, params: AlmgrenChrissParams | None = None):
        self.p = params or AlmgrenChrissParams()

    # ------------------------------------------------------------------
    # Optimal trajectory for a given risk aversion λ
    # ------------------------------------------------------------------
    def optimal_trajectory(self, lam: float) -> dict:
        """
        Compute the optimal liquidation trajectory for risk-aversion λ.

        Parameters
        ----------
        lam : float
            Risk-aversion parameter (higher → faster execution, lower
            market impact risk at cost of higher temporary impact).

        Returns
        -------
        dict with keys:
            'times'     : array of time points
            'inventory' : remaining shares at each time
            'trade_size': shares sold each period
            'E_cost'    : expected execution cost
            'Var_cost'  : variance of execution cost
        """
        p = self.p
        tau   = p.T / p.N                        # period length
        kappa = np.sqrt(lam * p.sigma**2 / p.eta) if p.eta > 0 else 0

        times = np.linspace(0, p.T, p.N + 1)

        if kappa < 1e-12:
            # Risk-neutral case: TWAP (uniform liquidation)
            inventory = p.X * (1.0 - times / p.T)
        else:
            inventory = p.X * np.sinh(kappa * (p.T - times)) / np.sinh(kappa * p.T)

        inventory = np.clip(inventory, 0, p.X)
        trade_size = -np.diff(inventory)   # shares sold each period (> 0)

        # Expected cost
        E_cost = (
            p.epsilon * p.X
            + 0.5 * p.gamma * p.X**2
            + p.eta * np.sum(trade_size**2 / tau)
        )

        # Variance of cost
        if kappa > 1e-12:
            Var_cost = (
                0.5 * p.sigma**2
                * p.X**2
                * tau
                * (
                    np.cosh(kappa * p.T / p.N) / np.sinh(kappa * p.T)
                    - 1.0 / (kappa * p.T)
                )
            )
        else:
            Var_cost = p.sigma**2 * p.X**2 * p.T / 6.0

        return {
            "times"     : times,
            "inventory" : inventory,
            "trade_size": trade_size,
            "E_cost"    : E_cost,
            "Var_cost"  : Var_cost,
            "lambda"    : lam,
        }

    # ------------------------------------------------------------------
    # Efficient Frontier
    # ------------------------------------------------------------------
    def efficient_frontier(
        self,
        n_points: int = 100,
        lam_min : float = 1e-8,
        lam_max : float = 1e-2,
    ) -> pd.DataFrame:
        """
        Trace the Almgren-Chriss efficient frontier.

        Each point represents a Pareto-optimal strategy for a given λ.

        Returns
        -------
        pd.DataFrame
            Columns: lambda, E_cost, std_cost, IS (Implementation Shortfall).
        """
        lambdas = np.logspace(np.log10(lam_min), np.log10(lam_max), n_points)
        records = []
        for lam in lambdas:
            traj = self.optimal_trajectory(lam)
            records.append({
                "lambda"  : lam,
                "E_cost"  : traj["E_cost"],
                "std_cost": np.sqrt(max(0, traj["Var_cost"])),
                "IS_bps"  : traj["E_cost"] / (self.p.X * self.p.S0) * 1e4,
            })
        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # TWAP benchmark trajectory
    # ------------------------------------------------------------------
    def twap_trajectory(self) -> dict:
        """Uniform (TWAP) liquidation trajectory (λ → 0)."""
        return self.optimal_trajectory(lam=1e-12)

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------
    def execution_summary(self, lam: float) -> dict:
        """Human-readable summary for a given risk-aversion λ."""
        traj = self.optimal_trajectory(lam)
        p    = self.p
        return {
            "Risk Aversion (λ)"            : f"{lam:.2e}",
            "Expected Cost ($)"            : f"{traj['E_cost']:,.2f}",
            "Cost StdDev ($)"              : f"{np.sqrt(max(0,traj['Var_cost'])):,.2f}",
            "Implementation Shortfall (bps)": f"{traj['E_cost']/(p.X*p.S0)*1e4:.2f}",
            "Initial Inventory"            : f"{p.X:,}",
            "Trading Horizon (days)"       : f"{p.T:.1f}",
            "Periods"                      : f"{p.N}",
        }
