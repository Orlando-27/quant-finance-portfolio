"""
Markowitz Mean-Variance Optimization Engine
=============================================

Classical mean-variance framework (Markowitz, 1952) with practical constraints.
Solved via CVXPY's disciplined convex programming (guaranteed global optimality).

Author: Jose Orlando Bobadilla Fuentes | CQF
"""

import numpy as np
import cvxpy as cp
import pandas as pd
from dataclasses import dataclass
from typing import Optional, List, Tuple


@dataclass
class OptimizationResult:
    """Container for portfolio optimization results."""
    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe_ratio: float
    asset_names: List[str]
    status: str = "optimal"

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "Asset": self.asset_names,
            "Weight": self.weights,
            "Weight_Pct": self.weights * 100
        }).sort_values("Weight", ascending=False)


class MarkowitzOptimizer:
    """
    Mean-Variance Portfolio Optimizer.

    Constructs efficient portfolios minimizing variance for a given target
    return, or maximizing Sharpe ratio (tangency portfolio).

    Usage:
        >>> opt = MarkowitzOptimizer(mu, Sigma, names, weight_bounds=(0, 0.4))
        >>> tangency = opt.max_sharpe(risk_free_rate=0.04)
        >>> frontier = opt.efficient_frontier(n_points=100)
    """

    def __init__(self, expected_returns, cov_matrix, asset_names=None,
                 weight_bounds=(0.0, 1.0), max_position=None):
        self.n = len(expected_returns)
        self.mu = np.array(expected_returns, dtype=np.float64)
        self.Sigma = np.array(cov_matrix, dtype=np.float64)
        self.asset_names = asset_names or [f"Asset_{i+1}" for i in range(self.n)]

        if self.Sigma.shape != (self.n, self.n):
            raise ValueError(f"Cov shape {self.Sigma.shape} != ({self.n},{self.n})")

        self.lb = weight_bounds[0] if weight_bounds else -np.inf
        self.ub = weight_bounds[1] if weight_bounds else np.inf
        if max_position is not None:
            self.ub = min(self.ub, max_position)

    def min_variance(self) -> OptimizationResult:
        """Global minimum variance portfolio (leftmost point of frontier)."""
        w = cp.Variable(self.n)
        prob = cp.Problem(
            cp.Minimize(cp.quad_form(w, self.Sigma)),
            [cp.sum(w) == 1, w >= self.lb, w <= self.ub]
        )
        prob.solve(solver=cp.OSQP, verbose=False)
        if prob.status != "optimal":
            raise ValueError(f"Failed: {prob.status}")
        wt = w.value
        ret = float(wt @ self.mu)
        vol = float(np.sqrt(wt @ self.Sigma @ wt))
        return OptimizationResult(wt, ret, vol, ret/vol if vol > 0 else 0,
                                  self.asset_names, prob.status)

    def max_sharpe(self, risk_free_rate=0.0) -> OptimizationResult:
        """
        Maximum Sharpe ratio (tangency) portfolio.
        Uses Cornuejols-Tutuncu transformation to convert to QP.
        """
        y = cp.Variable(self.n)
        prob = cp.Problem(
            cp.Minimize(cp.quad_form(y, self.Sigma)),
            [(self.mu - risk_free_rate) @ y == 1,
             y >= 1e-10 if self.lb >= 0 else self.lb]
        )
        prob.solve(solver=cp.OSQP, verbose=False)
        if prob.status != "optimal":
            raise ValueError(f"Failed: {prob.status}")
        wt = y.value / np.sum(y.value)
        ret = float(wt @ self.mu)
        vol = float(np.sqrt(wt @ self.Sigma @ wt))
        sr = (ret - risk_free_rate) / vol if vol > 0 else 0
        return OptimizationResult(wt, ret, vol, sr, self.asset_names, prob.status)

    def target_return(self, mu_target) -> Optional[OptimizationResult]:
        """Minimum variance portfolio for a given target return."""
        w = cp.Variable(self.n)
        prob = cp.Problem(
            cp.Minimize(cp.quad_form(w, self.Sigma)),
            [self.mu @ w >= mu_target, cp.sum(w) == 1,
             w >= self.lb, w <= self.ub]
        )
        prob.solve(solver=cp.OSQP, verbose=False)
        if prob.status != "optimal":
            return None
        wt = w.value
        ret = float(wt @ self.mu)
        vol = float(np.sqrt(wt @ self.Sigma @ wt))
        return OptimizationResult(wt, ret, vol, ret/vol if vol > 0 else 0,
                                  self.asset_names, prob.status)

    def efficient_frontier(self, n_points=100, risk_free_rate=0.0) -> pd.DataFrame:
        """Compute efficient frontier by sweeping target returns."""
        mv = self.min_variance()
        targets = np.linspace(mv.expected_return, float(np.max(self.mu)) * 0.99, n_points)
        results = []
        for mu_t in targets:
            r = self.target_return(mu_t)
            if r:
                results.append({
                    "return": r.expected_return,
                    "volatility": r.volatility,
                    "sharpe": (r.expected_return - risk_free_rate) / r.volatility,
                    "weights": r.weights
                })
        return pd.DataFrame(results)
