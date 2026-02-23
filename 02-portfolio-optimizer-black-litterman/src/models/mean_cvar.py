"""
Mean-CVaR Portfolio Optimization
==================================

CVaR (Conditional Value at Risk) = expected loss in the worst (1-beta) tail.
Reformulated as LP using Rockafellar-Uryasev (2000):

    min  alpha + [1/(T*(1-beta))] * sum(z_t)
    s.t. z_t >= -(R_t' * w) - alpha, z_t >= 0, w'*1 = 1

Author: Jose Orlando Bobadilla Fuentes | CQF
"""

import numpy as np
import cvxpy as cp
import pandas as pd
from typing import Optional, List
from dataclasses import dataclass


@dataclass
class CVaRResult:
    weights: np.ndarray
    expected_return: float
    cvar: float
    var: float
    status: str


class MeanCVaROptimizer:
    """
    CVaR portfolio optimizer using scenario-based LP.

    Usage:
        >>> opt = MeanCVaROptimizer(returns_matrix, confidence_level=0.95)
        >>> result = opt.optimize(target_return=0.08)
    """

    def __init__(self, returns_matrix, confidence_level=0.95, asset_names=None):
        self.R = np.array(returns_matrix, dtype=np.float64)
        self.T, self.n = self.R.shape
        self.beta = confidence_level
        self.mu = np.mean(self.R, axis=0)
        self.asset_names = asset_names or [f"Asset_{i}" for i in range(self.n)]

    def optimize(self, target_return=None, long_only=True, max_weight=1.0):
        """Minimize CVaR subject to return and weight constraints."""
        w = cp.Variable(self.n)
        alpha = cp.Variable()
        z = cp.Variable(self.T, nonneg=True)

        port_ret = self.R @ w
        cvar_obj = alpha + (1.0 / (self.T * (1.0 - self.beta))) * cp.sum(z)

        constraints = [
            z >= -port_ret - alpha,
            cp.sum(w) == 1,
            w <= max_weight,
        ]
        if long_only:
            constraints.append(w >= 0)
        if target_return is not None:
            constraints.append(self.mu @ w >= target_return)

        prob = cp.Problem(cp.Minimize(cvar_obj), constraints)
        prob.solve(solver=cp.ECOS, verbose=False)

        if prob.status not in ("optimal", "optimal_inaccurate"):
            raise ValueError(f"CVaR optimization failed: {prob.status}")

        wt = w.value
        return CVaRResult(
            weights=wt, expected_return=float(wt @ self.mu),
            cvar=float(cvar_obj.value), var=float(alpha.value),
            status=prob.status)

    def efficient_frontier_cvar(self, n_points=50, long_only=True):
        """Compute Mean-CVaR efficient frontier."""
        targets = np.linspace(float(np.min(self.mu)) * 1.1,
                              float(np.max(self.mu)) * 0.95, n_points)
        results = []
        for mu_t in targets:
            try:
                r = self.optimize(target_return=mu_t, long_only=long_only)
                results.append({"return": r.expected_return,
                                "cvar": r.cvar, "var": r.var, "weights": r.weights})
            except (ValueError, Exception):
                continue
        return pd.DataFrame(results)
