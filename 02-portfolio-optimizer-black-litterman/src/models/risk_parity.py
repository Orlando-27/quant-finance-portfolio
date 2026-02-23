"""
Risk Parity Portfolio Construction
=====================================

Equal risk contribution (ERC) portfolio: each asset contributes equally
to total portfolio risk.

    RC_i = w_i * (Sigma * w)_i / sigma_p = sigma_p / n  for all i

Author: Jose Orlando Bobadilla Fuentes | CQF

References:
    Maillard, S., Roncalli, T., & Teiletche, J. (2010). JoP.
    Roncalli, T. (2013). Introduction to Risk Parity and Budgeting.
"""

import numpy as np
from scipy.optimize import minimize
from typing import Optional, List
from dataclasses import dataclass


@dataclass
class RiskParityResult:
    weights: np.ndarray
    risk_contributions: np.ndarray
    portfolio_volatility: float
    asset_names: List[str]


class RiskParityOptimizer:
    """
    Equal Risk Contribution optimizer.

    Usage:
        >>> rp = RiskParityOptimizer(Sigma, asset_names=names)
        >>> result = rp.optimize()
    """

    def __init__(self, cov_matrix, risk_budgets=None, asset_names=None):
        self.Sigma = np.array(cov_matrix, dtype=np.float64)
        self.n = self.Sigma.shape[0]
        self.asset_names = asset_names or [f"Asset_{i}" for i in range(self.n)]
        self.budgets = (np.array(risk_budgets) / np.sum(risk_budgets)
                        if risk_budgets is not None
                        else np.ones(self.n) / self.n)

    def _risk_contributions(self, w):
        sigma_p = np.sqrt(w @ self.Sigma @ w)
        marginal = self.Sigma @ w / sigma_p
        return w * marginal

    def _objective(self, w):
        sigma_p_sq = w @ self.Sigma @ w
        rc = w * (self.Sigma @ w)
        target_rc = self.budgets * sigma_p_sq
        return float(np.sum((rc - target_rc) ** 2))

    def optimize(self) -> RiskParityResult:
        w0 = np.ones(self.n) / self.n
        result = minimize(
            self._objective, w0, method="SLSQP",
            bounds=[(1e-6, 1.0)] * self.n,
            constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        if not result.success:
            raise ValueError(f"Failed: {result.message}")

        wt = result.x / np.sum(result.x)
        rc = self._risk_contributions(wt)
        vol = np.sqrt(wt @ self.Sigma @ wt)
        return RiskParityResult(wt, rc, vol, self.asset_names)
