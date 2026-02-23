"""
Black-Litterman Asset Allocation Model
========================================

Bayesian framework combining market equilibrium with investor views.

Algorithm:
    1. Reverse-optimize implied equilibrium returns: pi = delta * Sigma * w_mkt
    2. Specify investor views as P*mu = Q + epsilon
    3. Compute posterior: mu_BL via Bayesian update formula
    4. Optimize portfolio using posterior returns

Author: Jose Orlando Bobadilla Fuentes | CQF

References:
    Black, F., & Litterman, R. (1992). Global Portfolio Optimization. FAJ.
    He, G., & Litterman, R. (1999). The Intuition Behind BL Portfolios. GSAM.
    Idzorek, T. (2005). A Step-By-Step Guide to the BL Model.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ViewSpecification:
    """
    Investor view specification.

    Absolute view: "Asset A returns 8%" -> P_row=[0,0,1,0], Q=0.08
    Relative view: "A outperforms B by 2%" -> P_row=[0,-1,1,0], Q=0.02
    """
    P_row: np.ndarray
    Q: float
    confidence: float = 0.5


class BlackLittermanModel:
    """
    Black-Litterman posterior return estimator.

    mu_BL = [(tau*Sigma)^{-1} + P'*Omega^{-1}*P]^{-1}
            * [(tau*Sigma)^{-1}*pi + P'*Omega^{-1}*Q]

    Usage:
        >>> bl = BlackLittermanModel(Sigma, w_mkt, risk_aversion=2.5, tau=0.025)
        >>> bl.add_absolute_view(asset_idx=2, return_view=0.10, confidence=0.8)
        >>> bl.add_relative_view(long_idx=0, short_idx=3, spread=0.03, confidence=0.6)
        >>> mu_post, Sigma_post = bl.posterior()
    """

    def __init__(self, cov_matrix, market_cap_weights, risk_aversion=2.5,
                 tau=0.025, asset_names=None):
        self.n = len(market_cap_weights)
        self.Sigma = np.array(cov_matrix, dtype=np.float64)
        self.w_mkt = np.array(market_cap_weights, dtype=np.float64)
        self.delta = risk_aversion
        self.tau = tau
        self.asset_names = asset_names or [f"Asset_{i}" for i in range(self.n)]

        # Implied equilibrium returns via reverse optimization
        self.pi = self.delta * self.Sigma @ self.w_mkt
        self._views: List[ViewSpecification] = []

    def implied_returns(self) -> pd.Series:
        """Returns that make market-cap weights optimal under MVO."""
        return pd.Series(self.pi, index=self.asset_names, name="Implied_Return")

    def add_absolute_view(self, asset_idx, return_view, confidence=0.5):
        """Add: 'Asset i will return X%.'"""
        P_row = np.zeros(self.n)
        P_row[asset_idx] = 1.0
        self._views.append(ViewSpecification(P_row, return_view, confidence))

    def add_relative_view(self, long_idx, short_idx, spread, confidence=0.5):
        """Add: 'Asset i outperforms Asset j by X%.'"""
        P_row = np.zeros(self.n)
        P_row[long_idx] = 1.0
        P_row[short_idx] = -1.0
        self._views.append(ViewSpecification(P_row, spread, confidence))

    def _build_view_matrices(self):
        """Construct P, Q, Omega using Idzorek (2005) confidence scaling."""
        k = len(self._views)
        P = np.zeros((k, self.n))
        Q = np.zeros(k)
        omega_diag = np.zeros(k)

        for i, v in enumerate(self._views):
            P[i, :] = v.P_row
            Q[i] = v.Q
            alpha = max(min(v.confidence, 0.999), 0.001)
            view_var = float(v.P_row @ (self.tau * self.Sigma) @ v.P_row.T)
            omega_diag[i] = view_var * (1.0 / alpha - 1.0)

        return P, Q, np.diag(omega_diag)

    def posterior(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute posterior expected returns and covariance.
        If no views, returns equilibrium prior.
        """
        if not self._views:
            return self.pi, (1 + self.tau) * self.Sigma

        P, Q, Omega = self._build_view_matrices()
        tS = self.tau * self.Sigma
        tS_inv = np.linalg.inv(tS)
        O_inv = np.linalg.inv(Omega)

        M_inv = tS_inv + P.T @ O_inv @ P
        M = np.linalg.inv(M_inv)

        mu_BL = M @ (tS_inv @ self.pi + P.T @ O_inv @ Q)
        Sigma_BL = self.Sigma + M

        return mu_BL, Sigma_BL

    def posterior_dataframe(self) -> pd.DataFrame:
        """Posterior returns with comparison to prior."""
        mu_BL, _ = self.posterior()
        return pd.DataFrame({
            "Asset": self.asset_names, "Equilibrium": self.pi,
            "Posterior": mu_BL, "Shift": mu_BL - self.pi,
            "Market_Weight": self.w_mkt,
        })
