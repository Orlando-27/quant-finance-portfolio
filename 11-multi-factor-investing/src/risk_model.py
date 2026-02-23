"""
Barra-Style Factor Risk Decomposition
======================================

Implements the fundamental factor risk model framework used by MSCI Barra
for risk attribution, portfolio construction, and risk budgeting.

Theory:
    Asset returns decompose into systematic (factor) and idiosyncratic parts:
        R = B * F + eps

    where R is (N x 1), B is (N x K) exposure matrix, F is (K x 1) factor
    returns, and eps is (N x 1) specific returns with diagonal covariance Delta.

    Portfolio variance:
        Var(R_p) = w' * V * w
        V = B * Sigma_F * B' + Delta

    where Sigma_F is (K x K) factor covariance and Delta is (N x N) diagonal.

    Risk decomposition:
        Total Risk = Factor Risk + Specific Risk
        Factor Risk = w' * B * Sigma_F * B' * w
        Specific Risk = w' * Delta * w

    Marginal contribution to risk (MCTR):
        MCTR_i = (V * w)_i / sigma_p
        Sum of w_i * MCTR_i = sigma_p (Euler decomposition)

    Active risk decomposition (relative to benchmark):
        TE^2 = h' * V * h,  where h = w_portfolio - w_benchmark

References:
    MSCI Barra GEM3 documentation, Grinold & Kahn (2000)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple


class BarraRiskModel:
    """
    Factor risk model for portfolio risk decomposition and attribution.

    Parameters
    ----------
    n_factors : int
        Number of risk factors K.
    """

    def __init__(self, n_factors: int = 6):
        self.n_factors = n_factors
        self.exposure_matrix = None    # (N x K) B matrix
        self.factor_covariance = None  # (K x K) Sigma_F
        self.specific_variance = None  # (N,) diagonal of Delta
        self.asset_names = None
        self.factor_names = None

    def fit(self, returns: pd.DataFrame, factors: pd.DataFrame,
            window: int = 60) -> 'BarraRiskModel':
        """
        Estimate the risk model components from historical data.

        Step 1: Cross-sectional regression at each date t to get
                time-varying exposures B_t and specific returns eps_t.
        Step 2: Estimate factor covariance Sigma_F from factor return
                time series (exponentially weighted optional).
        Step 3: Estimate specific variance Delta from cross-sectional
                variance of residuals.

        Parameters
        ----------
        returns : pd.DataFrame
            (T x N) asset returns.
        factors : pd.DataFrame
            (T x K) factor returns.
        window : int
            Estimation window for covariance (most recent observations).

        Returns
        -------
        self
        """
        self.asset_names = returns.columns.tolist()
        self.factor_names = factors.columns.tolist()
        T, N = returns.shape
        K = factors.shape[1]

        # Use most recent window for estimation
        ret_w = returns.iloc[-window:]
        fac_w = factors.iloc[-window:]

        # Cross-sectional regression: R_t = B_t * F_t + eps_t
        # Aggregate exposure via time-series regression for each asset
        B = np.zeros((N, K))
        specific_var = np.zeros(N)

        X = fac_w.values  # (W x K)
        X_aug = np.column_stack([np.ones(len(X)), X])

        for i in range(N):
            y = ret_w.iloc[:, i].values
            valid = ~np.isnan(y)
            if valid.sum() < K + 5:
                B[i] = 0.0
                specific_var[i] = ret_w.iloc[:, i].var()
                continue
            coef = np.linalg.lstsq(X_aug[valid], y[valid], rcond=None)[0]
            B[i] = coef[1:]
            resid = y[valid] - X_aug[valid] @ coef
            specific_var[i] = resid.var()

        # Factor covariance (annualized)
        sigma_F = fac_w.cov().values * 12

        self.exposure_matrix = pd.DataFrame(
            B, index=self.asset_names, columns=self.factor_names
        )
        self.factor_covariance = pd.DataFrame(
            sigma_F, index=self.factor_names, columns=self.factor_names
        )
        self.specific_variance = pd.Series(
            specific_var * 12, index=self.asset_names
        )
        return self

    def portfolio_risk(self, weights: np.ndarray) -> Dict[str, float]:
        """
        Decompose portfolio risk into factor and specific components.

        Parameters
        ----------
        weights : np.ndarray
            (N,) portfolio weight vector.

        Returns
        -------
        dict
            total_risk, factor_risk, specific_risk (all annualized vol),
            pct_factor, pct_specific (percentage contributions).
        """
        B = self.exposure_matrix.values
        Sigma_F = self.factor_covariance.values
        Delta = np.diag(self.specific_variance.values)

        # Factor risk
        Bw = B.T @ weights  # (K,) factor exposure of portfolio
        factor_var = Bw @ Sigma_F @ Bw
        specific_var = weights @ Delta @ weights
        total_var = factor_var + specific_var

        total_risk = np.sqrt(total_var)
        factor_risk = np.sqrt(max(factor_var, 0))
        specific_risk = np.sqrt(max(specific_var, 0))

        return {
            "total_risk": total_risk,
            "factor_risk": factor_risk,
            "specific_risk": specific_risk,
            "pct_factor": factor_var / total_var * 100 if total_var > 0 else 0,
            "pct_specific": specific_var / total_var * 100 if total_var > 0 else 0,
        }

    def factor_risk_contribution(self, weights: np.ndarray) -> pd.DataFrame:
        """
        Marginal and percentage contribution of each factor to portfolio risk.

        Uses Euler decomposition: sum of risk contributions = total factor risk.

        Parameters
        ----------
        weights : np.ndarray
            (N,) portfolio weights.

        Returns
        -------
        pd.DataFrame
            Factor-level risk contributions with columns:
            exposure, marginal_contribution, percentage_contribution.
        """
        B = self.exposure_matrix.values
        Sigma_F = self.factor_covariance.values
        Bw = B.T @ weights  # (K,) portfolio factor exposures
        factor_var = Bw @ Sigma_F @ Bw
        factor_vol = np.sqrt(max(factor_var, 1e-12))

        # Marginal contribution: d(sigma_factor)/d(Bw_k)
        mctr = (Sigma_F @ Bw) / factor_vol

        # Risk contribution: Bw_k * MCTR_k
        rc = Bw * mctr
        pct_rc = rc / factor_vol * 100

        result = pd.DataFrame({
            "Exposure": Bw,
            "Marginal Contribution": mctr,
            "Risk Contribution": rc,
            "Pct Contribution": pct_rc,
        }, index=self.factor_names)

        return result

    def active_risk_decomposition(self, w_portfolio: np.ndarray,
                                   w_benchmark: np.ndarray) -> Dict:
        """
        Decompose tracking error into factor and specific components.

        Parameters
        ----------
        w_portfolio : np.ndarray
            (N,) portfolio weights.
        w_benchmark : np.ndarray
            (N,) benchmark weights.

        Returns
        -------
        dict
            tracking_error, factor_te, specific_te, active_factor_contributions.
        """
        h = w_portfolio - w_benchmark
        B = self.exposure_matrix.values
        Sigma_F = self.factor_covariance.values
        Delta = np.diag(self.specific_variance.values)

        Bh = B.T @ h
        factor_var = Bh @ Sigma_F @ Bh
        specific_var = h @ Delta @ h
        total_var = factor_var + specific_var
        te = np.sqrt(max(total_var, 0))

        # Active factor contributions
        factor_te = np.sqrt(max(factor_var, 0))
        if factor_te > 1e-12:
            mctr_active = (Sigma_F @ Bh) / factor_te
            rc_active = Bh * mctr_active
        else:
            rc_active = np.zeros(len(self.factor_names))

        return {
            "tracking_error": te,
            "factor_te": factor_te,
            "specific_te": np.sqrt(max(specific_var, 0)),
            "active_factor_exposures": pd.Series(Bh, index=self.factor_names),
            "active_factor_risk_contrib": pd.Series(
                rc_active, index=self.factor_names
            ),
        }
