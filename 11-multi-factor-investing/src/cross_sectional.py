"""
Fama-MacBeth Cross-Sectional Regression
========================================

Implements the two-pass Fama-MacBeth (1973) procedure for estimating
factor risk premia from a panel of asset returns and characteristics.

Methodology:
    Pass 1 (Time-series regression):
        For each asset i over a rolling window of size W:
            R_it - R_ft = alpha_i + sum_k beta_ik * F_kt + eps_it
        This yields time-varying beta estimates hat{beta}_{ik,t}.

    Pass 2 (Cross-sectional regression):
        At each date t, run cross-sectional OLS across all assets:
            R_it = gamma_0t + sum_k gamma_kt * hat{beta}_{ik,t-1} + eta_it
        Using lagged betas avoids contemporaneous estimation bias.

    Risk premia estimation:
        hat{lambda}_k = (1/T) * sum_t gamma_kt
        with Shanken (1992) correction for errors-in-variables:
        Var(hat{lambda}) = (1/T) * [Sigma_gamma
                           + (1 + hat{lambda}' Sigma_F^{-1} hat{lambda}) * Sigma_eta]

    The Shanken correction accounts for the fact that betas are estimated
    with error in Pass 1, which biases downward the standard errors from
    a naive time-series average of the gamma estimates.

References:
    Fama & MacBeth (1973), Shanken (1992)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from scipy import stats as sp_stats
import statsmodels.api as sm


class FamaMacBeth:
    """
    Fama-MacBeth two-pass cross-sectional regression with Shanken correction.

    Parameters
    ----------
    rolling_window : int
        Window size (in periods) for Pass 1 time-series beta estimation.
    min_observations : int
        Minimum number of non-NaN observations required for an asset to be
        included in the cross-sectional regression at any given date.
    use_shanken : bool
        If True, apply Shanken (1992) errors-in-variables correction
        to standard errors.
    """

    def __init__(self, rolling_window: int = 60, min_observations: int = 36,
                 use_shanken: bool = True):
        self.rolling_window = rolling_window
        self.min_obs = min_observations
        self.use_shanken = use_shanken
        self.betas = None
        self.gammas = None
        self.risk_premia = None
        self.shanken_se = None

    def estimate_betas(self, returns: pd.DataFrame,
                       factors: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Pass 1: Rolling time-series regression to estimate factor betas.

        For each asset i, runs OLS of excess returns on factor returns
        over a rolling window. The resulting beta panel is used as the
        independent variable in Pass 2.

        Parameters
        ----------
        returns : pd.DataFrame
            (T x N) asset excess returns.
        factors : pd.DataFrame
            (T x K) factor returns (must be aligned with returns index).

        Returns
        -------
        dict
            Keys are factor names, values are (T x N) DataFrames of
            rolling beta estimates. Entries are NaN where the rolling
            window has insufficient data.
        """
        T, N = returns.shape
        K = factors.shape[1]
        dates = returns.index
        beta_panels = {f: pd.DataFrame(index=dates, columns=returns.columns,
                                        dtype=float)
                       for f in factors.columns}

        for end in range(self.rolling_window, T):
            start = end - self.rolling_window
            window_dates = dates[start:end]
            F_w = factors.loc[window_dates].values  # (W x K)
            X = np.column_stack([np.ones(len(window_dates)), F_w])

            for col_idx, stock in enumerate(returns.columns):
                y = returns.iloc[start:end, col_idx].values
                valid = ~np.isnan(y)
                if valid.sum() < self.min_obs:
                    continue
                y_v, X_v = y[valid], X[valid]
                try:
                    coef = np.linalg.lstsq(X_v, y_v, rcond=None)[0]
                    for k, fname in enumerate(factors.columns):
                        beta_panels[fname].iloc[end, col_idx] = coef[k + 1]
                except np.linalg.LinAlgError:
                    continue

        self.betas = beta_panels
        return beta_panels

    def cross_sectional_regression(self, returns: pd.DataFrame,
                                    factors: pd.DataFrame) -> pd.DataFrame:
        """
        Pass 2: Cross-sectional regressions and risk premia estimation.

        At each date t (after the burn-in window), regresses cross-section
        of asset returns on lagged beta estimates from Pass 1.

        Parameters
        ----------
        returns : pd.DataFrame
            (T x N) asset excess returns.
        factors : pd.DataFrame
            (T x K) factor returns for Shanken correction.

        Returns
        -------
        pd.DataFrame
            (T_eff x K+1) panel of gamma estimates. Column 0 is the
            intercept (pricing error), columns 1..K are factor premia.
        """
        if self.betas is None:
            self.estimate_betas(returns, factors)

        dates = returns.index
        fnames = list(factors.columns)
        K = len(fnames)
        gamma_cols = ["intercept"] + fnames
        gammas = pd.DataFrame(index=dates, columns=gamma_cols, dtype=float)

        for t_idx in range(self.rolling_window + 1, len(dates)):
            t = dates[t_idx]
            t_lag = dates[t_idx - 1]

            # Collect lagged betas for all stocks
            beta_mat = np.column_stack([
                self.betas[f].loc[t_lag].values for f in fnames
            ])  # (N x K)
            y_t = returns.loc[t].values  # (N,)

            # Drop stocks with missing betas or returns
            valid = ~(np.isnan(beta_mat).any(axis=1) | np.isnan(y_t))
            if valid.sum() < K + 5:
                continue

            X_cs = np.column_stack([np.ones(valid.sum()), beta_mat[valid]])
            y_cs = y_t[valid]

            try:
                coef = np.linalg.lstsq(X_cs, y_cs, rcond=None)[0]
                gammas.loc[t] = coef
            except np.linalg.LinAlgError:
                continue

        self.gammas = gammas.dropna()
        self._compute_risk_premia(factors)
        return self.gammas

    def _compute_risk_premia(self, factors: pd.DataFrame) -> None:
        """
        Compute risk premia estimates with Shanken-corrected standard errors.

        The naive standard error from the time-series variance of gamma_kt
        understates true uncertainty because betas are estimated, not observed.
        The Shanken (1992) correction inflates variance by a factor:
            c = 1 + hat{lambda}' * Sigma_F^{-1} * hat{lambda}

        This ensures valid inference for testing whether lambda_k != 0.
        """
        g = self.gammas.astype(float)
        T_eff = len(g)
        lambda_hat = g.mean()
        sigma_gamma = g.cov() / T_eff  # naive covariance of means

        if self.use_shanken and factors is not None:
            common = g.index.intersection(factors.index)
            F_common = factors.loc[common]
            sigma_F = F_common.cov()
            lam_vec = lambda_hat.iloc[1:].values  # exclude intercept
            try:
                sigma_F_inv = np.linalg.inv(sigma_F.values)
                c = 1.0 + lam_vec @ sigma_F_inv @ lam_vec
            except np.linalg.LinAlgError:
                c = 1.0
            # Shanken-adjusted covariance
            shanken_cov = sigma_gamma * c
        else:
            shanken_cov = sigma_gamma
            c = 1.0

        se = np.sqrt(np.diag(shanken_cov.values))
        t_stats = lambda_hat.values / se
        p_vals = 2 * (1 - sp_stats.t.cdf(np.abs(t_stats), df=T_eff - 1))

        self.risk_premia = pd.DataFrame({
            "Risk Premium": lambda_hat.values,
            "Std Error (Shanken)": se,
            "t-statistic": t_stats,
            "p-value": p_vals,
            "Ann. Premium (x12)": lambda_hat.values * 12,
        }, index=g.columns)
        self.shanken_se = se
        self._shanken_correction_factor = c

    def get_summary(self) -> str:
        """Return formatted summary of risk premia estimation."""
        if self.risk_premia is None:
            return "Run cross_sectional_regression() first."

        lines = [
            "=" * 70,
            "FAMA-MACBETH RISK PREMIA ESTIMATION",
            "=" * 70,
            f"Rolling window: {self.rolling_window} periods",
            f"Effective obs: {len(self.gammas)} cross-sections",
            f"Shanken correction factor: {self._shanken_correction_factor:.4f}",
            "-" * 70,
            self.risk_premia.to_string(float_format="{:.6f}".format),
            "-" * 70,
            "Significance: *** p<0.01, ** p<0.05, * p<0.10",
            "=" * 70,
        ]
        return "\n".join(lines)
