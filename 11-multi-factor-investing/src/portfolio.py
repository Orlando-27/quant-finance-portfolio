"""
Factor Portfolio Construction & Optimization
=============================================

Implements multiple portfolio construction methodologies in factor return
space, including equal-weight, inverse-volatility, risk parity, and
mean-variance optimization.

Methodologies:
    1. Equal-Weight: w_k = 1/K for all factors.
       Simple, robust, but ignores correlation structure.

    2. Inverse-Volatility: w_k = (1/sigma_k) / sum_j(1/sigma_j)
       Tilts toward lower-volatility factors. Ignores correlations.

    3. Risk Parity: Equalizes risk contribution from each factor.
       Solves: RC_k = w_k * (Sigma * w)_k / (w' * Sigma * w) = 1/K
       via Sequential Least Squares Programming (SLSQP).

    4. Mean-Variance (Markowitz): Maximize Sharpe or minimize variance
       subject to constraints. Optionally uses ML-predicted expected
       returns for dynamic allocation.

    5. Maximum Diversification: Maximize the diversification ratio
       DR = (w' * sigma) / sqrt(w' * Sigma * w)

References:
    Maillard, Roncalli & Teiletche (2010) - Risk parity
    Choueifaty & Coignard (2008) - Maximum diversification
    Ang (2014) - Factor investing framework
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from scipy.optimize import minimize


class FactorPortfolio:
    """
    Multi-strategy factor portfolio optimizer.

    Parameters
    ----------
    factor_returns : pd.DataFrame
        (T x K) historical factor return series.
    risk_free_rate : float
        Annual risk-free rate (default 2%).
    """

    def __init__(self, factor_returns: pd.DataFrame,
                 risk_free_rate: float = 0.02):
        self.factor_returns = factor_returns
        self.rf = risk_free_rate / 12  # monthly
        self.K = factor_returns.shape[1]
        self.factor_names = factor_returns.columns.tolist()
        self._update_moments()

    def _update_moments(self, window: Optional[int] = None) -> None:
        """Compute factor return moments from specified window."""
        data = self.factor_returns if window is None \
            else self.factor_returns.iloc[-window:]
        self.mu = data.mean().values      # (K,) monthly expected returns
        self.Sigma = data.cov().values     # (K x K) monthly covariance
        self.vol = data.std().values       # (K,) monthly volatilities

    def equal_weight(self) -> np.ndarray:
        """Equal-weight allocation across K factors."""
        return np.ones(self.K) / self.K

    def inverse_volatility(self) -> np.ndarray:
        """
        Inverse-volatility weighting.
        Assigns weight proportional to 1/sigma_k, normalized to sum to 1.
        Tilts portfolio toward lower-risk factors.
        """
        inv_vol = 1.0 / np.maximum(self.vol, 1e-8)
        return inv_vol / inv_vol.sum()

    def risk_parity(self) -> np.ndarray:
        """
        Risk parity: equalize risk contribution from each factor.

        The risk contribution of factor k is:
            RC_k = w_k * (Sigma * w)_k / sigma_p

        Risk parity requires RC_k = sigma_p / K for all k.

        Solved via minimization of sum of squared differences between
        actual and target risk contributions, subject to w >= 0, sum(w) = 1.
        """
        target_rc = 1.0 / self.K

        def objective(w):
            port_var = w @ self.Sigma @ w
            if port_var < 1e-12:
                return 1e6
            port_vol = np.sqrt(port_var)
            marginal = self.Sigma @ w
            rc = w * marginal / port_vol
            rc_pct = rc / port_vol
            return np.sum((rc_pct - target_rc) ** 2)

        w0 = self.inverse_volatility()
        bounds = [(0.01, 1.0)] * self.K
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        result = minimize(objective, w0, method="SLSQP",
                          bounds=bounds, constraints=constraints,
                          options={"maxiter": 1000, "ftol": 1e-12})
        return result.x / result.x.sum()

    def mean_variance(self, target: str = "max_sharpe",
                      expected_returns: Optional[np.ndarray] = None,
                      max_weight: float = 0.40) -> np.ndarray:
        """
        Mean-variance optimization in factor space.

        Parameters
        ----------
        target : str
            'max_sharpe' or 'min_variance'.
        expected_returns : np.ndarray, optional
            Override historical mean with ML-predicted returns.
        max_weight : float
            Maximum weight per factor (prevents corner solutions).
        """
        mu = expected_returns if expected_returns is not None else self.mu

        if target == "min_variance":
            def objective(w):
                return w @ self.Sigma @ w
        else:
            def objective(w):
                port_ret = w @ mu - self.rf
                port_vol = np.sqrt(w @ self.Sigma @ w)
                return -port_ret / max(port_vol, 1e-8)

        w0 = self.equal_weight()
        bounds = [(0.0, max_weight)] * self.K
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        result = minimize(objective, w0, method="SLSQP",
                          bounds=bounds, constraints=constraints,
                          options={"maxiter": 1000})
        return result.x / result.x.sum()

    def maximum_diversification(self) -> np.ndarray:
        """
        Maximize the diversification ratio:
            DR = (w' * sigma) / sqrt(w' * Sigma * w)

        Higher DR means the portfolio exploits imperfect correlations
        more effectively, achieving greater diversification benefit.
        """
        def neg_div_ratio(w):
            num = w @ self.vol
            denom = np.sqrt(w @ self.Sigma @ w)
            return -num / max(denom, 1e-8)

        w0 = self.equal_weight()
        bounds = [(0.01, 0.50)] * self.K
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        result = minimize(neg_div_ratio, w0, method="SLSQP",
                          bounds=bounds, constraints=constraints)
        return result.x / result.x.sum()

    def compute_all_strategies(self,
                               expected_returns: Optional[np.ndarray] = None
                               ) -> pd.DataFrame:
        """
        Compute weights for all portfolio strategies.

        Returns
        -------
        pd.DataFrame
            (K x S) matrix of factor weights for each strategy.
        """
        strategies = {
            "Equal Weight": self.equal_weight(),
            "Inverse Vol": self.inverse_volatility(),
            "Risk Parity": self.risk_parity(),
            "Max Sharpe": self.mean_variance("max_sharpe", expected_returns),
            "Min Variance": self.mean_variance("min_variance"),
            "Max Diversification": self.maximum_diversification(),
        }
        return pd.DataFrame(strategies, index=self.factor_names)

    def portfolio_metrics(self, weights: np.ndarray,
                          label: str = "") -> Dict:
        """
        Compute portfolio-level performance metrics.

        Parameters
        ----------
        weights : np.ndarray
            (K,) factor allocation weights.
        label : str
            Strategy label for identification.

        Returns
        -------
        dict
            ann_return, ann_vol, sharpe, max_drawdown, calmar, turnover_proxy.
        """
        port_ret = self.factor_returns.values @ weights
        port_series = pd.Series(port_ret, index=self.factor_returns.index)

        ann_ret = port_series.mean() * 12
        ann_vol = port_series.std() * np.sqrt(12)
        sharpe = (ann_ret - self.rf * 12) / ann_vol if ann_vol > 0 else 0
        cum = (1 + port_series).cumprod()
        max_dd = (cum / cum.cummax() - 1).min()
        calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0
        skew = port_series.skew()
        kurt = port_series.kurtosis()

        return {
            "Strategy": label,
            "Ann. Return": ann_ret,
            "Ann. Volatility": ann_vol,
            "Sharpe Ratio": sharpe,
            "Max Drawdown": max_dd,
            "Calmar Ratio": calmar,
            "Skewness": skew,
            "Excess Kurtosis": kurt,
        }
