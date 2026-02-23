"""
================================================================================
TRADITIONAL PORTFOLIO STRATEGY BASELINES
================================================================================
Implements classical strategies for benchmarking against RL agents.

Strategies:
    1. Equal Weight (1/N)   -- DeMiguel et al. (2009), surprisingly hard to beat
    2. Risk Parity          -- Weight inversely proportional to volatility
    3. Mean-Variance        -- Markowitz with Ledoit-Wolf shrinkage covariance
    4. Momentum             -- Long top-k assets by 12-month return
    5. Buy-and-Hold         -- Initial equal weight, no rebalancing
    6. Minimum Variance     -- Portfolio minimizing total variance

Each baseline returns a weight time series that can be compared with
RL agent outputs using the same performance metrics.

Author: Jose Orlando Bobadilla Fuentes, CQF | MSc AI
================================================================================
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Optional, Dict


class EqualWeightBaseline:
    """
    1/N portfolio: allocate equally across all assets at every rebalance.

    Despite its simplicity, DeMiguel et al. (2009) showed that 1/N
    outperforms most optimized portfolios out of sample due to estimation
    error in expected returns and covariances.
    """

    def __init__(self, n_assets: int, rebalance_freq: int = 21):
        self.n = n_assets
        self.freq = rebalance_freq

    def generate_weights(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Return constant equal-weight allocation."""
        w = np.ones(self.n) / self.n
        weights = pd.DataFrame(
            np.tile(w, (len(returns), 1)),
            index=returns.index,
            columns=returns.columns,
        )
        return weights


class RiskParityBaseline:
    """
    Risk parity: weight each asset inversely proportional to its volatility.

    w_i = (1/sigma_i) / sum(1/sigma_j)

    This equalizes the marginal risk contribution of each asset,
    ensuring that no single asset dominates portfolio risk.

    Parameters
    ----------
    lookback : int
        Volatility estimation window in days (default 63 = 1 quarter).
    rebalance_freq : int
        Days between rebalancing (default 21 = monthly).
    """

    def __init__(self, lookback: int = 63, rebalance_freq: int = 21):
        self.lookback = lookback
        self.freq = rebalance_freq

    def generate_weights(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Compute time-varying risk parity weights."""
        n = returns.shape[1]
        weights = pd.DataFrame(
            np.ones((len(returns), n)) / n,
            index=returns.index,
            columns=returns.columns,
        )

        for i in range(self.lookback, len(returns), self.freq):
            window = returns.iloc[max(0, i - self.lookback):i]
            vols = window.std() * np.sqrt(252)
            vols = vols.replace(0, np.nan).fillna(vols.mean())

            inv_vol = 1.0 / vols
            w = inv_vol / inv_vol.sum()
            w = w.values

            end = min(i + self.freq, len(returns))
            weights.iloc[i:end] = w

        return weights


class MeanVarianceBaseline:
    """
    Markowitz mean-variance optimization with Ledoit-Wolf shrinkage.

    Solves: max w'mu - (lambda/2) * w'Sigma*w
    subject to: sum(w) = 1, w >= 0

    Uses shrinkage covariance to mitigate estimation error.

    Parameters
    ----------
    risk_aversion : float
        Risk aversion parameter (default 2.0).
    lookback : int
        Estimation window (default 252 = 1 year).
    rebalance_freq : int
        Rebalancing frequency in days (default 63 = quarterly).
    """

    def __init__(
        self,
        risk_aversion: float = 2.0,
        lookback: int = 252,
        rebalance_freq: int = 63,
    ):
        self.gamma = risk_aversion
        self.lookback = lookback
        self.freq = rebalance_freq

    @staticmethod
    def _ledoit_wolf_shrinkage(returns: pd.DataFrame) -> np.ndarray:
        """Ledoit-Wolf linear shrinkage toward identity."""
        X = returns.values
        n, p = X.shape
        S = np.cov(X, rowvar=False) * 252  # Annualize
        mu_hat = np.trace(S) / p
        F = mu_hat * np.eye(p)

        # Optimal shrinkage intensity (simplified)
        d2 = np.sum((S - F) ** 2) / p
        X_centered = X - X.mean(axis=0)
        b2 = 0
        for i in range(n):
            xi = X_centered[i:i+1]
            b2 += np.sum((xi.T @ xi - S) ** 2)
        b2 = b2 / (n ** 2 * p)

        delta = max(0, min(1, b2 / (d2 + 1e-10)))
        return delta * F + (1 - delta) * S

    def generate_weights(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Compute time-varying mean-variance optimal weights."""
        n = returns.shape[1]
        weights = pd.DataFrame(
            np.ones((len(returns), n)) / n,
            index=returns.index,
            columns=returns.columns,
        )

        for i in range(self.lookback, len(returns), self.freq):
            window = returns.iloc[max(0, i - self.lookback):i]
            mu = window.mean().values * 252
            Sigma = self._ledoit_wolf_shrinkage(window)

            # Quadratic programming via scipy
            def neg_utility(w):
                return -(w @ mu - 0.5 * self.gamma * w @ Sigma @ w)

            constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
            bounds = [(0, 0.4)] * n  # Long-only, max 40% per asset

            x0 = np.ones(n) / n
            result = minimize(
                neg_utility, x0, method="SLSQP",
                bounds=bounds, constraints=constraints,
            )
            w = result.x if result.success else x0

            end = min(i + self.freq, len(returns))
            weights.iloc[i:end] = w

        return weights


class MomentumBaseline:
    """
    Cross-sectional momentum: long top-k assets by trailing 12-month return.

    Parameters
    ----------
    top_k : int
        Number of assets in the long portfolio (default 3).
    lookback : int
        Momentum lookback in days (default 252 = 12 months).
    rebalance_freq : int
        Rebalancing frequency in days (default 21 = monthly).
    """

    def __init__(
        self, top_k: int = 3, lookback: int = 252, rebalance_freq: int = 21
    ):
        self.top_k = top_k
        self.lookback = lookback
        self.freq = rebalance_freq

    def generate_weights(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Compute momentum-based weights."""
        n = returns.shape[1]
        weights = pd.DataFrame(
            np.ones((len(returns), n)) / n,
            index=returns.index,
            columns=returns.columns,
        )

        for i in range(self.lookback, len(returns), self.freq):
            window = returns.iloc[max(0, i - self.lookback):i]
            cum_ret = (1 + window).prod() - 1

            # Top-k assets by cumulative return
            top_assets = cum_ret.nlargest(self.top_k).index
            w = np.zeros(n)
            for col in top_assets:
                idx = returns.columns.get_loc(col)
                w[idx] = 1.0 / self.top_k

            end = min(i + self.freq, len(returns))
            weights.iloc[i:end] = w

        return weights


class MinimumVarianceBaseline:
    """
    Minimum variance portfolio: minimize w'Sigma*w subject to sum(w)=1.

    This requires NO return estimates, only the covariance matrix,
    making it robust to return estimation error.
    """

    def __init__(self, lookback: int = 252, rebalance_freq: int = 63):
        self.lookback = lookback
        self.freq = rebalance_freq

    def generate_weights(self, returns: pd.DataFrame) -> pd.DataFrame:
        n = returns.shape[1]
        weights = pd.DataFrame(
            np.ones((len(returns), n)) / n,
            index=returns.index,
            columns=returns.columns,
        )

        for i in range(self.lookback, len(returns), self.freq):
            window = returns.iloc[max(0, i - self.lookback):i]
            Sigma = window.cov().values * 252

            def portfolio_var(w):
                return w @ Sigma @ w

            constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
            bounds = [(0, 0.4)] * n

            x0 = np.ones(n) / n
            result = minimize(
                portfolio_var, x0, method="SLSQP",
                bounds=bounds, constraints=constraints,
            )
            w = result.x if result.success else x0

            end = min(i + self.freq, len(returns))
            weights.iloc[i:end] = w

        return weights


def run_all_baselines(returns: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Run all baseline strategies and return weight DataFrames."""
    n = returns.shape[1]
    return {
        "Equal Weight": EqualWeightBaseline(n).generate_weights(returns),
        "Risk Parity": RiskParityBaseline().generate_weights(returns),
        "Mean-Variance": MeanVarianceBaseline().generate_weights(returns),
        "Momentum": MomentumBaseline(top_k=min(3, n)).generate_weights(returns),
        "Min Variance": MinimumVarianceBaseline().generate_weights(returns),
    }
