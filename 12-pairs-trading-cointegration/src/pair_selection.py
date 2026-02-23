"""
Pair Selection Methods
======================

Implements distance-based (Gatev et al., 2006) and cointegration-based
pair selection algorithms for systematic statistical arbitrage.

The selection process operates on a formation period to identify candidate
pairs, which are then traded during a subsequent trading period. This
temporal separation prevents in-sample overfitting.

References:
    Gatev, Goetzmann & Rouwenhorst (2006), Vidyamurthy (2004)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from itertools import combinations
from src.cointegration import EngleGranger


class PairSelector:
    """
    Systematic pair selection using distance and cointegration methods.

    Parameters
    ----------
    method : str
        'distance' for SSD-based selection, 'cointegration' for
        Engle-Granger based selection, 'both' for intersection.
    significance : float
        Significance level for cointegration test (default 0.05).
    min_half_life : float
        Minimum acceptable half-life in days (default 5).
    max_half_life : float
        Maximum acceptable half-life in days (default 60).
    top_k : int
        Number of top pairs to select (default 20).
    """

    def __init__(self, method: str = "cointegration",
                 significance: float = 0.05,
                 min_half_life: float = 5.0,
                 max_half_life: float = 60.0,
                 top_k: int = 20):
        self.method = method
        self.significance = significance
        self.min_hl = min_half_life
        self.max_hl = max_half_life
        self.top_k = top_k
        self.pair_scores = None
        self.selected_pairs = None

    def select(self, prices: pd.DataFrame) -> List[Dict]:
        """
        Identify tradeable pairs from a universe of price series.

        Parameters
        ----------
        prices : pd.DataFrame
            (T x N) matrix of price series (columns = tickers).

        Returns
        -------
        list of dict
            Each dict contains: pair (tuple), hedge_ratio, half_life,
            adf_stat, adf_pvalue, ssd (sum of squared deviations),
            and other diagnostics.
        """
        tickers = prices.columns.tolist()
        all_pairs = list(combinations(tickers, 2))
        log_prices = np.log(prices)

        results = []
        for i, (t1, t2) in enumerate(all_pairs):
            p1 = log_prices[t1].dropna()
            p2 = log_prices[t2].dropna()
            common = p1.index.intersection(p2.index)
            if len(common) < 60:
                continue

            pair_info = {"pair": (t1, t2)}

            # Distance metric: SSD on normalized prices
            norm1 = prices[t1].loc[common] / prices[t1].loc[common].iloc[0]
            norm2 = prices[t2].loc[common] / prices[t2].loc[common].iloc[0]
            pair_info["ssd"] = float(((norm1 - norm2) ** 2).sum())

            # Cointegration test
            eg = EngleGranger(significance=self.significance)
            coint_res = eg.test(p1.loc[common], p2.loc[common])
            pair_info["cointegrated"] = coint_res.get("cointegrated", False)
            pair_info["adf_stat"] = coint_res.get("adf_stat", np.nan)
            pair_info["adf_pvalue"] = coint_res.get("adf_pvalue", 1.0)
            pair_info["hedge_ratio"] = coint_res.get("hedge_ratio", np.nan)

            # Half-life estimation from spread
            if isinstance(coint_res.get("residuals"), pd.Series):
                spread = coint_res["residuals"]
                if len(spread) > 10:
                    hl = self._estimate_half_life(spread.values)
                    pair_info["half_life"] = hl
                else:
                    pair_info["half_life"] = np.nan
            else:
                pair_info["half_life"] = np.nan

            # Hurst exponent for mean-reversion strength
            if isinstance(coint_res.get("residuals"), pd.Series):
                pair_info["hurst"] = self._hurst_exponent(
                    coint_res["residuals"].values
                )
            else:
                pair_info["hurst"] = np.nan

            # Tradeable flag: cointegrated with valid half-life
            pair_info["tradeable"] = (
                pair_info["cointegrated"]
                and self.min_hl < pair_info.get("half_life", 999) < self.max_hl
            )

            results.append(pair_info)

        self.pair_scores = pd.DataFrame(results)

        # Select top pairs based on method
        if self.method == "distance":
            selected = self.pair_scores.nsmallest(self.top_k, "ssd")
        elif self.method == "cointegration":
            tradeable = self.pair_scores[self.pair_scores["tradeable"]]
            selected = tradeable.nsmallest(
                min(self.top_k, len(tradeable)), "adf_pvalue"
            )
        else:
            tradeable = self.pair_scores[self.pair_scores["tradeable"]]
            tradeable = tradeable.sort_values("ssd")
            selected = tradeable.head(min(self.top_k, len(tradeable)))

        self.selected_pairs = selected.to_dict("records")
        return self.selected_pairs

    @staticmethod
    def _estimate_half_life(spread: np.ndarray) -> float:
        """
        Estimate mean-reversion half-life from AR(1) regression on spread.

        S_t = c + phi * S_{t-1} + eps_t
        Half-life = -ln(2) / ln(phi)
        """
        y = spread[1:]
        x = spread[:-1]
        x_aug = np.column_stack([np.ones(len(x)), x])
        try:
            coef = np.linalg.lstsq(x_aug, y, rcond=None)[0]
            phi = coef[1]
            if 0 < phi < 1:
                return -np.log(2) / np.log(phi)
            else:
                return 999.0
        except np.linalg.LinAlgError:
            return 999.0

    @staticmethod
    def _hurst_exponent(series: np.ndarray, max_lag: int = 40) -> float:
        """
        Estimate the Hurst exponent via the rescaled range (R/S) method.

        H < 0.5: mean-reverting (favorable for pairs trading)
        H = 0.5: random walk
        H > 0.5: trending
        """
        lags = range(2, min(max_lag, len(series) // 4))
        tau = []
        rs = []

        for lag in lags:
            sub = series[:lag * (len(series) // lag)]
            sub = sub.reshape(-1, lag)
            means = sub.mean(axis=1, keepdims=True)
            devs = sub - means
            cum_devs = devs.cumsum(axis=1)
            R = cum_devs.max(axis=1) - cum_devs.min(axis=1)
            S = sub.std(axis=1, ddof=1)
            valid = S > 1e-10
            if valid.sum() > 0:
                rs_ratio = (R[valid] / S[valid]).mean()
                if rs_ratio > 0:
                    tau.append(lag)
                    rs.append(rs_ratio)

        if len(tau) < 3:
            return 0.5

        log_tau = np.log(tau)
        log_rs = np.log(rs)
        try:
            coef = np.polyfit(log_tau, log_rs, 1)
            return coef[0]
        except (np.linalg.LinAlgError, ValueError):
            return 0.5
