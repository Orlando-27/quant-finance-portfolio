"""
Illiquidity & Liquidity Risk Models
=====================================
Implements cross-sectional and time-series illiquidity measures.

Models:
    - Amihud (2002) ILLIQ ratio
    - Kyle (1985) lambda (price impact coefficient)
    - Pastor-Stambaugh (2002) liquidity factor
    - Turnover-based liquidity (Lo & Wang, 2000)

References:
    Amihud, Y. (2002). JFM 5(1), 31-56.
    Kyle, A.S. (1985). Econometrica 53(6), 1315-1335.
    Pastor, L. & Stambaugh, R.F. (2003). JPE 111(3), 642-685.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


class IlliquidityModels:
    """Illiquidity measurement and liquidity risk analytics."""

    # ------------------------------------------------------------------
    # Amihud (2002) ILLIQ Ratio
    # ------------------------------------------------------------------
    @staticmethod
    def amihud_illiq(
        returns      : pd.Series,
        dollar_volume: pd.Series,
        window       : int = 252,
    ) -> pd.Series:
        """
        Amihud (2002) daily illiquidity ratio:

            ILLIQ_t = |r_t| / DollarVolume_t

        Rolling mean gives the average price impact per dollar traded.
        Higher ILLIQ → less liquid (illiquidity premium candidate).

        Parameters
        ----------
        returns       : pd.Series  Daily returns (not percentage).
        dollar_volume : pd.Series  Daily dollar trading volume (price * shares).
        window        : int        Rolling window (252 = annual).

        Returns
        -------
        pd.Series
            Rolling Amihud ILLIQ (×10⁶ for readability).
        """
        illiq = (returns.abs() / dollar_volume.replace(0, np.nan)) * 1e6
        illiq_roll = illiq.rolling(window, min_periods=20).mean()
        illiq_roll.name = "amihud_illiq_1e6"
        return illiq_roll

    # ------------------------------------------------------------------
    # Kyle's Lambda
    # ------------------------------------------------------------------
    @staticmethod
    def kyle_lambda(
        prices    : pd.Series,
        volume    : pd.Series,
        window    : int = 20,
    ) -> pd.Series:
        """
        Estimate Kyle's (1985) lambda via rolling OLS regression:

            Δp_t = α + λ * x_t + ε_t

        where x_t = signed_volume = direction_t * volume_t.

        λ measures permanent price impact per unit of signed order flow.
        Higher λ → lower depth / more informed trading.

        Parameters
        ----------
        prices : pd.Series  Mid-prices.
        volume : pd.Series  Unsigned volume.
        window : int        Rolling OLS window.

        Returns
        -------
        pd.Series
            Rolling Kyle's lambda estimates.
        """
        dp        = prices.diff()
        direction = np.sign(dp).replace(0, np.nan).ffill().fillna(1)
        signed_vol = direction * volume

        lambdas = []
        idx     = prices.index

        for i in range(window, len(prices)):
            y = dp.iloc[i - window: i].dropna()
            x = signed_vol.iloc[i - window: i].loc[y.index]
            if len(y) < 5:
                lambdas.append(np.nan)
                continue
            slope, _, _, _, _ = stats.linregress(x, y)
            lambdas.append(slope)

        series = pd.Series(
            [np.nan] * window + lambdas,
            index=idx,
            name="kyle_lambda",
        )
        return series

    # ------------------------------------------------------------------
    # Pastor-Stambaugh (2002) Gamma (Liquidity Beta)
    # ------------------------------------------------------------------
    @staticmethod
    def pastor_stambaugh_gamma(
        returns   : pd.Series,
        volume    : pd.Series,
        window    : int = 60,
    ) -> pd.Series:
        """
        Pastor-Stambaugh (2003) liquidity measure gamma_i.

        Estimates the sensitivity of next-day returns to signed turnover:

            r_{t+1}^e = θ + φ * r_t + γ * sign(r_t^e) * v_t + ε_{t+1}

        A more negative γ indicates higher liquidity (stronger reversal
        after volume-driven moves). Used to construct the P-S liquidity
        factor in asset pricing.

        Returns
        -------
        pd.Series
            Rolling γ estimates (more negative = more liquid).
        """
        signed_vol = np.sign(returns) * volume
        gammas     = []
        idx        = returns.index

        for i in range(window, len(returns)):
            r_t   = returns.iloc[i - window: i]
            sv_t  = signed_vol.iloc[i - window: i]
            r_tp1 = returns.iloc[i - window + 1: i + 1]

            if len(r_t) < 10:
                gammas.append(np.nan)
                continue

            X = np.column_stack([np.ones(len(r_t)), r_t.values, sv_t.values])
            try:
                coef, _, _, _ = np.linalg.lstsq(X, r_tp1.values, rcond=None)
                gammas.append(coef[2])
            except np.linalg.LinAlgError:
                gammas.append(np.nan)

        series = pd.Series(
            [np.nan] * window + gammas,
            index=idx,
            name="pastor_stambaugh_gamma",
        )
        return series

    # ------------------------------------------------------------------
    # Turnover-Based Liquidity
    # ------------------------------------------------------------------
    @staticmethod
    def turnover_liquidity(
        volume    : pd.Series,
        shares_out: float = 1.0,
        window    : int = 21,
    ) -> pd.Series:
        """
        Turnover ratio: shares traded / shares outstanding (rolling mean).

        Higher turnover indicates greater liquidity. Used in the
        Lo-Wang (2000) liquidity decomposition.
        """
        turnover = volume / shares_out
        turn_roll = turnover.rolling(window).mean()
        turn_roll.name = "turnover_ratio"
        return turn_roll

    # ------------------------------------------------------------------
    # Composite Liquidity Score
    # ------------------------------------------------------------------
    @staticmethod
    def composite_liquidity_score(
        amihud  : pd.Series,
        kyle_lam: pd.Series,
        turnover: pd.Series,
    ) -> pd.Series:
        """
        Composite liquidity score (rank-based Z-score combination).

        Combines Amihud (illiquidity), Kyle lambda (impact), and
        turnover (activity) into a single standardized score.

        Score > 0 → above-average liquidity; < 0 → below-average.
        """
        def _zscore(s: pd.Series) -> pd.Series:
            return (s - s.expanding().mean()) / s.expanding().std().replace(0, 1)

        # Note: Amihud and Kyle are illiquidity, so invert sign
        score = (
            -_zscore(amihud)
            - _zscore(kyle_lam)
            + _zscore(turnover)
        ) / 3.0
        score.name = "composite_liquidity_score"
        return score
