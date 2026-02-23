"""
Momentum Signal Generators
============================

Implements two canonical momentum strategies:

1. Time-Series Momentum (TSMOM) -- Moskowitz, Ooi & Pedersen (2012)
   Each asset's own past return predicts its future return.
   Position: sign(trailing return) scaled by inverse volatility.

2. Cross-Sectional Momentum (CS-MOM) -- Jegadeesh & Titman (1993)
   Rank assets by trailing return. Long winners, short losers.
   Standard 12-1 specification: 12-month lookback, skip most recent month.

Both signals support multiple lookback windows and include the
"momentum quality" filter from Gray & Vogel (2016) to distinguish
smooth vs. choppy momentum paths.

Author: Jose Orlando Bobadilla Fuentes, CQF, MSc AI
"""

import numpy as np
import pandas as pd
from typing import Optional


class TimeSeriesMomentum:
    """
    Time-Series Momentum (TSMOM) signal generator.

    Following Moskowitz, Ooi & Pedersen (2012), the signal for each asset
    is based on its own trailing return. Positions are scaled by inverse
    EWMA volatility to achieve volatility targeting.

    Parameters
    ----------
    lookback_days : int
        Number of trading days for return calculation (default 252 = 12 months).
    vol_lookback : int
        EWMA volatility estimation window in days (default 60).
    vol_target : float
        Annualized volatility target for position sizing (default 0.10 = 10%).
    skip_days : int
        Number of most recent days to skip (short-term reversal filter).
    """

    def __init__(
        self,
        lookback_days: int = 252,
        vol_lookback: int = 60,
        vol_target: float = 0.10,
        skip_days: int = 21,
    ):
        self.lookback_days = lookback_days
        self.vol_lookback = vol_lookback
        self.vol_target = vol_target
        self.skip_days = skip_days

    def compute_signal(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Compute TSMOM signal for each asset.

        The raw signal is sign(cumulative return over lookback window,
        excluding the most recent skip_days). The signal is then scaled
        by the ratio of vol_target to estimated EWMA volatility.

        Parameters
        ----------
        returns : pd.DataFrame
            Daily returns (dates x assets).

        Returns
        -------
        pd.DataFrame
            TSMOM signal values. Positive = long, negative = short.
            Magnitude reflects volatility-scaled position size.
        """
        # Trailing cumulative return excluding skip period
        # r_{t-skip-lookback : t-skip}
        cum_ret = returns.rolling(window=self.lookback_days).sum()
        if self.skip_days > 0:
            cum_ret_skip = returns.rolling(window=self.skip_days).sum()
            cum_ret = cum_ret - cum_ret_skip

        # Raw directional signal: +1 or -1
        raw_signal = np.sign(cum_ret)

        # EWMA volatility for position sizing
        # Using pandas ewm with span = vol_lookback
        ewma_vol = returns.ewm(
            span=self.vol_lookback, min_periods=self.vol_lookback
        ).std() * np.sqrt(252)

        # Scale signal by inverse volatility (volatility targeting)
        # w = (sigma_target / sigma_hat) * sign(cum_ret)
        vol_scale = self.vol_target / ewma_vol.replace(0, np.nan)
        signal = raw_signal * vol_scale

        # Cap individual position at 3x vol-target ratio
        signal = signal.clip(-3.0, 3.0)

        return signal

    def compute_momentum_quality(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Compute momentum quality score (Gray & Vogel, 2016).

        Momentum quality measures the smoothness of the return path.
        High-quality momentum has a higher percentage of positive-return
        days within the lookback window, indicating a consistent trend
        rather than a choppy, erratic path.

        The quality score is:
            Q = (fraction of positive days - 0.5) * 2

        Values near +1 indicate very smooth upward momentum,
        values near -1 indicate smooth downward momentum,
        values near 0 indicate choppy/noisy returns.

        Parameters
        ----------
        returns : pd.DataFrame
            Daily returns.

        Returns
        -------
        pd.DataFrame
            Momentum quality score in [-1, 1].
        """
        pos_frac = (returns > 0).astype(float).rolling(
            window=self.lookback_days
        ).mean()

        quality = (pos_frac - 0.5) * 2.0
        return quality


class CrossSectionalMomentum:
    """
    Cross-Sectional Momentum (Jegadeesh-Titman) signal generator.

    At each date, ranks assets by their trailing J-month return
    (skipping the most recent month to avoid short-term reversal).
    The signal is a normalized rank score in [-1, +1].

    Parameters
    ----------
    lookback_months : int
        Formation period in months (default 12).
    skip_months : int
        Months to skip (default 1, the "12-1" specification).
    """

    def __init__(self, lookback_months: int = 12, skip_months: int = 1):
        self.lookback_days = lookback_months * 21   # ~21 trading days/month
        self.skip_days = skip_months * 21

    def compute_signal(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Compute cross-sectional momentum rank signal.

        For each date, computes trailing return over the formation
        period (excluding skip period), then converts to a normalized
        rank score. Top-ranked assets get positive signal (long),
        bottom-ranked get negative signal (short).

        The rank is normalized to [-1, +1]:
            signal = 2 * (rank - 1) / (N - 1) - 1

        Parameters
        ----------
        returns : pd.DataFrame
            Daily returns (dates x assets).

        Returns
        -------
        pd.DataFrame
            Cross-sectional momentum signal in [-1, +1].
        """
        # Trailing return over full period
        full_ret = returns.rolling(window=self.lookback_days).sum()
        # Trailing return over skip period
        skip_ret = returns.rolling(window=self.skip_days).sum()
        # Formation period return = full - skip
        formation_ret = full_ret - skip_ret

        # Cross-sectional rank at each date
        # rank(ascending=True): worst performer = rank 1, best = rank N
        ranks = formation_ret.rank(axis=1, ascending=True, method="average")
        n_assets = returns.shape[1]

        # Normalize to [-1, +1]
        signal = 2.0 * (ranks - 1.0) / max(n_assets - 1.0, 1.0) - 1.0

        return signal

    def compute_winner_loser_spread(
        self, returns: pd.DataFrame, quantile: float = 0.2
    ) -> pd.Series:
        """
        Compute the long-short spread: winners minus losers return.

        At each date, identifies the top and bottom quantile of assets
        by formation-period return, then computes the equal-weighted
        spread of their next-period returns.

        Parameters
        ----------
        returns : pd.DataFrame
            Daily returns.
        quantile : float
            Fraction for winner/loser portfolios (default 0.2 = quintiles).

        Returns
        -------
        pd.Series
            Daily winner-minus-loser return spread.
        """
        full_ret = returns.rolling(window=self.lookback_days).sum()
        skip_ret = returns.rolling(window=self.skip_days).sum()
        formation_ret = full_ret - skip_ret

        n_assets = returns.shape[1]
        n_top = max(int(n_assets * quantile), 1)

        spread = pd.Series(0.0, index=returns.index)

        for t in range(self.lookback_days + 1, len(returns)):
            row = formation_ret.iloc[t]
            valid = row.dropna()
            if len(valid) < 3:
                continue
            sorted_assets = valid.sort_values()
            losers = sorted_assets.index[:n_top]
            winners = sorted_assets.index[-n_top:]

            ret_t = returns.iloc[t]
            long_ret = ret_t[winners].mean()
            short_ret = ret_t[losers].mean()
            spread.iloc[t] = long_ret - short_ret

        return spread
