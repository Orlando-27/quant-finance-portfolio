"""
Market Regime Detection
========================

Identifies market regimes to dynamically blend momentum and mean-reversion
signals. Three complementary regime indicators:

1. Volatility Regime:
   - Low vol -> trending markets -> favor momentum
   - High vol -> choppy/crisis -> favor mean reversion or reduce exposure

2. Dispersion Regime:
   - High cross-sectional dispersion -> favor cross-sectional momentum
   - Low dispersion -> factor crowding -> favor mean reversion

3. Autocorrelation Regime:
   - Positive autocorrelation -> trending -> favor TSMOM
   - Negative autocorrelation -> mean-reverting -> favor MR signals

The composite regime score determines the blending alpha between
momentum (alpha -> 1) and mean reversion (alpha -> 0).

Author: Jose Orlando Bobadilla Fuentes, CQF, MSc AI
"""

import numpy as np
import pandas as pd
from typing import Tuple


class VolatilityRegime:
    """
    Volatility-based regime detector.

    Compares short-term realized volatility (EWMA) against its long-term
    moving average. The ratio identifies whether current conditions are
    elevated (crisis/choppy) or subdued (trending/calm).

    Parameters
    ----------
    short_window : int
        EWMA span for short-term vol estimation (default 21).
    long_window : int
        Rolling window for long-term vol average (default 252).
    """

    def __init__(self, short_window: int = 21, long_window: int = 252):
        self.short_window = short_window
        self.long_window = long_window

    def compute_regime(self, returns: pd.DataFrame) -> pd.Series:
        """
        Compute volatility regime indicator.

        Returns a score in [0, 1]:
            0 -> extremely low vol (strong trend regime)
            0.5 -> normal vol
            1 -> extremely high vol (crisis/mean-reversion regime)

        Parameters
        ----------
        returns : pd.DataFrame
            Daily returns for multiple assets.

        Returns
        -------
        pd.Series
            Volatility regime score.
        """
        # Cross-sectional average volatility
        avg_returns = returns.mean(axis=1)

        short_vol = avg_returns.ewm(span=self.short_window).std() * np.sqrt(252)
        long_vol = avg_returns.rolling(window=self.long_window).std() * np.sqrt(252)

        # Ratio: short_vol / long_vol, capped and normalized
        ratio = short_vol / long_vol.replace(0, np.nan)

        # Map ratio to [0, 1]: ratio=0.5 -> 0, ratio=1.0 -> 0.5, ratio=2.0 -> 1.0
        score = (ratio - 0.5) / 1.5
        score = score.clip(0.0, 1.0)

        return score


class DispersionRegime:
    """
    Cross-sectional return dispersion regime detector.

    High dispersion indicates strong factor differentiation (good for
    cross-sectional momentum). Low dispersion indicates factor crowding
    (favors mean reversion or reduced exposure).

    Parameters
    ----------
    lookback : int
        Rolling window for dispersion estimation (default 63 ~ 3 months).
    """

    def __init__(self, lookback: int = 63):
        self.lookback = lookback

    def compute_regime(self, returns: pd.DataFrame) -> pd.Series:
        """
        Compute dispersion regime indicator.

        Returns a score in [0, 1]:
            0 -> low dispersion (crowded, favor MR)
            1 -> high dispersion (differentiated, favor momentum)

        Parameters
        ----------
        returns : pd.DataFrame
            Daily returns.

        Returns
        -------
        pd.Series
            Dispersion regime score.
        """
        # Cross-sectional standard deviation at each date
        cs_std = returns.std(axis=1)

        # Rolling percentile rank of current dispersion
        rolling_disp = cs_std.rolling(window=self.lookback)
        # Use rank within the rolling window as a percentile
        rank_pct = cs_std.rolling(window=self.lookback).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1],
            raw=False,
        )

        return rank_pct.clip(0.0, 1.0)


class AutocorrelationRegime:
    """
    Return autocorrelation regime detector.

    Positive first-order autocorrelation indicates trending behavior
    (momentum-friendly). Negative autocorrelation indicates mean-reversion.

    Uses cross-sectional average of individual asset autocorrelations
    to get a market-wide regime indicator.

    Parameters
    ----------
    lookback : int
        Rolling window for autocorrelation estimation (default 63).
    """

    def __init__(self, lookback: int = 63):
        self.lookback = lookback

    def compute_regime(self, returns: pd.DataFrame) -> pd.Series:
        """
        Compute autocorrelation regime indicator.

        Returns a score in [0, 1]:
            0 -> strong negative autocorrelation (mean-reverting)
            0.5 -> zero autocorrelation (random walk)
            1 -> strong positive autocorrelation (trending)

        Parameters
        ----------
        returns : pd.DataFrame
            Daily returns.

        Returns
        -------
        pd.Series
            Autocorrelation regime score.
        """
        # Compute rolling AC(1) for each asset, then average
        def _rolling_ac1(series):
            """First-order autocorrelation over a rolling window."""
            return series.rolling(window=self.lookback).apply(
                lambda x: pd.Series(x).autocorr(lag=1) if len(x) > 5 else 0.0,
                raw=False,
            )

        ac_per_asset = returns.apply(_rolling_ac1)
        avg_ac = ac_per_asset.mean(axis=1)

        # Map from [-1, 1] to [0, 1]
        score = (avg_ac + 1.0) / 2.0
        return score.clip(0.0, 1.0)


class RegimeDetector:
    """
    Composite regime detector combining volatility, dispersion, and
    autocorrelation indicators.

    The composite regime score determines how to blend momentum and
    mean-reversion signals:
        high score (near 1.0) -> favor momentum
        low score (near 0.0) -> favor mean reversion

    Parameters
    ----------
    w_vol : float
        Weight for volatility regime (default 0.35).
    w_disp : float
        Weight for dispersion regime (default 0.30).
    w_ac : float
        Weight for autocorrelation regime (default 0.35).
    """

    def __init__(
        self,
        w_vol: float = 0.35,
        w_disp: float = 0.30,
        w_ac: float = 0.35,
    ):
        self.w_vol = w_vol
        self.w_disp = w_disp
        self.w_ac = w_ac
        self.vol_regime = VolatilityRegime()
        self.disp_regime = DispersionRegime()
        self.ac_regime = AutocorrelationRegime()

    def compute_regime_scores(
        self, returns: pd.DataFrame
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Compute individual and composite regime scores.

        Parameters
        ----------
        returns : pd.DataFrame
            Daily returns.

        Returns
        -------
        composite : pd.Series
            Overall momentum-vs-MR regime score in [0, 1].
        components : pd.DataFrame
            Individual regime scores (volatility, dispersion, autocorrelation).
        """
        # Volatility regime: high vol -> 1, but we want high vol -> favor MR
        # So we invert: 1 - vol_score for the composite
        vol_score = self.vol_regime.compute_regime(returns)
        vol_mom_score = 1.0 - vol_score  # High vol -> low momentum alpha

        disp_score = self.disp_regime.compute_regime(returns)
        ac_score = self.ac_regime.compute_regime(returns)

        composite = (
            self.w_vol * vol_mom_score
            + self.w_disp * disp_score
            + self.w_ac * ac_score
        )

        components = pd.DataFrame({
            "volatility": vol_score,
            "dispersion": disp_score,
            "autocorrelation": ac_score,
            "composite_mom_alpha": composite,
        })

        return composite, components

    def get_regime_label(self, composite_score: float) -> str:
        """
        Convert composite score to human-readable regime label.

        Parameters
        ----------
        composite_score : float
            Regime score in [0, 1].

        Returns
        -------
        str
            Regime label.
        """
        if composite_score > 0.65:
            return "TRENDING (Momentum)"
        elif composite_score < 0.35:
            return "MEAN-REVERTING"
        else:
            return "TRANSITIONAL"
