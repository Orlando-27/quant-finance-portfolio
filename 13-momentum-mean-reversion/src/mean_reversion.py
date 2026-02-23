"""
Mean Reversion Signal Generators
==================================

Implements three complementary mean-reversion signals:

1. Z-Score: distance from rolling mean in standard deviations
   (Ornstein-Uhlenbeck inspired)
2. RSI (Relative Strength Index): Wilder's momentum oscillator
   identifying overbought/oversold conditions
3. Bollinger Band %B: position within Bollinger Bands

Each signal produces a standardized output in [-1, +1] where:
    -1 = maximum oversold (buy signal for mean reversion)
    +1 = maximum overbought (sell signal for mean reversion)

For portfolio construction, the mean-reversion signal is inverted:
    positive composite = expect price increase (oversold -> go long)
    negative composite = expect price decrease (overbought -> go short)

Author: Jose Orlando Bobadilla Fuentes, CQF, MSc AI
"""

import numpy as np
import pandas as pd


class ZScoreSignal:
    """
    Z-Score mean reversion signal.

    Computes the standardized distance of price from its rolling mean:
        z = (P - MA) / rolling_std

    Trading logic:
        |z| > entry_threshold -> enter mean-reversion trade
        |z| < exit_threshold  -> exit trade

    Parameters
    ----------
    lookback : int
        Rolling window for mean and std estimation (trading days).
    entry_z : float
        Z-score threshold for entry (absolute value).
    exit_z : float
        Z-score threshold for exit (absolute value).
    """

    def __init__(self, lookback: int = 60, entry_z: float = 2.0, exit_z: float = 0.5):
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z

    def compute_zscore(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Compute rolling z-score for each asset.

        Parameters
        ----------
        prices : pd.DataFrame
            Daily price levels.

        Returns
        -------
        pd.DataFrame
            Z-scores at each date for each asset.
        """
        ma = prices.rolling(window=self.lookback).mean()
        std = prices.rolling(window=self.lookback).std()
        z = (prices - ma) / std.replace(0, np.nan)
        return z

    def compute_signal(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Compute mean-reversion signal from z-scores.

        Signal is clipped to [-1, +1] and inverted for portfolio use:
            z > entry -> signal = -1 (short / expect reversion down)
            z < -entry -> signal = +1 (long / expect reversion up)
            |z| < exit -> signal = 0 (no position)

        For intermediate values, the signal is linearly interpolated.

        Parameters
        ----------
        prices : pd.DataFrame
            Daily prices.

        Returns
        -------
        pd.DataFrame
            Mean-reversion signal in [-1, +1].
        """
        z = self.compute_zscore(prices)

        # Continuous signal: negative z-score -> positive signal (buy)
        # Normalize by entry threshold and clip
        signal = -z / self.entry_z
        signal = signal.clip(-1.0, 1.0)

        # Zero out signals within exit band (dead zone)
        mask_deadzone = z.abs() < self.exit_z
        signal[mask_deadzone] = 0.0

        return signal


class RSISignal:
    """
    Relative Strength Index (RSI) mean-reversion signal.

    RSI = 100 - 100 / (1 + RS)
    where RS = EWMA(gains) / EWMA(losses) over N periods.

    Oversold: RSI < oversold_level -> expect bounce (long)
    Overbought: RSI > overbought_level -> expect pullback (short)

    Parameters
    ----------
    period : int
        RSI calculation period (default 14).
    oversold : float
        Oversold threshold (default 30).
    overbought : float
        Overbought threshold (default 70).
    """

    def __init__(self, period: int = 14, oversold: float = 30.0, overbought: float = 70.0):
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def compute_rsi(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Compute RSI for each asset using Wilder's smoothing method.

        Parameters
        ----------
        prices : pd.DataFrame
            Daily prices.

        Returns
        -------
        pd.DataFrame
            RSI values in [0, 100].
        """
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)

        # Wilder's smoothing (equivalent to EMA with alpha=1/period)
        avg_gain = gain.ewm(alpha=1.0 / self.period, min_periods=self.period).mean()
        avg_loss = loss.ewm(alpha=1.0 / self.period, min_periods=self.period).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100.0 - 100.0 / (1.0 + rs)
        return rsi

    def compute_signal(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Compute mean-reversion signal from RSI.

        Maps RSI to [-1, +1]:
            RSI < oversold  -> positive signal (long, expect bounce)
            RSI > overbought -> negative signal (short, expect pullback)
            Between thresholds -> linearly interpolated through zero

        Parameters
        ----------
        prices : pd.DataFrame
            Daily prices.

        Returns
        -------
        pd.DataFrame
            Mean-reversion signal.
        """
        rsi = self.compute_rsi(prices)
        midpoint = (self.overbought + self.oversold) / 2.0
        half_range = (self.overbought - self.oversold) / 2.0

        # Map: RSI=oversold -> +1, RSI=overbought -> -1
        signal = -(rsi - midpoint) / half_range
        signal = signal.clip(-1.0, 1.0)
        return signal


class BollingerBandSignal:
    """
    Bollinger Band %B mean-reversion signal.

    %B = (Price - Lower Band) / (Upper Band - Lower Band)

    Where:
        Middle Band = SMA(lookback)
        Upper Band = Middle + k * std(lookback)
        Lower Band = Middle - k * std(lookback)

    %B near 0 -> price near lower band (oversold)
    %B near 1 -> price near upper band (overbought)

    Parameters
    ----------
    lookback : int
        Rolling window for SMA and std (default 20).
    num_std : float
        Number of standard deviations for bands (default 2.0).
    """

    def __init__(self, lookback: int = 20, num_std: float = 2.0):
        self.lookback = lookback
        self.num_std = num_std

    def compute_bands(self, prices: pd.DataFrame) -> dict:
        """
        Compute Bollinger Bands.

        Parameters
        ----------
        prices : pd.DataFrame
            Daily prices.

        Returns
        -------
        dict
            Keys: 'middle', 'upper', 'lower', 'pct_b', 'bandwidth'.
        """
        middle = prices.rolling(window=self.lookback).mean()
        std = prices.rolling(window=self.lookback).std()

        upper = middle + self.num_std * std
        lower = middle - self.num_std * std

        band_width = upper - lower
        pct_b = (prices - lower) / band_width.replace(0, np.nan)

        bandwidth_pct = band_width / middle.replace(0, np.nan)

        return {
            "middle": middle,
            "upper": upper,
            "lower": lower,
            "pct_b": pct_b,
            "bandwidth": bandwidth_pct,
        }

    def compute_signal(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Compute mean-reversion signal from Bollinger %B.

        Maps %B to [-1, +1]:
            %B near 0 -> +1 (oversold, buy)
            %B near 1 -> -1 (overbought, sell)

        Parameters
        ----------
        prices : pd.DataFrame
            Daily prices.

        Returns
        -------
        pd.DataFrame
            Mean-reversion signal.
        """
        bands = self.compute_bands(prices)
        pct_b = bands["pct_b"]

        # Map: %B=0 -> +1 (buy), %B=1 -> -1 (sell), %B=0.5 -> 0
        signal = 1.0 - 2.0 * pct_b
        signal = signal.clip(-1.0, 1.0)
        return signal


class CompositeMeanReversion:
    """
    Composite mean-reversion signal combining Z-Score, RSI, and Bollinger.

    The composite is a weighted average of the three individual signals,
    providing a more robust mean-reversion indicator than any single metric.

    Parameters
    ----------
    w_zscore : float
        Weight for z-score signal (default 0.4).
    w_rsi : float
        Weight for RSI signal (default 0.3).
    w_bollinger : float
        Weight for Bollinger signal (default 0.3).
    """

    def __init__(
        self,
        w_zscore: float = 0.4,
        w_rsi: float = 0.3,
        w_bollinger: float = 0.3,
    ):
        self.w_zscore = w_zscore
        self.w_rsi = w_rsi
        self.w_bollinger = w_bollinger
        self.zscore_gen = ZScoreSignal()
        self.rsi_gen = RSISignal()
        self.bb_gen = BollingerBandSignal()

    def compute_signal(
        self, prices: pd.DataFrame, returns: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute composite mean-reversion signal.

        Parameters
        ----------
        prices : pd.DataFrame
            Daily prices.
        returns : pd.DataFrame
            Daily returns (unused here but kept for interface consistency).

        Returns
        -------
        pd.DataFrame
            Composite mean-reversion signal in [-1, +1].
        """
        s_z = self.zscore_gen.compute_signal(prices)
        s_rsi = self.rsi_gen.compute_signal(prices)
        s_bb = self.bb_gen.compute_signal(prices)

        composite = (
            self.w_zscore * s_z + self.w_rsi * s_rsi + self.w_bollinger * s_bb
        )
        return composite.clip(-1.0, 1.0)
