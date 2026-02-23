"""
Bid-Ask Spread Estimation Models
=================================
Implements classic and modern spread estimators from microstructure theory.

Models:
    - Roll (1984): serial covariance implicit spread
    - Corwin & Schultz (2012): high-low range estimator
    - Realized/effective spread decomposition
    - Quoted spread from OHLC proxy

References:
    Roll, R. (1984). JF 39(4), 1127-1139.
    Corwin, S.A. & Schultz, P. (2012). JF 67(2), 719-759.
    Glosten, L.R. & Harris, L.E. (1988). JFE 21(1), 123-142.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


class SpreadModels:
    """Collection of bid-ask spread estimation methods."""

    # ------------------------------------------------------------------
    # Roll (1984) Implicit Spread
    # ------------------------------------------------------------------
    @staticmethod
    def roll_spread(prices: pd.Series, window: int = 60) -> pd.Series:
        """
        Compute the Roll (1984) implicit effective spread.

        Assumes a simple dealer model where the effective spread is:
            s = 2 * sqrt(-Cov(Δp_t, Δp_{t-1}))

        When the serial covariance is positive (momentum), the estimator
        is set to NaN (non-identified).

        Parameters
        ----------
        prices : pd.Series
            Daily closing prices.
        window : int
            Rolling window in trading days.

        Returns
        -------
        pd.Series
            Roll spread estimates (in price units, annualized by convention
            as fraction of mid-price).
        """
        dp = prices.diff()

        def _roll(x: np.ndarray) -> float:
            # Require sufficient non-NaN observations
            x = x[~np.isnan(x)]
            if len(x) < 10:
                return np.nan
            cov = np.cov(x[1:], x[:-1])[0, 1]
            if cov >= 0:
                return np.nan          # not identified (momentum regime)
            return 2.0 * np.sqrt(-cov)

        spread = dp.rolling(window).apply(_roll, raw=True)
        # Express as % of mid-price
        spread_pct = spread / prices.rolling(window).mean()
        spread_pct.name = "roll_spread_pct"
        return spread_pct

    # ------------------------------------------------------------------
    # Corwin-Schultz (2012) High-Low Estimator
    # ------------------------------------------------------------------
    @staticmethod
    def corwin_schultz_spread(
        high: pd.Series,
        low: pd.Series,
    ) -> pd.Series:
        """
        Corwin & Schultz (2012) high-low bid-ask spread estimator.

        Uses the ratio of the two-day high-low range to the one-day range.
        Intuition: a wider two-day range relative to one-day indicates larger
        bid-ask bounce.

            β = [ln(H_t/L_t)]² + [ln(H_{t-1}/L_{t-1})]²
            γ = [ln(max(H_t,H_{t-1}) / min(L_t,L_{t-1}))]²
            α = (√(2β) - √β) / (3 - 2√2) - √(γ / (3 - 2√2))
            S  = 2*(exp(α) - 1) / (1 + exp(α))

        Parameters
        ----------
        high, low : pd.Series
            Daily high and low prices.

        Returns
        -------
        pd.Series
            Corwin-Schultz spread (proportion of mid-price). Negative
            estimates (noise-floor artefacts) are set to zero.
        """
        ln_hl  = np.log(high / low)
        beta   = ln_hl**2 + ln_hl.shift(1)**2

        h2 = pd.concat([high, high.shift(1)], axis=1).max(axis=1)
        l2 = pd.concat([low,  low.shift(1)],  axis=1).min(axis=1)
        gamma  = np.log(h2 / l2) ** 2

        sqrt2m1 = np.sqrt(2) - 1
        denom   = 3.0 - 2.0 * np.sqrt(2)

        alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / denom \
              - np.sqrt(gamma / denom)
        spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
        spread = spread.clip(lower=0.0)   # floor at zero
        spread.name = "corwin_schultz_spread_pct"
        return spread

    # ------------------------------------------------------------------
    # Quoted Spread Proxy (OHLC)
    # ------------------------------------------------------------------
    @staticmethod
    def quoted_spread_ohlc(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> pd.Series:
        """
        Simple quoted spread proxy from daily OHLC data.

        Quoted spread ≈ (High - Low) / ((High + Low) / 2)

        Parameters
        ----------
        high, low, close : pd.Series
            Daily OHLC data.

        Returns
        -------
        pd.Series
            Quoted spread as fraction of mid-price.
        """
        mid  = (high + low) / 2.0
        spread = (high - low) / mid
        spread.name = "quoted_spread_pct"
        return spread

    # ------------------------------------------------------------------
    # Effective Spread Decomposition
    # ------------------------------------------------------------------
    @staticmethod
    def effective_spread_decomposition(
        prices: pd.Series,
        window: int = 60,
    ) -> pd.DataFrame:
        """
        Decompose the effective spread into:
            Effective Spread  = Price Impact + Realized Spread

        Using 5-minute reversal proxy with daily data:
            Realized Spread    = 2 * d_t * (p_{t+1} - p_t)     (positive for MM)
            Adverse Selection  = 2 * d_t * (p_{t+1} - m_t)     (information cost)

        Returns
        -------
        pd.DataFrame
            Columns: effective_spread, price_impact, realized_spread (all %).
        """
        mid  = prices
        dp   = mid.diff()
        # Classify trade direction via tick rule
        direction = np.sign(dp).replace(0, np.nan).ffill().fillna(1)

        eff_spread  = 2.0 * direction * dp / mid
        price_impact = direction * mid.diff().shift(-1) / mid
        real_spread  = eff_spread - price_impact

        result = pd.DataFrame({
            "effective_spread": eff_spread.rolling(window).mean(),
            "price_impact"    : price_impact.rolling(window).mean(),
            "realized_spread" : real_spread.rolling(window).mean(),
        })
        return result

    # ------------------------------------------------------------------
    # Comparison summary
    # ------------------------------------------------------------------
    @staticmethod
    def spread_comparison(
        prices: pd.Series,
        high: pd.Series,
        low: pd.Series,
    ) -> pd.DataFrame:
        """Return all spread estimates in a single DataFrame."""
        roll = SpreadModels.roll_spread(prices)
        cs   = SpreadModels.corwin_schultz_spread(high, low)
        quot = SpreadModels.quoted_spread_ohlc(high, low, prices)
        return pd.concat([roll, cs, quot], axis=1)
