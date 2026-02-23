"""
Order Flow Analysis
====================
Trade classification and order flow imbalance metrics.

Implements:
    - Tick rule & Lee-Ready (1991) trade classification
    - Order Flow Imbalance (OFI)
    - VPIN (Volume-Synchronized PIN, Easley et al. 2012)
    - Autocorrelation analysis of order flow

References:
    Lee, C.M.C. & Ready, M.J. (1991). JF 46(2), 733-764.
    Easley, D., Lopez de Prado, M. & O'Hara, M. (2012). RFS 25(5), 1457-1493.
    Cont, R., Kukanov, A. & Stoikov, S. (2014). JF 69(4), 1457-1498.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


class OrderFlowAnalyzer:
    """Order flow imbalance and toxicity metrics."""

    # ------------------------------------------------------------------
    # Trade Classification – Tick Rule
    # ------------------------------------------------------------------
    @staticmethod
    def tick_rule(prices: pd.Series) -> pd.Series:
        """
        Lee-Ready (1991) tick rule for trade direction classification.

        Rules:
            Uptick   (+1): p_t > p_{t-1}  → buyer-initiated
            Downtick (-1): p_t < p_{t-1}  → seller-initiated
            Zero-tick     : direction of last non-zero tick

        Parameters
        ----------
        prices : pd.Series
            Transaction prices (tick or OHLC close).

        Returns
        -------
        pd.Series
            Trade direction: +1 (buy), -1 (sell).
        """
        dp        = prices.diff()
        direction = np.sign(dp)
        direction = direction.replace(0, np.nan).ffill().fillna(1)
        direction.name = "trade_direction"
        return direction.astype(float)

    # ------------------------------------------------------------------
    # Order Flow Imbalance (OFI)
    # ------------------------------------------------------------------
    @staticmethod
    def order_flow_imbalance(
        prices  : pd.Series,
        volume  : pd.Series,
        window  : int = 20,
    ) -> pd.Series:
        """
        Compute the Order Flow Imbalance (OFI) of Cont et al. (2014).

        OFI measures the net signed volume over a rolling window,
        normalized by total volume:

            OFI_t = Σ_{τ} d_τ * v_τ / Σ_{τ} v_τ

        where d_τ ∈ {-1, +1} is trade direction and v_τ is trade size.

        A value near +1 indicates predominant buying pressure;
        near -1 indicates selling pressure.

        Returns
        -------
        pd.Series
            Rolling OFI in [-1, +1].
        """
        direction   = OrderFlowAnalyzer.tick_rule(prices)
        signed_vol  = direction * volume
        ofi = signed_vol.rolling(window).sum() / volume.rolling(window).sum()
        ofi.name = "order_flow_imbalance"
        return ofi

    # ------------------------------------------------------------------
    # VPIN (Volume-Synchronized PIN)
    # ------------------------------------------------------------------
    @staticmethod
    def vpin(
        prices    : pd.Series,
        volume    : pd.Series,
        bucket_vol: float | None = None,
        n_buckets : int   = 50,
    ) -> pd.Series:
        """
        Compute VPIN (Volume-Synchronized PIN) of Easley, Lopez de Prado
        & O'Hara (2012).

        VPIN approximates the probability of informed trading by measuring
        the imbalance between buy and sell volumes in equal-volume buckets:

            VPIN = (1/n) * Σ |V_b^τ - V_s^τ| / V_bucket

        Algorithm (bulk-volume classification):
            1. Divide volume into equal-volume buckets.
            2. Classify each bucket's volume into buy/sell using the
               standardized price change (Z-score proxy for P(buy)).
            3. Compute |V_b - V_s| / V_bucket per bucket.
            4. VPIN = rolling mean over n_buckets.

        Parameters
        ----------
        prices     : pd.Series
        volume     : pd.Series
        bucket_vol : float | None
            Target volume per bucket. If None, uses total / (n_buckets * 5).
        n_buckets  : int
            Number of buckets in rolling average.

        Returns
        -------
        pd.Series
            VPIN index aligned to original price series index.
        """
        if bucket_vol is None:
            bucket_vol = volume.sum() / (n_buckets * 5)

        ret        = prices.pct_change().fillna(0)
        sigma_ret  = ret.expanding().std().replace(0, ret.std())

        # Bulk volume classification: V_b ≈ V * Φ(Δp / σ)
        z_score  = ret / sigma_ret.replace(0, 1e-10)
        prob_buy = norm.cdf(z_score)
        v_buy    = volume * prob_buy
        v_sell   = volume * (1.0 - prob_buy)

        # Accumulate into fixed-volume buckets
        records   = []
        cum_vol   = 0.0
        cum_buy   = 0.0
        cum_sell  = 0.0

        for i, (vol_i, vb_i, vs_i) in enumerate(
            zip(volume.values, v_buy.values, v_sell.values)
        ):
            cum_vol  += vol_i
            cum_buy  += vb_i
            cum_sell += vs_i

            while cum_vol >= bucket_vol:
                imbalance = abs(cum_buy - cum_sell) / bucket_vol
                records.append((prices.index[i], imbalance))
                cum_vol  -= bucket_vol
                ratio     = max(0.0, min(1.0, cum_vol / max(vol_i, 1e-10)))
                cum_buy   = v_buy.values[i]  * ratio
                cum_sell  = v_sell.values[i] * ratio

        if not records:
            return pd.Series(np.nan, index=prices.index, name="vpin")

        bucket_df = pd.DataFrame(records, columns=["date", "imbalance"])
        bucket_df = bucket_df.set_index("date")
        vpin_vals = bucket_df["imbalance"].rolling(n_buckets).mean()

        # Reindex to original time series (forward-fill within gaps)
        vpin_vals = vpin_vals[~vpin_vals.index.duplicated(keep="last")]
        vpin_series = vpin_vals.reindex(prices.index, method="ffill")
        vpin_series.name = "vpin"
        return vpin_series

    # ------------------------------------------------------------------
    # OFI Autocorrelation
    # ------------------------------------------------------------------
    @staticmethod
    def ofi_autocorrelation(ofi: pd.Series, max_lag: int = 20) -> pd.Series:
        """
        Compute autocorrelation of OFI at lags 1..max_lag.

        Persistent positive autocorrelation indicates order-splitting
        (iceberg orders); negative autocorrelation indicates mean-reversion
        in order flow (contrarian market-makers).

        Returns
        -------
        pd.Series
            ACF values indexed by lag.
        """
        ofi_clean = ofi.dropna()
        acf = pd.Series(
            [ofi_clean.autocorr(lag=lag) for lag in range(1, max_lag + 1)],
            index=range(1, max_lag + 1),
            name="ofi_acf",
        )
        return acf
