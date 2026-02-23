"""
================================================================================
SENTIMENT FEATURE ENGINEERING & SIGNAL CONSTRUCTION
================================================================================
Transforms raw document-level sentiment scores into tradeable asset-level
signals through aggregation, smoothing, and composite construction.

Pipeline:
    Raw scores (per document) -> Daily aggregation (per ticker)
    -> EWMA smoothing -> Momentum & dispersion features
    -> Cross-sectional z-score -> Composite signal

Key features constructed:
    1. Sentiment Level    : EWMA of daily average sentiment
    2. Sentiment Momentum : Short MA minus long MA of sentiment
    3. Sentiment Dispersion: Rolling std of document scores (disagreement)
    4. News Volume        : Article count (attention proxy)
    5. Composite Signal   : Weighted z-score of all features

Author: Jose Orlando Bobadilla Fuentes, CQF | MSc AI
================================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from scipy import stats


class SentimentFeatureEngine:
    """
    Constructs tradeable sentiment features from scored news data.

    The engine aggregates document-level sentiment to daily asset-level
    signals, applies EWMA smoothing, and constructs composite signals
    suitable for systematic portfolio construction.

    Parameters
    ----------
    ewma_halflife : int
        Half-life in days for EWMA smoothing (default 5 trading days).
    momentum_fast : int
        Fast window for sentiment momentum (default 5 days).
    momentum_slow : int
        Slow window for sentiment momentum (default 21 days).
    dispersion_window : int
        Rolling window for sentiment dispersion (default 10 days).
    zscore_window : int
        Lookback for cross-sectional z-score (default 63 days = 1 quarter).
    min_articles : int
        Minimum articles per day to compute valid sentiment (else NaN).
    composite_weights : dict, optional
        Weights for composite signal components.
        Default: level=0.4, momentum=0.3, dispersion=0.2, volume=0.1.
    """

    def __init__(
        self,
        ewma_halflife: int = 5,
        momentum_fast: int = 5,
        momentum_slow: int = 21,
        dispersion_window: int = 10,
        zscore_window: int = 63,
        min_articles: int = 1,
        composite_weights: Optional[Dict[str, float]] = None,
    ):
        self.ewma_halflife = ewma_halflife
        self.momentum_fast = momentum_fast
        self.momentum_slow = momentum_slow
        self.dispersion_window = dispersion_window
        self.zscore_window = zscore_window
        self.min_articles = min_articles
        self.composite_weights = composite_weights or {
            "level": 0.4,
            "momentum": 0.3,
            "dispersion": 0.2,
            "volume": 0.1,
        }

    def aggregate_daily(
        self,
        scored_news: pd.DataFrame,
        score_column: str = "ensemble_score",
        date_column: str = "date",
        ticker_column: str = "ticker",
    ) -> pd.DataFrame:
        """
        Aggregate document-level scores to daily asset-level metrics.

        For each (date, ticker), computes:
            - mean score (average sentiment)
            - median score (robust central tendency)
            - std score (intraday disagreement)
            - count (number of articles)
            - max/min score (extreme sentiment)

        Parameters
        ----------
        scored_news : pd.DataFrame
            Must contain score_column, date_column, ticker_column.
        score_column : str
            Column with sentiment scores.

        Returns
        -------
        pd.DataFrame indexed by (date, ticker) with aggregated features.
        """
        # Ensure date column is datetime date (not datetime)
        df = scored_news.copy()
        if date_column not in df.columns and "datetime" in df.columns:
            df[date_column] = pd.to_datetime(df["datetime"]).dt.date

        agg = df.groupby([date_column, ticker_column])[score_column].agg(
            sent_mean="mean",
            sent_median="median",
            sent_std="std",
            sent_min="min",
            sent_max="max",
            n_articles="count",
        ).reset_index()

        # Fill NaN std (single-article days) with 0
        agg["sent_std"] = agg["sent_std"].fillna(0.0)

        # Mark days with insufficient coverage
        agg.loc[agg["n_articles"] < self.min_articles, "sent_mean"] = np.nan

        return agg

    def compute_features(
        self,
        daily_sentiment: pd.DataFrame,
        tickers: Optional[list] = None,
    ) -> pd.DataFrame:
        """
        Compute full feature set for each ticker from daily aggregates.

        Features per ticker:
            1. sent_ewma      : EWMA-smoothed sentiment level
            2. sent_momentum   : Fast MA - Slow MA of sentiment
            3. sent_dispersion : Rolling std of daily sentiment (disagreement)
            4. news_volume     : EWMA-smoothed article count
            5. sent_zscore     : Rolling z-score of EWMA sentiment
            6. volume_zscore   : Rolling z-score of news volume

        Parameters
        ----------
        daily_sentiment : pd.DataFrame
            Output of aggregate_daily().
        tickers : list, optional
            Subset of tickers to process. None = all.

        Returns
        -------
        pd.DataFrame with features for each (date, ticker).
        """
        if tickers is None:
            tickers = daily_sentiment["ticker"].unique().tolist()

        frames = []

        for ticker in tickers:
            tk = daily_sentiment[daily_sentiment["ticker"] == ticker].copy()
            tk = tk.sort_values("date")
            tk = tk.set_index("date")

            # 1. EWMA-smoothed sentiment level
            tk["sent_ewma"] = (
                tk["sent_mean"]
                .ewm(halflife=self.ewma_halflife, min_periods=1)
                .mean()
            )

            # 2. Sentiment momentum: fast MA - slow MA
            fast = tk["sent_mean"].rolling(self.momentum_fast, min_periods=1).mean()
            slow = tk["sent_mean"].rolling(self.momentum_slow, min_periods=1).mean()
            tk["sent_momentum"] = fast - slow

            # 3. Sentiment dispersion (disagreement proxy)
            #    Rolling std of daily mean sentiment captures day-to-day swings
            tk["sent_dispersion"] = (
                tk["sent_mean"]
                .rolling(self.dispersion_window, min_periods=3)
                .std()
            )

            # 4. News volume (EWMA-smoothed article count)
            tk["news_volume"] = (
                tk["n_articles"]
                .ewm(halflife=self.ewma_halflife, min_periods=1)
                .mean()
            )

            # 5. Rolling z-score of sentiment level
            roll_mean = tk["sent_ewma"].rolling(self.zscore_window, min_periods=10).mean()
            roll_std = tk["sent_ewma"].rolling(self.zscore_window, min_periods=10).std()
            tk["sent_zscore"] = (tk["sent_ewma"] - roll_mean) / (roll_std + 1e-8)

            # 6. Rolling z-score of news volume
            vol_mean = tk["news_volume"].rolling(self.zscore_window, min_periods=10).mean()
            vol_std = tk["news_volume"].rolling(self.zscore_window, min_periods=10).std()
            tk["volume_zscore"] = (tk["news_volume"] - vol_mean) / (vol_std + 1e-8)

            tk["ticker"] = ticker
            frames.append(tk.reset_index())

        return pd.concat(frames, ignore_index=True)

    def build_composite_signal(
        self,
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Construct the composite sentiment signal from individual features.

        The composite signal is a weighted sum of z-scored components:

            C_t = w_level * z(EWMA) + w_momentum * z(momentum)
                  + w_dispersion * (-z(dispersion)) + w_volume * z(volume)

        Note: Dispersion enters with a NEGATIVE sign because high disagreement
        typically precedes negative returns (Antweiler & Frank, 2004).

        Parameters
        ----------
        features : pd.DataFrame
            Output of compute_features().

        Returns
        -------
        pd.DataFrame with added 'composite_signal' column.
        """
        df = features.copy()
        w = self.composite_weights

        # Normalize momentum and dispersion to z-scores within each ticker
        for col in ["sent_momentum", "sent_dispersion"]:
            roll_m = df.groupby("ticker")[col].transform(
                lambda x: x.rolling(self.zscore_window, min_periods=10).mean()
            )
            roll_s = df.groupby("ticker")[col].transform(
                lambda x: x.rolling(self.zscore_window, min_periods=10).std()
            )
            df[f"{col}_z"] = (df[col] - roll_m) / (roll_s + 1e-8)

        # Composite signal (dispersion enters negatively)
        df["composite_signal"] = (
            w["level"] * df["sent_zscore"]
            + w["momentum"] * df["sent_momentum_z"]
            - w["dispersion"] * df["sent_dispersion_z"]
            + w["volume"] * df["volume_zscore"]
        )

        # Winsorize extreme values at +/- 3 sigma
        df["composite_signal"] = df["composite_signal"].clip(-3, 3)

        return df

    def build_cross_sectional_signal(
        self,
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute cross-sectional rank signal across all tickers on each date.

        Rank_t^i = rank(composite_t^i) / N_t

        This transforms the signal into a uniform [0, 1] distribution
        on each cross-section, making it robust to time-varying sentiment
        levels across different market regimes.

        Parameters
        ----------
        features : pd.DataFrame
            Must contain 'composite_signal', 'date', 'ticker'.

        Returns
        -------
        pd.DataFrame with added 'cs_rank' column in [0, 1].
        """
        df = features.copy()
        df["cs_rank"] = df.groupby("date")["composite_signal"].rank(pct=True)
        return df
