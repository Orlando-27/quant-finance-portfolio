"""
Feature Engineering for Financial Time Series
================================================

Computes technical indicators and prepares windowed datasets for
sequence models (LSTM, GRU, Transformer).

Author: Jose Orlando Bobadilla Fuentes | CQF
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple


class FeatureEngineer:
    """
    Constructs feature matrix from OHLCV data with technical indicators.

    Features:
        - Log returns, realized volatility
        - SMA, EMA (multiple windows)
        - RSI, MACD, Bollinger Bands
        - Volume-weighted metrics

    Usage:
        >>> fe = FeatureEngineer(lookback=60)
        >>> X_train, y_train, X_test, y_test, scaler = fe.prepare(df, split=0.8)
    """

    def __init__(self, lookback: int = 60):
        self.lookback = lookback
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_names = []

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to OHLCV DataFrame."""
        feat = pd.DataFrame(index=df.index)

        # Returns and volatility
        feat["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
        feat["vol_10"] = feat["log_return"].rolling(10).std()
        feat["vol_20"] = feat["log_return"].rolling(20).std()

        # Moving averages
        for w in [5, 10, 20, 50]:
            feat[f"sma_{w}"] = df["Close"].rolling(w).mean() / df["Close"] - 1
            feat[f"ema_{w}"] = df["Close"].ewm(span=w).mean() / df["Close"] - 1

        # RSI
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        feat["rsi"] = 100 - 100 / (1 + rs)

        # MACD
        ema12 = df["Close"].ewm(span=12).mean()
        ema26 = df["Close"].ewm(span=26).mean()
        feat["macd"] = (ema12 - ema26) / df["Close"]
        feat["macd_signal"] = feat["macd"].ewm(span=9).mean()

        # Bollinger Bands width
        sma20 = df["Close"].rolling(20).mean()
        std20 = df["Close"].rolling(20).std()
        feat["bb_width"] = (2 * std20) / sma20

        # Volume features
        if "Volume" in df.columns:
            feat["vol_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()

        feat["target"] = feat["log_return"].shift(-1)  # Next-day return
        self.feature_names = [c for c in feat.columns if c != "target"]

        return feat.dropna()

    def create_sequences(self, data: np.ndarray, target: np.ndarray
                         ) -> Tuple[np.ndarray, np.ndarray]:
        """Create lookback windows for sequence models."""
        X, y = [], []
        for i in range(self.lookback, len(data)):
            X.append(data[i - self.lookback:i])
            y.append(target[i])
        return np.array(X), np.array(y)

    def prepare(self, df: pd.DataFrame, train_split: float = 0.8):
        """
        Full pipeline: features -> scale -> sequence -> train/test split.
        Walk-forward: train on first train_split fraction, test on rest.
        """
        feat = self.compute_features(df)
        features = feat[self.feature_names].values
        target = feat["target"].values

        split_idx = int(len(features) * train_split)

        # Fit scaler on training data only (no look-ahead)
        self.scaler.fit(features[:split_idx])
        scaled = self.scaler.transform(features)

        X, y = self.create_sequences(scaled, target)

        # Adjust split for lookback offset
        split_adj = split_idx - self.lookback
        X_train, y_train = X[:split_adj], y[:split_adj]
        X_test, y_test = X[split_adj:], y[split_adj:]

        return X_train, y_train, X_test, y_test
