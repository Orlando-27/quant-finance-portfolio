"""
Data Acquisition & Synthetic Tick Generator
=============================================
Provides:
    - MarketDataLoader: Download OHLCV from yfinance
    - SyntheticTickGenerator: Generate realistic intraday tick data
      via a modified compound Poisson process with diurnal volume pattern.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import time as dtime


class MarketDataLoader:
    """Download and prepare market data for microstructure analysis."""

    def __init__(self, tickers: list[str], start: str, end: str):
        self.tickers = tickers
        self.start   = start
        self.end     = end
        self._cache: dict[str, pd.DataFrame] = {}

    def fetch(self, ticker: str) -> pd.DataFrame:
        """Download daily OHLCV data for a single ticker."""
        if ticker in self._cache:
            return self._cache[ticker]

        data = yf.download(
            ticker,
            start     = self.start,
            end       = self.end,
            auto_adjust=True,
            progress  = False,
        )
        if data.empty:
            raise ValueError(f"No data returned for {ticker}")

        data.columns = [c[0] if isinstance(c, tuple) else c for c in data.columns]
        data.index   = pd.to_datetime(data.index)
        data["Return"]       = data["Close"].pct_change()
        data["DollarVolume"] = data["Close"] * data["Volume"]
        data.dropna(subset=["Return"], inplace=True)
        self._cache[ticker] = data
        return data

    def fetch_all(self) -> dict[str, pd.DataFrame]:
        """Download all tickers and return as dict."""
        return {t: self.fetch(t) for t in self.tickers}


class SyntheticTickGenerator:
    """
    Generate synthetic intraday tick data with realistic microstructure.

    Model:
        - Prices follow a geometric Brownian motion with diurnal volatility
        - Volume follows a U-shaped intraday pattern (high open/close)
        - Spreads widen at open/close and during high-volatility periods
        - Arrival process: non-homogeneous Poisson with λ(t) ∝ diurnal_vol(t)
    """

    def __init__(
        self,
        S0          : float = 100.0,
        daily_vol   : float = 0.015,
        daily_vol_bps: float = 5.0,  # half-spread in bps (average)
        n_ticks     : int   = 5_000,
        seed        : int   = 42,
    ):
        self.S0          = S0
        self.daily_vol   = daily_vol
        self.half_spread = S0 * daily_vol_bps / 1e4
        self.n_ticks     = n_ticks
        self.rng         = np.random.default_rng(seed)

    def _diurnal_lambda(self, t: np.ndarray) -> np.ndarray:
        """
        U-shaped intraday intensity: higher at open (t≈0) and close (t≈1).
        t ∈ [0, 1] represents the fraction of the trading day.
        """
        return 1.5 + 1.5 * (np.cos(2.0 * np.pi * t) ** 2)

    def generate(self) -> pd.DataFrame:
        """
        Generate a synthetic trading day of tick data.

        Returns
        -------
        pd.DataFrame
            Columns: timestamp, price, bid, ask, volume, direction.
        """
        t_uniform = np.linspace(0, 1, self.n_ticks)
        intensity  = self._diurnal_lambda(t_uniform)
        intensity  = intensity / intensity.mean()      # normalise

        # Diurnal volatility (higher at open/close)
        diurnal_vol = self.daily_vol * np.sqrt(intensity) / np.sqrt(self.n_ticks)

        # Geometric Brownian Motion price path
        shocks   = self.rng.normal(0, 1, self.n_ticks)
        log_ret  = diurnal_vol * shocks
        prices   = self.S0 * np.exp(np.cumsum(log_ret))

        # Diurnal spread (wider at open/close)
        spread_factor = 0.5 + 0.5 * np.abs(np.cos(2.0 * np.pi * t_uniform))
        half_spread   = self.half_spread * spread_factor

        bid = prices - half_spread
        ask = prices + half_spread

        # Diurnal volume pattern
        base_vol = 100.0 + 200.0 * (np.cos(2.0 * np.pi * t_uniform) ** 2)
        volume   = np.maximum(1, self.rng.poisson(base_vol))

        # Trade direction (correlated with price movement)
        direction = np.where(shocks > 0, 1, -1).astype(float)

        # Build timestamps for a single trading day (9:30–16:00)
        trading_seconds = int(6.5 * 3600)
        tick_times = np.linspace(0, trading_seconds - 1, self.n_ticks).astype(int)
        open_time  = pd.Timestamp("2024-01-15 09:30:00")
        timestamps = [open_time + pd.Timedelta(seconds=int(s)) for s in tick_times]

        return pd.DataFrame({
            "timestamp": timestamps,
            "price"    : prices,
            "bid"      : bid,
            "ask"      : ask,
            "volume"   : volume,
            "direction": direction,
        }).set_index("timestamp")

    def aggregate_to_bars(
        self,
        ticks    : pd.DataFrame,
        freq     : str = "5T",
    ) -> pd.DataFrame:
        """
        Aggregate tick data to OHLCV bars.

        Parameters
        ----------
        ticks : pd.DataFrame   Tick data from generate().
        freq  : str            Pandas offset alias (e.g. '1T', '5T', '15T').
        """
        bars = ticks["price"].resample(freq).ohlc()
        bars["volume"]       = ticks["volume"].resample(freq).sum()
        bars["signed_volume"] = (ticks["direction"] * ticks["volume"]).resample(freq).sum()
        bars["vwap"]         = (
            (ticks["price"] * ticks["volume"]).resample(freq).sum()
            / bars["volume"].replace(0, np.nan)
        )
        bars.dropna(inplace=True)
        return bars
