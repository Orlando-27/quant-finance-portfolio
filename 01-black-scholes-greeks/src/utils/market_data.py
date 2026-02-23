"""
Market Data Acquisition Module
================================

Fetches option chains and historical data via Yahoo Finance.

Author: Jose Orlando Bobadilla Fuentes | CQF
"""

import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


class MarketDataProvider:
    """
    Interface to Yahoo Finance for options analysis.

    Usage:
        >>> provider = MarketDataProvider("AAPL")
        >>> chain = provider.get_option_chain()
        >>> hist = provider.get_historical_prices(period="1y")
    """

    def __init__(self, ticker: str):
        self.ticker = ticker.upper()
        self._yf = yf.Ticker(self.ticker)

    @property
    def current_price(self) -> float:
        hist = self._yf.history(period="1d")
        if hist.empty:
            raise ValueError(f"No data for {self.ticker}")
        return float(hist["Close"].iloc[-1])

    @property
    def dividend_yield(self) -> float:
        return self._yf.info.get("dividendYield", 0.0) or 0.0

    def get_available_expirations(self) -> list:
        return list(self._yf.options)

    def get_option_chain(self, expiration=None):
        """Retrieve calls and puts DataFrames for a given expiration."""
        if expiration is None:
            exps = self.get_available_expirations()
            if not exps:
                raise ValueError(f"No options for {self.ticker}")
            expiration = exps[0]
        chain = self._yf.option_chain(expiration)
        calls, puts = chain.calls.copy(), chain.puts.copy()
        spot = self.current_price
        for df in [calls, puts]:
            df["moneyness"] = df["strike"] / spot
            df["mid_price"] = (df["bid"] + df["ask"]) / 2.0
        return calls, puts

    def get_historical_prices(self, period="1y", interval="1d"):
        """Fetch OHLCV with computed log returns and realized vol."""
        df = self._yf.history(period=period, interval=interval)
        if df.empty:
            raise ValueError(f"No historical data for {self.ticker}")
        df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
        df["realized_vol_20d"] = df["log_return"].rolling(20).std() * np.sqrt(252)
        return df

    @staticmethod
    def get_risk_free_rate() -> float:
        """3-month T-bill yield proxy for risk-free rate."""
        try:
            irx = yf.Ticker("^IRX")
            h = irx.history(period="5d")
            if not h.empty:
                return float(h["Close"].iloc[-1]) / 100.0
        except Exception:
            pass
        return 0.05
