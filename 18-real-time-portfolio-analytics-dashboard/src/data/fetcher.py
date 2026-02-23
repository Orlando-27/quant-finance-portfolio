# =============================================================================
# src/data/fetcher.py | Project 18 | Jose Orlando Bobadilla Fuentes | CQF
# Market data acquisition via yfinance with caching and error handling
# =============================================================================
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")
import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Optional

def fetch_prices(tickers: List[str], period: str = "1y") -> pd.DataFrame:
    """Download adjusted close prices. Returns DataFrame indexed by date."""
    try:
        raw = yf.download(tickers, period=period, auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"]
        else:
            prices = raw[["Close"]] if "Close" in raw.columns else raw
        prices = prices.dropna(how="all").ffill().bfill()
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=tickers[0])
        return prices
    except Exception as e:
        raise RuntimeError(f"Failed to fetch prices: {e}")

def fetch_returns(tickers: List[str], period: str = "1y") -> pd.DataFrame:
    """Log returns from adjusted close prices."""
    prices = fetch_prices(tickers, period)
    return np.log(prices / prices.shift(1)).dropna()

def fetch_benchmark(ticker: str = "SPY", period: str = "1y") -> pd.Series:
    """Fetch benchmark returns."""
    rets = fetch_returns([ticker], period)
    return rets.iloc[:, 0].rename(ticker)

def fetch_info(ticker: str) -> dict:
    """Basic info for a single ticker."""
    try:
        info = yf.Ticker(ticker).info
        return {
            "name":    info.get("longName", ticker),
            "sector":  info.get("sector", "Unknown"),
            "market_cap": info.get("marketCap", 0),
            "currency": info.get("currency", "USD"),
        }
    except Exception:
        return {"name": ticker, "sector": "Unknown", "market_cap": 0, "currency": "USD"}
