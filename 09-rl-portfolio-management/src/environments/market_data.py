"""
================================================================================
MARKET DATA LOADER AND FEATURE ENGINEERING
================================================================================
Downloads historical price data and constructs the feature matrix used
as the observation space for the RL environment.

Feature categories:
    1. Returns        : 1d, 5d, 21d log returns per asset
    2. Volatility     : 5d, 21d, 63d realized volatility per asset
    3. Momentum       : RSI(14), MACD signal per asset
    4. Correlations   : 21d rolling pairwise correlations (upper triangle)
    5. Market regime  : Cross-sectional dispersion, avg correlation

All features are computed with NO lookahead bias -- each observation at
time t uses only data available up to and including time t.

Author: Jose Orlando Bobadilla Fuentes, CQF | MSc AI
================================================================================
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import List, Optional, Tuple, Dict


class MarketDataLoader:
    """
    Downloads and preprocesses market data for the RL environment.

    Parameters
    ----------
    tickers : list of str
        Asset universe (e.g., ['SPY', 'TLT', 'GLD', 'QQQ']).
    start_date : str
        Data start date ('YYYY-MM-DD'). Should include lookback buffer.
    end_date : str
        Data end date.
    risk_free_ticker : str
        Proxy for risk-free rate (default: 'SHV' -- short-term treasury ETF).
    """

    def __init__(
        self,
        tickers: List[str],
        start_date: str = "2010-01-01",
        end_date: str = "2023-12-31",
        risk_free_ticker: str = "SHV",
    ):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.rf_ticker = risk_free_ticker
        self.n_assets = len(tickers)

        # Data containers
        self.prices = None
        self.returns = None
        self.features = None

    def download(self) -> pd.DataFrame:
        """Download adjusted close prices from Yahoo Finance."""
        all_tickers = self.tickers + [self.rf_ticker]
        data = yf.download(
            all_tickers,
            start=self.start_date,
            end=self.end_date,
            auto_adjust=True,
        )
        if isinstance(data.columns, pd.MultiIndex):
            self.prices = data["Close"][self.tickers].dropna()
        else:
            self.prices = data[["Close"]].dropna()
            self.prices.columns = self.tickers

        self.returns = np.log(self.prices / self.prices.shift(1)).dropna()
        return self.prices

    def generate_synthetic(
        self,
        n_days: int = 2520,
        mu: Optional[np.ndarray] = None,
        sigma: Optional[np.ndarray] = None,
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        Generate synthetic multi-asset price data for testing.

        Uses correlated geometric Brownian motion with realistic parameters.

        Parameters
        ----------
        n_days : int
            Number of trading days (default 2520 = ~10 years).
        mu : np.ndarray, optional
            Annualized expected returns per asset.
        sigma : np.ndarray, optional
            Annualized volatilities per asset.
        seed : int
            Random seed.
        """
        rng = np.random.RandomState(seed)
        n = self.n_assets

        # Default parameters calibrated to realistic asset classes
        if mu is None:
            mu = np.array([0.08, 0.03, 0.05, 0.10, 0.06][:n])
        if sigma is None:
            sigma = np.array([0.16, 0.07, 0.15, 0.20, 0.12][:n])

        # Generate correlation matrix with moderate correlations
        # Use random correlation via Cholesky of a structured matrix
        base_corr = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                c = rng.uniform(0.1, 0.6)
                base_corr[i, j] = c
                base_corr[j, i] = c

        # Ensure positive definiteness
        eigvals = np.linalg.eigvalsh(base_corr)
        if eigvals.min() < 0:
            base_corr += (-eigvals.min() + 0.01) * np.eye(n)
            d = np.sqrt(np.diag(base_corr))
            base_corr = base_corr / np.outer(d, d)

        # Covariance matrix
        cov = np.outer(sigma, sigma) * base_corr / 252

        # Daily drift
        daily_mu = mu / 252 - 0.5 * sigma ** 2 / 252

        # Simulate correlated returns
        L = np.linalg.cholesky(cov)
        Z = rng.standard_normal((n_days, n))
        daily_returns = daily_mu + Z @ L.T

        # Convert to prices
        dates = pd.bdate_range("2014-01-01", periods=n_days)
        log_prices = np.cumsum(daily_returns, axis=0)
        prices = 100 * np.exp(log_prices)

        self.prices = pd.DataFrame(prices, index=dates, columns=self.tickers)
        self.returns = pd.DataFrame(daily_returns, index=dates, columns=self.tickers)
        return self.prices

    def compute_features(self) -> pd.DataFrame:
        """
        Compute the full feature matrix for the RL environment.

        Returns
        -------
        pd.DataFrame with multi-level columns (feature_type, asset/pair).
        Each row corresponds to one trading day.
        """
        if self.returns is None:
            raise ValueError("Call download() or generate_synthetic() first.")

        ret = self.returns
        frames = {}

        # --- 1. Multi-horizon returns ---
        for h in [1, 5, 21]:
            for ticker in self.tickers:
                col = f"ret_{h}d_{ticker}"
                frames[col] = ret[ticker].rolling(h).sum()

        # --- 2. Realized volatility ---
        for h in [5, 21, 63]:
            for ticker in self.tickers:
                col = f"vol_{h}d_{ticker}"
                frames[col] = ret[ticker].rolling(h).std() * np.sqrt(252)

        # --- 3. RSI(14) per asset ---
        for ticker in self.tickers:
            delta = ret[ticker]
            gain = delta.where(delta > 0, 0.0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
            rs = gain / (loss + 1e-10)
            frames[f"rsi_{ticker}"] = 100 - 100 / (1 + rs)

        # --- 4. MACD signal per asset ---
        for ticker in self.tickers:
            price = self.prices[ticker]
            ema12 = price.ewm(span=12, adjust=False).mean()
            ema26 = price.ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            # Normalize by price to make scale-invariant
            frames[f"macd_{ticker}"] = (macd_line - signal_line) / price

        # --- 5. Rolling correlations (21d, upper triangle) ---
        if self.n_assets > 1:
            for i in range(self.n_assets):
                for j in range(i + 1, self.n_assets):
                    t1, t2 = self.tickers[i], self.tickers[j]
                    corr = ret[t1].rolling(21).corr(ret[t2])
                    frames[f"corr_{t1}_{t2}"] = corr

        # --- 6. Cross-sectional features ---
        # Average pairwise correlation (regime indicator)
        if self.n_assets > 1:
            corr_cols = [c for c in frames if c.startswith("corr_")]
            corr_df = pd.DataFrame({c: frames[c] for c in corr_cols})
            frames["avg_correlation"] = corr_df.mean(axis=1)

        # Cross-sectional return dispersion
        frames["cs_dispersion"] = ret.std(axis=1)

        # Combine into single DataFrame
        self.features = pd.DataFrame(frames).dropna()
        return self.features

    def get_train_test_split(
        self,
        test_ratio: float = 0.2,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test periods (temporal, no shuffling).

        Returns
        -------
        train_returns, test_returns, train_features, test_features
        """
        if self.features is None:
            self.compute_features()

        # Align returns and features on common dates
        common_idx = self.returns.index.intersection(self.features.index)
        ret = self.returns.loc[common_idx]
        feat = self.features.loc[common_idx]

        split_idx = int(len(common_idx) * (1 - test_ratio))

        return (
            ret.iloc[:split_idx],
            ret.iloc[split_idx:],
            feat.iloc[:split_idx],
            feat.iloc[split_idx:],
        )

    @property
    def feature_dim(self) -> int:
        """Total number of market features."""
        if self.features is None:
            return 0
        return self.features.shape[1]
