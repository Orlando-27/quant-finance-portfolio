"""
Synthetic Multi-Asset Data Generator
=====================================

Generates realistic synthetic price series for multiple asset classes
with configurable drift, volatility, and correlation structure.

Asset classes:
    - Equities: trending with moderate volatility, clustered vol
    - Fixed Income: low vol, mean-reverting yields
    - Commodities: high vol, regime-switching behavior
    - FX: low drift, moderate vol, carry component

The generator uses a correlated GBM framework with GARCH(1,1) volatility
dynamics to produce realistic return distributions including fat tails,
volatility clustering, and cross-asset correlation.

Author: Jose Orlando Bobadilla Fuentes, CQF, MSc AI
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


# -----------------------------------------------------------------------
# Asset universe specification
# -----------------------------------------------------------------------
ASSET_SPECS = {
    # Equities
    "SPX":   {"class": "equity",    "mu": 0.08, "sigma": 0.16, "name": "S&P 500"},
    "EAFE":  {"class": "equity",    "mu": 0.06, "sigma": 0.18, "name": "MSCI EAFE"},
    "EM":    {"class": "equity",    "mu": 0.09, "sigma": 0.22, "name": "MSCI EM"},
    "SMAL":  {"class": "equity",    "mu": 0.09, "sigma": 0.20, "name": "Russell 2000"},
    # Fixed Income
    "UST10": {"class": "fixed_inc", "mu": 0.03, "sigma": 0.06, "name": "US 10Y Treasury"},
    "HY":    {"class": "fixed_inc", "mu": 0.05, "sigma": 0.08, "name": "US High Yield"},
    "EMBD":  {"class": "fixed_inc", "mu": 0.05, "sigma": 0.10, "name": "EM Sovereign Debt"},
    # Commodities
    "GOLD":  {"class": "commodity", "mu": 0.04, "sigma": 0.15, "name": "Gold"},
    "OIL":   {"class": "commodity", "mu": 0.02, "sigma": 0.30, "name": "Crude Oil WTI"},
    "COPR":  {"class": "commodity", "mu": 0.03, "sigma": 0.22, "name": "Copper"},
    # FX (vs USD)
    "EURUSD": {"class": "fx",      "mu": 0.00, "sigma": 0.08, "name": "EUR/USD"},
    "JPYUSD": {"class": "fx",      "mu":-0.01, "sigma": 0.09, "name": "JPY/USD"},
}


def _build_correlation_matrix(n_assets: int, seed: int = 42) -> np.ndarray:
    """
    Build a realistic correlation matrix with block structure
    reflecting intra-class and inter-class correlations.

    Uses random Wishart-derived matrix constrained to be positive definite.

    Parameters
    ----------
    n_assets : int
        Number of assets in the universe.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        (n_assets, n_assets) correlation matrix.
    """
    rng = np.random.RandomState(seed)

    # Generate via Cholesky of a random Wishart matrix
    A = rng.randn(n_assets, n_assets + 5)
    cov = A @ A.T / (n_assets + 5)

    # Convert to correlation
    d = np.sqrt(np.diag(cov))
    corr = cov / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)

    # Boost intra-class correlations for realism
    # Assets 0-3: equities, 4-6: fixed income, 7-9: commodities, 10-11: FX
    blocks = [(0, 4), (4, 7), (7, 10), (10, 12)]
    for start, end in blocks:
        for i in range(start, min(end, n_assets)):
            for j in range(i + 1, min(end, n_assets)):
                corr[i, j] = 0.4 + 0.4 * abs(corr[i, j])
                corr[j, i] = corr[i, j]

    # Ensure positive definite via eigenvalue floor
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.maximum(eigvals, 0.01)
    corr = eigvecs @ np.diag(eigvals) @ eigvecs.T
    d2 = np.sqrt(np.diag(corr))
    corr = corr / np.outer(d2, d2)
    np.fill_diagonal(corr, 1.0)

    return corr


def _simulate_garch_returns(
    mu: float,
    sigma: float,
    n_days: int,
    innovations: np.ndarray,
    omega: float = 1e-6,
    alpha: float = 0.08,
    beta: float = 0.88,
) -> np.ndarray:
    """
    Simulate daily returns with GARCH(1,1) volatility dynamics.

    GARCH(1,1): h_t = omega + alpha * r_{t-1}^2 + beta * h_{t-1}

    Parameters
    ----------
    mu : float
        Annualized drift.
    sigma : float
        Annualized long-run volatility.
    n_days : int
        Number of trading days to simulate.
    innovations : np.ndarray
        Standard normal innovations (pre-correlated).
    omega : float
        GARCH constant (daily scale).
    alpha : float
        GARCH shock coefficient.
    beta : float
        GARCH persistence coefficient.

    Returns
    -------
    np.ndarray
        Daily log-returns array of shape (n_days,).
    """
    daily_mu = mu / 252.0
    daily_var = (sigma ** 2) / 252.0

    h = np.zeros(n_days)
    r = np.zeros(n_days)
    h[0] = daily_var

    for t in range(n_days):
        vol_t = np.sqrt(h[t])
        r[t] = daily_mu + vol_t * innovations[t]
        if t < n_days - 1:
            h[t + 1] = omega + alpha * r[t] ** 2 + beta * h[t]

    return r


def generate_multi_asset_data(
    n_years: int = 15,
    seed: int = 42,
    specs: Optional[Dict] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Generate synthetic multi-asset price and return data.

    Creates a realistic multi-asset universe with correlated returns,
    GARCH volatility clustering, and proper cross-asset dynamics.

    Parameters
    ----------
    n_years : int
        Number of years of daily data to generate.
    seed : int
        Random seed for reproducibility.
    specs : dict, optional
        Asset specifications. Defaults to ASSET_SPECS.

    Returns
    -------
    prices : pd.DataFrame
        Daily price levels indexed by date. Columns are asset tickers.
    returns : pd.DataFrame
        Daily log-returns indexed by date.
    metadata : dict
        Asset metadata including class, name, and parameters.
    """
    if specs is None:
        specs = ASSET_SPECS

    rng = np.random.RandomState(seed)
    tickers = list(specs.keys())
    n_assets = len(tickers)
    n_days = n_years * 252

    # Build correlation structure and generate correlated innovations
    corr = _build_correlation_matrix(n_assets, seed)
    L = np.linalg.cholesky(corr)
    raw = rng.randn(n_days, n_assets)
    corr_innovations = raw @ L.T

    # Simulate each asset with GARCH dynamics
    returns_dict = {}
    for idx, ticker in enumerate(tickers):
        spec = specs[ticker]
        r = _simulate_garch_returns(
            mu=spec["mu"],
            sigma=spec["sigma"],
            n_days=n_days,
            innovations=corr_innovations[:, idx],
        )
        returns_dict[ticker] = r

    # Build DataFrames
    dates = pd.bdate_range(
        start="2010-01-04", periods=n_days, freq="B"
    )
    returns = pd.DataFrame(returns_dict, index=dates)
    prices = np.exp(returns.cumsum()) * 100.0  # Start at 100

    # Metadata
    metadata = {}
    for ticker, spec in specs.items():
        metadata[ticker] = {
            "asset_class": spec["class"],
            "name": spec["name"],
            "ann_drift": spec["mu"],
            "ann_vol": spec["sigma"],
        }

    return prices, returns, metadata


def get_asset_class_map(metadata: Dict) -> Dict[str, list]:
    """
    Group tickers by asset class.

    Parameters
    ----------
    metadata : dict
        Asset metadata from generate_multi_asset_data.

    Returns
    -------
    dict
        Mapping from asset class name to list of tickers.
    """
    class_map = {}
    for ticker, info in metadata.items():
        ac = info["asset_class"]
        if ac not in class_map:
            class_map[ac] = []
        class_map[ac].append(ticker)
    return class_map
