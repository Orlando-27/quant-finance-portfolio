"""
Yield Curve Data Acquisition
==============================
Provides:
    - US Treasury yield data via yfinance (^IRX, ^FVX, ^TNX, ^TYX proxies)
    - Colombian TES synthetic data (calibrated to typical BanRep curves)
    - SyntheticYieldCurve: configurable panel generator for testing

Note on Colombian TES:
    The Banco de la República publishes TES yield curves via its API at
    https://www.banrep.gov.co/es/estadisticas/tasas-colocacion-dtf-ibc
    For reproducibility this module generates calibrated synthetic TES data
    based on historical BanRep curve parameters (2015-2024 averages).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
import warnings

warnings.filterwarnings("ignore")

# US Treasury tenors available via yfinance tickers
_US_TENORS = {
    0.25: "^IRX",    # 13-week T-bill
    2.0 : "^TWO",    # 2-year note
    5.0 : "^FVX",    # 5-year note
    10.0: "^TNX",    # 10-year note
    30.0: "^TYX",    # 30-year bond
}


class USTreasuryLoader:
    """Download US Treasury yield data from yfinance."""

    def __init__(self, start: str = "2015-01-01", end: str = "2024-12-31"):
        self.start = start
        self.end   = end

    def fetch(self) -> pd.DataFrame:
        """
        Download available Treasury yields; resample to month-end.

        Returns
        -------
        pd.DataFrame  (dates x tenors) with decimal yields.
        """
        frames = {}
        for tenor, ticker in _US_TENORS.items():
            try:
                df = yf.download(ticker, start=self.start, end=self.end,
                                 auto_adjust=True, progress=False)
                if df.empty:
                    continue
                close = df["Close"] if "Close" in df.columns else df.iloc[:, 0]
                if hasattr(close, 'columns'):
                    close = close.iloc[:, 0]
                close = close.dropna()
                # Convert from percent to decimal
                frames[tenor] = close / 100.0
            except Exception:
                continue

        if not frames:
            return pd.DataFrame()

        panel = pd.DataFrame(frames)
        panel.index = pd.to_datetime(panel.index)
        # Month-end resample
        panel = panel.resample("ME").last().dropna(how="all")
        panel.columns = [float(c) for c in panel.columns]
        return panel.sort_index()


class SyntheticYieldCurve:
    """
    Generate synthetic yield curve panels for testing and demonstration.

    Two modes:
        'us'  : Calibrated to US Treasury parameters (pre/post-2022 hiking).
        'tes' : Calibrated to Colombian TES parameters (BanRep 2015-2024).
    """

    # Colombian TES calibration (average BanRep NS parameters 2015-2024)
    _TES_PARAMS = {
        "beta0_mean"  : 0.085,   # ~8.5% long-run level
        "beta0_std"   : 0.015,
        "beta1_mean"  : -0.030,  # moderate upward slope
        "beta1_std"   : 0.010,
        "beta2_mean"  : 0.005,
        "beta2_std"   : 0.008,
        "lam"         : 1.20,    # hump around 5-7 years
        "ar1"         : 0.92,    # high persistence (typical for EM)
        "tenors"      : [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0],
    }

    _US_PARAMS = {
        "beta0_mean"  : 0.040,
        "beta0_std"   : 0.012,
        "beta1_mean"  : -0.015,
        "beta1_std"   : 0.008,
        "beta2_mean"  : 0.003,
        "beta2_std"   : 0.005,
        "lam"         : 0.60,
        "ar1"         : 0.95,
        "tenors"      : [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0],
    }

    def __init__(
        self,
        mode       : str = "us",
        n_periods  : int = 120,     # months
        start_date : str = "2015-01-01",
        seed       : int = 42,
    ):
        self.mode       = mode
        self.n_periods  = n_periods
        self.start_date = start_date
        self.rng        = np.random.default_rng(seed)
        self.params     = self._TES_PARAMS if mode == "tes" else self._US_PARAMS

    def generate(self) -> pd.DataFrame:
        """
        Generate a synthetic monthly yield panel via NS factor simulation.

        Factors follow correlated AR(1) processes with realistic parameters.

        Returns
        -------
        pd.DataFrame  (n_periods x n_tenors) yield panel.
        """
        from models.nelson_siegel import _ns_loadings

        p   = self.params
        T   = self.n_periods
        ar1 = p["ar1"]

        # Correlated factor innovations (realistic covariance structure)
        cov = np.array([
            [0.0004, -0.0002,  0.0001],
            [-0.0002,  0.0003, -0.0001],
            [0.0001, -0.0001,  0.0002],
        ])
        try:
            chol = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            chol = np.diag(np.sqrt(np.diag(cov)))

        eps   = (chol @ self.rng.standard_normal((3, T))).T  # [T x 3]

        b0 = np.zeros(T)
        b1 = np.zeros(T)
        b2 = np.zeros(T)

        b0[0] = p["beta0_mean"]
        b1[0] = p["beta1_mean"]
        b2[0] = p["beta2_mean"]

        for t in range(1, T):
            b0[t] = p["beta0_mean"] * (1-ar1) + ar1 * b0[t-1] + eps[t, 0]
            b1[t] = p["beta1_mean"] * (1-ar1) + ar1 * b1[t-1] + eps[t, 1]
            b2[t] = p["beta2_mean"] * (1-ar1) + ar1 * b2[t-1] + eps[t, 2]

        # Ensure positivity of level factor (yields can't be < -1% here)
        b0 = np.maximum(b0, 0.005)

        tenors = np.array(p["tenors"], dtype=float)
        L      = _ns_loadings(tenors, p["lam"])          # [n_tenors x 3]
        betas  = np.column_stack([b0, b1, b2])           # [T x 3]
        yields = betas @ L.T                             # [T x n_tenors]

        # Add small i.i.d. measurement noise
        noise  = self.rng.normal(0, 5e-4, yields.shape)
        yields = np.maximum(yields + noise, 0.001)

        idx    = pd.date_range(self.start_date, periods=T, freq="MS")
        return pd.DataFrame(yields, index=idx,
                            columns=[float(t) for t in tenors])
