"""
Diebold-Li VAR Forecasting Model
==================================
Implements the Diebold & Li (2006) two-step yield curve forecasting
approach: fit NS factors monthly, then forecast via VAR(p).

References:
    Diebold, F.X. & Li, C. (2006). JoE 130(2), 337-364.
    Lütkepohl, H. (2005). New Introduction to Multiple Time Series.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import warnings
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")


class DieboldLiVAR:
    """
    Diebold-Li (2006) dynamic Nelson-Siegel with VAR forecasting.

    Two-step procedure:
        Step 1: Extract NS factors {β₀ₜ, β₁ₜ, β₂ₜ} via OLS (fixed λ).
        Step 2: Fit VAR(p) to factor time series; forecast h steps ahead.
        Step 3: Reconstruct yield curve forecast from factor forecasts.
    """

    def __init__(self, max_lags: int = 6, lam: float = 0.0609):
        self.max_lags = max_lags
        self.lam      = lam           # fixed Diebold-Li lambda
        self._var_fit = None
        self._factors_: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Fit VAR to factor series
    # ------------------------------------------------------------------
    def fit(self, factors: pd.DataFrame) -> "DieboldLiVAR":
        """
        Fit a VAR(p) model to the NS factor time series.

        Parameters
        ----------
        factors : pd.DataFrame
            (T x 3) DataFrame with columns [beta0_level, beta1_slope,
            beta2_curv] — output of diebold_li_fit_panel().

        Returns
        -------
        self (fitted)
        """
        self._factors_ = factors.dropna()

        # ADF test for stationarity (informational)
        self._adf_results_ = {}
        for col in self._factors_.columns:
            stat, pval, *_ = adfuller(self._factors_[col].dropna(), maxlag=4)
            self._adf_results_[col] = {"adf_stat": stat, "p_value": pval}

        # VAR fitting with lag order selection by AIC
        model = VAR(self._factors_.values)
        try:
            lag_sel = model.select_order(maxlags=self.max_lags)
            best_lag = lag_sel.aic
            if best_lag == 0:
                best_lag = 1
        except Exception:
            best_lag = 1

        self._var_fit  = model.fit(best_lag)
        self._best_lag = best_lag
        return self

    # ------------------------------------------------------------------
    # Forecast factors
    # ------------------------------------------------------------------
    def forecast_factors(self, h: int = 12) -> pd.DataFrame:
        """
        Forecast the three NS factors h steps ahead.

        Returns
        -------
        pd.DataFrame  (h x 3) factor forecasts.
        """
        if self._var_fit is None:
            raise RuntimeError("Fit VAR first. Call fit().")

        lag_vals = self._factors_.values[-self._best_lag:]
        fc       = self._var_fit.forecast(lag_vals, steps=h)

        # Build forecast index (monthly frequency assumed)
        last_date = self._factors_.index[-1]
        fc_index  = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=h,
            freq="MS",
        )
        return pd.DataFrame(
            fc,
            index  = fc_index,
            columns= self._factors_.columns,
        )

    # ------------------------------------------------------------------
    # Reconstruct yield curve from factors
    # ------------------------------------------------------------------
    def reconstruct_curve(
        self,
        factors_df: pd.DataFrame,
        tenors    : np.ndarray,
    ) -> pd.DataFrame:
        """
        Reconstruct yield curves from a factors DataFrame.

        Parameters
        ----------
        factors_df : pd.DataFrame  (T x 3) with NS factors.
        tenors     : np.ndarray    Maturities in years.

        Returns
        -------
        pd.DataFrame  (T x len(tenors)) yield curve matrix.
        """
        from models.nelson_siegel import _ns_loadings
        L  = _ns_loadings(tenors, self.lam)    # [n_tenors x 3]
        F  = factors_df.values                 # [T x 3]
        Y  = F @ L.T                           # [T x n_tenors]
        return pd.DataFrame(Y, index=factors_df.index,
                            columns=[f"{t:.2f}y" for t in tenors])

    # ------------------------------------------------------------------
    # Forecast yield curves
    # ------------------------------------------------------------------
    def forecast_curves(
        self,
        tenors: np.ndarray,
        h     : int = 12,
    ) -> pd.DataFrame:
        """
        End-to-end yield curve forecast: factor forecast → curve reconstruction.

        Returns
        -------
        pd.DataFrame  (h x len(tenors)) forecast yield curves.
        """
        fc_factors = self.forecast_factors(h)
        return self.reconstruct_curve(fc_factors, tenors)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    def summary(self) -> dict:
        """Model summary statistics."""
        if self._var_fit is None:
            return {}
        return {
            "optimal_lag" : self._best_lag,
            "aic"         : self._var_fit.aic,
            "bic"         : self._var_fit.bic,
            "n_obs"       : self._var_fit.nobs,
            "adf_tests"   : self._adf_results_,
        }
