"""
Nelson-Siegel and Nelson-Siegel-Svensson Yield Curve Models
=============================================================
Implements parametric yield curve fitting, bootstrap confidence
bands, and diagnostic statistics.

Models:
    - Nelson & Siegel (1987): 4-parameter parsimonious model
    - Svensson (1994): 6-parameter extension with second hump
    - Diebold & Li (2006): fixed-lambda two-step estimation

References:
    Nelson, C.R. & Siegel, A.F. (1987). JoB 60(4), 473-489.
    Svensson, L.E.O. (1994). NBER Working Paper 4871.
    Diebold, F.X. & Li, C. (2006). JoE 130(2), 337-364.
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from scipy.optimize import minimize, differential_evolution
from typing import Optional

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Loading functions
# ---------------------------------------------------------------------------
def _ns_loadings(tau: np.ndarray, lam: float) -> np.ndarray:
    """
    Nelson-Siegel loading matrix  [N x 3].

    Columns: [L(τ), S(τ), C(τ)] where:
        L(τ) = 1                              (level – constant)
        S(τ) = (1 - e^{-λτ}) / (λτ)          (slope – decaying)
        C(τ) = S(τ) - e^{-λτ}                (curvature – hump)
    """
    tau  = np.atleast_1d(tau).astype(float)
    lam  = float(lam)
    lt   = lam * tau
    # Avoid division by zero at τ=0
    s    = np.where(lt < 1e-10, 1.0, (1.0 - np.exp(-lt)) / lt)
    c    = s - np.exp(-lt)
    return np.column_stack([np.ones_like(tau), s, c])


def _nss_loadings(tau: np.ndarray, lam1: float, lam2: float) -> np.ndarray:
    """
    Nelson-Siegel-Svensson loading matrix [N x 4].

    Extends NS with a second curvature term at decay λ₂.
    """
    L3 = _ns_loadings(tau, lam1)   # [N, 3]: L, S, C1
    lt2 = lam2 * np.atleast_1d(tau)
    s2  = np.where(lt2 < 1e-10, 1.0, (1.0 - np.exp(-lt2)) / lt2)
    c2  = s2 - np.exp(-lt2)
    return np.column_stack([L3, c2])   # [N, 4]


# ---------------------------------------------------------------------------
# NS Model
# ---------------------------------------------------------------------------
@dataclass
class NSParams:
    """Fitted Nelson-Siegel parameters."""
    beta0 : float = 0.05    # level
    beta1 : float = -0.02   # slope
    beta2 : float = 0.01    # curvature
    lam   : float = 1.50    # decay
    rmse  : float = np.nan
    r2    : float = np.nan

    def as_array(self) -> np.ndarray:
        return np.array([self.beta0, self.beta1, self.beta2, self.lam])


class NelsonSiegel:
    """
    Nelson-Siegel (1987) yield curve model.

    Fitting is done via non-linear least squares using scipy.optimize.
    For each candidate λ, the β parameters are estimated by OLS
    (since the model is linear in betas given λ).
    """

    BOUNDS = [(0.001, 0.25), (-0.20, 0.20), (-0.20, 0.20), (0.01, 10.0)]

    def __init__(self):
        self.params: Optional[NSParams] = None

    # ------------------------------------------------------------------
    # Core yield function
    # ------------------------------------------------------------------
    @staticmethod
    def yield_curve(
        tau   : np.ndarray,
        beta0 : float,
        beta1 : float,
        beta2 : float,
        lam   : float,
    ) -> np.ndarray:
        """Evaluate NS yield at maturities tau (years)."""
        L = _ns_loadings(tau, lam)
        return L @ np.array([beta0, beta1, beta2])

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit(
        self,
        tenors : np.ndarray,
        yields : np.ndarray,
        method : str = "grid",
        n_grid : int = 50,
    ) -> NSParams:
        """
        Fit NS model to observed (tenor, yield) pairs.

        Strategy: grid search over λ ∈ [0.1, 5.0], OLS for betas at
        each λ, select λ with minimum RMSE. Then refine with L-BFGS-B.

        Parameters
        ----------
        tenors : array-like  Maturities in years.
        yields : array-like  Observed yields (decimal, not percent).
        method : 'grid' | 'de'  'de' = differential evolution (slower).
        n_grid : int  Number of λ grid points.

        Returns
        -------
        NSParams  Fitted parameter object.
        """
        tenors = np.asarray(tenors, dtype=float)
        yields = np.asarray(yields, dtype=float)

        def _sse(lam: float) -> tuple[np.ndarray, float]:
            """OLS betas for fixed lambda; returns (betas, SSE)."""
            L    = _ns_loadings(tenors, lam)
            beta, _, _, _ = np.linalg.lstsq(L, yields, rcond=None)
            yhat = L @ beta
            sse  = np.sum((yields - yhat) ** 2)
            return beta, sse

        if method == "grid":
            lam_grid = np.linspace(0.1, 5.0, n_grid)
            best_lam, best_sse = lam_grid[0], np.inf
            for lam in lam_grid:
                _, sse = _sse(lam)
                if sse < best_sse:
                    best_sse, best_lam = sse, lam
            # Refine with local optimiser
            res = minimize(
                lambda x: _sse(x[0])[1],
                x0     = [best_lam],
                bounds = [(0.01, 10.0)],
                method = "L-BFGS-B",
            )
            best_lam = float(res.x[0])
        else:
            # Differential evolution – more robust but slower
            res = differential_evolution(
                lambda x: _sse(x[0])[1],
                bounds  = [(0.01, 10.0)],
                seed    = 42,
                tol     = 1e-8,
            )
            best_lam = float(res.x[0])

        betas, _ = _sse(best_lam)
        yhat     = _ns_loadings(tenors, best_lam) @ betas
        resid    = yields - yhat
        rmse     = np.sqrt(np.mean(resid ** 2))
        ss_tot   = np.sum((yields - yields.mean()) ** 2)
        r2       = 1.0 - np.sum(resid ** 2) / ss_tot if ss_tot > 0 else np.nan

        self.params = NSParams(
            beta0=float(betas[0]),
            beta1=float(betas[1]),
            beta2=float(betas[2]),
            lam  =best_lam,
            rmse =rmse,
            r2   =r2,
        )
        return self.params

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def predict(self, tau: np.ndarray) -> np.ndarray:
        """Evaluate fitted curve at maturities tau."""
        if self.params is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        p = self.params
        return self.yield_curve(tau, p.beta0, p.beta1, p.beta2, p.lam)

    # ------------------------------------------------------------------
    # Bootstrap confidence bands
    # ------------------------------------------------------------------
    def bootstrap_bands(
        self,
        tenors     : np.ndarray,
        yields     : np.ndarray,
        tau_grid   : np.ndarray,
        n_iter     : int   = 500,
        ci         : float = 0.95,
        seed       : int   = 0,
    ) -> dict:
        """
        Residual bootstrap for NS parameter uncertainty.

        Algorithm:
            1. Fit NS to observed yields → residuals ε̂
            2. For each iteration:
               a. Sample residuals with replacement → ε*
               b. Construct y* = ŷ + ε*
               c. Re-fit NS → predicted curve at tau_grid
            3. Return pointwise CI across tau_grid.

        Returns
        -------
        dict with keys: mean_curve, lower, upper, all_curves (n_iter x len(tau_grid))
        """
        if self.params is None:
            self.fit(tenors, yields)

        rng     = np.random.default_rng(seed)
        yhat    = self.predict(tenors)
        resid   = yields - yhat
        n       = len(tenors)
        curves  = np.zeros((n_iter, len(tau_grid)))

        for i in range(n_iter):
            eps    = rng.choice(resid, size=n, replace=True)
            y_boot = yhat + eps
            try:
                m = NelsonSiegel()
                m.fit(tenors, y_boot)
                curves[i] = m.predict(tau_grid)
            except Exception:
                curves[i] = self.predict(tau_grid)

        alpha = 1.0 - ci
        return {
            "mean_curve": curves.mean(axis=0),
            "lower"     : np.percentile(curves, 100 * alpha / 2,   axis=0),
            "upper"     : np.percentile(curves, 100 * (1 - alpha / 2), axis=0),
            "all_curves": curves,
        }

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def diagnostics(
        self,
        tenors: np.ndarray,
        yields: np.ndarray,
    ) -> dict:
        """Compute fit diagnostics: RMSE, MAE, R², residuals by tenor."""
        if self.params is None:
            raise RuntimeError("Fit the model first.")
        yhat  = self.predict(tenors)
        resid = yields - yhat
        ss_tot = np.sum((yields - yields.mean()) ** 2)
        return {
            "rmse"   : float(np.sqrt(np.mean(resid ** 2))),
            "mae"    : float(np.mean(np.abs(resid))),
            "r2"     : float(1.0 - np.sum(resid**2) / ss_tot) if ss_tot > 0 else np.nan,
            "max_err": float(np.max(np.abs(resid))),
            "resid"  : resid,
            "yhat"   : yhat,
        }


# ---------------------------------------------------------------------------
# NSS Model
# ---------------------------------------------------------------------------
@dataclass
class NSSParams:
    """Fitted Nelson-Siegel-Svensson parameters."""
    beta0 : float = 0.05
    beta1 : float = -0.02
    beta2 : float = 0.01
    beta3 : float = 0.005
    lam1  : float = 1.50
    lam2  : float = 5.00
    rmse  : float = np.nan
    r2    : float = np.nan

    def as_array(self) -> np.ndarray:
        return np.array([self.beta0, self.beta1, self.beta2,
                         self.beta3, self.lam1,  self.lam2])


class NelsonSiegelSvensson:
    """
    Nelson-Siegel-Svensson (1994) yield curve model.

    Extends NS with a fourth parameter β₃ and second decay λ₂
    to capture more complex curve shapes.
    """

    def __init__(self):
        self.params: Optional[NSSParams] = None

    @staticmethod
    def yield_curve(
        tau  : np.ndarray,
        beta0: float, beta1: float,
        beta2: float, beta3: float,
        lam1 : float, lam2 : float,
    ) -> np.ndarray:
        """Evaluate NSS yield at maturities tau."""
        L = _nss_loadings(tau, lam1, lam2)
        return L @ np.array([beta0, beta1, beta2, beta3])

    def fit(
        self,
        tenors : np.ndarray,
        yields : np.ndarray,
    ) -> NSSParams:
        """
        Fit NSS via grid search over (λ₁, λ₂) with OLS for betas.

        Constraint: λ₁ ≠ λ₂ (otherwise the two curvature terms are
        collinear and the model reduces to NS).
        """
        tenors = np.asarray(tenors, dtype=float)
        yields = np.asarray(yields, dtype=float)

        lam_grid = np.linspace(0.1, 8.0, 20)
        best_sse = np.inf
        best_lam1, best_lam2 = 1.5, 5.0

        for l1 in lam_grid:
            for l2 in lam_grid:
                if abs(l1 - l2) < 0.3:   # avoid collinearity
                    continue
                L    = _nss_loadings(tenors, l1, l2)
                try:
                    beta, _, _, _ = np.linalg.lstsq(L, yields, rcond=None)
                    sse  = float(np.sum((yields - L @ beta) ** 2))
                    if sse < best_sse:
                        best_sse  = sse
                        best_lam1 = l1
                        best_lam2 = l2
                except np.linalg.LinAlgError:
                    continue

        # Local refinement
        def _obj(x: np.ndarray) -> float:
            l1, l2 = x
            if abs(l1 - l2) < 0.1:
                return 1e12
            L    = _nss_loadings(tenors, l1, l2)
            beta, _, _, _ = np.linalg.lstsq(L, yields, rcond=None)
            return float(np.sum((yields - L @ beta) ** 2))

        res = minimize(
            _obj,
            x0     = [best_lam1, best_lam2],
            bounds = [(0.01, 10.0), (0.01, 10.0)],
            method = "L-BFGS-B",
        )
        l1, l2 = float(res.x[0]), float(res.x[1])
        L      = _nss_loadings(tenors, l1, l2)
        betas, _, _, _ = np.linalg.lstsq(L, yields, rcond=None)
        yhat   = L @ betas
        resid  = yields - yhat
        rmse   = float(np.sqrt(np.mean(resid ** 2)))
        ss_tot = float(np.sum((yields - yields.mean()) ** 2))
        r2     = float(1.0 - np.sum(resid**2) / ss_tot) if ss_tot > 0 else np.nan

        self.params = NSSParams(
            beta0=float(betas[0]), beta1=float(betas[1]),
            beta2=float(betas[2]), beta3=float(betas[3]),
            lam1=l1, lam2=l2,
            rmse=rmse, r2=r2,
        )
        return self.params

    def predict(self, tau: np.ndarray) -> np.ndarray:
        if self.params is None:
            raise RuntimeError("Model not fitted.")
        p = self.params
        return self.yield_curve(tau, p.beta0, p.beta1,
                                p.beta2, p.beta3, p.lam1, p.lam2)

    def forward_curve(self, tau: np.ndarray) -> np.ndarray:
        """
        Instantaneous forward rate implied by NSS:

            f(τ) = β₀ + β₁·e^{-λ₁τ}
                       + β₂·λ₁τ·e^{-λ₁τ}
                       + β₃·λ₂τ·e^{-λ₂τ}
        """
        if self.params is None:
            raise RuntimeError("Model not fitted.")
        p   = self.params
        tau = np.asarray(tau, dtype=float)
        fwd = (p.beta0
               + p.beta1 * np.exp(-p.lam1 * tau)
               + p.beta2 * p.lam1 * tau * np.exp(-p.lam1 * tau)
               + p.beta3 * p.lam2 * tau * np.exp(-p.lam2 * tau))
        return fwd


# ---------------------------------------------------------------------------
# Diebold-Li fixed-lambda estimation
# ---------------------------------------------------------------------------
def diebold_li_fit_panel(
    tenors      : np.ndarray,
    yields_panel: pd.DataFrame,
    lam         : float = 0.0609,  # Diebold-Li canonical value (30y US)
) -> pd.DataFrame:
    """
    Diebold-Li (2006) two-step estimation on a panel of yield curves.

    Step 1: Fix λ and compute OLS betas for each date.
    Step 2: Model beta dynamics with VAR (done in var_forecast.py).

    Parameters
    ----------
    tenors        : np.ndarray  Maturities in years.
    yields_panel  : pd.DataFrame  (dates x tenors) yield panel.
    lam           : float  Fixed decay parameter.

    Returns
    -------
    pd.DataFrame  (dates x 3) factor time series: beta0, beta1, beta2.
    """
    L       = _ns_loadings(tenors, lam)    # [n_tenors x 3]
    betas   = []

    for _, row in yields_panel.iterrows():
        y = row.values.astype(float)
        if np.any(np.isnan(y)):
            betas.append([np.nan, np.nan, np.nan])
            continue
        b, _, _, _ = np.linalg.lstsq(L, y, rcond=None)
        betas.append(b.tolist())

    df = pd.DataFrame(betas,
                      index  = yields_panel.index,
                      columns= ["beta0_level", "beta1_slope", "beta2_curv"])
    return df
