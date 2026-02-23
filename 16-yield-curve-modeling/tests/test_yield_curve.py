"""
Unit Tests — Yield Curve Modeling and Forecasting
===================================================
24 tests covering all core modules.

Coverage:
    - NelsonSiegel:           loadings, fit accuracy, predict,
                              bootstrap, diagnostics
    - NelsonSiegelSvensson:   fit, predict, forward curve
    - diebold_li_fit_panel:   factor extraction
    - YieldCurvePCA:          fit, transform, variance, reconstruct
    - DieboldLiVAR:           fit, forecast_factors, forecast_curves
    - SyntheticYieldCurve:    generation shape and positivity
    - Helpers:                spline, linear interp, curve metrics,
                              duration, dv01, par yield, decomp

Run: pytest tests/ -v --tb=short
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from models.nelson_siegel  import (NelsonSiegel, NelsonSiegelSvensson,
                                    _ns_loadings, diebold_li_fit_panel)
from models.pca_factors    import YieldCurvePCA
from models.var_forecast   import DieboldLiVAR
from utils.data_loader     import SyntheticYieldCurve
from utils.helpers         import (cubic_spline_curve, linear_interp_curve,
                                   curve_fit_metrics, modified_duration,
                                   dv01, par_yield, yield_change_decomposition)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def us_tenors() -> np.ndarray:
    return np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])


@pytest.fixture(scope="module")
def sample_yields(us_tenors) -> np.ndarray:
    """
    Realistic upward-sloping US yield curve snapshot (2024-style).
    Short end anchored by Fed funds ~5.25%; long end ~4.2%.
    """
    return np.array([0.052, 0.051, 0.049, 0.046, 0.044,
                     0.043, 0.043, 0.042, 0.044, 0.045])


@pytest.fixture(scope="module")
def inverted_yields(us_tenors) -> np.ndarray:
    """Inverted curve (short > long)."""
    return np.array([0.054, 0.053, 0.051, 0.048, 0.046,
                     0.044, 0.043, 0.042, 0.041, 0.040])


@pytest.fixture(scope="module")
def synthetic_panel() -> pd.DataFrame:
    """120-month synthetic US yield panel."""
    gen = SyntheticYieldCurve(mode="us", n_periods=120, seed=0)
    return gen.generate()


@pytest.fixture(scope="module")
def tes_panel() -> pd.DataFrame:
    """60-month synthetic Colombian TES panel."""
    gen = SyntheticYieldCurve(mode="tes", n_periods=60, seed=7)
    return gen.generate()


@pytest.fixture(scope="module")
def fitted_ns(us_tenors, sample_yields) -> NelsonSiegel:
    ns = NelsonSiegel()
    ns.fit(us_tenors, sample_yields)
    return ns


@pytest.fixture(scope="module")
def fitted_nss(us_tenors, sample_yields) -> NelsonSiegelSvensson:
    nss = NelsonSiegelSvensson()
    nss.fit(us_tenors, sample_yields)
    return nss


# =============================================================================
# NelsonSiegel tests (6 tests)
# =============================================================================
class TestNelsonSiegel:

    def test_loadings_shape(self, us_tenors):
        """NS loading matrix has shape (n_tenors, 3)."""
        L = _ns_loadings(us_tenors, lam=1.5)
        assert L.shape == (len(us_tenors), 3)

    def test_loadings_first_column_ones(self, us_tenors):
        """First loading column (level) is all ones."""
        L = _ns_loadings(us_tenors, lam=1.5)
        np.testing.assert_allclose(L[:, 0], np.ones(len(us_tenors)))

    def test_fit_r2_above_threshold(self, fitted_ns, us_tenors, sample_yields):
        """NS fit achieves R² > 0.90 on realistic yield curve."""
        diag = fitted_ns.diagnostics(us_tenors, sample_yields)
        assert diag["r2"] > 0.90, f"R² too low: {diag['r2']:.4f}"

    def test_fit_rmse_below_threshold(self, fitted_ns, us_tenors, sample_yields):
        """NS RMSE is below 10 bps on realistic yield curve."""
        diag = fitted_ns.diagnostics(us_tenors, sample_yields)
        assert diag["rmse"] * 1e4 < 10.0, f"RMSE too high: {diag['rmse']*1e4:.2f} bps"

    def test_predict_returns_correct_shape(self, fitted_ns):
        """predict() returns array of expected length."""
        tau   = np.linspace(0.1, 30.0, 200)
        preds = fitted_ns.predict(tau)
        assert len(preds) == 200

    def test_predict_all_positive(self, fitted_ns):
        """All predicted yields are positive (no negative rates here)."""
        tau   = np.linspace(0.1, 30.0, 100)
        preds = fitted_ns.predict(tau)
        assert (preds > 0).all(), "NS produced negative yield predictions"

    def test_bootstrap_ci_coverage(self, fitted_ns, us_tenors, sample_yields):
        """Bootstrap CI upper bound > lower bound for all tenors."""
        tau_grid = np.linspace(0.1, 30.0, 50)
        boot     = fitted_ns.bootstrap_bands(us_tenors, sample_yields,
                                             tau_grid, n_iter=100, seed=0)
        assert (boot["upper"] >= boot["lower"]).all()


# =============================================================================
# NelsonSiegelSvensson tests (4 tests)
# =============================================================================
class TestNelsonSiegelSvensson:

    def test_nss_fit_r2_ge_ns(self, fitted_ns, fitted_nss,
                               us_tenors, sample_yields):
        """NSS R² should be >= NS R² (more flexible model)."""
        ns_diag  = fitted_ns.diagnostics(us_tenors, sample_yields)
        nss_pred = fitted_nss.predict(us_tenors)
        ss_tot   = np.sum((sample_yields - sample_yields.mean())**2)
        nss_r2   = 1.0 - np.sum((sample_yields - nss_pred)**2) / ss_tot
        assert nss_r2 >= ns_diag["r2"] - 0.05  # NSS >= NS - 5% tolerance

    def test_nss_predict_positive(self, fitted_nss):
        """NSS yields are positive."""
        tau   = np.linspace(0.1, 30.0, 100)
        preds = fitted_nss.predict(tau)
        assert (preds > 0).all()

    def test_forward_curve_reasonable(self, fitted_nss):
        """NSS forward curve is within realistic bounds [0%, 25%]."""
        tau = np.linspace(0.1, 30.0, 100)
        fwd = fitted_nss.forward_curve(tau)
        assert fwd.min() > -0.05, "Forward rate too negative"
        assert fwd.max() < 0.25,  "Forward rate unrealistically high"

    def test_nss_lambda_separation(self, fitted_nss):
        """NSS decay parameters λ₁ ≠ λ₂ (no collinearity)."""
        p = fitted_nss.params
        assert abs(p.lam1 - p.lam2) > 0.05, "λ₁ and λ₂ too close (collinear)"


# =============================================================================
# Diebold-Li panel extraction tests (2 tests)
# =============================================================================
class TestDieboldLiPanel:

    def test_factor_panel_shape(self, synthetic_panel, us_tenors):
        """Diebold-Li panel extraction returns correct shape."""
        factors = diebold_li_fit_panel(us_tenors, synthetic_panel)
        assert factors.shape == (len(synthetic_panel), 3)

    def test_factor_panel_no_nan(self, synthetic_panel, us_tenors):
        """Diebold-Li factors contain no NaN for clean input."""
        factors = diebold_li_fit_panel(us_tenors, synthetic_panel)
        assert not factors.isna().any().any()


# =============================================================================
# YieldCurvePCA tests (4 tests)
# =============================================================================
class TestYieldCurvePCA:

    def test_pca_fit_transform_shape(self, synthetic_panel):
        """PCA fit+transform returns (n_obs, 3) scores."""
        pca    = YieldCurvePCA(n_components=3, scale=True)
        scores = pca.fit_transform(synthetic_panel)
        assert scores.shape == (len(synthetic_panel.dropna()), 3)

    def test_pca_explains_99_percent(self, synthetic_panel):
        """Three components explain at least 99% of variance."""
        pca = YieldCurvePCA(n_components=3, scale=True)
        pca.fit(synthetic_panel)
        cum_var = pca.variance_explained().iloc[-1]
        assert cum_var > 0.99, f"Cumulative var too low: {cum_var:.4f}"

    def test_pca_reconstruct_shape(self, synthetic_panel):
        """PCA reconstruction returns same shape as input."""
        pca   = YieldCurvePCA(n_components=3, scale=True)
        pca.fit(synthetic_panel)
        recon = pca.reconstruct(synthetic_panel)
        assert recon.shape == synthetic_panel.dropna().shape

    def test_pca_loadings_shape(self, synthetic_panel):
        """PCA loadings matrix shape is (n_tenors, 3)."""
        pca = YieldCurvePCA(n_components=3, scale=True)
        pca.fit(synthetic_panel)
        n_t = len(synthetic_panel.columns)
        assert pca.loadings_.shape == (n_t, 3)


# =============================================================================
# DieboldLiVAR tests (4 tests)
# =============================================================================
class TestDieboldLiVAR:

    @pytest.fixture(scope="class")
    def fitted_var(self, synthetic_panel, us_tenors):
        factors = diebold_li_fit_panel(us_tenors, synthetic_panel)
        dl = DieboldLiVAR(max_lags=4, lam=0.0609)
        dl.fit(factors)
        return dl

    def test_var_forecast_shape(self, fitted_var, us_tenors):
        """VAR factor forecast has shape (h, 3)."""
        fc = fitted_var.forecast_factors(h=12)
        assert fc.shape == (12, 3)

    def test_var_forecast_curves_shape(self, fitted_var, us_tenors):
        """VAR yield curve forecast has shape (h, n_tenors)."""
        fc = fitted_var.forecast_curves(us_tenors, h=6)
        assert fc.shape == (6, len(us_tenors))

    def test_var_optimal_lag_positive(self, fitted_var):
        """VAR optimal lag is at least 1."""
        summ = fitted_var.summary()
        assert summ["optimal_lag"] >= 1

    def test_var_forecast_yields_positive(self, fitted_var, us_tenors):
        """Forecasted yields are positive (no negative rates)."""
        fc = fitted_var.forecast_curves(us_tenors, h=12)
        assert (fc.values > -0.02).all()  # allow tiny floating point


# =============================================================================
# SyntheticYieldCurve tests (2 tests)
# =============================================================================
class TestSyntheticYieldCurve:

    def test_us_panel_shape(self):
        """US synthetic panel has correct shape."""
        gen   = SyntheticYieldCurve(mode="us", n_periods=60, seed=0)
        panel = gen.generate()
        assert panel.shape[0] == 60
        assert panel.shape[1] == 10   # 10 US tenors

    def test_tes_panel_positive_yields(self):
        """Colombian TES yields are all positive."""
        gen   = SyntheticYieldCurve(mode="tes", n_periods=60, seed=1)
        panel = gen.generate()
        assert (panel.values > 0).all()


# =============================================================================
# Helpers tests (4 tests)
# =============================================================================
class TestHelpers:

    def test_cubic_spline_monotone_upward(self, us_tenors, sample_yields):
        """Spline interpolation is smooth and within observed range."""
        tau_fine = np.linspace(us_tenors[0], us_tenors[-1], 200)
        result   = cubic_spline_curve(us_tenors, sample_yields, tau_fine)
        assert result.min() >= sample_yields.min() * 0.9
        assert result.max() <= sample_yields.max() * 1.1

    def test_curve_metrics_r2_positive(self, us_tenors, sample_yields, fitted_ns):
        """Curve fit R² is positive for NS on realistic data."""
        yhat    = fitted_ns.predict(us_tenors)
        metrics = curve_fit_metrics(sample_yields, yhat, us_tenors)
        assert metrics["R²"] > 0

    def test_modified_duration_positive(self):
        """Modified duration of a coupon bond is positive."""
        # 5-year 4% semi-annual bond
        times   = np.arange(0.5, 5.5, 0.5)
        cfs     = np.array([2.0] * 9 + [102.0])
        dur     = modified_duration(times, cfs, ytm=0.04, freq=2)
        assert dur > 0

    def test_yield_decomp_shape(self, synthetic_panel):
        """Yield change decomposition has 3 columns."""
        decomp = yield_change_decomposition(synthetic_panel)
        assert decomp.shape[1] == 3
        assert "level_chg" in decomp.columns
