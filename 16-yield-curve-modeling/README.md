# Yield Curve Modeling and Forecasting

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![CQF](https://img.shields.io/badge/CQF-Certified-gold.svg)](https://www.cqf.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Author:** Jose Orlando Bobadilla Fuentes, CQF | MSc AI
**Role:** Senior Quantitative Portfolio Manager & Lead Data Scientist
**Organization:** Colfondos S.A. – Vicepresidencia de Inversiones
**Project:** 15 of 20 | Quantitative Finance Portfolio

---

## Overview

Institutional-grade yield curve modeling toolkit covering parametric
curve fitting (Nelson-Siegel, Nelson-Siegel-Svensson), factor analysis
(PCA), dynamic forecasting (Diebold-Li VAR), and uncertainty
quantification (bootstrap confidence bands). Applied to both US
Treasuries and Colombian TES sovereign bonds.

---

## Theoretical Background

### 1. Nelson-Siegel (1987) Model

The NS model represents the yield curve as a function of three factors:

```
y(τ) = β₀ + β₁ · (1 - e^{-λτ})/(λτ)
             + β₂ · [(1 - e^{-λτ})/(λτ) - e^{-λτ}]
```

**Factor interpretation (Litterman-Scheinkman, 1991):**
- β₀: **Level** – long-end yield, shifts entire curve up/down
- β₁: **Slope** – short minus long rate (yield curve steepness)
- β₂: **Curvature** – hump in the medium term
- λ:  **Decay** – controls where the hump peaks (≈ 1/λ years)

The three loading functions are monotonically decreasing (β₀),
starts at 1 and decays to 0 (β₁), and a bell-shaped hump (β₂).

### 2. Nelson-Siegel-Svensson (1994) Extension

Adds a second curvature term for greater flexibility:

```
y(τ) = β₀ + β₁·L(λ₁,τ) + β₂·C(λ₁,τ) + β₃·C(λ₂,τ)
```

where L and C are the standard NS loading functions evaluated at
two different decay parameters λ₁ and λ₂. Captures more complex
shapes (double humps, inverted curves with a bump).

### 3. Diebold-Li (2006) Dynamic Factor Model

Fixes λ at its cross-sectional median (≈1.5 for monthly data)
and treats the three NS factors as a VAR(1) system:

```
[β₀ₜ]   [μ₀]   [Φ₁₁ Φ₁₂ Φ₁₃] [β₀,ₜ₋₁]   [ε₁ₜ]
[β₁ₜ] = [μ₁] + [Φ₂₁ Φ₂₂ Φ₂₃] [β₁,ₜ₋₁] + [ε₂ₜ]
[β₂ₜ]   [μ₂]   [Φ₃₁ Φ₃₂ Φ₃₃] [β₂,ₜ₋₁]   [ε₃ₜ]
```

This two-step approach achieves competitive forecast accuracy
against DNS and affine term structure models.

### 4. PCA Decomposition

Following Litterman & Scheinkman (1991), PCA of yield changes
explains >99% of variance with three components:
- **PC1 (Level):** ~89% variance, all loadings same sign
- **PC2 (Slope):** ~8% variance, loadings change sign across tenor
- **PC3 (Curvature):** ~2% variance, U-shaped loadings

### 5. Bootstrap Confidence Bands

Residual bootstrap on NS/NSS fits:
1. Fit NS model → residuals ε̂_t
2. Re-sample residuals with replacement → ε*_t
3. Reconstruct yield curves → y*_t = ŷ_t + ε*_t
4. Re-fit NS for each bootstrap sample → β̂*
5. Report 2.5th and 97.5th percentile bands across maturities

---

## Project Structure

```
15-yield-curve-modeling/
├── src/
│   ├── models/
│   │   ├── nelson_siegel.py          # NS and NSS fitting + diagnostics
│   │   ├── pca_factors.py            # PCA decomposition of yield changes
│   │   └── var_forecast.py           # Diebold-Li VAR forecasting
│   ├── utils/
│   │   ├── data_loader.py            # FRED / synthetic yield curve data
│   │   └── helpers.py                # Interpolation, duration, metrics
│   └── visualization/
│       └── charts.py                 # Dark-theme professional charts
├── tests/
│   └── test_yield_curve.py           # 22+ unit tests
├── main.py                           # Full pipeline with CLI
├── requirements.txt
├── setup.py
└── config/config.yaml
```

---

## Quick Start

```bash
cd ~/quant-finance-portfolio/15-yield-curve-modeling
pip install -r requirements.txt
python main.py --mode synthetic
python main.py --mode tes          # Colombian TES analysis
pytest tests/ -v --tb=short
```

---

## Key Outputs

| Chart | Description |
|-------|-------------|
| `01_ns_nss_fit.png` | NS vs NSS curve fit with residuals |
| `02_factor_dynamics.png` | Level/slope/curvature time series |
| `03_pca_analysis.png` | PCA loadings, variance explained, factor evolution |
| `04_var_forecast.png` | Diebold-Li VAR yield curve forecast with fan chart |
| `05_bootstrap_bands.png` | Bootstrap confidence bands + curve 3D surface |

---

## References

- Diebold, F.X. & Li, C. (2006). Forecasting the term structure. *Journal of Econometrics*.
- Litterman, R. & Scheinkman, J. (1991). Common factors affecting bond returns. *Journal of Fixed Income*.
- Nelson, C.R. & Siegel, A.F. (1987). Parsimonious modeling of yield curves. *Journal of Business*.
- Svensson, L.E.O. (1994). Estimating and interpreting forward interest rates. *NBER WP 4871*.

---

## License

MIT License. See [LICENSE](../LICENSE) for details.
