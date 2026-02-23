# Project 19 — CQF Concepts Explained: Interactive Academic Notebooks

**Author:** Jose Orlando Bobadilla Fuentes
**Credentials:** CQF (Certificate in Quantitative Finance) | MSc Artificial Intelligence
**Role:** Senior Quantitative Portfolio Manager & Lead Data Scientist
**Institution:** Colombian Pension Fund — Investment Division

---

## Overview

A rigorous, self-contained implementation of all 55 core topics of the CQF curriculum.
Every mathematical derivation is verified numerically so the theory can be *seen and
measured*, not merely read. All scripts run headless (Cloud Shell / server) and save
publication-quality dark-theme figures to `outputs/figures/`.

---

## Complete Module Index

### CQF Priority Concepts (Modules 51-55)
*Full derivations, numerical proofs, publication figures.*

| ID    | Script                              | Topic                                             |
|-------|-------------------------------------|---------------------------------------------------|
| 51    | `cqf_51_stochastic_calculus.py`     | Stochastic Calculus Visualized: BM, GBM, OU, QV  |
| 52    | `cqf_52_itos_lemma.py`              | Ito's Lemma: Taylor Expansion, Proof, Error Analysis |
| 53    | `cqf_53_black_scholes_derivation.py`| Black-Scholes: Delta Hedge, PDE, Heat Equation    |
| 54    | `cqf_54_greeks_intuition.py`        | Greeks: Geometric Intuition & 3D Surfaces         |
| 55    | `cqf_55_sabr_model.py`              | SABR: SDE, Hagan Approximation, Vol Smile         |

---

### Module 1 — Market Data Analysis (Modules 01-07)

| ID  | Script                          | Topic                                              |
|-----|---------------------------------|----------------------------------------------------|
| 01  | `m01_market_data.py`            | OHLCV Download, Splits & Dividend Adjustment       |
| 02  | `m02_returns.py`                | Log vs Simple Returns, Compounding Visualization   |
| 03  | `m03_stylized_facts.py`         | Jarque-Bera, Kurtosis, Skew, Volatility Clustering |
| 04  | `m04_qq_plots.py`               | Q-Q Plots, Fat Tails vs Normal Distribution        |
| 05  | `m05_volatility_ewma.py`        | Rolling Volatility & RiskMetrics EWMA (lambda=0.94)|
| 06  | `m06_stationarity.py`           | ADF, KPSS, Phillips-Perron Unit Root Tests         |
| 07  | `m07_autocorrelation.py`        | ACF, PACF, Ljung-Box Serial Correlation            |

---

### Module 2 — Stochastic Calculus & Simulation (Modules 08-12)

| ID  | Script                          | Topic                                              |
|-----|---------------------------------|----------------------------------------------------|
| 08  | `m08_gbm.py`                    | GBM Euler-Maruyama + Exact Discretisation          |
| 09  | `m09_itos_lemma.py`             | Ito Lemma: dS vs d(lnS) Numerical Verification     |
| 10  | `m10_mean_reversion.py`         | Ornstein-Uhlenbeck / Vasicek: Speed of Reversion   |
| 11  | `m11_correlated_paths.py`       | Cholesky Decomposition for Correlated Assets       |
| 12  | `m12_quasi_random.py`           | Pseudo-Random vs Sobol/Halton Convergence          |

---

### Module 3 — Options Pricing (Modules 13-18)

| ID  | Script                          | Topic                                              |
|-----|---------------------------------|----------------------------------------------------|
| 13  | `m13_binomial_trees.py`         | CRR Binomial Tree: European & American Options     |
| 14  | `m14_binomial_conv.py`          | Binomial Convergence to Black-Scholes as N -> inf  |
| 15  | `m15_bs_formula.py`             | BSM Formula: Calls, Puts, Put-Call Parity          |
| 16  | `m16_greeks_surfaces.py`        | 3D Greek Surfaces: Delta, Gamma, Vega, Theta, Rho  |
| 17  | `m17_implied_vol.py`            | Implied Volatility: Newton-Raphson & Brent Solver  |
| 18  | `m18_vol_smile.py`              | Volatility Smile, Skew & Term Structure            |

---

### Module 4 — Numerical Methods: PDEs & Monte Carlo (Modules 19-25)

| ID  | Script                          | Topic                                              |
|-----|---------------------------------|----------------------------------------------------|
| 19  | `m19_mc_vanilla.py`             | Monte Carlo Vanilla Options & Confidence Bands     |
| 20  | `m20_variance_reduction.py`     | Antithetic Variates & Control Variates             |
| 21  | `m21_fd_explicit.py`            | Explicit Finite Differences: BS PDE on Grid        |
| 22  | `m22_fd_crank_nicolson.py`      | Crank-Nicolson Implicit Scheme (Unconditionally Stable) |
| 23  | `m23_asian_options.py`          | Asian Options: Arithmetic Average MC               |
| 24  | `m24_barrier_options.py`        | Barrier Options: Knock-In / Knock-Out              |
| 25  | `m25_lookback_options.py`       | Lookback Options: Path Maximum & Minimum           |

---

### Module 5 — Fixed Income & Yield Curves (Modules 26-30)

| ID  | Script                          | Topic                                              |
|-----|---------------------------------|----------------------------------------------------|
| 26  | `m26_yield_bootstrap.py`        | Bootstrapping Zero-Coupon Yield Curve              |
| 27  | `m27_duration_convexity.py`     | DV01, Modified Duration & Convexity                |
| 28  | `m28_pca_rates.py`              | PCA on Yield Curves: Level, Slope, Curvature       |
| 29  | `m29_vasicek.py`                | Vasicek Model: Calibration, Simulation, Bond Price |
| 30  | `m30_hjm.py`                    | Heath-Jarrow-Morton: Forward Rate Evolution & Drift|

---

### Module 6 — Portfolio Theory & Risk Management (Modules 31-35)

| ID  | Script                          | Topic                                              |
|-----|---------------------------------|----------------------------------------------------|
| 31  | `m31_markowitz.py`              | Efficient Frontier, Min-Var & Max-Sharpe Weights   |
| 32  | `m32_black_litterman.py`        | Black-Litterman: Views, Posterior Mean & Covariance|
| 33  | `m33_hrp.py`                    | Hierarchical Risk Parity: Clustering & Quasi-Diag  |
| 34  | `m34_var_parametric.py`         | Historical & Parametric VaR at 95% / 99%           |
| 35  | `m35_var_mc_cvar.py`            | Monte Carlo VaR & Expected Shortfall (CVaR)        |

---

### Module 7 — Supervised Machine Learning (Modules 36-40)

| ID  | Script                          | Topic                                              |
|-----|---------------------------------|----------------------------------------------------|
| 36  | `m36_factor_engineering.py`     | Momentum, Value & Volatility Factor Pipeline       |
| 37  | `m37_ridge_lasso.py`            | Ridge / Lasso Regularised Return Prediction        |
| 38  | `m38_random_forests.py`         | Random Forests: Classification & Feature Importance|
| 39  | `m39_gradient_boosting.py`      | XGBoost / LightGBM Ensemble Trading Signals        |
| 40  | `m40_purged_kfold.py`           | Purged K-Fold: Embargo Gap & Leakage Prevention    |

---

### Module 8 — Deep Learning & NLP (Modules 41-45)

| ID  | Script                          | Topic                                              |
|-----|---------------------------------|----------------------------------------------------|
| 41  | `m41_mlp.py`                    | Feedforward Neural Network: Architecture & Training|
| 42  | `m42_rnn_lstm.py`               | RNN / LSTM: Long-Term Memory in Sequences          |
| 43  | `m43_autoencoders.py`           | Autoencoders: Anomaly Detection & Vol Surface      |
| 44  | `m44_nlp_sentiment.py`          | NLP Sentiment: Financial Headlines Analysis        |
| 45  | `m45_word2vec.py`               | Word2Vec: Financial Text Embeddings                |

---

### Module 9 — Backtesting & Algorithmic Trading (Modules 46-50)

| ID  | Script                          | Topic                                              |
|-----|---------------------------------|----------------------------------------------------|
| 46  | `m46_vectorbt.py`               | Vectorised Backtesting: MA Crossover               |
| 47  | `m47_event_backtest.py`         | Event-Driven Simulation: Commissions & Slippage    |
| 48  | `m48_alphalens.py`              | Factor IC, Signal Decay & Alphalens Tearsheet      |
| 49  | `m49_pyfolio.py`                | Sharpe, Sortino, Drawdowns & Rolling Performance   |
| 50  | `m50_pairs_trading.py`          | Cointegration, Z-Score & Mean-Reversion Signals    |

---

## Running the Modules

```bash
# Install dependencies
pip install -r requirements.txt

# Run CQF priority scripts (51-55) — start here
python src/cqf_core/cqf_51_stochastic_calculus.py
python src/cqf_core/cqf_52_itos_lemma.py
python src/cqf_core/cqf_53_black_scholes_derivation.py
python src/cqf_core/cqf_54_greeks_intuition.py
python src/cqf_core/cqf_55_sabr_model.py

# Run a specific thematic module
python src/m08_gbm/m08_gbm.py

# Run all 55 modules
bash run_all.sh
```

---

## Output Structure

```
outputs/
  figures/
    cqf51_bm_paths.png
    cqf51_gbm_envelope.png
    cqf51_ou_reversion.png
    cqf51_quadratic_variation.png
    cqf52_ito_paths.png
    cqf52_error_distribution.png
    cqf53_bs_pde_derivation.png
    cqf53_fd_vs_analytic.png
    cqf54_delta_surface.png
    cqf54_gamma_surface.png
    cqf54_vega_surface.png
    cqf54_theta_surface.png
    cqf55_sabr_smile.png
    cqf55_sabr_calibration.png
    m01_ohlcv.png
    m01_adjusted_prices.png
    ... (one or more figures per module)
```

---

## Technical Standards

- **Language:** Python 3.10+
- **Backend:** matplotlib Agg (headless / Cloud Shell)
- **Style:** dark professional theme via `src/common/style.py`
- **Documentation:** LaTeX-style math annotations on every figure
- **Tests:** pytest unit tests in `tests/`
- **No external data dependency for modules 51-55** (fully synthetic)

---

## Repository Structure

```
19-cqf-concepts-explained/
├── README.md
├── requirements.txt
├── run_all.sh
├── src/
│   ├── common/
│   │   └── style.py
│   ├── cqf_core/
│   │   ├── cqf_51_stochastic_calculus.py
│   │   ├── cqf_52_itos_lemma.py
│   │   ├── cqf_53_black_scholes_derivation.py
│   │   ├── cqf_54_greeks_intuition.py
│   │   └── cqf_55_sabr_model.py
│   ├── m01_market_data/
│   ├── m02_returns/
│   │   ... (modules 01-50)
│   └── m50_pairs_trading/
├── tests/
├── notebooks/
└── outputs/figures/
```
