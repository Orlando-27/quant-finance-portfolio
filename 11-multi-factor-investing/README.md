# Multi-Factor Investing with Machine Learning

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-grade quantitative framework for systematic factor investing,
combining classical financial economics with modern machine learning techniques.

---

## Theoretical Foundation

### 1. Factor Models in Asset Pricing

The Arbitrage Pricing Theory (Ross, 1976) posits that expected returns are a
linear function of exposures to systematic risk factors:

    E[R_i] - R_f = sum_k (beta_ik * lambda_k)

where beta_ik is asset i's exposure to factor k, and lambda_k is the risk
premium for bearing factor k exposure.

### 2. Fama-French Factor Construction

**Three-Factor Model (1993):**

    R_i - R_f = alpha_i + beta_MKT * MKT + beta_SMB * SMB + beta_HML * HML + eps_i

- MKT: Market excess return (value-weighted market minus risk-free rate)
- SMB: Small Minus Big (size premium from 2x3 independent sort on size and B/M)
- HML: High Minus Low (value premium from 2x3 independent sort on size and B/M)

**Five-Factor Extension (2015):**

    R_i - R_f = alpha + b*MKT + s*SMB + h*HML + r*RMW + c*CMA + eps

- RMW: Robust Minus Weak (profitability: high vs low operating profit)
- CMA: Conservative Minus Aggressive (investment: low vs high asset growth)

**Carhart Four-Factor (1997):** adds UMD (Up Minus Down, 12-1 month momentum)

### 3. Fama-MacBeth Two-Pass Regression (1973)

**Pass 1 (Time-series):** For each asset i, estimate factor betas using
rolling-window OLS:

    R_it - R_ft = alpha_i + sum_k beta_ik * F_kt + eps_it

**Pass 2 (Cross-sectional):** At each date t, regress cross-section of
returns on estimated betas:

    R_it = gamma_0t + sum_k gamma_kt * hat{beta}_ik + eta_it

The time-series average of gamma_kt estimates risk premium lambda_k.
Standard errors require Shanken (1992) correction for errors-in-variables:

    Var(hat{lambda}) = (1/T) * [Sigma_gamma + (1 + lambda' Sigma_F^{-1} lambda) * Sigma_eta]

### 4. Barra-Style Risk Decomposition

Total portfolio variance decomposes into factor and specific components:

    Var(R_p) = w' * (B * Sigma_F * B' + Delta) * w

where B is the (N x K) exposure matrix, Sigma_F is (K x K) factor covariance,
and Delta is (N x N) diagonal specific variance matrix.

Active risk (tracking error):

    TE^2 = h' * (B * Sigma_F * B' + Delta) * h

where h = w_portfolio - w_benchmark.

### 5. ML Factor Timing

Traditional models assume constant factor premia. ML allows conditional
estimation:

- **Regime Detection (HMM):** Hidden Markov Model with K states captures
  distinct market regimes (risk-on, risk-off, transition) with different
  factor return distributions per state.
- **Factor Return Prediction:** Gradient Boosting and Random Forest trained
  on macro indicators (yield curve slope, credit spreads, VIX, PMI) to
  forecast next-period factor returns via walk-forward cross-validation.
- **Dynamic Allocation:** Portfolio weights conditioned on predicted factor
  returns through mean-variance or risk-budgeting optimization.

### 6. Risk Parity in Factor Space

Equal risk contribution across factors:

    w_k^{RP} such that RC_k = w_k * (Sigma * w)_k / (w' * Sigma * w) = 1/K

Solved via sequential least squares or Newton method on the RC equations.

---

## Project Structure

    11-multi-factor-investing/
    +-- src/
    |   +-- __init__.py
    |   +-- factors.py            # Factor construction and FF replication
    |   +-- cross_sectional.py    # Fama-MacBeth regression with Shanken correction
    |   +-- risk_model.py         # Barra-style factor risk decomposition
    |   +-- ml_timing.py          # ML factor timing (RF, XGB, HMM regimes)
    |   +-- portfolio.py          # Factor portfolio construction and optimization
    |   +-- backtesting.py        # Walk-forward backtesting engine
    |   +-- visualization/
    |       +-- __init__.py
    |       +-- factor_plots.py   # Publication-quality visualizations
    +-- tests/
    |   +-- test_factors.py
    +-- outputs/figures/
    +-- README.md
    +-- requirements.txt
    +-- setup.py

---

## Key Differentiators

- Full Fama-French 3/5 and Carhart 4-factor replication from synthetic cross-section
- Fama-MacBeth two-pass regression with Shanken-corrected standard errors
- Barra-style factor + specific risk decomposition with marginal contributions
- ML factor timing via walk-forward validation (strict no-lookahead)
- HMM regime detection with state-dependent factor distributions
- Risk parity and mean-variance optimization in factor return space
- Comprehensive backtesting with turnover constraints and transaction costs

---

## References

- Ross, S. (1976). The Arbitrage Theory of Capital Asset Pricing. JET.
- Fama, E. & French, K. (1993). Common Risk Factors in the Returns on Stocks and Bonds. JFE.
- Fama, E. & MacBeth, J. (1973). Risk, Return, and Equilibrium. JPE.
- Carhart, M. (1997). On Persistence in Mutual Fund Performance. JoF.
- Fama, E. & French, K. (2015). A Five-Factor Asset Pricing Model. JFE.
- Shanken, J. (1992). On the Estimation of Beta-Pricing Models. RFS.
- Ang, A. (2014). Asset Management: A Systematic Approach to Factor Investing. OUP.
- Asness, C., Moskowitz, T. & Pedersen, L. (2013). Value and Momentum Everywhere. JoF.
- Gu, S., Kelly, B. & Xiu, D. (2020). Empirical Asset Pricing via Machine Learning. RFS.
- MSCI Barra. Global Equity Model (GEM3). MSCI Research.
