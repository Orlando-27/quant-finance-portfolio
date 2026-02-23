# Momentum & Mean Reversion Multi-Asset Strategy

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CQF](https://img.shields.io/badge/CQF-Certified-darkblue.svg)](https://www.cqf.com/)

A production-grade quantitative trading framework that combines **time-series momentum (TSMOM)**, **cross-sectional momentum**, and **mean reversion** signals with adaptive regime detection and dynamic signal blending across multiple asset classes.

---

## Theoretical Foundation

### 1. Time-Series Momentum (TSMOM)

Following Moskowitz, Ooi & Pedersen (2012), TSMOM exploits the autocorrelation
structure of asset returns. The signal for asset i at time t:

$$TSMOM_{i,t} = \text{sign}\left(\sum_{k=1}^{K} r_{i,t-k}\right)$$

The position is scaled by inverse volatility (volatility targeting):

$$w_{i,t} = \frac{\sigma_{target}}{\hat{\sigma}_{i,t}} \cdot TSMOM_{i,t}$$

where sigma_hat is estimated via exponentially weighted moving average (EWMA).

### 2. Cross-Sectional Momentum (Jegadeesh-Titman)

At each rebalancing date, rank N assets by trailing J-month return. Go long
the top decile, short the bottom decile, hold for K months:

$$CS\_MOM_{i,t} = \text{rank}\left(\sum_{k=1}^{J} r_{i,t-k}\right)$$

The 12-1 specification (12-month lookback, skip most recent month) is the
standard academic benchmark to avoid short-term reversal contamination.

### 3. Mean Reversion Signals

Three complementary mean-reversion indicators:

**Z-Score (Ornstein-Uhlenbeck):**
$$z_{i,t} = \frac{P_{i,t} - \mu_{i,L}}{\sigma_{i,L}}$$

where mu and sigma are computed over lookback window L. Entry when |z| > 2,
exit when |z| < 0.5.

**Relative Strength Index (RSI):**
$$RSI_{i,t} = 100 - \frac{100}{1 + RS_{i,t}}$$

where RS = average gain / average loss over N periods. Oversold < 30,
overbought > 70.

**Bollinger Band Width:**
$$BB\%_{i,t} = \frac{P_{i,t} - (MA_L - k \cdot \sigma_L)}{2k \cdot \sigma_L}$$

Values near 0 indicate proximity to lower band (potential long), near 1
proximity to upper band (potential short).

### 4. Regime Detection

The framework identifies market regimes using three indicators:
- **Volatility Regime**: EWMA volatility vs. long-term average
- **Dispersion Regime**: Cross-sectional return dispersion (high dispersion favors momentum)
- **Autocorrelation Regime**: Rolling first-order autocorrelation (positive favors TSMOM)

Regime state determines the dynamic blend between momentum and mean-reversion.

### 5. Signal Blending & Portfolio Construction

The composite signal combines momentum and mean-reversion with regime-dependent weights:

$$S_{i,t} = \alpha_t \cdot MOM_{i,t} + (1 - \alpha_t) \cdot MR_{i,t}$$

where alpha_t is determined by the detected regime (high alpha in trending
regimes, low alpha in mean-reverting regimes).

### 6. Risk Management

- EWMA volatility scaling to target constant portfolio volatility
- Maximum position limits per asset and per asset class
- Drawdown control: reduce exposure when trailing drawdown exceeds threshold
- Turnover constraints to control transaction costs

---

## Project Structure

```
13-momentum-mean-reversion/
    src/
        __init__.py
        momentum.py           # TSMOM & cross-sectional momentum signals
        mean_reversion.py     # Z-score, RSI, Bollinger mean reversion
        regime.py             # Regime detection (volatility, dispersion, autocorrelation)
        portfolio.py          # Signal blending, portfolio construction, risk mgmt
        backtesting.py        # Walk-forward backtesting engine with costs
        data_generator.py     # Synthetic multi-asset data generation
        visualization/
            __init__.py
            strategy_plots.py # Publication-quality visualizations
    tests/
        test_momentum.py      # Comprehensive unit tests
    outputs/figures/           # Generated plots
    README.md
    main.py
    requirements.txt
    setup.py
    .gitignore
```

---

## Key Differentiators

- TSMOM with EWMA volatility scaling following Moskowitz et al. (2012)
- Cross-sectional momentum with 12-1 specification and skip-month filter
- Three complementary mean-reversion signals (z-score, RSI, Bollinger)
- Adaptive regime detection using volatility, dispersion, and autocorrelation
- Dynamic signal blending: momentum-heavy in trending regimes, MR-heavy in ranging
- Multi-asset class support: equities, fixed income, commodities, FX
- Walk-forward backtesting with transaction costs and slippage
- Drawdown-based risk overlay with position sizing controls
- Publication-quality dark-theme visualizations

---

## References

- Moskowitz, T., Ooi, Y.H. & Pedersen, L.H. (2012). "Time Series Momentum." JFE
- Jegadeesh, N. & Titman, S. (1993). "Returns to Buying Winners and Selling Losers." JoF
- Asness, C., Moskowitz, T. & Pedersen, L.H. (2013). "Value and Momentum Everywhere." JoF
- Gray, W. & Vogel, J. (2016). "Quantitative Momentum." Wiley Finance
- Baltas, N. & Kosowski, R. (2013). "Momentum Strategies in Futures Markets." SSRN
- DeMiguel, V., Garlappi, L. & Uppal, R. (2009). "Optimal vs Naive Diversification." RFS
- Barroso, P. & Santa-Clara, P. (2015). "Momentum Has Its Moments." JFE
- Daniel, K. & Moskowitz, T. (2016). "Momentum Crashes." JFE
- Lo, A. (2004). "The Adaptive Markets Hypothesis." JPM
- Hamilton, J. (1989). "A New Approach to the Economic Analysis of Nonstationary Time Series." Econometrica
