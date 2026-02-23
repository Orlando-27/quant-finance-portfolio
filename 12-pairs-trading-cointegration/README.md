# Pairs Trading Strategy with Cointegration Analysis

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-grade statistical arbitrage framework for pairs trading,
combining classical cointegration theory with modern adaptive techniques
including Kalman filtering and Ornstein-Uhlenbeck process calibration.

---

## Theoretical Foundation

### 1. Statistical Arbitrage and Pairs Trading

Pairs trading exploits temporary deviations from long-run equilibrium
between co-moving securities. The strategy is market-neutral by
construction: for each dollar long in stock A, approximately beta
dollars are short in stock B, where beta is the cointegrating (hedge)
ratio. Profit accrues when the spread mean-reverts.

The core assumption is cointegration: while individual prices are
non-stationary I(1), a linear combination is stationary I(0):

    S_t = P_A,t - beta * P_B,t ~ I(0)

### 2. Cointegration Testing

**Engle-Granger Two-Step Procedure (1987):**

Step 1: Estimate the cointegrating relationship via OLS:
    P_A,t = alpha + beta * P_B,t + eps_t

Step 2: Test the residuals eps_t for stationarity using the
Augmented Dickey-Fuller (ADF) test:
    Delta(eps_t) = gamma * eps_{t-1} + sum_j c_j * Delta(eps_{t-j}) + u_t

    H0: gamma = 0 (no cointegration, unit root in residuals)
    H1: gamma < 0 (cointegration, residuals are stationary)

Critical values follow the MacKinnon (1991) distribution, which differs
from standard ADF tables because the residuals are estimated, not observed.

**Johansen Procedure (1991):**

Tests for cointegration rank r in a VAR(p) system:
    Delta(Y_t) = Pi * Y_{t-1} + sum_j Gamma_j * Delta(Y_{t-j}) + eps_t

where Pi = alpha * beta' has reduced rank r < n. The trace statistic:
    lambda_trace(r0) = -T * sum_{i=r0+1}^{n} ln(1 - hat{lambda}_i)

tests H0: rank(Pi) <= r0 against H1: rank(Pi) > r0.

Johansen is preferred for systems with more than 2 variables and avoids
the normalization problem inherent in Engle-Granger.

### 3. Ornstein-Uhlenbeck Process

The spread S_t follows a continuous-time mean-reverting process:
    dS_t = kappa * (theta - S_t) * dt + sigma * dW_t

where:
- kappa > 0: speed of mean reversion
- theta: long-run equilibrium level
- sigma: volatility of the spread

**Discrete-time estimation (AR(1) on spread):**
    S_t = c + phi * S_{t-1} + eps_t

    kappa = -ln(phi) / Delta_t
    theta = c / (1 - phi)
    sigma = std(eps) * sqrt(-2*ln(phi) / (Delta_t * (1 - phi^2)))

**Half-life of mean reversion:**
    tau_{1/2} = ln(2) / kappa

The half-life determines the expected holding period and is critical
for position sizing and risk management. Tradeable pairs typically
have half-lives between 5 and 60 trading days.

### 4. Kalman Filter for Adaptive Hedge Ratios

The static OLS hedge ratio assumes a constant beta, which fails in
practice due to structural shifts. The Kalman filter treats beta as
a latent state variable evolving according to:

    State equation:     beta_t = beta_{t-1} + w_t,  w_t ~ N(0, Q)
    Observation eq:     P_A,t = alpha_t + beta_t * P_B,t + v_t,  v_t ~ N(0, R)

The filter recursively updates beta_t given new price observations,
producing a time-varying hedge ratio that adapts to regime changes.

### 5. Pair Selection Methods

**Distance Method (Gatev et al., 2006):**
1. Normalize all price series to unit starting value
2. Compute sum of squared differences (SSD) for all (N choose 2) pairs
3. Select the K pairs with smallest SSD in the formation period
4. Trade these pairs during the subsequent trading period

**Cointegration Method:**
1. For all candidate pairs, test Engle-Granger cointegration
2. Filter by: ADF p-value < 0.05, half-life in [5, 60] days
3. Rank by: highest ADF t-statistic (strongest cointegration)
4. Optionally cross-validate with Johansen trace statistic

### 6. Trading Signal Generation

**Z-score signal:**
    z_t = (S_t - mu_S) / sigma_S

where mu_S and sigma_S are rolling mean and standard deviation of the spread.

**Entry/exit rules:**
- Enter long spread:  z_t < -z_entry (spread is cheap)
- Enter short spread: z_t > +z_entry (spread is rich)
- Exit position:      |z_t| < z_exit  (spread reverted)
- Stop-loss:          |z_t| > z_stop  (spread diverging)

Typical parameters: z_entry = 2.0, z_exit = 0.5, z_stop = 4.0.

---

## Project Structure

    12-pairs-trading-cointegration/
    +-- src/
    |   +-- __init__.py
    |   +-- cointegration.py      # Engle-Granger & Johansen tests
    |   +-- pair_selection.py     # Distance & cointegration pair selection
    |   +-- ornstein_uhlenbeck.py # OU process calibration & simulation
    |   +-- kalman_filter.py      # Adaptive hedge ratio estimation
    |   +-- strategy.py           # Z-score trading signals & position mgmt
    |   +-- backtesting.py        # Walk-forward pairs trading backtest
    |   +-- visualization/
    |       +-- __init__.py
    |       +-- pairs_plots.py    # Publication-quality visualizations
    +-- tests/
    |   +-- test_pairs.py
    +-- outputs/figures/
    +-- README.md
    +-- requirements.txt
    +-- setup.py

---

## Key Differentiators

- Engle-Granger and Johansen cointegration with MacKinnon critical values
- Ornstein-Uhlenbeck calibration with maximum likelihood estimation
- Kalman filter for time-varying hedge ratio (adaptive beta)
- Distance-based and cointegration-based pair selection with cross-validation
- Z-score signals with dynamic thresholds and stop-loss
- Walk-forward backtest with formation/trading period split
- Transaction costs, slippage modeling, and capacity analysis
- Comprehensive regime-aware performance attribution

---

## References

- Engle, R. & Granger, C. (1987). Co-Integration and Error Correction. Econometrica.
- Johansen, S. (1991). Estimation and Hypothesis Testing of Cointegration Vectors. Econometrica.
- MacKinnon, J. (1991). Critical Values for Cointegration Tests. Queen's Economics.
- Gatev, E., Goetzmann, W. & Rouwenhorst, K. (2006). Pairs Trading. RFS.
- Vidyamurthy, G. (2004). Pairs Trading: Quantitative Methods and Analysis. Wiley.
- Elliott, R., Van der Hoek, J. & Malcolm, W. (2005). Pairs Trading. QF.
- Avellaneda, M. & Lee, J. (2010). Statistical Arbitrage in the US Equities Market. QF.
- Hamilton, J. (1994). Time Series Analysis. Princeton University Press.
