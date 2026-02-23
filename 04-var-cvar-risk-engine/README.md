# Value at Risk and CVaR Multi-Method Risk Engine

**Author:** Jose Orlando Bobadilla Fuentes | CQF
**Category:** Risk Management | Tier 1 - Core Quantitative Finance

---

## Theoretical Foundation

### Value at Risk (VaR)

VaR at confidence level alpha is the loss threshold such that the probability
of exceeding it is (1 - alpha):

    P(L > VaR_alpha) = 1 - alpha

### Methods Implemented

**1. Historical Simulation:**
Non-parametric. Uses empirical distribution of past returns.
`VaR = -percentile(returns, (1-alpha)*100)`

**2. Parametric (Variance-Covariance):**
Assumes normal distribution: `VaR = -mu + z_alpha * sigma`
Extended with Cornish-Fisher expansion for non-normal returns.

**3. Monte Carlo Simulation:**
Generates future scenarios from fitted distribution (normal or GARCH).

**4. GARCH(1,1) Conditional VaR:**
Captures volatility clustering:
`sigma_t^2 = omega + alpha*epsilon_{t-1}^2 + beta*sigma_{t-1}^2`

### Conditional Value at Risk (CVaR / Expected Shortfall)

`CVaR_alpha = E[L | L > VaR_alpha]`

CVaR is a coherent risk measure (subadditive), unlike VaR.

### Backtesting

**Kupiec (1995) POF Test:** Tests if VaR exceedance rate matches expected.
LR = -2*ln[(1-p)^(T-x) * p^x] + 2*ln[(1-x/T)^(T-x) * (x/T)^x]

**Christoffersen (1998) Independence Test:** Tests if violations are independent.

---

## References

- Jorion, P. (2007). *Value at Risk.* McGraw-Hill.
- McNeil, A., Frey, R., & Embrechts, P. (2015). *Quantitative Risk Management.* Princeton.
- Kupiec, P. (1995). Techniques for Verifying the Accuracy of Risk Models. JoD.
- Christoffersen, P. (1998). Evaluating Interval Forecasts. IER.
