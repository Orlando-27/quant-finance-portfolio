# Credit Risk Modeling

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CQF](https://img.shields.io/badge/CQF-Quantitative%20Finance-darkgreen.svg)](https://www.cqf.com/)

**Production-grade credit risk framework implementing structural models (Merton),
reduced-form models (Jarrow-Turnbull), portfolio credit risk (Vasicek, CreditMetrics),
CDS pricing, and Credit VaR with Monte Carlo simulation.**

---

## Theoretical Foundation

### 1. Structural Models: Merton (1974)

The Merton model treats a firm's equity as a European call option on its assets.
If firm value V follows geometric Brownian motion:

    dV = mu * V * dt + sigma_V * V * dW

Then equity E at maturity T with debt face value D is:

    E = V * N(d1) - D * exp(-r*T) * N(d2)

where:

    d1 = [ln(V/D) + (r + sigma_V^2 / 2) * T] / (sigma_V * sqrt(T))
    d2 = d1 - sigma_V * sqrt(T)

The probability of default (PD) under the risk-neutral measure is N(-d2),
and under the physical measure is N(-d2_phys) where d2 uses mu instead of r.

The distance to default (DD) measures how many standard deviations the firm
is from the default point:

    DD = [ln(V/D) + (mu - sigma_V^2 / 2) * T] / (sigma_V * sqrt(T))

### 2. Reduced-Form Models: Jarrow-Turnbull (1995)

Unlike structural models, reduced-form models treat default as an exogenous
Poisson process with intensity lambda(t). The survival probability is:

    Q(t, T) = exp(-integral from t to T of lambda(s) ds)

For constant intensity:

    Q(t, T) = exp(-lambda * (T - t))

The price of a defaultable zero-coupon bond is:

    P_risky(t, T) = P_riskfree(t, T) * [Q(t,T) + (1-Q(t,T)) * R]

where R is the recovery rate. The hazard rate can be bootstrapped from
observed credit spreads:

    lambda = spread / (1 - R)

### 3. Portfolio Credit Risk: Vasicek Single-Factor Model

The Vasicek model assumes each obligor's asset return X_i follows:

    X_i = sqrt(rho) * Z + sqrt(1-rho) * epsilon_i

where Z ~ N(0,1) is the systematic factor, epsilon_i ~ N(0,1) is
idiosyncratic risk, and rho is the asset correlation.

Conditional on Z, defaults are independent with conditional PD:

    PD(Z) = N( (N_inv(PD) - sqrt(rho) * Z) / sqrt(1-rho) )

The Vasicek large-portfolio loss distribution has the analytical form:

    P(L <= x) = N( (sqrt(1-rho) * N_inv(x) - N_inv(PD)) / sqrt(rho) )

This is the theoretical basis for the Basel II/III IRB formula.

### 4. CDS Pricing

A Credit Default Swap is priced by equating the premium leg to the
protection leg:

    Premium Leg = s * sum_{i=1}^{N} Delta_i * P(0,t_i) * Q(0,t_i)
    Protection Leg = (1-R) * sum_{i=1}^{N} P(0,t_i) * [Q(0,t_{i-1}) - Q(0,t_i)]

The fair CDS spread s* equates these two legs:

    s* = Protection Leg / Risky Annuity

### 5. CreditMetrics Framework

CreditMetrics estimates portfolio credit risk by:
1. Simulating correlated asset returns via Cholesky decomposition
2. Mapping returns to rating transitions using migration thresholds
3. Revaluing each position under the new rating
4. Aggregating to compute portfolio loss distribution

### 6. Credit VaR

Credit VaR at confidence alpha is:

    Credit_VaR_alpha = Quantile_alpha(Loss) - Expected_Loss

This captures unexpected loss beyond what is provisioned for (expected loss).

---

## Project Structure

    credit-risk-modeling/
    |-- main.py                    # Full pipeline demonstration
    |-- src/
    |   |-- __init__.py
    |   |-- models/
    |   |   |-- __init__.py
    |   |   |-- merton.py          # Merton structural model
    |   |   |-- reduced_form.py    # Jarrow-Turnbull hazard rate model
    |   |   |-- vasicek.py         # Vasicek single-factor portfolio model
    |   |   |-- creditmetrics.py   # CreditMetrics migration framework
    |   |   |-- cds_pricing.py     # CDS spread and valuation
    |   |-- credit_var.py          # Credit VaR Monte Carlo engine
    |   |-- utils.py               # Rating data, helpers, visualization
    |-- tests/
    |   |-- test_credit_risk.py    # Comprehensive unit tests
    |-- requirements.txt
    |-- setup.py
    |-- .gitignore

---

## Key Features

- Merton model with equity-implied calibration (iterative solver for V and sigma_V)
- Hazard rate bootstrapping from CDS term structure
- Vasicek analytical loss distribution with Basel IRB formula
- CreditMetrics with Cholesky-correlated migration simulation
- Full CDS pricing (par spread, mark-to-market, DV01)
- Monte Carlo Credit VaR with importance sampling
- Transition matrix generation and validation

---

## References

- Merton, R. (1974). On the Pricing of Corporate Debt. Journal of Finance.
- Jarrow, R. & Turnbull, S. (1995). Pricing Derivatives on Financial Securities
  Subject to Credit Risk. Journal of Finance.
- Vasicek, O. (2002). The Distribution of Loan Portfolio Value. Risk.
- Gupton, G., Finger, C., & Bhatia, M. (1997). CreditMetrics Technical Document.
- Hull, J. (2018). Options, Futures, and Other Derivatives, Ch. 24-25.
- Bluhm, C., Overbeck, L., & Wagner, C. (2010). Introduction to Credit Risk Modeling.
- Basel Committee (2006). International Convergence of Capital Measurement
  and Capital Standards (Basel II IRB Approach).

---

## Author

**Jose Orlando Bobadilla Fuentes, CQF**
Senior Quantitative Portfolio Manager & Lead Data Scientist
