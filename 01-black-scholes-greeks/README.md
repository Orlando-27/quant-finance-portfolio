# Black-Scholes Options Pricing and Greeks Engine

**Author:** Jose Orlando Bobadilla Fuentes | CQF
**Category:** Derivatives Pricing | Tier 1 - Core Quantitative Finance

---

## Theoretical Foundation

### The Black-Scholes-Merton Framework

The Black-Scholes model (1973) provides a closed-form solution for European
option pricing under the following assumptions:

1. The underlying asset follows a geometric Brownian motion (GBM):
   `dS = mu * S * dt + sigma * S * dW`
2. No arbitrage opportunities exist in the market
3. Continuous trading with no transaction costs
4. Constant risk-free rate and volatility over the option's life
5. The underlying pays no dividends (extended by Merton for continuous dividends)

### Closed-Form Solutions

**European Call:**
`C = S * exp(-q*T) * N(d1) - K * exp(-r*T) * N(d2)`

**European Put:**
`P = K * exp(-r*T) * N(-d2) - S * exp(-q*T) * N(-d1)`

Where:
- `d1 = [ln(S/K) + (r - q + 0.5*sigma^2)*T] / (sigma * sqrt(T))`
- `d2 = d1 - sigma * sqrt(T)`
- `N(.)` is the standard normal CDF
- `q` is the continuous dividend yield

### The Greeks

| Greek | Definition | Formula (Call) |
|-------|-----------|----------------|
| Delta | dC/dS | exp(-qT) * N(d1) |
| Gamma | d2C/dS2 | exp(-qT) * n(d1) / (S * sigma * sqrt(T)) |
| Vega | dC/dsigma | S * exp(-qT) * n(d1) * sqrt(T) |
| Theta | dC/dT | -(S*sigma*exp(-qT)*n(d1))/(2*sqrt(T)) - r*K*exp(-rT)*N(d2) + q*S*exp(-qT)*N(d1) |
| Rho | dC/dr | K * T * exp(-rT) * N(d2) |

### Implied Volatility

Implied volatility is extracted by inverting the BS formula using:
1. **Newton-Raphson** with Vega as the derivative (primary solver)
2. **Brent's method** as a robust fallback
3. **Brenner-Subrahmanyam** approximation for initial guess

### American Options (CRR Binomial Tree)

For American options with early exercise, we implement the Cox-Ross-Rubinstein
(1979) binomial tree with Richardson extrapolation for convergence acceleration.

---

## Project Structure

```
01-black-scholes-greeks/
├── src/
│   ├── models/
│   │   ├── black_scholes.py      # Core BS pricing engine (OOP)
│   │   ├── greeks.py             # Analytical Greeks calculator
│   │   ├── implied_volatility.py # IV solver (Newton-Raphson + Brent)
│   │   └── binomial_tree.py      # CRR American options pricer
│   ├── utils/
│   │   └── market_data.py        # Yahoo Finance data acquisition
│   └── visualization/
│       ├── surface_plots.py      # 3D volatility surface renderer
│       └── greeks_charts.py      # Greeks sensitivity charts
├── tests/
│   └── test_black_scholes.py
├── main.py                        # CLI entry point
├── requirements.txt
├── setup.py
└── README.md
```

---

## Quick Start

```bash
cd 01-black-scholes-greeks
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python main.py
```

---

## References

- Black, F., & Scholes, M. (1973). *The Pricing of Options and Corporate Liabilities.* JPE.
- Cox, J. C., Ross, S. A., & Rubinstein, M. (1979). *Option Pricing: A Simplified Approach.* JFE.
- Hull, J. C. (2022). *Options, Futures, and Other Derivatives.* 11th Ed. Pearson.
- Jaeckel, P. (2015). *Let's Be Rational.* Wilmott Magazine.
- Wilmott, P. (2006). *Paul Wilmott on Quantitative Finance.* 2nd Ed. Wiley.
