# Stochastic Processes Simulator and Visualizer

**Author:** Jose Orlando Bobadilla Fuentes | CQF
**Category:** Stochastic Calculus | Tier 1 - Core Quantitative Finance

---

## Theoretical Foundation

### 1. Standard Brownian Motion (Wiener Process)

W(t) has: W(0)=0, independent increments, W(t)-W(s) ~ N(0, t-s), continuous paths.

### 2. Geometric Brownian Motion (GBM)

`dS = mu*S*dt + sigma*S*dW`
Solution: `S(t) = S(0)*exp((mu - sigma^2/2)*t + sigma*W(t))`
Application: Black-Scholes stock price model.

### 3. Ornstein-Uhlenbeck (OU) Process

`dX = kappa*(theta - X)*dt + sigma*dW`
Mean-reverting. Application: Vasicek interest rate model, pairs trading.

### 4. Cox-Ingersoll-Ross (CIR) Process

`dX = kappa*(theta - X)*dt + sigma*sqrt(X)*dW`
Non-negative, mean-reverting. Application: interest rates, volatility modeling.
Feller condition: 2*kappa*theta >= sigma^2 ensures positivity.

### 5. Heston Stochastic Volatility

`dS = mu*S*dt + sqrt(v)*S*dW_1`
`dv = kappa*(theta - v)*dt + xi*sqrt(v)*dW_2`
`dW_1*dW_2 = rho*dt`
Application: options pricing with volatility smile.

### 6. Merton Jump-Diffusion

`dS/S = (mu - lambda*k)*dt + sigma*dW + J*dN`
N(t) is Poisson process, J ~ N(mu_J, sigma_J).
Application: modeling sudden market shocks.

---

## References

- Shreve, S. (2004). *Stochastic Calculus for Finance II.* Springer.
- Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering.* Springer.
- Heston, S. (1993). *A Closed-Form Solution for Options with Stochastic Volatility.* RFS.
