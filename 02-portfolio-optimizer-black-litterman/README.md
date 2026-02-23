# Portfolio Optimization: Black-Litterman and Mean-CVaR

**Author:** Jose Orlando Bobadilla Fuentes | CQF
**Category:** Asset Allocation | Tier 1 - Core Quantitative Finance

---

## Theoretical Foundation

### 1. Markowitz Mean-Variance Optimization (1952)

The classical framework solves:

    min  w' * Sigma * w
    s.t. w' * mu >= mu_target, w' * 1 = 1, w >= 0

**Limitation:** Extreme sensitivity to expected return estimates.

### 2. Black-Litterman Model (1992)

Bayesian approach combining market equilibrium (prior) with investor views:

1. **Equilibrium returns:** `pi = delta * Sigma * w_mkt`
2. **Views:** `P * mu = Q + epsilon`, `epsilon ~ N(0, Omega)`
3. **Posterior:** `mu_BL = [(tau*Sigma)^{-1} + P'*Omega^{-1}*P]^{-1} * [(tau*Sigma)^{-1}*pi + P'*Omega^{-1}*Q]`

### 3. Mean-CVaR Optimization (Rockafellar & Uryasev, 2000)

CVaR measures expected loss in the worst alpha-percentile. Reformulated as LP:

    min  alpha + [1/(T*(1-beta))] * sum(z_t)
    s.t. z_t >= -(R_t' * w) - alpha, z_t >= 0, w'*1 = 1

### 4. Risk Parity

Equal risk contribution: `RC_i = w_i * (Sigma*w)_i / sigma_p` for all i.

---

## Project Structure

```
02-portfolio-optimizer-black-litterman/
├── src/models/
│   ├── markowitz.py          # Classical MVO
│   ├── black_litterman.py    # BL with views
│   ├── mean_cvar.py          # CVaR optimization via CVXPY
│   └── risk_parity.py        # Equal risk contribution
├── src/utils/
│   └── performance_metrics.py
├── src/visualization/
│   └── efficient_frontier.py
├── tests/
├── main.py
└── README.md
```

## References

- Markowitz, H. (1952). *Portfolio Selection.* Journal of Finance.
- Black, F., & Litterman, R. (1992). *Global Portfolio Optimization.* FAJ.
- Rockafellar, R.T., & Uryasev, S. (2000). *Optimization of CVaR.* JoR.
- Meucci, A. (2005). *Risk and Asset Allocation.* Springer.
- Roncalli, T. (2013). *Introduction to Risk Parity and Budgeting.* CRC Press.
