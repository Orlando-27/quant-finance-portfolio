# Monte Carlo Engine for Exotic Derivatives

**Author:** Jose Orlando Bobadilla Fuentes | CQF
**Category:** Numerical Methods | Tier 1 - Core Quantitative Finance

---

## Theoretical Foundation

### Monte Carlo for Option Pricing

Under risk-neutral measure Q:
`V = exp(-r*T) * E^Q[h({S_t})]`

Estimated by simulating N paths and averaging discounted payoffs.
Standard error: O(1/sqrt(N)), independent of dimension.

### GBM Path Simulation

`S(t+dt) = S(t) * exp((r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)`, Z ~ N(0,1)

### Exotic Options

**Asian:** Payoff on average price. Arithmetic (no closed-form) and geometric
(log-normal, used as control variate).

**Barrier:** Activates/deactivates when path crosses barrier H.
Broadie-Glasserman-Kou (1997) continuity correction for discrete monitoring.

**Lookback:** Payoff on path extremum (max or min).

### Variance Reduction

**Antithetic Variates:** Pair each path Z with -Z. Reduces variance when
payoff is monotone.

**Control Variates:** Use geometric Asian (known E[C]) as control:
`V_cv = V_arith - beta * (V_geo_mc - V_geo_exact)`

---

## Project Structure

```
03-monte-carlo-exotic-derivatives/
├── src/models/
│   ├── path_generator.py      # GBM paths (vectorized)
│   ├── asian_options.py       # Arithmetic/geometric, CV, antithetic
│   ├── barrier_options.py     # 4 types + continuity correction
│   └── lookback_options.py    # Floating/fixed strike
├── src/visualization/
│   └── convergence_plots.py   # Convergence and sample paths
├── tests/
├── main.py
└── README.md
```

## References

- Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering.* Springer.
- Broadie, M., Glasserman, P., & Kou, S.G. (1997). *Continuity Correction for Discrete Barrier Options.* MF.
- Kemna, A.G.Z., & Vorst, A.C.F. (1990). *Pricing Method Based on Average Asset Values.* JBF.
