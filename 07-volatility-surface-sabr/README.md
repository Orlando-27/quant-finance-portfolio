# Volatility Surface & SABR Calibration

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CQF](https://img.shields.io/badge/CQF-Quantitative%20Finance-darkgreen.svg)](https://www.cqf.com/)

**Production-grade framework for implied volatility surface construction, SABR model calibration, and volatility smile analytics with arbitrage-free interpolation.**

---

## Theoretical Foundation

### 1. Implied Volatility and the Smile

The Black-Scholes model assumes constant volatility, yet market prices reveal a **volatility smile** -- implied volatility varies with strike and maturity. For a European call with market price $C^{mkt}$, the implied volatility $\sigma_{imp}$ solves:

$$C^{BS}(S, K, T, r, \sigma_{imp}) = C^{mkt}$$

This inversion is performed numerically (Newton-Raphson or Brent's method). The resulting surface $\sigma_{imp}(K, T)$ encodes the market's risk-neutral expectations about the distribution of future returns.

### 2. Smile Phenomenology

Equity markets typically exhibit a **volatility skew**: OTM puts trade at higher implied volatility than ATM options due to crash risk premium. FX markets show a symmetric **smile**. The shape is characterized by:

- **Level**: ATM volatility $\sigma_{ATM}$
- **Skew**: $\partial \sigma / \partial K$ at ATM (risk reversal)
- **Curvature**: $\partial^2 \sigma / \partial K^2$ at ATM (butterfly spread)
- **Term structure**: $\sigma_{ATM}(T)$ across maturities

### 3. SABR Model (Hagan et al., 2002)

The SABR (Stochastic Alpha Beta Rho) model is the industry standard for smile interpolation in fixed income and FX:

$$dF_t = \hat{\sigma}_t F_t^\beta \, dW_t^1$$
$$d\hat{\sigma}_t = \alpha \hat{\sigma}_t \, dW_t^2$$
$$dW_t^1 \cdot dW_t^2 = \rho \, dt$$

where:
- $F_t$ is the forward price
- $\hat{\sigma}_t$ is the stochastic volatility
- $\beta \in [0, 1]$ controls the backbone (CEV exponent)
- $\alpha$ is the vol-of-vol
- $\rho \in (-1, 1)$ is the correlation between forward and vol

### 4. SVI Parametrization (Gatheral, 2004)

The raw SVI parametrizes total implied variance as:

$$w(k) = a + b\left(\rho(k - m) + \sqrt{(k - m)^2 + \sigma^2}\right)$$

where $k = \ln(K/F)$ is log-moneyness and $(a, b, \rho, m, \sigma)$ are the 5 parameters.

### 5. Arbitrage Constraints

A valid volatility surface must satisfy:
- **No calendar spread arbitrage**: total variance $w(k, T) = \sigma^2(k, T) \cdot T$ must be non-decreasing in $T$
- **No butterfly arbitrage**: the density $\frac{\partial^2 C}{\partial K^2} \geq 0$ must be non-negative
- **Call spread constraint**: $-1 \leq \frac{\partial C}{\partial K} \leq 0$

---

## Project Structure

```
vol-surface-sabr/
├── src/
│   ├── __init__.py
│   ├── implied_vol.py          # IV extraction: Newton-Raphson, Brent, Jaeckel
│   ├── sabr.py                 # SABR model: Hagan, Obloj, calibration
│   ├── svi.py                  # SVI parametrization & arbitrage-free fit
│   ├── surface.py              # Vol surface construction & interpolation
│   ├── local_vol.py            # Dupire local volatility
│   ├── vanna_volga.py          # Vanna-Volga pricing & interpolation
│   ├── arbitrage.py            # Arbitrage diagnostics & constraints
│   └── visualization.py        # 3D surfaces, smile plots, term structure
├── tests/
│   └── test_vol_surface.py     # Unit tests
├── README.md
├── requirements.txt
├── setup.py
└── .gitignore
```

---

## Quick Start

```python
from src.sabr import SABRModel
from src.surface import VolSurface
from src.svi import SVIModel

# SABR calibration to market quotes
sabr = SABRModel(beta=0.5)
params = sabr.calibrate(
    forward=100.0,
    strikes=[85, 90, 95, 100, 105, 110, 115],
    market_vols=[0.25, 0.22, 0.20, 0.18, 0.19, 0.21, 0.24],
    expiry=0.5
)
print(f"alpha={params.params.alpha:.4f}, rho={params.params.rho:.4f}, nu={params.params.nu:.4f}")
```

---

## References

- Hagan, P.S. et al. (2002). *Managing Smile Risk*. Wilmott Magazine.
- Obloj, J. (2008). *Fine-Tune Your Smile: Correction to Hagan et al.*
- Gatheral, J. (2004). *A Parsimonious Arbitrage-Free Implied Volatility Parametrization*.
- Gatheral, J. & Jacquier, A. (2014). *Arbitrage-Free SVI Volatility Surfaces*.
- Dupire, B. (1994). *Pricing with a Smile*. Risk Magazine.
- Castagna, A. & Mercurio, F. (2007). *The Vanna-Volga Method for Implied Volatilities*.
- Jaeckel, P. (2015). *Let's Be Rational*. Wilmott Magazine.

---

## Author

**Jose Orlando Bobadilla Fuentes**
CQF | Senior Quantitative Portfolio Manager
[LinkedIn](https://www.linkedin.com/in/jose-orlando-bobadilla-fuentes-aa418a116) | [GitHub](https://github.com/joseorlandobf)
