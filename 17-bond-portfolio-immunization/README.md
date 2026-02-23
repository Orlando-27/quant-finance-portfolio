# Bond Portfolio Immunization

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CQF](https://img.shields.io/badge/CQF-Quantitative%20Finance-darkgreen.svg)](https://www.cqf.com/)

**Production-grade fixed-income immunization framework** implementing classical duration matching, cash flow matching, and key rate duration analysis to protect bond portfolios against interest rate risk.

---

## Theoretical Foundation

### 1. Bond Pricing

The price of a fixed-coupon bond paying coupon $C$ semi-annually with face $F$ and $n$ periods is:

$$P = \sum_{t=1}^{n} \frac{C/2}{(1 + y/2)^t} + \frac{F}{(1 + y/2)^n}$$

For continuous compounding with yield $y$:

$$P = \sum_{t} CF_t \cdot e^{-y \cdot t}$$

### 2. Duration & Convexity

**Macaulay Duration** — the weighted-average time to cash flows:

$$D_{Mac} = \frac{\sum_t t \cdot PV(CF_t)}{P}$$

**Modified Duration** — price sensitivity to yield:

$$D_{mod} = \frac{D_{Mac}}{1 + y/m} \approx -\frac{1}{P}\frac{dP}{dy}$$

**Dollar Duration (DV01)**:

$$DV01 = D_{mod} \cdot P \cdot 0.0001$$

**Convexity** — second-order sensitivity:

$$C = \frac{1}{P} \frac{d^2P}{dy^2} = \frac{\sum_t t(t+1) \cdot PV(CF_t)}{P \cdot (1+y/m)^2}$$

**Full price approximation** for yield shock $\Delta y$:

$$\frac{\Delta P}{P} \approx -D_{mod} \cdot \Delta y + \frac{1}{2} C \cdot (\Delta y)^2$$

### 3. Redington Immunization (Classical)

A bond portfolio is immunized against small parallel yield shifts if and only if:

1. **PV matching**: $PV_{assets} = PV_{liabilities}$
2. **Duration matching**: $D_{assets} = D_{liabilities}$
3. **Convexity condition**: $C_{assets} \geq C_{liabilities}$ (ensures surplus protection)

### 4. Cash Flow Matching (Dedication)

Given liability cash flows $\{L_t\}$, find a bond portfolio with holdings $\{x_j\}$ such that:

$$\sum_j x_j \cdot CF_{j,t} \geq L_t \quad \forall t$$

This is a linear program. The dedication strategy eliminates reinvestment risk entirely.

### 5. Key Rate Durations

Key Rate Durations (KRD) measure sensitivity to movements at specific maturities $\tau_k$ on the yield curve:

$$KRD_k = -\frac{1}{P}\frac{\partial P}{\partial y_k}$$

where $y_k$ is the spot rate at key maturity $\tau_k$. The sum satisfies:

$$\sum_k KRD_k = D_{mod}$$

KRDs enable hedging of non-parallel curve shifts (steepening, flattening, butterfly).

### 6. Portfolio Structures

| Structure | Description | Best For |
|-----------|-------------|----------|
| **Bullet** | Bonds concentrated near liability date | Predictable single liability |
| **Barbell** | Short & long bonds, low intermediate | High convexity, active mgmt |
| **Ladder** | Equal maturities across horizon | Reinvestment risk diversification |
| **Matched** | Optimized to liability cash flows | Pension/insurance dedication |

---

## Project Structure

```
16-bond-portfolio-immunization/
├── src/
│   ├── bond.py               # Bond pricing engine (price, YTM, duration, convexity)
│   ├── immunization.py       # Redington + cash flow matching optimizer
│   ├── key_rate_duration.py  # KRD / DV01 by tenor bucket
│   ├── portfolio.py          # Portfolio aggregation & scenario analysis
│   └── visualization.py     # Dark-theme publication charts
├── tests/
│   ├── test_bond.py
│   ├── test_immunization.py
│   └── test_krd.py
├── data/                     # Bond universe CSVs
├── notebooks/                # Interactive exploration
├── outputs/
│   ├── figures/              # PNG charts (dark theme)
│   └── reports/              # Summary JSON/CSV
├── config/
│   └── settings.yaml
├── main.py                   # Full pipeline entry point
├── requirements.txt
└── README.md
```

---

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

---

## Key Results

- Immunized portfolios against ±300 bps parallel yield shifts
- KRD analysis across 8 curve tenors (1m, 3m, 6m, 1y, 2y, 5y, 10y, 30y)
- Cash flow matching LP achieves liability dedication at minimum cost
- Convexity advantage of barbell over bullet quantified

---

## Author

**Jose Orlando Bobadilla Fuentes**  
CQF (Certificate in Quantitative Finance) | MSc Artificial Intelligence  
Senior Quantitative Portfolio Manager & Lead Data Scientist  
