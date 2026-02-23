# Market Microstructure Analysis

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![CQF](https://img.shields.io/badge/CQF-Certified-gold.svg)](https://www.cqf.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Author:** Jose Orlando Bobadilla Fuentes, CQF | MSc AI
**Role:** Senior Quantitative Portfolio Manager & Lead Data Scientist
**Organization:** Colfondos S.A. – Vicepresidencia de Inversiones
**Project:** 14 of 20 | Quantitative Finance Portfolio

---

## Overview

Institutional-grade market microstructure analysis toolkit covering bid-ask
spread decomposition, order flow toxicity, illiquidity measurement, and
optimal execution modeling using classic and modern microstructure theory.

---

## Theoretical Background

### 1. Bid-Ask Spread Models

**Roll (1984) Implicit Spread Estimator**
The Roll model recovers the effective spread from the serial covariance of
price changes under the assumption of a simple market-maker model:

```
s_Roll = 2 * sqrt(-Cov(Δp_t, Δp_{t-1}))
```

**Corwin-Schultz (2012) High-Low Estimator**
Uses the ratio of two-day to one-day high-low ranges:

```
S = (2*(exp(α) - 1)) / (1 + exp(α))
α = (sqrt(2β) - sqrt(β)) / (3 - 2√2) - sqrt(γ / (3 - 2√2))
```

**Realized Spread & Price Impact Decomposition**
Effective spread = Adverse selection component + Realized spread

### 2. Order Flow & Trade Classification

**Tick Rule (Lee & Ready, 1991)**
Classifies trades as buyer/seller-initiated based on price movements:
- Uptick → buy-initiated (+1)
- Downtick → sell-initiated (-1)
- Zero-tick → direction of last non-zero tick

**VPIN (Easley, Lopez de Prado & O'Hara, 2012)**
Volume-Synchronized PIN measures order flow toxicity:

```
VPIN = |V_b - V_s| / V_bucket
```

Elevated VPIN signals informed trading and predicts flash crash events.

### 3. Illiquidity Measures

**Amihud (2002) Illiquidity Ratio**
```
ILLIQ_t = (1/D) * Σ |r_{d}| / Volume_{d}
```
Price impact per unit of dollar volume traded.

**Kyle's Lambda (1985)**
Regresses price changes on signed order flow to estimate permanent impact:
```
Δp_t = α + λ * OF_t + ε_t
```

**Pastor-Stambaugh (2002) Liquidity Factor**
Signed order flow predicts next-day returns, capturing a liquidity premium.

### 4. Market Impact & Optimal Execution

**Almgren-Chriss (2001) Framework**
Minimizes expected shortfall + λ * variance of execution cost:

```
E[Cost] = γ * x² * σ² * T + η * Σ (v_k)²
Optimal trajectory: x(t) = x₀ * sinh(κ(T-t)) / sinh(κT)
```

where κ = sqrt(λ * σ² / η), balancing market impact vs. timing risk.

---

## Project Structure

```
14-market-microstructure/
├── src/
│   ├── models/
│   │   ├── spread_models.py          # Roll, Corwin-Schultz, Glosten-Harris
│   │   ├── order_flow.py             # VPIN, OFI, trade classification
│   │   ├── illiquidity.py            # Amihud, Kyle lambda, Pastor-Stambaugh
│   │   └── market_impact.py         # Almgren-Chriss optimal execution
│   ├── utils/
│   │   ├── data_loader.py            # yfinance wrapper + synthetic tick gen
│   │   └── helpers.py               # rolling stats, autocorrelation utils
│   └── visualization/
│       └── charts.py                # Dark-theme professional plots
├── tests/
│   └── test_microstructure.py       # 20+ unit tests
├── main.py                          # Full analysis pipeline
├── requirements.txt
├── setup.py
└── config/config.yaml
```

---

## Quick Start

```bash
cd ~/quant-finance-portfolio/14-market-microstructure
pip install -r requirements.txt
python main.py --tickers SPY QQQ IWM --start 2020-01-01 --end 2024-12-31
pytest tests/ -v --tb=short
```

---

## Key Outputs

| Module | Output |
|--------|--------|
| Spread Models | Roll, Corwin-Schultz time-series; effective vs quoted spread |
| Order Flow | VPIN heatmap; OFI autocorrelation; trade classification |
| Illiquidity | Amihud ratio; Kyle λ; liquidity-adjusted returns |
| Market Impact | Almgren-Chriss efficient frontier; optimal TWAP trajectory |
| Intraday | Volume profile; VWAP bands; intraday seasonality |

---

## References

- Amihud, Y. (2002). Illiquidity and stock returns. *Journal of Financial Markets*.
- Almgren, R., & Chriss, N. (2001). Optimal execution of portfolio transactions. *Journal of Risk*.
- Easley, D., Lopez de Prado, M., & O'Hara, M. (2012). Flow toxicity and liquidity. *Review of Financial Studies*.
- Kyle, A.S. (1985). Continuous auctions and insider trading. *Econometrica*.
- Roll, R. (1984). A simple implicit measure of the effective bid-ask spread. *Journal of Finance*.

---

## License

MIT License. See [LICENSE](../LICENSE) for details.
