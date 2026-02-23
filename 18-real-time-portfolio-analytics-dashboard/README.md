# Project 18 — Real-Time Portfolio Analytics Dashboard

**Author:** Jose Orlando Bobadilla Fuentes | CQF | MSc AI
**Institution:** Colombian Pension Fund — Investment Division

## Overview
Production-grade Streamlit dashboard for real-time portfolio risk analytics,
performance attribution, and alert management.

## Features
- Live market data via yfinance
- Risk metrics: VaR, CVaR, Sharpe, Sortino, Calmar, Max Drawdown
- Brinson-Hood-Beebower performance attribution
- Alert engine: VaR breach, drawdown, volatility spike
- Dark-theme Plotly charts with professional watermark
- Multi-asset correlation analysis

## Usage
```bash
pip install -r requirements.txt
streamlit run app.py
# Cloud Shell: Web Preview -> Port 8501
```

## Tests
```bash
pytest tests/ -v
```

## Architecture
```
src/
  data/fetcher.py          Market data (yfinance)
  analytics/risk.py        Risk metrics engine
  analytics/attribution.py BHB attribution
  alerts/manager.py        Alert system
  visualization/charts.py  Plotly dark-theme charts
app.py                     Streamlit main app
```
