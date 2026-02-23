# Project 20: Financial Modeling Templates

**Author:** Jose Orlando Bobadilla Fuentes
**Credentials:** CQF | MSc Artificial Intelligence
**Role:** Senior Quantitative Portfolio Manager & Lead Data Scientist
**Institution:** Colombian Pension Fund -- Vicepresidencia de Inversiones

---

## Overview

A production-grade financial modeling toolkit implementing the six core
valuation and analysis frameworks used in investment banking, private
equity, and corporate finance. Each module is a self-contained, fully
documented implementation with institutional-quality visualizations.

This is the **20th and final project** of a comprehensive quantitative
finance GitHub portfolio demonstrating expertise across derivatives
pricing, portfolio optimization, machine learning, risk management,
and financial modeling.

---

## Modules

| # | Module | Description | Figures |
|---|--------|-------------|---------|
| 1 | **DCF Valuation Engine** | WACC, FCF projection, Gordon Growth & Exit Multiple terminal value, EV-to-equity bridge, sensitivity heatmaps, football field | 6 |
| 2 | **LBO Model** | Sources & Uses, multi-tranche debt schedule with cash sweep, exit valuation, IRR/MOIC, value creation bridge, returns by hold period | 6 |
| 3 | **Three-Statement Model** | Fully-linked IS/BS/CF with circular reference solver (interest-debt-cash loop), revolver auto-draw, PP&E roll-forward, key ratios dashboard | 6 |
| 4 | **Comparable Analysis** | Trading comps (10 peers, 8 multiples), precedent transactions with time-decay weighting, regression-based valuation, bootstrap confidence intervals | 6 |
| 5 | **Merger Model** | Accretion/dilution analysis, purchase price allocation, synergy phase-in, contribution analysis, breakeven synergy, EPS bridge waterfall | 6 |
| 6 | **Sensitivity & Scenario** | Tornado charts, two-way data tables, 50,000-trial Monte Carlo with correlated inputs (Cholesky), scenario manager with probability weighting, spider chart | 8 |

**Total: 38 publication-quality figures**

---

## Technical Highlights

- **Circular Reference Solver**: The three-statement model resolves the
  interest-debt-cash circular dependency via fixed-point iteration,
  converging within 5-10 iterations to $1 tolerance.

- **Correlated Monte Carlo**: The sensitivity module uses Cholesky
  decomposition of a 7x7 correlation matrix to generate realistic
  joint distributions of model inputs, producing full valuation
  distributions with VaR/CVaR risk metrics.

- **Time-Decay Weighted Comps**: Transaction comparables are weighted
  by recency using exponential decay, reflecting the greater relevance
  of recent deal pricing.

- **Bootstrap Confidence Intervals**: 10,000-iteration bootstrap on
  peer multiples provides statistically rigorous valuation ranges
  beyond simple mean/median analysis.

---

## Directory Structure

```
20_financial_modeling_templates/
    src/
        common/
            __init__.py
            style.py                 # Dark-theme config, watermark, helpers
            finance_utils.py         # Core financial calculations
        dcf_valuation.py             # Module 1: DCF Engine
        lbo_model.py                 # Module 2: LBO Model
        three_statement_model.py     # Module 3: 3-Statement Model
        comparable_analysis.py       # Module 4: Comparable Analysis
        merger_model.py              # Module 5: Merger Model
        sensitivity_analysis.py      # Module 6: Scenario Analysis
        __init__.py
    outputs/
        figures/
            dcf/                     # 6 DCF figures
            lbo/                     # 6 LBO figures
            ts/                      # 6 Three-Statement figures
            comps/                   # 6 Comparable Analysis figures
            merger/                  # 6 Merger Model figures
            scenario/                # 8 Scenario Analysis figures
        reports/
    tests/
        __init__.py
        test_finance_utils.py        # Unit tests (30+ test cases)
        test_dcf_integration.py      # Integration tests
    docs/
    notebooks/
    run_all.sh                       # Execute all 6 modules
    requirements.txt
    README.md
```

---

## Quick Start

```bash
# Clone and navigate
cd 20_financial_modeling_templates

# Install dependencies
pip install -r requirements.txt

# Run individual modules
python src/dcf_valuation.py
python src/lbo_model.py
python src/three_statement_model.py
python src/comparable_analysis.py
python src/merger_model.py
python src/sensitivity_analysis.py

# Or run everything at once
bash run_all.sh

# Run tests
pytest tests/ -v
```

---

## Requirements

- Python 3.10+
- numpy, scipy, pandas
- matplotlib, seaborn
- yfinance (optional, for live data)
- cvxpy (optional, for optimization extensions)
- pytest (testing)

All figures use the `Agg` backend for headless/Cloud Shell execution.

---

## Visualization Standards

All figures follow a consistent institutional aesthetic:

- **Background**: #0a0a0a (near-black)
- **Axes**: #111111
- **Grid**: #1a1a1a, alpha 0.3
- **Text**: #e0e0e0
- **Palette**: Blues, greens, ambers -- designed for dark backgrounds
- **Watermark**: "Jose O. Bobadilla | CQF"
- **DPI**: 150 (publication quality)

---

## References

- Damodaran, A. (2012). *Investment Valuation*, 3rd ed., Wiley.
- Koller, T., Goedhart, M., Wessels, D. (2020). *Valuation*, 7th ed., McKinsey.
- Rosenbaum, J. & Pearl, J. (2020). *Investment Banking*, 3rd ed., Wiley.
- Benninga, S. (2014). *Financial Modeling*, 4th ed., MIT Press.
- Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*, Springer.
- Berk, J. & DeMarzo, P. (2019). *Corporate Finance*, 5th ed., Pearson.

---

## License

MIT License. See repository root for details.

---

*Project 20 of 20 -- Portfolio Complete.*
