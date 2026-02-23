"""
Momentum & Mean Reversion Multi-Asset Strategy
===============================================

Quantitative trading framework combining time-series momentum (TSMOM),
cross-sectional momentum, and mean reversion signals with adaptive
regime detection and dynamic signal blending.

Modules:
    momentum        - TSMOM and cross-sectional momentum signal generators
    mean_reversion  - Z-score, RSI, and Bollinger Band mean reversion signals
    regime          - Market regime detection (volatility, dispersion, autocorrelation)
    portfolio       - Signal blending, portfolio construction, risk management
    backtesting     - Walk-forward backtesting engine with transaction costs
    data_generator  - Synthetic multi-asset universe for testing

Author: Jose Orlando Bobadilla Fuentes, CQF, MSc AI
"""

__version__ = "1.0.0"
__author__ = "Jose Orlando Bobadilla Fuentes"
