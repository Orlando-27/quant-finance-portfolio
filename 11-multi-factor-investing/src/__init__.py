"""
Multi-Factor Investing with Machine Learning
=============================================

Production-grade quantitative framework for systematic factor investing.
Combines classical financial economics (Fama-French, Fama-MacBeth, Barra)
with modern machine learning (gradient boosting, HMM regime detection).

Modules:
    factors          - Factor construction and Fama-French 3/5 replication
    cross_sectional  - Fama-MacBeth two-pass regression with Shanken correction
    risk_model       - Barra-style factor + specific risk decomposition
    ml_timing        - ML factor timing with walk-forward validation
    portfolio        - Factor portfolio construction and optimization
    backtesting      - Walk-forward backtesting engine with transaction costs

Author: Jose Orlando Bobadilla Fuentes, CQF, MSc AI
"""

from src.factors import FactorConstructor, FamaFrenchReplicator
from src.cross_sectional import FamaMacBeth
from src.risk_model import BarraRiskModel
from src.ml_timing import FactorTimingML
from src.portfolio import FactorPortfolio
from src.backtesting import FactorBacktester

__version__ = "1.0.0"
__author__ = "Jose Orlando Bobadilla Fuentes"
