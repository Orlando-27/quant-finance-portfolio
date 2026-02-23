"""
Pairs Trading Strategy with Cointegration Analysis
===================================================

Production-grade statistical arbitrage framework combining cointegration
theory with adaptive estimation techniques for systematic pairs trading.

Modules:
    cointegration      - Engle-Granger & Johansen cointegration tests
    pair_selection     - Distance & cointegration-based pair identification
    ornstein_uhlenbeck - OU process calibration, half-life, simulation
    kalman_filter      - Adaptive hedge ratio via Kalman filter
    strategy           - Z-score signal generation & position management
    backtesting        - Walk-forward pairs trading backtest engine

Author: Jose Orlando Bobadilla Fuentes, CQF, MSc AI
"""

from src.cointegration import EngleGranger, JohansenTest
from src.pair_selection import PairSelector
from src.ornstein_uhlenbeck import OrnsteinUhlenbeck
from src.kalman_filter import KalmanHedgeRatio
from src.strategy import PairsTradingStrategy
from src.backtesting import PairsBacktester

__version__ = "1.0.0"
__author__ = "Jose Orlando Bobadilla Fuentes"
