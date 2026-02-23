"""
Volatility Surface & SABR Calibration
======================================
Framework for implied volatility surface construction, SABR/SVI calibration,
local volatility extraction, and arbitrage-free interpolation.

Author: Jose Orlando Bobadilla Fuentes, CQF
"""

from src.implied_vol import (
    implied_vol_newton, implied_vol_brent, implied_vol_rational,
    black_scholes_price, black_price, extract_iv_surface,
)
from src.sabr import SABRModel, SABRParams, SABRCalibrationResult
from src.svi import SVIModel, SVIParams, SVIFitResult
from src.surface import VolSurface, SliceData, SurfaceDiagnostics
from src.local_vol import LocalVolSurface
from src.vanna_volga import VannaVolga, VannaVolgaQuotes
from src.arbitrage import ArbitrageChecker, ArbitrageDiagnostics

__version__ = "1.0.0"
__author__ = "Jose Orlando Bobadilla Fuentes"

__all__ = [
    "implied_vol_newton", "implied_vol_brent", "implied_vol_rational",
    "black_scholes_price", "black_price", "extract_iv_surface",
    "SABRModel", "SABRParams", "SABRCalibrationResult",
    "SVIModel", "SVIParams", "SVIFitResult",
    "VolSurface", "SliceData", "SurfaceDiagnostics",
    "LocalVolSurface",
    "VannaVolga", "VannaVolgaQuotes",
    "ArbitrageChecker", "ArbitrageDiagnostics",
]
