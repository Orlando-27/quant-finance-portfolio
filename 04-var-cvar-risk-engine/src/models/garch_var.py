"""
GARCH(1,1) Conditional VaR
==============================

Captures volatility clustering via:
    sigma_t^2 = omega + alpha * epsilon_{t-1}^2 + beta * sigma_{t-1}^2

Conditional VaR uses the time-varying volatility estimate for more
responsive risk measurement during volatile periods.

Author: Jose Orlando Bobadilla Fuentes | CQF
"""

import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import norm
from dataclasses import dataclass
from typing import Optional


@dataclass
class GARCHVaRResult:
    var_series: pd.Series
    cvar_series: pd.Series
    conditional_vol: pd.Series
    params: dict
    confidence: float


class GARCHVaRModel:
    """
    GARCH(1,1)-based conditional VaR.

    Fits GARCH(1,1) to returns, then computes time-varying VaR using
    the conditional volatility forecast.

    Usage:
        >>> model = GARCHVaRModel(returns_series)
        >>> result = model.fit_and_compute(confidence=0.99)
    """

    def __init__(self, returns: pd.Series, dist: str = "normal"):
        """
        Parameters:
            returns: Pandas Series of daily returns (with DatetimeIndex)
            dist: Error distribution: 'normal', 't', or 'skewt'
        """
        self.returns = returns
        self.dist = dist
        self.model = None
        self.fit_result = None

    def fit(self, p: int = 1, q: int = 1):
        """
        Fit GARCH(p,q) model.

        The arch library parameterization:
            r_t = mu + epsilon_t
            epsilon_t = sigma_t * z_t, z_t ~ D(0,1)
            sigma_t^2 = omega + alpha_1*epsilon_{t-1}^2 + beta_1*sigma_{t-1}^2
        """
        self.model = arch_model(
            self.returns * 100,  # arch expects percentage returns
            mean="Constant", vol="Garch", p=p, q=q, dist=self.dist)
        self.fit_result = self.model.fit(disp="off")
        return self.fit_result

    def conditional_var(self, confidence: float = 0.99,
                        portfolio_value: float = 1e6) -> GARCHVaRResult:
        """
        Compute conditional VaR from GARCH volatility forecasts.

        VaR_t = -(mu + z_alpha * sigma_t)
        where sigma_t is the GARCH conditional standard deviation.
        """
        if self.fit_result is None:
            self.fit()

        cond_vol = self.fit_result.conditional_volatility / 100  # back to decimal
        mu = self.fit_result.params.get("mu", 0) / 100

        z = norm.ppf(1 - confidence)
        var_series = -(mu + z * cond_vol) * portfolio_value
        cvar_factor = norm.pdf(z) / (1 - confidence)
        cvar_series = (-mu + cvar_factor * cond_vol) * portfolio_value

        params = {
            "omega": self.fit_result.params.get("omega", None),
            "alpha[1]": self.fit_result.params.get("alpha[1]", None),
            "beta[1]": self.fit_result.params.get("beta[1]", None),
            "persistence": (self.fit_result.params.get("alpha[1]", 0)
                           + self.fit_result.params.get("beta[1]", 0)),
            "log_likelihood": self.fit_result.loglikelihood,
            "AIC": self.fit_result.aic,
            "BIC": self.fit_result.bic,
        }

        return GARCHVaRResult(
            var_series=pd.Series(var_series, index=self.returns.index),
            cvar_series=pd.Series(cvar_series, index=self.returns.index),
            conditional_vol=pd.Series(cond_vol, index=self.returns.index),
            params=params, confidence=confidence)
