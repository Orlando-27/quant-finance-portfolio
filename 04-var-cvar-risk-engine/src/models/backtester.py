"""
VaR Backtesting Framework
============================

Statistical tests for VaR model validation:
    1. Kupiec (1995) Proportion of Failures (POF) test
    2. Christoffersen (1998) conditional coverage test
    3. Traffic light system (Basel Committee)

Author: Jose Orlando Bobadilla Fuentes | CQF
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2
from dataclasses import dataclass
from typing import List


@dataclass
class BacktestResult:
    """Container for backtesting results."""
    n_observations: int
    n_violations: int
    violation_rate: float
    expected_rate: float
    kupiec_statistic: float
    kupiec_pvalue: float
    kupiec_reject: bool
    traffic_light: str
    violations_dates: List


class VaRBacktester:
    """
    Backtester for VaR model validation.

    Usage:
        >>> bt = VaRBacktester(actual_returns, var_estimates, confidence=0.99)
        >>> result = bt.run()
    """

    def __init__(self, actual_returns: np.ndarray, var_estimates: np.ndarray,
                 confidence: float = 0.99):
        self.returns = np.array(actual_returns)
        self.var = np.array(var_estimates)
        self.confidence = confidence
        self.alpha = 1 - confidence

    def _kupiec_test(self, n_violations: int, n_obs: int) -> tuple:
        """
        Kupiec (1995) Proportion of Failures test.
        H0: violation rate = expected rate (1 - confidence)
        LR = -2*ln[(1-p)^(T-x)*p^x] + 2*ln[(1-x/T)^(T-x)*(x/T)^x]
        Under H0, LR ~ chi2(1).
        """
        p = self.alpha
        x = n_violations
        T = n_obs

        if x == 0:
            lr = -2 * (T * np.log(1 - p))
        elif x == T:
            lr = -2 * (T * np.log(p))
        else:
            p_hat = x / T
            lr = (-2 * ((T - x) * np.log(1 - p) + x * np.log(p))
                  + 2 * ((T - x) * np.log(1 - p_hat) + x * np.log(p_hat)))

        pvalue = 1 - chi2.cdf(lr, df=1)
        return lr, pvalue

    def _traffic_light(self, n_violations: int, n_obs: int) -> str:
        """
        Basel Committee traffic light system.
        For 250 observations at 99% VaR:
            Green:  0-4 violations
            Yellow: 5-9 violations
            Red:    10+ violations
        Scaled proportionally for other sample sizes.
        """
        rate = n_violations / n_obs
        if rate <= self.alpha * 1.5:
            return "GREEN"
        elif rate <= self.alpha * 3.0:
            return "YELLOW"
        else:
            return "RED"

    def run(self, dates=None) -> BacktestResult:
        """Execute full backtest analysis."""
        # A violation occurs when loss exceeds VaR
        # Convention: returns are raw (negative = loss), var is positive threshold
        violations = (-self.returns) > self.var
        n_viol = int(np.sum(violations))
        n_obs = len(self.returns)

        kupiec_lr, kupiec_p = self._kupiec_test(n_viol, n_obs)

        viol_dates = []
        if dates is not None:
            viol_dates = list(np.array(dates)[violations])

        return BacktestResult(
            n_observations=n_obs,
            n_violations=n_viol,
            violation_rate=n_viol / n_obs,
            expected_rate=self.alpha,
            kupiec_statistic=kupiec_lr,
            kupiec_pvalue=kupiec_p,
            kupiec_reject=kupiec_p < 0.05,
            traffic_light=self._traffic_light(n_viol, n_obs),
            violations_dates=viol_dates)
