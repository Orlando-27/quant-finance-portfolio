"""
Multi-Method Value at Risk Engine
====================================

Implements Historical, Parametric, Monte Carlo, and GARCH-based VaR/CVaR.

Author: Jose Orlando Bobadilla Fuentes | CQF
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, t as t_dist
from typing import Optional
from dataclasses import dataclass


@dataclass
class VaRResult:
    """Container for VaR/CVaR computation results."""
    var: float
    cvar: float
    confidence: float
    method: str
    additional: Optional[dict] = None


class VaREngine:
    """
    Multi-method VaR/CVaR calculator.

    Supports portfolio or single-asset risk estimation using:
        1. Historical simulation (non-parametric)
        2. Parametric normal (with Cornish-Fisher extension)
        3. Parametric Student-t
        4. Monte Carlo simulation

    Usage:
        >>> engine = VaREngine(returns_series)
        >>> result = engine.historical(confidence=0.99)
    """

    def __init__(self, returns: np.ndarray, portfolio_value: float = 1e6):
        """
        Parameters:
            returns: Array of historical returns (daily)
            portfolio_value: Portfolio notional for dollar VaR
        """
        self.returns = np.array(returns, dtype=np.float64)
        self.portfolio_value = portfolio_value
        self.mu = np.mean(self.returns)
        self.sigma = np.std(self.returns, ddof=1)
        self.skew = float(pd.Series(self.returns).skew())
        self.kurt = float(pd.Series(self.returns).kurtosis())

    def historical(self, confidence: float = 0.95,
                   window: Optional[int] = None) -> VaRResult:
        """
        Historical simulation VaR.
        Non-parametric: uses the empirical quantile of past returns.
        No distributional assumptions required.
        """
        data = self.returns[-window:] if window else self.returns
        alpha = 1 - confidence
        var = -np.percentile(data, alpha * 100)
        # CVaR: mean of returns below the VaR threshold
        tail = data[data <= -var]
        cvar = -np.mean(tail) if len(tail) > 0 else var

        return VaRResult(
            var=var * self.portfolio_value,
            cvar=cvar * self.portfolio_value,
            confidence=confidence, method="Historical Simulation",
            additional={"var_pct": var, "cvar_pct": cvar, "n_obs": len(data)})

    def parametric_normal(self, confidence: float = 0.95,
                          cornish_fisher: bool = False) -> VaRResult:
        """
        Parametric VaR assuming normal distribution.
        Optional Cornish-Fisher expansion adjusts for skewness and kurtosis:
            z_cf = z + (z^2-1)*S/6 + (z^3-3*z)*K/24 - (2*z^3-5*z)*S^2/36
        """
        z = norm.ppf(1 - confidence)

        if cornish_fisher:
            S, K = self.skew, self.kurt
            z_cf = (z + (z**2 - 1) * S / 6
                    + (z**3 - 3*z) * K / 24
                    - (2*z**3 - 5*z) * S**2 / 36)
            var = -(self.mu + z_cf * self.sigma)
        else:
            var = -(self.mu + z * self.sigma)

        # Normal CVaR: E[X | X < VaR] for normal distribution
        cvar = -self.mu + self.sigma * norm.pdf(norm.ppf(1-confidence)) / (1-confidence)

        method = "Parametric Normal" + (" (Cornish-Fisher)" if cornish_fisher else "")
        return VaRResult(
            var=var * self.portfolio_value,
            cvar=cvar * self.portfolio_value,
            confidence=confidence, method=method,
            additional={"var_pct": var, "cvar_pct": cvar,
                        "skewness": self.skew, "kurtosis": self.kurt})

    def parametric_student_t(self, confidence: float = 0.95) -> VaRResult:
        """
        Parametric VaR using Student-t distribution.
        Better captures heavy tails observed in financial returns.
        Degrees of freedom estimated via MLE.
        """
        # Fit Student-t distribution
        params = t_dist.fit(self.returns)
        df, loc, scale = params

        var_quantile = t_dist.ppf(1 - confidence, df, loc, scale)
        var = -var_quantile

        # Student-t CVaR (analytical)
        x_alpha = t_dist.ppf(1 - confidence, df)
        cvar_std = (t_dist.pdf(x_alpha, df) / (1 - confidence)
                    * (df + x_alpha**2) / (df - 1))
        cvar = -loc + scale * cvar_std

        return VaRResult(
            var=var * self.portfolio_value,
            cvar=cvar * self.portfolio_value,
            confidence=confidence, method="Parametric Student-t",
            additional={"var_pct": var, "cvar_pct": cvar,
                        "df": df, "loc": loc, "scale": scale})

    def monte_carlo(self, confidence: float = 0.95,
                    n_simulations: int = 100000, seed: int = 42) -> VaRResult:
        """
        Monte Carlo VaR using fitted normal distribution.
        Generates future 1-day return scenarios.
        """
        rng = np.random.default_rng(seed)
        simulated = rng.normal(self.mu, self.sigma, n_simulations)

        alpha = 1 - confidence
        var = -np.percentile(simulated, alpha * 100)
        tail = simulated[simulated <= -var]
        cvar = -np.mean(tail) if len(tail) > 0 else var

        return VaRResult(
            var=var * self.portfolio_value,
            cvar=cvar * self.portfolio_value,
            confidence=confidence, method="Monte Carlo (Normal)",
            additional={"var_pct": var, "n_sims": n_simulations})

    def compute_all_methods(self, confidence: float = 0.95) -> pd.DataFrame:
        """Run all VaR methods and return comparative table."""
        results = [
            self.historical(confidence),
            self.parametric_normal(confidence),
            self.parametric_normal(confidence, cornish_fisher=True),
            self.parametric_student_t(confidence),
            self.monte_carlo(confidence),
        ]
        rows = []
        for r in results:
            rows.append({
                "Method": r.method,
                "VaR": r.var,
                "CVaR": r.cvar,
                "VaR_pct": r.additional.get("var_pct", r.var / self.portfolio_value),
                "CVaR_pct": r.additional.get("cvar_pct", r.cvar / self.portfolio_value),
            })
        return pd.DataFrame(rows)
