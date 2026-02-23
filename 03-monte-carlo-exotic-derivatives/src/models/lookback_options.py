"""
Lookback Option Pricing via Monte Carlo
==========================================

Floating strike call: S_T - min(S_t)
Floating strike put: max(S_t) - S_T
Fixed strike call: max(max(S_t) - K, 0)

Author: Jose Orlando Bobadilla Fuentes | CQF
"""

import numpy as np
from dataclasses import dataclass
from .path_generator import GBMPathGenerator, SimulationConfig


@dataclass
class LookbackResult:
    price: float
    std_error: float
    ci_lower: float
    ci_upper: float
    n_paths: int
    option_desc: str


class LookbackOptionPricer:
    def __init__(self, config, K=100.0):
        self.config = config
        self.K = K
        self.generator = GBMPathGenerator(config)

    def _compute(self, payoffs, desc):
        disc = np.exp(-self.config.r * self.config.T) * payoffs
        p = np.mean(disc)
        se = np.std(disc, ddof=1) / np.sqrt(self.config.n_paths)
        return LookbackResult(p, se, p-1.96*se, p+1.96*se,
                              self.config.n_paths, desc)

    def floating_strike_call(self):
        """Payoff = S_T - min(S_t): buy at lowest observed price."""
        paths = self.generator.generate()
        return self._compute(paths[:, -1] - np.min(paths, axis=1),
                             "Floating Call: S_T - min(S_t)")

    def floating_strike_put(self):
        """Payoff = max(S_t) - S_T: sell at highest observed price."""
        paths = self.generator.generate()
        return self._compute(np.max(paths, axis=1) - paths[:, -1],
                             "Floating Put: max(S_t) - S_T")

    def fixed_strike_call(self):
        """Payoff = max(max(S_t) - K, 0)."""
        paths = self.generator.generate()
        return self._compute(np.maximum(np.max(paths, axis=1) - self.K, 0),
                             f"Fixed Call: max(max(S_t)-{self.K}, 0)")
