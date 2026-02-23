"""
Barrier Option Pricing via Monte Carlo
=========================================

Types: Down-and-Out/In, Up-and-Out/In.
In-Out parity: V_in + V_out = V_vanilla.
Broadie-Glasserman-Kou (1997) continuity correction for discrete monitoring.

Author: Jose Orlando Bobadilla Fuentes | CQF
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass
from .path_generator import GBMPathGenerator, SimulationConfig

BROADIE_BETA = 0.5826  # -zeta(1/2) / sqrt(2*pi)


class BarrierType(Enum):
    DOWN_AND_OUT = "down_and_out"
    DOWN_AND_IN = "down_and_in"
    UP_AND_OUT = "up_and_out"
    UP_AND_IN = "up_and_in"


@dataclass
class BarrierOptionResult:
    price: float
    std_error: float
    ci_lower: float
    ci_upper: float
    n_paths: int
    barrier_type: str
    knock_frequency: float


class BarrierOptionPricer:
    """
    MC pricer for barrier options with continuity correction.

    Usage:
        >>> pricer = BarrierOptionPricer(config, K=100, H=90,
        ...          barrier_type=BarrierType.DOWN_AND_OUT)
        >>> result = pricer.price_call()
    """

    def __init__(self, config, K, H, barrier_type, use_correction=True):
        self.config = config
        self.K = K
        self.barrier_type = barrier_type
        self.generator = GBMPathGenerator(config)

        if use_correction:
            corr = np.exp(BROADIE_BETA * config.sigma * np.sqrt(config.dt))
            if barrier_type in (BarrierType.DOWN_AND_OUT, BarrierType.DOWN_AND_IN):
                self.H = H / corr
            else:
                self.H = H * corr
        else:
            self.H = H

    def _breached(self, paths):
        if self.barrier_type in (BarrierType.DOWN_AND_OUT, BarrierType.DOWN_AND_IN):
            return np.any(paths <= self.H, axis=1)
        return np.any(paths >= self.H, axis=1)

    def _price(self, vanilla_payoff, paths):
        breached = self._breached(paths)
        if self.barrier_type in (BarrierType.DOWN_AND_OUT, BarrierType.UP_AND_OUT):
            payoffs = vanilla_payoff * (~breached).astype(float)
        else:
            payoffs = vanilla_payoff * breached.astype(float)

        disc = np.exp(-self.config.r * self.config.T) * payoffs
        p = np.mean(disc)
        se = np.std(disc, ddof=1) / np.sqrt(self.config.n_paths)
        return BarrierOptionResult(
            p, se, p-1.96*se, p+1.96*se, self.config.n_paths,
            self.barrier_type.value, np.mean(breached))

    def price_call(self):
        paths = self.generator.generate()
        return self._price(np.maximum(paths[:, -1] - self.K, 0), paths)

    def price_put(self):
        paths = self.generator.generate()
        return self._price(np.maximum(self.K - paths[:, -1], 0), paths)
