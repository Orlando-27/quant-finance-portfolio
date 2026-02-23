"""
Geometric Brownian Motion Path Generator
==========================================

Vectorized GBM simulation under risk-neutral measure Q.
S(t+dt) = S(t) * exp((r - q - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)

Author: Jose Orlando Bobadilla Fuentes | CQF
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class SimulationConfig:
    """
    Attributes:
        S0: Initial spot price
        r: Risk-free rate (annualized)
        sigma: Volatility (annualized)
        T: Time to maturity (years)
        n_steps: Time steps per path
        n_paths: Number of simulated paths
        q: Dividend yield
        seed: Random seed for reproducibility
    """
    S0: float
    r: float
    sigma: float
    T: float
    n_steps: int = 252
    n_paths: int = 100000
    q: float = 0.0
    seed: Optional[int] = None

    @property
    def dt(self) -> float:
        return self.T / self.n_steps


class GBMPathGenerator:
    """
    Vectorized GBM path generator.
    Output shape: (n_paths, n_steps + 1) where column 0 = S0.

    Usage:
        >>> gen = GBMPathGenerator(config)
        >>> paths = gen.generate()  # (100000, 253)
    """

    def __init__(self, config: SimulationConfig):
        self.config = config
        self._rng = np.random.default_rng(config.seed)

    def generate(self) -> np.ndarray:
        """Generate full price paths using exact log-normal solution."""
        c = self.config
        drift = (c.r - c.q - 0.5 * c.sigma**2) * c.dt
        diffusion = c.sigma * np.sqrt(c.dt)

        Z = self._rng.standard_normal((c.n_paths, c.n_steps))
        log_inc = drift + diffusion * Z
        log_paths = np.concatenate(
            [np.zeros((c.n_paths, 1)), np.cumsum(log_inc, axis=1)], axis=1)
        return c.S0 * np.exp(log_paths)

    def generate_antithetic(self) -> tuple:
        """Generate paired paths (Z, -Z) for variance reduction."""
        c = self.config
        drift = (c.r - c.q - 0.5 * c.sigma**2) * c.dt
        diffusion = c.sigma * np.sqrt(c.dt)

        Z = self._rng.standard_normal((c.n_paths, c.n_steps))

        def _build(z):
            lp = np.concatenate(
                [np.zeros((c.n_paths, 1)),
                 np.cumsum(drift + diffusion * z, axis=1)], axis=1)
            return c.S0 * np.exp(lp)

        return _build(Z), _build(-Z)

    def terminal_values(self) -> np.ndarray:
        """Generate only S(T) without storing full paths."""
        c = self.config
        Z = self._rng.standard_normal(c.n_paths)
        drift = (c.r - c.q - 0.5 * c.sigma**2) * c.T
        return c.S0 * np.exp(drift + c.sigma * np.sqrt(c.T) * Z)
