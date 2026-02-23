"""
Stochastic Process Simulators
================================

Implements exact and Euler-Maruyama discretizations for fundamental
stochastic differential equations in quantitative finance.

Each process class provides:
    - simulate(): Generate n_paths sample paths
    - theoretical_mean(): Analytical expected value function
    - theoretical_var(): Analytical variance function

Author: Jose Orlando Bobadilla Fuentes | CQF
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class SimConfig:
    """Shared simulation configuration."""
    T: float = 1.0
    n_steps: int = 1000
    n_paths: int = 500
    seed: int = 42


class StochasticProcess(ABC):
    """Abstract base class for all stochastic processes."""

    @abstractmethod
    def simulate(self, config: SimConfig) -> tuple:
        """Returns (time_grid, paths) where paths.shape = (n_paths, n_steps+1)."""
        pass

    @abstractmethod
    def theoretical_mean(self, t: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def theoretical_var(self, t: np.ndarray) -> np.ndarray:
        pass


class WienerProcess(StochasticProcess):
    """
    Standard Brownian Motion / Wiener Process.
    dW(t), W(0) = 0
    E[W(t)] = 0, Var[W(t)] = t
    """

    def simulate(self, config: SimConfig) -> tuple:
        rng = np.random.default_rng(config.seed)
        dt = config.T / config.n_steps
        t = np.linspace(0, config.T, config.n_steps + 1)

        dW = rng.normal(0, np.sqrt(dt), (config.n_paths, config.n_steps))
        W = np.zeros((config.n_paths, config.n_steps + 1))
        W[:, 1:] = np.cumsum(dW, axis=1)
        return t, W

    def theoretical_mean(self, t):
        return np.zeros_like(t)

    def theoretical_var(self, t):
        return t


class GeometricBrownianMotion(StochasticProcess):
    """
    GBM: dS = mu*S*dt + sigma*S*dW
    Exact solution: S(t) = S0 * exp((mu - sigma^2/2)*t + sigma*W(t))
    """

    def __init__(self, S0: float = 100, mu: float = 0.08, sigma: float = 0.20):
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma

    def simulate(self, config: SimConfig) -> tuple:
        rng = np.random.default_rng(config.seed)
        dt = config.T / config.n_steps
        t = np.linspace(0, config.T, config.n_steps + 1)

        Z = rng.standard_normal((config.n_paths, config.n_steps))
        log_inc = (self.mu - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * Z
        log_paths = np.concatenate(
            [np.zeros((config.n_paths, 1)), np.cumsum(log_inc, axis=1)], axis=1)
        return t, self.S0 * np.exp(log_paths)

    def theoretical_mean(self, t):
        return self.S0 * np.exp(self.mu * t)

    def theoretical_var(self, t):
        return self.S0**2 * np.exp(2*self.mu*t) * (np.exp(self.sigma**2*t) - 1)


class OrnsteinUhlenbeck(StochasticProcess):
    """
    OU Process: dX = kappa*(theta - X)*dt + sigma*dW
    Mean-reverting to theta with speed kappa.
    """

    def __init__(self, X0: float = 0.05, kappa: float = 3.0,
                 theta: float = 0.05, sigma: float = 0.02):
        self.X0 = X0
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma

    def simulate(self, config: SimConfig) -> tuple:
        rng = np.random.default_rng(config.seed)
        dt = config.T / config.n_steps
        t = np.linspace(0, config.T, config.n_steps + 1)

        X = np.zeros((config.n_paths, config.n_steps + 1))
        X[:, 0] = self.X0

        # Exact discretization for OU
        exp_kdt = np.exp(-self.kappa * dt)
        var_dt = (self.sigma**2 / (2*self.kappa)) * (1 - exp_kdt**2)

        for i in range(config.n_steps):
            X[:, i+1] = (self.theta + (X[:, i] - self.theta) * exp_kdt
                         + np.sqrt(var_dt) * rng.standard_normal(config.n_paths))
        return t, X

    def theoretical_mean(self, t):
        return self.theta + (self.X0 - self.theta) * np.exp(-self.kappa * t)

    def theoretical_var(self, t):
        return (self.sigma**2 / (2*self.kappa)) * (1 - np.exp(-2*self.kappa*t))


class CIRProcess(StochasticProcess):
    """
    CIR: dX = kappa*(theta - X)*dt + sigma*sqrt(X)*dW
    Non-negative if 2*kappa*theta >= sigma^2 (Feller condition).
    """

    def __init__(self, X0: float = 0.05, kappa: float = 2.0,
                 theta: float = 0.05, sigma: float = 0.1):
        self.X0 = X0
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.feller = 2 * kappa * theta >= sigma**2

    def simulate(self, config: SimConfig) -> tuple:
        rng = np.random.default_rng(config.seed)
        dt = config.T / config.n_steps
        t = np.linspace(0, config.T, config.n_steps + 1)

        X = np.zeros((config.n_paths, config.n_steps + 1))
        X[:, 0] = self.X0

        # Full truncation Euler scheme (Lord et al., 2010)
        for i in range(config.n_steps):
            X_pos = np.maximum(X[:, i], 0)
            dW = rng.standard_normal(config.n_paths) * np.sqrt(dt)
            X[:, i+1] = (X[:, i]
                         + self.kappa * (self.theta - X_pos) * dt
                         + self.sigma * np.sqrt(X_pos) * dW)
            X[:, i+1] = np.maximum(X[:, i+1], 0)
        return t, X

    def theoretical_mean(self, t):
        return self.theta + (self.X0 - self.theta) * np.exp(-self.kappa * t)

    def theoretical_var(self, t):
        ekt = np.exp(-self.kappa * t)
        return (self.X0 * self.sigma**2 * ekt / self.kappa * (1 - ekt)
                + self.theta * self.sigma**2 / (2*self.kappa) * (1 - ekt)**2)


class HestonModel(StochasticProcess):
    """
    Heston (1993) stochastic volatility model.
    dS = mu*S*dt + sqrt(v)*S*dW1
    dv = kappa*(theta-v)*dt + xi*sqrt(v)*dW2
    corr(dW1,dW2) = rho
    """

    def __init__(self, S0=100, v0=0.04, mu=0.05, kappa=2.0,
                 theta=0.04, xi=0.3, rho=-0.7):
        self.S0 = S0
        self.v0 = v0
        self.mu = mu
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho

    def simulate(self, config: SimConfig) -> tuple:
        rng = np.random.default_rng(config.seed)
        dt = config.T / config.n_steps
        t = np.linspace(0, config.T, config.n_steps + 1)

        S = np.zeros((config.n_paths, config.n_steps + 1))
        v = np.zeros((config.n_paths, config.n_steps + 1))
        S[:, 0] = self.S0
        v[:, 0] = self.v0

        for i in range(config.n_steps):
            Z1 = rng.standard_normal(config.n_paths)
            Z2 = self.rho * Z1 + np.sqrt(1 - self.rho**2) * rng.standard_normal(config.n_paths)

            v_pos = np.maximum(v[:, i], 0)
            sqrt_v = np.sqrt(v_pos)

            S[:, i+1] = S[:, i] * np.exp(
                (self.mu - 0.5*v_pos)*dt + sqrt_v*np.sqrt(dt)*Z1)
            v[:, i+1] = np.maximum(
                v[:, i] + self.kappa*(self.theta - v_pos)*dt
                + self.xi*sqrt_v*np.sqrt(dt)*Z2, 0)

        return t, S, v  # Returns 3 items: t, price paths, vol paths

    def theoretical_mean(self, t):
        return self.S0 * np.exp(self.mu * t)

    def theoretical_var(self, t):
        return np.full_like(t, np.nan)  # No simple closed-form


class MertonJumpDiffusion(StochasticProcess):
    """
    Merton (1976) jump-diffusion:
    dS/S = (mu - lambda*k)*dt + sigma*dW + J*dN
    N ~ Poisson(lambda*dt), J ~ N(mu_J, sigma_J)
    k = exp(mu_J + sigma_J^2/2) - 1
    """

    def __init__(self, S0=100, mu=0.08, sigma=0.15,
                 lam=1.0, mu_J=-0.05, sigma_J=0.10):
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
        self.lam = lam
        self.mu_J = mu_J
        self.sigma_J = sigma_J
        self.k = np.exp(mu_J + 0.5*sigma_J**2) - 1

    def simulate(self, config: SimConfig) -> tuple:
        rng = np.random.default_rng(config.seed)
        dt = config.T / config.n_steps
        t = np.linspace(0, config.T, config.n_steps + 1)

        S = np.zeros((config.n_paths, config.n_steps + 1))
        S[:, 0] = self.S0

        drift = (self.mu - self.lam * self.k - 0.5*self.sigma**2) * dt

        for i in range(config.n_steps):
            Z = rng.standard_normal(config.n_paths)
            N = rng.poisson(self.lam * dt, config.n_paths)
            J = np.sum([rng.normal(self.mu_J, self.sigma_J, config.n_paths)
                        for _ in range(int(np.max(N)))], axis=0) if np.max(N) > 0 else 0

            # Proper jump handling per path
            jump_sum = np.zeros(config.n_paths)
            for path in range(config.n_paths):
                if N[path] > 0:
                    jump_sum[path] = np.sum(
                        rng.normal(self.mu_J, self.sigma_J, N[path]))

            S[:, i+1] = S[:, i] * np.exp(
                drift + self.sigma*np.sqrt(dt)*Z + jump_sum)
        return t, S

    def theoretical_mean(self, t):
        return self.S0 * np.exp(self.mu * t)

    def theoretical_var(self, t):
        return np.full_like(t, np.nan)
