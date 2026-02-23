"""
================================================================================
CUSTOM GYMNASIUM ENVIRONMENT FOR PORTFOLIO MANAGEMENT
================================================================================
A fully-featured trading environment following the Gymnasium API that
simulates multi-asset portfolio management with realistic mechanics.

Key features:
    - Continuous action space (target portfolio weights)
    - Configurable reward functions (log-return, diff Sharpe, risk-adj, Sortino)
    - Proportional and fixed transaction costs
    - Position limits and turnover constraints
    - State includes portfolio weights, market features, risk metrics
    - Episode termination on drawdown breach

The environment satisfies the Gymnasium interface:
    observation, info = env.reset()
    observation, reward, terminated, truncated, info = env.step(action)

Reference:
    Jiang, Z. et al. (2017). A Deep RL Framework for Financial
    Portfolio Management Problem. arXiv:1706.10059.

Author: Jose Orlando Bobadilla Fuentes, CQF | MSc AI
================================================================================
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any


class PortfolioEnv(gym.Env):
    """
    Gymnasium environment for portfolio management.

    Observation space: [portfolio_weights, cash, market_features, risk_features]
    Action space: target allocation weights in [0, 1]^n (softmax-normalized)

    Parameters
    ----------
    returns : pd.DataFrame
        Daily log returns matrix (T x n_assets).
    features : pd.DataFrame
        Market feature matrix (T x n_features), aligned with returns.
    initial_cash : float
        Starting portfolio value (default 1.0 = normalized).
    transaction_cost : float
        Proportional cost per unit traded (default 0.001 = 10 bps).
    fixed_cost : float
        Fixed cost per rebalancing event (default 0.0).
    reward_type : str
        'log_return', 'differential_sharpe', 'risk_adjusted', 'sortino'.
    risk_aversion : float
        Lambda for risk-adjusted and Sortino rewards (default 1.0).
    vol_target : float
        Annualized volatility target for risk-adjusted reward (default 0.15).
    max_drawdown : float
        Episode terminates if drawdown exceeds this (default 0.25 = 25%).
    max_position : float
        Maximum weight for any single asset (default 0.40).
    lookback : int
        Number of past observations included in state (default 1).
    normalize_obs : bool
        Apply running normalization to observations (default True).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        returns: pd.DataFrame,
        features: pd.DataFrame,
        initial_cash: float = 1.0,
        transaction_cost: float = 0.001,
        fixed_cost: float = 0.0,
        reward_type: str = "differential_sharpe",
        risk_aversion: float = 1.0,
        vol_target: float = 0.15,
        max_drawdown: float = 0.25,
        max_position: float = 0.40,
        lookback: int = 1,
        normalize_obs: bool = True,
    ):
        super().__init__()

        self.returns = returns.values
        self.features = features.values
        self.dates = returns.index
        self.tickers = returns.columns.tolist()
        self.n_assets = len(self.tickers)
        self.n_features = features.shape[1]
        self.n_steps = len(returns)

        # Environment parameters
        self.initial_cash = initial_cash
        self.tc_rate = transaction_cost
        self.fixed_cost = fixed_cost
        self.reward_type = reward_type
        self.risk_aversion = risk_aversion
        self.vol_target = vol_target
        self.max_dd = max_drawdown
        self.max_position = max_position
        self.lookback = lookback
        self.normalize_obs = normalize_obs

        # Observation space dimension:
        #   portfolio weights (n) + cash (1) + market features (n_feat)
        #   + portfolio vol (1) + drawdown (1) + step_frac (1)
        self.obs_dim = self.n_assets + 1 + self.n_features + 3

        # Gymnasium spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.obs_dim,), dtype=np.float32,
        )
        # Action: unnormalized logits for softmax -> portfolio weights
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.n_assets,), dtype=np.float32,
        )

        # Running normalization statistics (Welford's algorithm)
        self._obs_mean = np.zeros(self.obs_dim)
        self._obs_var = np.ones(self.obs_dim)
        self._obs_count = 1e-4

        # State variables (set in reset)
        self._reset_state()

    def _reset_state(self):
        """Initialize all episode state variables."""
        self.step_idx = 0
        self.wealth = self.initial_cash
        self.peak_wealth = self.initial_cash
        self.weights = np.zeros(self.n_assets)  # Start fully in cash
        self.cash_weight = 1.0
        self.portfolio_returns = []

        # Differential Sharpe ratio accumulators
        self._A = 0.0  # EMA of returns
        self._B = 0.0  # EMA of squared returns
        self._eta = 0.01  # Adaptation rate

    def _get_observation(self) -> np.ndarray:
        """Construct the observation vector at current time step."""
        # Portfolio state
        port_state = np.concatenate([
            self.weights,
            [self.cash_weight],
        ])

        # Market features at current step
        if self.step_idx < len(self.features):
            mkt_feat = self.features[self.step_idx]
        else:
            mkt_feat = np.zeros(self.n_features)

        # Risk features
        if len(self.portfolio_returns) > 5:
            port_vol = np.std(self.portfolio_returns[-21:]) * np.sqrt(252)
        else:
            port_vol = 0.0

        drawdown = (self.peak_wealth - self.wealth) / self.peak_wealth
        step_frac = self.step_idx / self.n_steps

        risk_feat = np.array([port_vol, drawdown, step_frac])

        obs = np.concatenate([port_state, mkt_feat, risk_feat]).astype(np.float32)

        # Replace NaN/Inf with 0
        obs = np.nan_to_num(obs, nan=0.0, posinf=3.0, neginf=-3.0)

        # Running normalization
        if self.normalize_obs:
            obs = self._normalize(obs)

        return obs

    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        """Apply running mean/variance normalization (Welford's online)."""
        self._obs_count += 1
        delta = obs - self._obs_mean
        self._obs_mean += delta / self._obs_count
        delta2 = obs - self._obs_mean
        self._obs_var += (delta * delta2 - self._obs_var) / self._obs_count
        std = np.sqrt(self._obs_var + 1e-8)
        return (obs - self._obs_mean) / std

    def _action_to_weights(self, action: np.ndarray) -> np.ndarray:
        """
        Convert raw action to valid portfolio weights via softmax.

        Ensures: weights >= 0, sum(weights) <= 1, max(weights) <= max_position.
        The residual (1 - sum) is allocated to cash.
        """
        # Softmax for non-negative weights
        exp_a = np.exp(action - np.max(action))  # Numerical stability
        weights = exp_a / (exp_a.sum() + 1e-8)

        # Apply concentration limit
        weights = np.clip(weights, 0.0, self.max_position)

        # Renormalize so total allocation <= 1.0
        total = weights.sum()
        if total > 1.0:
            weights = weights / total

        return weights

    def _compute_reward(self, port_return: float) -> float:
        """
        Compute reward based on the configured reward function.

        Parameters
        ----------
        port_return : float
            Single-period portfolio log return.
        """
        if self.reward_type == "log_return":
            # Simple wealth growth
            return port_return

        elif self.reward_type == "differential_sharpe":
            # Moody & Saffell (2001) incremental Sharpe ratio
            dA = port_return - self._A
            dB = port_return ** 2 - self._B

            denominator = (self._B - self._A ** 2)
            if denominator > 1e-10:
                D = (self._B * dA - 0.5 * self._A * dB) / (denominator ** 1.5)
            else:
                D = port_return  # Fallback for initial steps

            # Update accumulators
            self._A += self._eta * dA
            self._B += self._eta * dB

            return float(D)

        elif self.reward_type == "risk_adjusted":
            # Penalize volatility above target
            if len(self.portfolio_returns) > 5:
                recent_vol = np.std(self.portfolio_returns[-21:]) * np.sqrt(252)
                penalty = self.risk_aversion * max(0, recent_vol - self.vol_target) ** 2
            else:
                penalty = 0.0
            return port_return - penalty

        elif self.reward_type == "sortino":
            # Penalize downside returns quadratically
            if port_return < 0:
                return port_return - self.risk_aversion * port_return ** 2
            return port_return

        else:
            return port_return

    def _compute_transaction_cost(
        self, old_weights: np.ndarray, new_weights: np.ndarray
    ) -> float:
        """
        Compute total transaction cost from rebalancing.

        TC = tc_rate * sum(|w_new - w_old|) + fixed_cost * I(rebalance)
        """
        turnover = np.sum(np.abs(new_weights - old_weights))
        proportional = self.tc_rate * turnover * self.wealth
        fixed = self.fixed_cost if turnover > 0.01 else 0.0
        return proportional + fixed

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        self._reset_state()

        obs = self._get_observation()
        info = {
            "wealth": self.wealth,
            "weights": self.weights.copy(),
            "step": self.step_idx,
        }
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step: rebalance portfolio, observe market return, compute reward.

        Parameters
        ----------
        action : np.ndarray of shape (n_assets,)
            Raw action (converted to weights via softmax).

        Returns
        -------
        observation, reward, terminated, truncated, info
        """
        # Convert action to valid portfolio weights
        new_weights = self._action_to_weights(action)

        # Transaction cost from rebalancing
        tc = self._compute_transaction_cost(self.weights, new_weights)
        self.wealth -= tc

        # Update weights
        old_weights = self.weights.copy()
        self.weights = new_weights
        self.cash_weight = 1.0 - self.weights.sum()

        # Observe market return for this step
        if self.step_idx < self.n_steps:
            asset_returns = self.returns[self.step_idx]  # Log returns
        else:
            asset_returns = np.zeros(self.n_assets)

        # Portfolio return (weighted sum of asset returns + cash at 0)
        port_return = float(np.dot(self.weights, asset_returns))
        self.portfolio_returns.append(port_return)

        # Update wealth
        self.wealth *= np.exp(port_return)
        self.peak_wealth = max(self.peak_wealth, self.wealth)

        # Drift weights due to differential asset returns
        # w_i^{t+1} = w_i^t * exp(r_i) / sum(w_j * exp(r_j))
        if self.weights.sum() > 1e-8:
            drifted = self.weights * np.exp(asset_returns)
            total = drifted.sum() + self.cash_weight
            self.weights = drifted / total
            self.cash_weight = self.cash_weight / total

        # Advance time
        self.step_idx += 1

        # Compute reward
        reward = self._compute_reward(port_return)

        # Termination conditions
        drawdown = (self.peak_wealth - self.wealth) / self.peak_wealth
        terminated = drawdown > self.max_dd  # Drawdown breach
        truncated = self.step_idx >= self.n_steps  # End of data

        # Observation for next step
        obs = self._get_observation()

        info = {
            "wealth": self.wealth,
            "weights": self.weights.copy(),
            "cash": self.cash_weight,
            "port_return": port_return,
            "transaction_cost": tc,
            "turnover": np.sum(np.abs(new_weights - old_weights)),
            "drawdown": drawdown,
            "step": self.step_idx,
            "date": str(self.dates[min(self.step_idx - 1, len(self.dates) - 1)]),
        }

        return obs, float(reward), terminated, truncated, info

    def get_episode_stats(self) -> Dict:
        """Compute summary statistics for the completed episode."""
        rets = np.array(self.portfolio_returns)
        if len(rets) == 0:
            return {}

        ann_ret = np.mean(rets) * 252
        ann_vol = np.std(rets) * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

        cum = np.cumprod(1 + rets)
        peak = np.maximum.accumulate(cum)
        dd = (cum - peak) / peak
        max_dd = np.abs(dd.min())

        return {
            "total_return": float(self.wealth / self.initial_cash - 1),
            "ann_return": float(ann_ret),
            "ann_volatility": float(ann_vol),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_dd),
            "n_steps": len(rets),
            "final_wealth": float(self.wealth),
        }
