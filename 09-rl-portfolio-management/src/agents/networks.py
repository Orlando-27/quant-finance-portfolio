"""
================================================================================
SHARED NEURAL NETWORK ARCHITECTURES FOR RL AGENTS
================================================================================
Implements actor and critic networks used by PPO, A2C, and DDPG.

Architecture design choices for financial RL:
    - Layer normalization (more stable than batch norm for RL)
    - Orthogonal initialization (Andrychowicz et al., 2020)
    - Moderate depth (2-3 hidden layers) to prevent overfitting
    - Tanh activations for bounded outputs
    - Separate actor/critic networks (no weight sharing)

The actor outputs:
    - PPO/A2C: mean + log_std of Gaussian policy (continuous actions)
    - DDPG: deterministic action (mu)

The critic outputs:
    - PPO/A2C: state value V(s)
    - DDPG: action value Q(s, a)

Author: Jose Orlando Bobadilla Fuentes, CQF | MSc AI
================================================================================
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple


def orthogonal_init(layer: nn.Linear, gain: float = np.sqrt(2)):
    """
    Apply orthogonal initialization (recommended for RL).

    Orthogonal init preserves gradient norms during backpropagation,
    leading to more stable training in deep RL.

    Reference:
        Saxe, A. et al. (2014). Exact Solutions to the Nonlinear
        Dynamics of Learning in Deep Linear Neural Networks.
    """
    nn.init.orthogonal_(layer.weight, gain=gain)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, 0.0)
    return layer


class ActorNetwork(nn.Module):
    """
    Stochastic actor network for PPO and A2C.

    Outputs the mean and log standard deviation of a diagonal Gaussian
    policy over portfolio weights.

    Architecture:
        obs -> FC(hidden) -> LayerNorm -> ReLU -> FC(hidden) -> LayerNorm
        -> ReLU -> FC(n_actions) -> mean
        + learnable log_std parameter

    Parameters
    ----------
    obs_dim : int
        Observation space dimension.
    action_dim : int
        Number of assets (action space dimension).
    hidden_dim : int
        Hidden layer size (default 256).
    n_layers : int
        Number of hidden layers (default 2).
    log_std_init : float
        Initial value for log standard deviation (default -0.5).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
        log_std_init: float = -0.5,
    ):
        super().__init__()

        layers = []
        in_dim = obs_dim
        for _ in range(n_layers):
            layers.append(orthogonal_init(nn.Linear(in_dim, hidden_dim)))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        self.features = nn.Sequential(*layers)

        # Mean head: outputs logits for softmax -> portfolio weights
        self.mean_head = orthogonal_init(
            nn.Linear(hidden_dim, action_dim), gain=0.01
        )

        # Learnable log standard deviation (state-independent)
        self.log_std = nn.Parameter(
            torch.ones(action_dim) * log_std_init
        )

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns
        -------
        mean : (batch, action_dim) -- policy mean
        log_std : (action_dim,) -- log standard deviation
        """
        h = self.features(obs)
        mean = self.mean_head(h)
        return mean, self.log_std.expand_as(mean)

    def get_distribution(
        self, obs: torch.Tensor
    ) -> torch.distributions.Normal:
        """Return the Gaussian policy distribution."""
        mean, log_std = self.forward(obs)
        std = torch.exp(log_std.clamp(-5, 2))
        return torch.distributions.Normal(mean, std)

    def sample_action(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action and compute log probability.

        Returns
        -------
        action : (batch, action_dim)
        log_prob : (batch,)
        """
        dist = self.get_distribution(obs)
        action = dist.rsample()  # Reparameterized for gradient flow
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob


class CriticNetwork(nn.Module):
    """
    State-value critic network V(s) for PPO and A2C.

    Architecture:
        obs -> FC(hidden) -> LayerNorm -> ReLU -> FC(hidden) -> LayerNorm
        -> ReLU -> FC(1) -> value

    Parameters
    ----------
    obs_dim : int
        Observation space dimension.
    hidden_dim : int
        Hidden layer size.
    n_layers : int
        Number of hidden layers.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
    ):
        super().__init__()

        layers = []
        in_dim = obs_dim
        for _ in range(n_layers):
            layers.append(orthogonal_init(nn.Linear(in_dim, hidden_dim)))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        layers.append(orthogonal_init(nn.Linear(hidden_dim, 1), gain=1.0))
        self.network = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return state value V(s), shape (batch, 1)."""
        return self.network(obs)


class DeterministicActor(nn.Module):
    """
    Deterministic actor mu(s) for DDPG.

    Outputs raw action logits that are transformed to portfolio weights
    via softmax in the environment.

    Architecture:
        obs -> FC -> LayerNorm -> ReLU -> FC -> LayerNorm -> ReLU
        -> FC -> Tanh -> scaled action
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
    ):
        super().__init__()

        layers = []
        in_dim = obs_dim
        for _ in range(n_layers):
            layers.append(orthogonal_init(nn.Linear(in_dim, hidden_dim)))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        layers.append(orthogonal_init(nn.Linear(hidden_dim, action_dim), gain=0.01))
        layers.append(nn.Tanh())

        self.network = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return deterministic action mu(s), shape (batch, action_dim)."""
        return self.network(obs)


class QNetwork(nn.Module):
    """
    Action-value critic Q(s, a) for DDPG.

    Concatenates state and action, then estimates the Q-value.

    Architecture:
        [obs, action] -> FC -> LayerNorm -> ReLU -> FC -> LayerNorm
        -> ReLU -> FC(1) -> Q-value
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
    ):
        super().__init__()

        layers = []
        in_dim = obs_dim + action_dim
        for _ in range(n_layers):
            layers.append(orthogonal_init(nn.Linear(in_dim, hidden_dim)))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        layers.append(orthogonal_init(nn.Linear(hidden_dim, 1), gain=1.0))
        self.network = nn.Sequential(*layers)

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Return Q(s, a), shape (batch, 1)."""
        x = torch.cat([obs, action], dim=-1)
        return self.network(x)
