"""
================================================================================
DEEP DETERMINISTIC POLICY GRADIENT (DDPG) AGENT
================================================================================
Off-policy actor-critic for continuous control (Lillicrap et al., 2016).

Key components:
    1. Deterministic actor mu(s; theta) -- outputs actions directly
    2. Q-critic Q(s, a; phi) -- estimates action value
    3. Experience replay buffer -- breaks temporal correlation
    4. Target networks -- stabilizes training via soft updates:
       theta_target = tau * theta + (1 - tau) * theta_target

Exploration uses Ornstein-Uhlenbeck noise process:
    dx_t = theta_OU * (mu_OU - x_t) * dt + sigma_OU * dW_t

This produces temporally correlated noise suitable for financial
environments where sudden regime changes should be explored smoothly.

Author: Jose Orlando Bobadilla Fuentes, CQF | MSc AI
================================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple, Optional
from collections import deque
import random

from src.agents.networks import DeterministicActor, QNetwork


class OUNoise:
    """
    Ornstein-Uhlenbeck process for exploration noise.

    Generates temporally correlated noise that decays toward zero,
    suitable for continuous control tasks. In the portfolio context,
    this encourages smooth exploration of neighboring weight configurations
    rather than random jumps.

    dx = theta * (mu - x) * dt + sigma * dW

    Parameters
    ----------
    action_dim : int
        Dimension of the action space.
    mu : float
        Long-run mean (default 0).
    theta : float
        Mean reversion speed (default 0.15).
    sigma : float
        Volatility of the noise process (default 0.2).
    """

    def __init__(
        self,
        action_dim: int,
        mu: float = 0.0,
        theta: float = 0.15,
        sigma: float = 0.2,
    ):
        self.action_dim = action_dim
        self.mu = mu * np.ones(action_dim)
        self.theta = theta
        self.sigma = sigma
        self.state = self.mu.copy()

    def reset(self):
        """Reset noise to long-run mean."""
        self.state = self.mu.copy()

    def sample(self) -> np.ndarray:
        """Generate next noise sample."""
        dx = (
            self.theta * (self.mu - self.state)
            + self.sigma * np.random.randn(self.action_dim)
        )
        self.state += dx
        return self.state.copy()


class ReplayBuffer:
    """
    Experience replay buffer for off-policy learning.

    Stores (s, a, r, s', done) transitions and samples random
    mini-batches for training. Random sampling breaks temporal
    correlations in sequential data.
    """

    def __init__(self, capacity: int = 100_000):
        self.buffer = deque(maxlen=capacity)

    def add(
        self, obs: np.ndarray, action: np.ndarray,
        reward: float, next_obs: np.ndarray, done: bool,
    ):
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(
        self, batch_size: int, device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """Sample random batch and convert to tensors."""
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)

        return {
            "obs": torch.FloatTensor(np.array(obs)).to(device),
            "actions": torch.FloatTensor(np.array(actions)).to(device),
            "rewards": torch.FloatTensor(rewards).unsqueeze(1).to(device),
            "next_obs": torch.FloatTensor(np.array(next_obs)).to(device),
            "dones": torch.FloatTensor(dones).unsqueeze(1).to(device),
        }

    def __len__(self) -> int:
        return len(self.buffer)


class DDPGAgent:
    """
    DDPG agent for portfolio management.

    Parameters
    ----------
    obs_dim : int
        Observation space dimension.
    action_dim : int
        Number of assets.
    lr_actor : float
        Actor learning rate (default 1e-4).
    lr_critic : float
        Critic learning rate (default 1e-3).
    gamma : float
        Discount factor (default 0.99).
    tau : float
        Target network soft update rate (default 0.005).
    buffer_size : int
        Replay buffer capacity (default 100,000).
    batch_size : int
        Training batch size (default 128).
    warmup_steps : int
        Steps before training starts (default 1000).
    noise_sigma : float
        OU noise volatility (default 0.2).
    noise_decay : float
        Multiplicative noise decay per episode (default 0.995).
    hidden_dim : int
        Hidden layer size.
    device : str
        'cuda' or 'cpu'.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        lr_actor: float = 1e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_size: int = 100_000,
        batch_size: int = 128,
        warmup_steps: int = 1000,
        noise_sigma: float = 0.2,
        noise_decay: float = 0.995,
        hidden_dim: int = 256,
        device: str = "auto",
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.noise_decay = noise_decay
        self.noise_scale = 1.0

        # Online networks
        self.actor = DeterministicActor(obs_dim, action_dim, hidden_dim).to(self.device)
        self.critic = QNetwork(obs_dim, action_dim, hidden_dim).to(self.device)

        # Target networks (initialized as copies)
        self.actor_target = DeterministicActor(obs_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = QNetwork(obs_dim, action_dim, hidden_dim).to(self.device)
        self._hard_update(self.actor_target, self.actor)
        self._hard_update(self.critic_target, self.critic)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Replay buffer and exploration noise
        self.buffer = ReplayBuffer(buffer_size)
        self.noise = OUNoise(action_dim, sigma=noise_sigma)

        self.total_steps = 0
        self.training_stats = []

    @staticmethod
    def _hard_update(target: nn.Module, source: nn.Module):
        """Copy all parameters from source to target."""
        target.load_state_dict(source.state_dict())

    def _soft_update(self, target: nn.Module, source: nn.Module):
        """
        Polyak averaging: theta_target = tau*theta + (1-tau)*theta_target.
        """
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(self.tau * sp.data + (1.0 - self.tau) * tp.data)

    @torch.no_grad()
    def select_action(
        self, obs: np.ndarray, deterministic: bool = False
    ) -> np.ndarray:
        """Select action with optional OU exploration noise."""
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        action = self.actor(obs_t).cpu().numpy()[0]

        if not deterministic:
            noise = self.noise.sample() * self.noise_scale
            action = action + noise
            action = np.clip(action, -1.0, 1.0)

        return action

    def store_transition(
        self, obs: np.ndarray, action: np.ndarray,
        reward: float, next_obs: np.ndarray, done: bool,
    ):
        """Store transition in replay buffer."""
        self.buffer.add(obs, action, reward, next_obs, done)
        self.total_steps += 1

    def update(self) -> Optional[Dict[str, float]]:
        """
        Perform one DDPG update step.

        Returns None if not enough data in buffer.
        """
        if len(self.buffer) < self.warmup_steps:
            return None

        batch = self.buffer.sample(self.batch_size, self.device)
        obs = batch["obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_obs = batch["next_obs"]
        dones = batch["dones"]

        # --- Critic update ---
        with torch.no_grad():
            next_actions = self.actor_target(next_obs)
            target_q = self.critic_target(next_obs, next_actions)
            y = rewards + self.gamma * (1 - dones) * target_q

        current_q = self.critic(obs, actions)
        critic_loss = nn.functional.mse_loss(current_q, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # --- Actor update ---
        pred_actions = self.actor(obs)
        actor_loss = -self.critic(obs, pred_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # --- Soft update target networks ---
        self._soft_update(self.actor_target, self.actor)
        self._soft_update(self.critic_target, self.critic)

        stats = {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "q_mean": current_q.mean().item(),
        }
        self.training_stats.append(stats)
        return stats

    def end_episode(self):
        """Called at episode end: decay noise, reset OU process."""
        self.noise_scale *= self.noise_decay
        self.noise.reset()

    def save(self, path: str):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic_target": self.critic_target.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_target.load_state_dict(ckpt["actor_target"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
