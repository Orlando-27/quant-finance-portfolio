"""
================================================================================
PROXIMAL POLICY OPTIMIZATION (PPO) AGENT
================================================================================
Implements PPO-Clip (Schulman et al., 2017) for continuous portfolio
management. PPO constrains policy updates within a trust region using
a clipped surrogate objective, preventing destructive large updates.

The clipped objective:

    L^{CLIP} = E[ min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t) ]

where r_t = pi_new(a|s) / pi_old(a|s) is the probability ratio and
A_t is the Generalized Advantage Estimate (GAE-lambda):

    A_t^{GAE} = sum_{l=0}^{T-t} (gamma * lambda)^l * delta_{t+l}
    delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)

Author: Jose Orlando Bobadilla Fuentes, CQF | MSc AI
================================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Optional, List, Tuple

from src.agents.networks import ActorNetwork, CriticNetwork


class RolloutBuffer:
    """
    Stores episode trajectories for PPO on-policy updates.

    Each rollout step stores: (obs, action, reward, done, log_prob, value).
    After collection, computes returns and advantages via GAE.
    """

    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.advantages = None
        self.returns = None

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
    ):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def compute_gae(
        self,
        last_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        """
        Compute Generalized Advantage Estimation (Schulman et al., 2016).

        GAE provides a bias-variance tradeoff in advantage estimation:
            lambda=0 -> TD(0) advantage (low variance, high bias)
            lambda=1 -> Monte Carlo advantage (high variance, low bias)
        """
        rewards = np.array(self.rewards)
        values = np.array(self.values + [last_value])
        dones = np.array(self.dones)
        T = len(rewards)

        advantages = np.zeros(T)
        gae = 0.0

        for t in reversed(range(T)):
            mask = 1.0 - dones[t]
            delta = rewards[t] + gamma * values[t + 1] * mask - values[t]
            gae = delta + gamma * gae_lambda * mask * gae
            advantages[t] = gae

        self.advantages = advantages
        self.returns = advantages + np.array(self.values)

    def get_batches(
        self, batch_size: int, device: torch.device
    ) -> List[Dict[str, torch.Tensor]]:
        """Yield mini-batches for PPO epoch updates."""
        n = len(self.observations)
        indices = np.random.permutation(n)

        batches = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = indices[start:end]

            batch = {
                "obs": torch.FloatTensor(
                    np.array(self.observations)[idx]
                ).to(device),
                "actions": torch.FloatTensor(
                    np.array(self.actions)[idx]
                ).to(device),
                "old_log_probs": torch.FloatTensor(
                    np.array(self.log_probs)[idx]
                ).to(device),
                "advantages": torch.FloatTensor(
                    self.advantages[idx]
                ).to(device),
                "returns": torch.FloatTensor(
                    self.returns[idx]
                ).to(device),
            }
            batches.append(batch)

        return batches

    def clear(self):
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()
        self.advantages = None
        self.returns = None


class PPOAgent:
    """
    PPO-Clip agent for portfolio management.

    Parameters
    ----------
    obs_dim : int
        Observation space dimension.
    action_dim : int
        Number of assets.
    lr_actor : float
        Actor learning rate (default 3e-4).
    lr_critic : float
        Critic learning rate (default 1e-3).
    gamma : float
        Discount factor (default 0.99).
    gae_lambda : float
        GAE lambda for advantage estimation (default 0.95).
    clip_eps : float
        PPO clipping parameter (default 0.2).
    entropy_coeff : float
        Entropy bonus coefficient for exploration (default 0.01).
    n_epochs : int
        PPO update epochs per rollout (default 10).
    batch_size : int
        Mini-batch size for PPO updates (default 64).
    max_grad_norm : float
        Gradient clipping norm (default 0.5).
    hidden_dim : int
        Hidden layer size for actor and critic.
    device : str
        'cuda' or 'cpu'.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coeff: float = 0.01,
        n_epochs: int = 10,
        batch_size: int = 64,
        max_grad_norm: float = 0.5,
        hidden_dim: int = 256,
        device: str = "auto",
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coeff = entropy_coeff
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm

        # Networks
        self.actor = ActorNetwork(obs_dim, action_dim, hidden_dim).to(self.device)
        self.critic = CriticNetwork(obs_dim, hidden_dim).to(self.device)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Rollout buffer
        self.buffer = RolloutBuffer()

        # Training stats
        self.training_stats = []

    @torch.no_grad()
    def select_action(
        self, obs: np.ndarray, deterministic: bool = False
    ) -> Tuple[np.ndarray, float, float]:
        """
        Select action given observation.

        Returns
        -------
        action : np.ndarray
        log_prob : float
        value : float
        """
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        if deterministic:
            mean, _ = self.actor(obs_t)
            action = mean.cpu().numpy()[0]
            log_prob = 0.0
        else:
            dist = self.actor.get_distribution(obs_t)
            action_t = dist.sample()
            log_prob = dist.log_prob(action_t).sum(dim=-1).item()
            action = action_t.cpu().numpy()[0]

        value = self.critic(obs_t).item()
        return action, log_prob, value

    def update(self) -> Dict[str, float]:
        """
        Perform PPO update using collected rollout data.

        Returns
        -------
        dict with training metrics (actor_loss, critic_loss, entropy, etc.)
        """
        # Compute GAE advantages
        last_obs = self.buffer.observations[-1]
        last_obs_t = torch.FloatTensor(last_obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            last_value = self.critic(last_obs_t).item()

        self.buffer.compute_gae(last_value, self.gamma, self.gae_lambda)

        # Normalize advantages (per-batch)
        adv = self.buffer.advantages
        self.buffer.advantages = (adv - adv.mean()) / (adv.std() + 1e-8)

        # PPO epoch loop
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for _ in range(self.n_epochs):
            batches = self.buffer.get_batches(self.batch_size, self.device)

            for batch in batches:
                obs = batch["obs"]
                actions = batch["actions"]
                old_log_probs = batch["old_log_probs"]
                advantages = batch["advantages"]
                returns = batch["returns"]

                # Current policy evaluation
                dist = self.actor.get_distribution(obs)
                new_log_probs = dist.log_prob(actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()

                # PPO clipped ratio
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages

                actor_loss = -torch.min(surr1, surr2).mean()
                actor_loss -= self.entropy_coeff * entropy

                # Critic loss (MSE)
                values = self.critic(obs).squeeze(-1)
                critic_loss = nn.functional.mse_loss(values, returns)

                # Update actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # Update critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.item()
                n_updates += 1

        # Clear buffer after update
        self.buffer.clear()

        stats = {
            "actor_loss": total_actor_loss / max(n_updates, 1),
            "critic_loss": total_critic_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
        }
        self.training_stats.append(stats)
        return stats

    def save(self, path: str):
        """Save model weights."""
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }, path)

    def load(self, path: str):
        """Load model weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
