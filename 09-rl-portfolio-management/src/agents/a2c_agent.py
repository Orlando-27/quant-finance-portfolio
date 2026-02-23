"""
================================================================================
ADVANTAGE ACTOR-CRITIC (A2C) AGENT
================================================================================
Synchronous variant of A3C (Mnih et al., 2016). Uses n-step returns
instead of GAE for advantage estimation and updates after every n steps.

Actor loss:   L_pi = -E[ log pi(a|s) * A(s,a) ] - beta * H(pi)
Critic loss:  L_V  = E[ (V(s) - R_t)^2 ]
A(s,a) = R_t - V(s)  where R_t is the n-step return

Author: Jose Orlando Bobadilla Fuentes, CQF | MSc AI
================================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple

from src.agents.networks import ActorNetwork, CriticNetwork


class A2CAgent:
    """
    Advantage Actor-Critic agent for portfolio management.

    Parameters
    ----------
    obs_dim : int
        Observation space dimension.
    action_dim : int
        Number of assets.
    lr : float
        Shared learning rate for actor and critic (default 7e-4).
    gamma : float
        Discount factor (default 0.99).
    n_steps : int
        Number of steps between updates (default 5).
    entropy_coeff : float
        Entropy bonus coefficient (default 0.01).
    value_coeff : float
        Critic loss coefficient (default 0.5).
    max_grad_norm : float
        Gradient clipping norm (default 0.5).
    hidden_dim : int
        Hidden layer size.
    device : str
        'cuda' or 'cpu'.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        lr: float = 7e-4,
        gamma: float = 0.99,
        n_steps: int = 5,
        entropy_coeff: float = 0.01,
        value_coeff: float = 0.5,
        max_grad_norm: float = 0.5,
        hidden_dim: int = 256,
        device: str = "auto",
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.gamma = gamma
        self.n_steps = n_steps
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        self.max_grad_norm = max_grad_norm

        # Networks
        self.actor = ActorNetwork(obs_dim, action_dim, hidden_dim).to(self.device)
        self.critic = CriticNetwork(obs_dim, hidden_dim).to(self.device)

        # Single optimizer for both networks
        self.optimizer = optim.RMSprop(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr, alpha=0.99, eps=1e-5,
        )

        # Step buffer
        self.obs_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.done_buffer = []
        self.log_prob_buffer = []

        self.training_stats = []

    @torch.no_grad()
    def select_action(
        self, obs: np.ndarray, deterministic: bool = False
    ) -> Tuple[np.ndarray, float, float]:
        """Select action, return (action, log_prob, value)."""
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

    def store_transition(
        self, obs: np.ndarray, action: np.ndarray,
        reward: float, done: bool, log_prob: float,
    ):
        """Store a single transition."""
        self.obs_buffer.append(obs)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)
        self.done_buffer.append(done)
        self.log_prob_buffer.append(log_prob)

    def should_update(self) -> bool:
        """Check if enough steps collected for an update."""
        return len(self.obs_buffer) >= self.n_steps

    def update(self) -> Dict[str, float]:
        """
        Perform A2C update using n-step returns.

        Computes n-step returns: R_t = r_t + gamma*r_{t+1} + ... + gamma^n * V(s_{t+n})
        Advantage: A_t = R_t - V(s_t)
        """
        obs_t = torch.FloatTensor(np.array(self.obs_buffer)).to(self.device)
        actions_t = torch.FloatTensor(np.array(self.action_buffer)).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_prob_buffer).to(self.device)

        rewards = np.array(self.reward_buffer)
        dones = np.array(self.done_buffer)

        # Bootstrap value for last state
        with torch.no_grad():
            last_obs = torch.FloatTensor(self.obs_buffer[-1]).unsqueeze(0).to(self.device)
            bootstrap = self.critic(last_obs).item() * (1 - dones[-1])

        # Compute n-step returns (backwards)
        n = len(rewards)
        returns = np.zeros(n)
        R = bootstrap
        for t in reversed(range(n)):
            R = rewards[t] + self.gamma * R * (1 - dones[t])
            returns[t] = R

        returns_t = torch.FloatTensor(returns).to(self.device)

        # Evaluate current policy and values
        dist = self.actor.get_distribution(obs_t)
        log_probs = dist.log_prob(actions_t).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1).mean()
        values = self.critic(obs_t).squeeze(-1)

        # Advantage
        advantages = returns_t - values.detach()

        # Actor loss
        actor_loss = -(log_probs * advantages).mean()
        actor_loss -= self.entropy_coeff * entropy

        # Critic loss
        critic_loss = nn.functional.mse_loss(values, returns_t)

        # Combined loss
        loss = actor_loss + self.value_coeff * critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            self.max_grad_norm,
        )
        self.optimizer.step()

        # Clear buffers
        self.obs_buffer.clear()
        self.action_buffer.clear()
        self.reward_buffer.clear()
        self.done_buffer.clear()
        self.log_prob_buffer.clear()

        stats = {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": entropy.item(),
            "mean_return": returns.mean(),
        }
        self.training_stats.append(stats)
        return stats

    def save(self, path: str):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
