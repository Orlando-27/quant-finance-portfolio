"""
================================================================================
TRAINING LOOP AND EXPERIMENT MANAGEMENT
================================================================================
Orchestrates the training of RL agents in the portfolio environment
with logging, checkpointing, and walk-forward evaluation.

Training procedure:
    1. Load market data and split into train/test
    2. Create environment with training data
    3. Train agent for N episodes, logging metrics each episode
    4. Evaluate on held-out test data
    5. Compare against traditional baselines

Author: Jose Orlando Bobadilla Fuentes, CQF | MSc AI
================================================================================
"""

import numpy as np
import time
from typing import Dict, Optional, List
from tqdm import tqdm

from src.environments.portfolio_env import PortfolioEnv
from src.environments.market_data import MarketDataLoader
from src.agents.ppo_agent import PPOAgent
from src.agents.a2c_agent import A2CAgent
from src.agents.ddpg_agent import DDPGAgent


class Trainer:
    """
    RL training loop for portfolio management.

    Parameters
    ----------
    tickers : list of str
        Asset universe.
    agent_type : str
        'ppo', 'a2c', or 'ddpg'.
    reward_type : str
        Reward function for the environment.
    n_episodes : int
        Number of training episodes.
    use_synthetic : bool
        If True, use synthetic data (no API dependency).
    seed : int
        Random seed for reproducibility.
    agent_params : dict, optional
        Override default agent hyperparameters.
    env_params : dict, optional
        Override default environment parameters.
    """

    AGENT_MAP = {
        "ppo": PPOAgent,
        "a2c": A2CAgent,
        "ddpg": DDPGAgent,
    }

    def __init__(
        self,
        tickers: List[str],
        agent_type: str = "ppo",
        reward_type: str = "differential_sharpe",
        n_episodes: int = 500,
        use_synthetic: bool = True,
        seed: int = 42,
        agent_params: Optional[Dict] = None,
        env_params: Optional[Dict] = None,
    ):
        self.tickers = tickers
        self.agent_type = agent_type
        self.reward_type = reward_type
        self.n_episodes = n_episodes
        self.seed = seed
        self.agent_params = agent_params or {}
        self.env_params = env_params or {}

        np.random.seed(seed)

        # Load data
        self.data_loader = MarketDataLoader(tickers)
        if use_synthetic:
            self.data_loader.generate_synthetic(seed=seed)
        else:
            self.data_loader.download()
        self.data_loader.compute_features()

        # Train/test split
        (
            self.train_returns, self.test_returns,
            self.train_features, self.test_features,
        ) = self.data_loader.get_train_test_split(test_ratio=0.2)

        # Create training environment
        env_kwargs = {
            "reward_type": self.reward_type,
            "transaction_cost": 0.001,
            "max_drawdown": 0.25,
            **self.env_params,
        }
        self.train_env = PortfolioEnv(
            self.train_returns, self.train_features, **env_kwargs
        )
        self.test_env = PortfolioEnv(
            self.test_returns, self.test_features, **env_kwargs
        )

        # Create agent
        obs_dim = self.train_env.obs_dim
        action_dim = len(tickers)
        AgentClass = self.AGENT_MAP[agent_type]
        self.agent = AgentClass(obs_dim, action_dim, **self.agent_params)

        # Training history
        self.episode_rewards = []
        self.episode_stats = []

    def _train_episode_ppo(self) -> Dict:
        """Run one PPO episode: collect rollout, then update."""
        obs, info = self.train_env.reset(seed=self.seed)
        total_reward = 0.0
        done = False

        while not done:
            action, log_prob, value = self.agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = self.train_env.step(action)
            done = terminated or truncated

            self.agent.buffer.add(obs, action, reward, done, log_prob, value)
            obs = next_obs
            total_reward += reward

        # Update after full episode
        update_stats = self.agent.update()
        episode_stats = self.train_env.get_episode_stats()
        episode_stats["total_reward"] = total_reward
        episode_stats.update(update_stats)
        return episode_stats

    def _train_episode_a2c(self) -> Dict:
        """Run one A2C episode: update every n_steps."""
        obs, info = self.train_env.reset(seed=self.seed)
        total_reward = 0.0
        done = False

        while not done:
            action, log_prob, value = self.agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = self.train_env.step(action)
            done = terminated or truncated

            self.agent.store_transition(obs, action, reward, done, log_prob)

            if self.agent.should_update():
                self.agent.update()

            obs = next_obs
            total_reward += reward

        # Final update for remaining steps
        if len(self.agent.obs_buffer) > 0:
            self.agent.update()

        episode_stats = self.train_env.get_episode_stats()
        episode_stats["total_reward"] = total_reward
        return episode_stats

    def _train_episode_ddpg(self) -> Dict:
        """Run one DDPG episode: store transitions, update each step."""
        obs, info = self.train_env.reset(seed=self.seed)
        total_reward = 0.0
        done = False

        while not done:
            action = self.agent.select_action(obs, deterministic=False)
            next_obs, reward, terminated, truncated, info = self.train_env.step(action)
            done = terminated or truncated

            self.agent.store_transition(obs, action, reward, next_obs, done)
            self.agent.update()

            obs = next_obs
            total_reward += reward

        self.agent.end_episode()
        episode_stats = self.train_env.get_episode_stats()
        episode_stats["total_reward"] = total_reward
        return episode_stats

    def train(self) -> List[Dict]:
        """
        Execute the full training loop.

        Returns
        -------
        List of episode statistics dictionaries.
        """
        train_fn = {
            "ppo": self._train_episode_ppo,
            "a2c": self._train_episode_a2c,
            "ddpg": self._train_episode_ddpg,
        }[self.agent_type]

        print(f"Training {self.agent_type.upper()} for {self.n_episodes} episodes")
        print(f"  Universe: {self.tickers}")
        print(f"  Reward: {self.reward_type}")
        print(f"  Train: {len(self.train_returns)} days | Test: {len(self.test_returns)} days")
        print("-" * 60)

        for ep in tqdm(range(self.n_episodes), desc=f"{self.agent_type.upper()}"):
            stats = train_fn()
            self.episode_stats.append(stats)
            self.episode_rewards.append(stats.get("total_reward", 0))

            # Log every 50 episodes
            if (ep + 1) % 50 == 0:
                recent = self.episode_stats[-50:]
                avg_ret = np.mean([s.get("ann_return", 0) for s in recent])
                avg_sr = np.mean([s.get("sharpe_ratio", 0) for s in recent])
                print(f"  Ep {ep+1:4d} | Avg Return: {avg_ret:+.2%} | "
                      f"Avg Sharpe: {avg_sr:.3f}")

        return self.episode_stats

    def evaluate(self, deterministic: bool = True) -> Dict:
        """
        Evaluate trained agent on test environment.

        Returns
        -------
        dict with episode stats on held-out test data.
        """
        obs, info = self.test_env.reset()
        done = False
        weights_history = []

        while not done:
            if self.agent_type == "ddpg":
                action = self.agent.select_action(obs, deterministic=deterministic)
            else:
                action, _, _ = self.agent.select_action(obs, deterministic=deterministic)

            obs, reward, terminated, truncated, info = self.test_env.step(action)
            done = terminated or truncated
            weights_history.append(info["weights"].copy())

        stats = self.test_env.get_episode_stats()
        stats["weights_history"] = np.array(weights_history)
        return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RL Portfolio Training")
    parser.add_argument("--agent", default="ppo", choices=["ppo", "a2c", "ddpg"])
    parser.add_argument("--reward", default="differential_sharpe")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tickers = ["SPY", "TLT", "GLD", "QQQ", "IWM"]

    trainer = Trainer(
        tickers=tickers,
        agent_type=args.agent,
        reward_type=args.reward,
        n_episodes=args.episodes,
        use_synthetic=True,
        seed=args.seed,
    )
    trainer.train()
    test_stats = trainer.evaluate()

    print("\n" + "=" * 60)
    print("  TEST EVALUATION")
    print("=" * 60)
    for k, v in test_stats.items():
        if k != "weights_history":
            print(f"  {k:20s}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
