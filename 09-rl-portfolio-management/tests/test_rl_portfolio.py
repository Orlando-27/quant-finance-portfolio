"""
================================================================================
UNIT TESTS -- REINFORCEMENT LEARNING FOR PORTFOLIO MANAGEMENT
================================================================================
Tests cover:
    1. Market data loader (synthetic generation, features, train/test split)
    2. Portfolio environment (reset, step, reward functions, constraints)
    3. Neural networks (forward pass shapes, initialization)
    4. RL agents (action selection, buffer operations, update)
    5. Baselines (weight generation, constraints)
    6. Evaluation metrics

Author: Jose Orlando Bobadilla Fuentes, CQF | MSc AI
================================================================================
"""

import pytest
import numpy as np
import pandas as pd
import torch


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def tickers():
    return ["SPY", "TLT", "GLD", "QQQ", "IWM"]

@pytest.fixture
def market_data(tickers):
    from src.environments.market_data import MarketDataLoader
    loader = MarketDataLoader(tickers)
    loader.generate_synthetic(n_days=500, seed=42)
    loader.compute_features()
    return loader

@pytest.fixture
def env(market_data):
    from src.environments.portfolio_env import PortfolioEnv
    ret = market_data.returns.iloc[:400]
    feat = market_data.features.iloc[:400]
    common = ret.index.intersection(feat.index)
    return PortfolioEnv(ret.loc[common], feat.loc[common])


# ============================================================================
# TEST: MARKET DATA
# ============================================================================

class TestMarketData:

    def test_synthetic_generation(self, market_data):
        assert market_data.prices.shape[1] == 5
        assert len(market_data.prices) == 2520

    def test_returns_computed(self, market_data):
        assert market_data.returns is not None
        assert market_data.returns.shape[1] == 5

    def test_features_computed(self, market_data):
        assert market_data.features is not None
        assert market_data.feature_dim > 0
        # No NaN in features (dropped during compute)
        assert not market_data.features.isna().any().any()

    def test_train_test_split(self, market_data):
        tr_ret, te_ret, tr_feat, te_feat = market_data.get_train_test_split(0.2)
        assert len(tr_ret) > len(te_ret)
        assert tr_ret.index[-1] < te_ret.index[0]  # No overlap


# ============================================================================
# TEST: PORTFOLIO ENVIRONMENT
# ============================================================================

class TestPortfolioEnv:

    def test_reset(self, env):
        obs, info = env.reset()
        assert obs.shape == (env.obs_dim,)
        assert info["wealth"] == 1.0

    def test_step(self, env):
        obs, _ = env.reset()
        action = np.random.randn(env.n_assets)
        next_obs, reward, terminated, truncated, info = env.step(action)
        assert next_obs.shape == obs.shape
        assert isinstance(reward, float)
        assert "weights" in info
        assert "turnover" in info

    def test_weights_valid(self, env):
        env.reset()
        action = np.random.randn(env.n_assets)
        _, _, _, _, info = env.step(action)
        w = info["weights"]
        assert np.all(w >= 0)
        assert w.sum() <= 1.0 + 1e-6

    def test_max_position_enforced(self, env):
        env.reset()
        # Extreme action favoring one asset
        action = np.zeros(env.n_assets)
        action[0] = 100.0
        _, _, _, _, info = env.step(action)
        assert info["weights"][0] <= env.max_position + 1e-6

    def test_episode_runs_to_completion(self, env):
        obs, _ = env.reset()
        done = False
        steps = 0
        while not done and steps < 500:
            action = np.random.randn(env.n_assets) * 0.1
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
        assert steps > 0

    def test_reward_types(self, market_data):
        from src.environments.portfolio_env import PortfolioEnv
        ret = market_data.returns.iloc[:100]
        feat = market_data.features.iloc[:100]
        common = ret.index.intersection(feat.index)

        for rtype in ["log_return", "differential_sharpe", "risk_adjusted", "sortino"]:
            env = PortfolioEnv(ret.loc[common], feat.loc[common], reward_type=rtype)
            obs, _ = env.reset()
            _, reward, _, _, _ = env.step(np.random.randn(env.n_assets))
            assert isinstance(reward, float)
            assert not np.isnan(reward)


# ============================================================================
# TEST: NEURAL NETWORKS
# ============================================================================

class TestNetworks:

    def test_actor_output_shape(self):
        from src.agents.networks import ActorNetwork
        actor = ActorNetwork(obs_dim=20, action_dim=5, hidden_dim=64)
        obs = torch.randn(4, 20)
        mean, log_std = actor(obs)
        assert mean.shape == (4, 5)
        assert log_std.shape == (4, 5)

    def test_critic_output_shape(self):
        from src.agents.networks import CriticNetwork
        critic = CriticNetwork(obs_dim=20, hidden_dim=64)
        obs = torch.randn(4, 20)
        value = critic(obs)
        assert value.shape == (4, 1)

    def test_deterministic_actor(self):
        from src.agents.networks import DeterministicActor
        actor = DeterministicActor(obs_dim=20, action_dim=5, hidden_dim=64)
        obs = torch.randn(4, 20)
        action = actor(obs)
        assert action.shape == (4, 5)
        # Tanh output bounded in [-1, 1]
        assert torch.all(action >= -1.0) and torch.all(action <= 1.0)

    def test_q_network(self):
        from src.agents.networks import QNetwork
        q_net = QNetwork(obs_dim=20, action_dim=5, hidden_dim=64)
        obs = torch.randn(4, 20)
        action = torch.randn(4, 5)
        q_val = q_net(obs, action)
        assert q_val.shape == (4, 1)


# ============================================================================
# TEST: RL AGENTS
# ============================================================================

class TestPPOAgent:

    def test_action_selection(self):
        from src.agents.ppo_agent import PPOAgent
        agent = PPOAgent(obs_dim=20, action_dim=5, hidden_dim=64)
        obs = np.random.randn(20).astype(np.float32)
        action, log_prob, value = agent.select_action(obs)
        assert action.shape == (5,)
        assert isinstance(log_prob, float)
        assert isinstance(value, float)

    def test_buffer_and_update(self):
        from src.agents.ppo_agent import PPOAgent
        agent = PPOAgent(obs_dim=10, action_dim=3, hidden_dim=32, batch_size=8)
        # Collect some data
        for _ in range(20):
            obs = np.random.randn(10).astype(np.float32)
            action, lp, val = agent.select_action(obs)
            agent.buffer.add(obs, action, 0.01, False, lp, val)
        stats = agent.update()
        assert "actor_loss" in stats
        assert "critic_loss" in stats


class TestDDPGAgent:

    def test_action_selection(self):
        from src.agents.ddpg_agent import DDPGAgent
        agent = DDPGAgent(obs_dim=20, action_dim=5, hidden_dim=64, warmup_steps=5)
        obs = np.random.randn(20).astype(np.float32)
        action = agent.select_action(obs, deterministic=False)
        assert action.shape == (5,)
        assert np.all(action >= -1.0) and np.all(action <= 1.0)

    def test_replay_buffer(self):
        from src.agents.ddpg_agent import ReplayBuffer
        buf = ReplayBuffer(capacity=100)
        for _ in range(50):
            buf.add(
                np.random.randn(10), np.random.randn(3),
                0.01, np.random.randn(10), False,
            )
        assert len(buf) == 50
        batch = buf.sample(16, torch.device("cpu"))
        assert batch["obs"].shape == (16, 10)


# ============================================================================
# TEST: BASELINES
# ============================================================================

class TestBaselines:

    def test_equal_weight(self):
        from src.baselines import EqualWeightBaseline
        ret = pd.DataFrame(
            np.random.randn(100, 4) * 0.01,
            columns=["A", "B", "C", "D"],
        )
        ew = EqualWeightBaseline(n_assets=4)
        w = ew.generate_weights(ret)
        assert w.shape == ret.shape
        assert np.allclose(w.iloc[0].values, 0.25)

    def test_risk_parity_weights_sum(self):
        from src.baselines import RiskParityBaseline
        ret = pd.DataFrame(
            np.random.randn(200, 3) * 0.01,
            columns=["X", "Y", "Z"],
        )
        rp = RiskParityBaseline(lookback=63, rebalance_freq=21)
        w = rp.generate_weights(ret)
        # Weights should sum to ~1.0
        assert np.allclose(w.sum(axis=1), 1.0, atol=1e-6)

    def test_momentum_top_k(self):
        from src.baselines import MomentumBaseline
        np.random.seed(42)
        ret = pd.DataFrame(
            np.random.randn(300, 5) * 0.01,
            columns=["A", "B", "C", "D", "E"],
        )
        mom = MomentumBaseline(top_k=2, lookback=252, rebalance_freq=21)
        w = mom.generate_weights(ret)
        # After lookback, at most 2 assets should have non-zero weight
        last_w = w.iloc[-1]
        assert (last_w > 0).sum() <= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
