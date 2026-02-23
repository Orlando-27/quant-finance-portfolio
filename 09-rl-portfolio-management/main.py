"""
Reinforcement Learning for Portfolio Management - Main Entry Point
===================================================================
Demonstrates the complete RL pipeline: environment setup, agent training
(PPO, A2C, DDPG), evaluation against traditional baselines (equal-weight,
mean-variance, risk parity), and comprehensive performance analysis.

Author: Jose Orlando Bobadilla Fuentes, CQF
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from src.environments.market_data import MarketDataPipeline
from src.environments.portfolio_env import PortfolioEnv
from src.agents.ppo_agent import PPOAgent
from src.agents.a2c_agent import A2CAgent
from src.agents.ddpg_agent import DDPGAgent
from src.baselines import BaselineStrategies
from src.trainer import RLTrainer
from src.evaluation import PortfolioEvaluator


def main():
    """Run the complete RL portfolio management pipeline."""
    print("=" * 70)
    print("REINFORCEMENT LEARNING FOR PORTFOLIO MANAGEMENT")
    print("=" * 70)

    np.random.seed(42)

    # --- Configuration ---
    config = {
        "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM",
                     "JNJ", "XOM", "GLD", "TLT", "VNQ"],
        "start_date": "2015-01-01",
        "end_date": "2024-12-31",
        "train_ratio": 0.7,
        "initial_capital": 1_000_000,
        "transaction_cost_bps": 10,
        "episodes": 100,
        "window_size": 60,
    }
    print(f"\n  Assets: {len(config['tickers'])} tickers")
    print(f"  Period: {config['start_date']} to {config['end_date']}")

    # --- Step 1: Data Pipeline ---
    print("\n[1/6] Loading and processing market data...")
    pipeline = MarketDataPipeline()
    data = pipeline.load(config["tickers"], config["start_date"], config["end_date"])
    train_data, test_data = pipeline.split(data, config["train_ratio"])
    print(f"  Train: {len(train_data)} days | Test: {len(test_data)} days")

    # --- Step 2: Environment ---
    print("\n[2/6] Creating portfolio environment...")
    train_env = PortfolioEnv(
        returns=train_data,
        window_size=config["window_size"],
        initial_capital=config["initial_capital"],
        transaction_cost_bps=config["transaction_cost_bps"],
    )
    test_env = PortfolioEnv(
        returns=test_data,
        window_size=config["window_size"],
        initial_capital=config["initial_capital"],
        transaction_cost_bps=config["transaction_cost_bps"],
    )
    state_dim = train_env.observation_space_dim
    action_dim = train_env.action_space_dim
    print(f"  State dim: {state_dim} | Action dim: {action_dim}")

    # --- Step 3: Train Agents ---
    print("\n[3/6] Training RL agents...")
    trainer = RLTrainer(episodes=config["episodes"])

    agents = {
        "PPO": PPOAgent(state_dim, action_dim),
        "A2C": A2CAgent(state_dim, action_dim),
        "DDPG": DDPGAgent(state_dim, action_dim),
    }

    for name, agent in agents.items():
        print(f"  Training {name}...", end=" ", flush=True)
        history = trainer.train(agent, train_env)
        print(f"final reward: {history[-1]:.4f}")

    # --- Step 4: Baselines ---
    print("\n[4/6] Computing baseline strategies...")
    baselines = BaselineStrategies(config["tickers"])
    baseline_results = {
        "Equal Weight": baselines.equal_weight(test_data),
        "Min Variance": baselines.min_variance(train_data, test_data),
        "Risk Parity": baselines.risk_parity(train_data, test_data),
    }
    for name, res in baseline_results.items():
        print(f"  {name:15s} | Return: {res['total_return']:7.2%} "
              f"| Sharpe: {res['sharpe']:.3f}")

    # --- Step 5: Evaluate on Test Set ---
    print("\n[5/6] Evaluating agents on test set...")
    evaluator = PortfolioEvaluator(initial_capital=config["initial_capital"])
    agent_results = {}
    for name, agent in agents.items():
        res = evaluator.evaluate(agent, test_env)
        agent_results[name] = res
        print(f"  {name:15s} | Return: {res['total_return']:7.2%} "
              f"| Sharpe: {res['sharpe']:.3f} | MaxDD: {res['max_drawdown']:.2%}")

    # --- Step 6: Summary ---
    print("\n[6/6] Comparative summary...")
    print("\n" + "=" * 70)
    print(f"{'Strategy':20s} {'Return':>10s} {'Sharpe':>10s} {'MaxDD':>10s} {'Calmar':>10s}")
    print("-" * 70)
    all_results = {**baseline_results, **agent_results}
    for name, res in all_results.items():
        calmar = abs(res['total_return'] / res['max_drawdown']) if res['max_drawdown'] != 0 else 0
        print(f"{name:20s} {res['total_return']:9.2%} {res['sharpe']:10.3f} "
              f"{res['max_drawdown']:9.2%} {calmar:10.3f}")

    print("=" * 70)
    print("PIPELINE COMPLETE")
    print("  All visualizations available via PortfolioEvaluator.plot_* methods")
    print("=" * 70)


if __name__ == "__main__":
    main()
