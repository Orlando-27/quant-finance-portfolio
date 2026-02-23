"""
Reinforcement Learning for Portfolio Management
================================================
Deep RL framework for learning dynamic portfolio allocation policies
through interaction with a realistic market environment.

Modules:
    environments.portfolio_env - Custom Gymnasium trading environment
    environments.market_data   - Data loading and feature engineering
    agents.ppo_agent           - Proximal Policy Optimization
    agents.a2c_agent           - Advantage Actor-Critic
    agents.ddpg_agent          - Deep Deterministic Policy Gradient
    agents.networks            - Shared neural network architectures
    baselines                  - Traditional strategy benchmarks
    trainer                    - Training loop and experiment management
    evaluation                 - Performance comparison and visualization

Author: Jose Orlando Bobadilla Fuentes, CQF | MSc AI
"""

__version__ = "1.0.0"
__author__ = "Jose Orlando Bobadilla Fuentes"
