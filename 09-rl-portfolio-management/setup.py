"""
Setup for Reinforcement Learning for Portfolio Management.

Author: Jose Orlando Bobadilla Fuentes, CQF | MSc AI
"""
from setuptools import setup, find_packages

setup(
    name="rl-portfolio-management",
    version="1.0.0",
    author="Jose Orlando Bobadilla Fuentes",
    description=(
        "Deep RL agents (PPO, A2C, DDPG) for dynamic portfolio allocation "
        "in a custom Gymnasium environment with realistic market mechanics."
    ),
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "torch>=2.0.0",
        "gymnasium>=0.29.0",
        "yfinance>=0.2.28",
        "matplotlib>=3.7.0",
    ],
    keywords=[
        "reinforcement-learning", "portfolio-management", "ppo", "ddpg",
        "quantitative-finance", "deep-learning", "gymnasium",
    ],
)
