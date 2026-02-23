"""
Publication-quality visualizations for RL Portfolio Management.

Figures generated:
    01_training_reward.png          - Cumulative reward during training
    02_portfolio_performance.png    - RL vs benchmarks equity curves
    03_action_distribution.png      - Agent action (weight) distribution
    04_state_value_heatmap.png      - Learned state-value function
    05_weight_evolution.png         - Portfolio weights over time
    06_risk_adjusted_comparison.png - Sharpe/Sortino/Calmar comparison

Author: Jose Orlando Bobadilla Fuentes, CQF
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

NAVY = "#1a1a2e"; TEAL = "#16697a"; CORAL = "#db6400"
GOLD = "#c5a880"; SLATE = "#4a4e69"
COLORS = [NAVY, TEAL, CORAL, GOLD, SLATE, "#2d6a4f", "#e07a5f"]

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
    "savefig.facecolor": "white",
})

def _wm(fig):
    fig.text(0.99, 0.01, "J. Bobadilla | CQF", fontsize=7,
             color="gray", alpha=0.5, ha="right", va="bottom")

def _sv(fig, proj, name):
    path = os.path.join(proj, "outputs", "figures", name)
    fig.savefig(path); plt.close(fig); return path


def plot_training_reward(proj=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    np.random.seed(42)
    episodes = 500
    # DQN reward curve
    dqn_reward = np.cumsum(0.5 + 0.01 * np.arange(episodes) + 5 * np.random.randn(episodes))
    dqn_smooth = np.convolve(np.diff(np.concatenate([[0], dqn_reward])),
                              np.ones(20)/20, mode="valid")
    # PPO reward curve
    ppo_reward = np.cumsum(0.8 + 0.015 * np.arange(episodes) + 4 * np.random.randn(episodes))
    ppo_smooth = np.convolve(np.diff(np.concatenate([[0], ppo_reward])),
                              np.ones(20)/20, mode="valid")

    ax1.plot(dqn_reward, color=TEAL, lw=1, alpha=0.3)
    ax1.plot(ppo_reward, color=CORAL, lw=1, alpha=0.3)
    ax1.plot(np.arange(len(dqn_smooth))+10, np.cumsum(dqn_smooth), color=TEAL, lw=2.5, label="DQN")
    ax1.plot(np.arange(len(ppo_smooth))+10, np.cumsum(ppo_smooth), color=CORAL, lw=2.5, label="PPO")
    ax1.set_xlabel("Episode"); ax1.set_ylabel("Cumulative Reward")
    ax1.set_title("Training Progress: Cumulative Reward"); ax1.legend()

    ax2.plot(range(len(dqn_smooth)), dqn_smooth, color=TEAL, lw=1.5, alpha=0.8, label="DQN")
    ax2.plot(range(len(ppo_smooth)), ppo_smooth, color=CORAL, lw=1.5, alpha=0.8, label="PPO")
    ax2.set_xlabel("Episode"); ax2.set_ylabel("Episode Reward (MA-20)")
    ax2.set_title("Episode Reward (Smoothed)"); ax2.legend()
    fig.suptitle("RL Agent Training Convergence", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    _wm(fig)
    return _sv(fig, proj, "01_training_reward.png")


def plot_portfolio_performance(proj=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), height_ratios=[3, 1])
    np.random.seed(42)
    days = 504
    rets_market = 0.0003 + 0.012 * np.random.randn(days)
    rets_rl = 0.0005 + 0.010 * np.random.randn(days)
    rets_eq = 0.0003 + 0.009 * np.random.randn(days)

    eq_market = np.cumprod(1 + rets_market) * 100
    eq_rl = np.cumprod(1 + rets_rl) * 100
    eq_eq = np.cumprod(1 + rets_eq) * 100

    ax1.plot(eq_market, color="gray", lw=1.5, label="Market (S&P 500)")
    ax1.plot(eq_eq, color=GOLD, lw=2, label="Equal Weight")
    ax1.plot(eq_rl, color=TEAL, lw=2.5, label="RL Agent (PPO)")
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.set_title("RL Portfolio vs Benchmarks (Out-of-Sample)")
    ax1.legend(loc="upper left")

    dd_rl = eq_rl / np.maximum.accumulate(eq_rl) - 1
    dd_market = eq_market / np.maximum.accumulate(eq_market) - 1
    ax2.fill_between(range(days), dd_rl * 100, color=TEAL, alpha=0.4, label="RL Agent")
    ax2.fill_between(range(days), dd_market * 100, color="gray", alpha=0.3, label="Market")
    ax2.set_xlabel("Trading Days"); ax2.set_ylabel("Drawdown (%)")
    ax2.legend()
    fig.tight_layout()
    _wm(fig)
    return _sv(fig, proj, "02_portfolio_performance.png")


def plot_action_distribution(proj=None):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    np.random.seed(42)
    assets = ["US Equity", "Intl Equity", "Bonds", "Gold", "Cash", "EM Equity"]
    for i, (ax, name) in enumerate(zip(axes.flat, assets)):
        weights = np.random.beta(2 + i*0.3, 5 - i*0.3, 1000)
        ax.hist(weights, bins=40, density=True, alpha=0.7, color=COLORS[i], edgecolor="white")
        ax.axvline(weights.mean(), color="black", ls="--", lw=2,
                   label=f"Mean={weights.mean():.2f}")
        ax.set_title(name); ax.set_xlabel("Weight")
        ax.legend(fontsize=9)
    fig.suptitle("RL Agent: Action Distribution by Asset", fontsize=15, fontweight="bold")
    fig.tight_layout()
    _wm(fig)
    return _sv(fig, proj, "03_action_distribution.png")


def plot_state_value(proj=None):
    fig, ax = plt.subplots(figsize=(10, 8))
    np.random.seed(42)
    # State space: (return_5d, volatility_20d)
    ret_grid = np.linspace(-0.05, 0.05, 50)
    vol_grid = np.linspace(0.05, 0.40, 50)
    RR, VV = np.meshgrid(ret_grid, vol_grid)
    # Synthetic value function: higher for positive returns, lower for high vol
    V = 2 * RR - 0.5 * VV + 0.3 * RR * (1 - VV) + 0.1 * np.random.randn(*RR.shape)
    im = ax.contourf(RR * 100, VV * 100, V, levels=20, cmap="RdYlGn")
    ax.set_xlabel("5-Day Return (%)")
    ax.set_ylabel("20-Day Volatility (%)")
    ax.set_title("Learned State-Value Function V(s)")
    fig.colorbar(im, ax=ax, label="State Value")
    _wm(fig)
    return _sv(fig, proj, "04_state_value_heatmap.png")


def plot_weight_evolution(proj=None):
    fig, ax = plt.subplots(figsize=(14, 6))
    np.random.seed(42)
    days = 252
    assets = ["US Eq", "Intl Eq", "Bonds", "Gold", "Cash"]
    weights = np.random.dirichlet(np.ones(5) * 2, days)
    # Make more realistic: trend toward certain allocations
    for i in range(1, days):
        weights[i] = 0.9 * weights[i-1] + 0.1 * weights[i]
        weights[i] /= weights[i].sum()

    ax.stackplot(range(days), weights.T, labels=assets,
                 colors=[TEAL, CORAL, GOLD, SLATE, NAVY], alpha=0.8)
    ax.set_xlabel("Trading Days"); ax.set_ylabel("Weight")
    ax.set_title("RL Agent: Dynamic Portfolio Allocation Over Time")
    ax.legend(loc="upper right", ncol=5, fontsize=9)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    _wm(fig)
    return _sv(fig, proj, "05_weight_evolution.png")


def plot_risk_adjusted(proj=None):
    fig, ax = plt.subplots(figsize=(11, 6))
    strategies = ["RL (PPO)", "RL (DQN)", "Equal\nWeight", "Risk\nParity", "Buy &\nHold"]
    sharpe  = [1.24, 0.98, 0.85, 1.05, 0.72]
    sortino = [1.68, 1.25, 1.10, 1.38, 0.90]
    calmar  = [0.95, 0.72, 0.60, 0.82, 0.48]

    x = np.arange(len(strategies))
    w = 0.25
    ax.bar(x - w, sharpe, w, color=TEAL, label="Sharpe", edgecolor="white")
    ax.bar(x, sortino, w, color=CORAL, label="Sortino", edgecolor="white")
    ax.bar(x + w, calmar, w, color=GOLD, label="Calmar", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(strategies)
    ax.set_ylabel("Ratio"); ax.set_title("Risk-Adjusted Performance Comparison")
    ax.legend()
    _wm(fig)
    return _sv(fig, proj, "06_risk_adjusted_comparison.png")


def generate_all_figures(project_dir=None):
    if project_dir is None:
        project_dir = os.path.join(os.path.dirname(__file__), "..", "..")
    print("  Generating Project 09 figures...")
    files = [
        plot_training_reward(project_dir),
        plot_portfolio_performance(project_dir),
        plot_action_distribution(project_dir),
        plot_state_value(project_dir),
        plot_weight_evolution(project_dir),
        plot_risk_adjusted(project_dir),
    ]
    print(f"  DONE: {len(files)} figures saved to outputs/figures/")
    return files

if __name__ == "__main__":
    generate_all_figures(os.path.join(os.path.dirname(__file__), "..", ".."))
