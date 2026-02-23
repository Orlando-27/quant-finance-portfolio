"""
================================================================================
PERFORMANCE COMPARISON: RL AGENTS VS TRADITIONAL BASELINES
================================================================================
Evaluates all strategies on the same held-out test period and produces
a comprehensive comparison report with visualization.

Author: Jose Orlando Bobadilla Fuentes, CQF | MSc AI
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, Optional


class StrategyEvaluator:
    """
    Unified evaluation framework for RL agents and traditional baselines.

    Parameters
    ----------
    returns : pd.DataFrame
        Test-period daily returns (dates x assets).
    risk_free_rate : float
        Annualized risk-free rate (default 0.04).
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        risk_free_rate: float = 0.04,
    ):
        self.returns = returns
        self.rf = risk_free_rate
        self.results = {}

    def add_strategy(
        self,
        name: str,
        weights: pd.DataFrame,
    ):
        """
        Add a strategy (RL or baseline) for evaluation.

        Parameters
        ----------
        name : str
            Strategy label.
        weights : pd.DataFrame
            Daily portfolio weights (dates x assets).
        """
        # Align weights with returns
        common = self.returns.index.intersection(weights.index)
        w = weights.loc[common]
        r = self.returns.loc[common]

        # Portfolio daily returns
        port_ret = (r * w).sum(axis=1)

        # Metrics
        ann_ret = port_ret.mean() * 252
        ann_vol = port_ret.std() * np.sqrt(252)
        sharpe = (ann_ret - self.rf) / ann_vol if ann_vol > 0 else 0

        # Sortino
        down = port_ret[port_ret < 0]
        down_vol = down.std() * np.sqrt(252) if len(down) > 0 else 1e-8
        sortino = (ann_ret - self.rf) / down_vol

        # Drawdown
        cum = (1 + port_ret).cumprod()
        peak = cum.expanding().max()
        dd = (cum - peak) / peak
        max_dd = abs(dd.min())
        calmar = ann_ret / max_dd if max_dd > 0 else 0

        # Turnover
        turnover = w.diff().abs().sum(axis=1).mean() * 252

        self.results[name] = {
            "ann_return": ann_ret,
            "ann_volatility": ann_vol,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown": max_dd,
            "calmar": calmar,
            "ann_turnover": turnover,
            "cumulative": cum,
            "drawdown": dd,
            "daily_returns": port_ret,
        }

    def summary_table(self) -> pd.DataFrame:
        """Return summary metrics as a formatted DataFrame."""
        rows = []
        for name, r in self.results.items():
            rows.append({
                "Strategy": name,
                "Ann. Return": f"{r['ann_return']:.2%}",
                "Volatility": f"{r['ann_volatility']:.2%}",
                "Sharpe": f"{r['sharpe']:.3f}",
                "Sortino": f"{r['sortino']:.3f}",
                "Max DD": f"{r['max_drawdown']:.2%}",
                "Calmar": f"{r['calmar']:.3f}",
                "Turnover": f"{r['ann_turnover']:.1f}x",
            })
        return pd.DataFrame(rows).set_index("Strategy")

    def plot_cumulative(
        self,
        figsize: tuple = (14, 7),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot cumulative returns for all strategies."""
        fig, ax = plt.subplots(figsize=figsize)
        for name, r in self.results.items():
            ax.plot(r["cumulative"].index, r["cumulative"].values,
                    label=name, linewidth=1.8)
        ax.set_title("Cumulative Performance: RL Agents vs Baselines",
                      fontweight="bold", fontsize=14)
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Return")
        ax.legend(fontsize=10, loc="upper left")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def plot_drawdowns(
        self,
        figsize: tuple = (14, 7),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot drawdown comparison."""
        fig, ax = plt.subplots(figsize=figsize)
        for name, r in self.results.items():
            ax.plot(r["drawdown"].index, r["drawdown"].values,
                    label=name, linewidth=1.2, alpha=0.8)
        ax.set_title("Drawdown Comparison", fontweight="bold", fontsize=14)
        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def plot_training_curves(
        self,
        episode_stats: list,
        figsize: tuple = (14, 10),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot training convergence curves."""
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

        episodes = range(1, len(episode_stats) + 1)

        # Panel 1: Episode reward
        rewards = [s.get("total_reward", 0) for s in episode_stats]
        axes[0].plot(episodes, rewards, alpha=0.3, color="#1f77b4")
        # Rolling average
        window = min(50, len(rewards) // 5 + 1)
        if window > 1:
            rolling = pd.Series(rewards).rolling(window).mean()
            axes[0].plot(episodes, rolling, color="#d62728", linewidth=2,
                         label=f"{window}-ep rolling avg")
        axes[0].set_ylabel("Episode Reward")
        axes[0].set_title("Training Progress", fontweight="bold")
        axes[0].legend()

        # Panel 2: Sharpe ratio
        sharpes = [s.get("sharpe_ratio", 0) for s in episode_stats]
        axes[1].plot(episodes, sharpes, alpha=0.3, color="#2ca02c")
        if window > 1:
            rolling_sr = pd.Series(sharpes).rolling(window).mean()
            axes[1].plot(episodes, rolling_sr, color="#d62728", linewidth=2)
        axes[1].set_ylabel("Sharpe Ratio")
        axes[1].axhline(0, color="black", linewidth=0.5, linestyle="--")

        # Panel 3: Max drawdown
        dds = [s.get("max_drawdown", 0) for s in episode_stats]
        axes[2].plot(episodes, dds, alpha=0.3, color="#9467bd")
        if window > 1:
            rolling_dd = pd.Series(dds).rolling(window).mean()
            axes[2].plot(episodes, rolling_dd, color="#d62728", linewidth=2)
        axes[2].set_ylabel("Max Drawdown")
        axes[2].set_xlabel("Episode")
        axes[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig
