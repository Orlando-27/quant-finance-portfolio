"""
================================================================================
UTILITIES: VISUALIZATION, CONFIGURATION, AND HELPERS
================================================================================
Professional plotting functions for sentiment analysis diagnostics
and strategy performance visualization.

Author: Jose Orlando Bobadilla Fuentes, CQF | MSc AI
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Optional, Dict


# Global plot style
plt.rcParams.update({
    "figure.figsize": (14, 7),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})


def plot_cumulative_performance(
    cumulative: pd.DataFrame,
    title: str = "Cumulative Performance: Sentiment Strategy vs Benchmark",
    figsize: tuple = (14, 7),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot cumulative return curves for strategy and benchmark.

    Parameters
    ----------
    cumulative : pd.DataFrame
        Columns are portfolio names, index is dates, values are
        cumulative wealth (starting at 1.0).
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = {"Strategy (Gross)": "#1f77b4", "Strategy (Net)": "#2ca02c",
              "Benchmark": "#d62728"}

    for col in cumulative.columns:
        ax.plot(cumulative.index, cumulative[col], label=col,
                color=colors.get(col, None), linewidth=1.8)

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.legend(fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    fig.autofmt_xdate()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_drawdown(
    returns: pd.Series,
    title: str = "Strategy Drawdown",
    figsize: tuple = (14, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot underwater (drawdown) chart."""
    cum = (1 + returns).cumprod()
    peak = cum.expanding().max()
    dd = (cum - peak) / peak

    fig, ax = plt.subplots(figsize=figsize)
    ax.fill_between(dd.index, dd.values, 0, color="#d62728", alpha=0.4)
    ax.plot(dd.index, dd.values, color="#d62728", linewidth=0.8)
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel("Drawdown")
    ax.set_xlabel("Date")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_sentiment_time_series(
    features: pd.DataFrame,
    ticker: str,
    figsize: tuple = (14, 10),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot sentiment features over time for a single ticker.

    Panels:
        1. EWMA sentiment level + momentum
        2. Sentiment dispersion (disagreement)
        3. News volume
        4. Composite signal
    """
    tk = features[features["ticker"] == ticker].copy()
    tk = tk.set_index("date").sort_index()

    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)

    # Panel 1: Sentiment level
    axes[0].plot(tk.index, tk["sent_ewma"], color="#1f77b4", label="EWMA Level")
    if "sent_momentum" in tk.columns:
        ax2 = axes[0].twinx()
        ax2.bar(tk.index, tk["sent_momentum"], alpha=0.3, color="#ff7f0e",
                label="Momentum", width=1)
        ax2.set_ylabel("Momentum")
        ax2.legend(loc="upper left")
    axes[0].set_ylabel("Sentiment")
    axes[0].set_title(f"Sentiment Analysis: {ticker}", fontweight="bold")
    axes[0].legend(loc="upper right")

    # Panel 2: Dispersion
    axes[1].fill_between(tk.index, tk["sent_dispersion"], alpha=0.4, color="#9467bd")
    axes[1].set_ylabel("Dispersion")
    axes[1].set_title("Sentiment Disagreement")

    # Panel 3: Volume
    axes[2].bar(tk.index, tk["news_volume"], alpha=0.6, color="#2ca02c", width=1)
    axes[2].set_ylabel("Articles/Day")
    axes[2].set_title("News Volume (EWMA)")

    # Panel 4: Composite signal
    colors = np.where(tk["composite_signal"] > 0, "#2ca02c", "#d62728")
    axes[3].bar(tk.index, tk["composite_signal"], color=colors, alpha=0.7, width=1)
    axes[3].axhline(0, color="black", linewidth=0.5)
    axes[3].set_ylabel("Signal")
    axes[3].set_title("Composite Sentiment Signal")
    axes[3].set_xlabel("Date")

    fig.autofmt_xdate()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_ic_analysis(
    ic_series: pd.Series,
    title: str = "Information Coefficient Analysis",
    figsize: tuple = (14, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot IC time series, rolling IC, and IC distribution.

    Panels:
        1. Daily IC with rolling 21-day mean
        2. IC histogram with statistics annotation
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

    # Panel 1: IC time series
    ax1.bar(ic_series.index, ic_series.values, alpha=0.4, color="#1f77b4", width=1)
    rolling_ic = ic_series.rolling(21).mean()
    ax1.plot(ic_series.index, rolling_ic, color="#d62728",
             linewidth=2, label="21-day Rolling Mean")
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.set_title(title, fontweight="bold")
    ax1.set_ylabel("IC (Spearman)")
    ax1.legend()

    # Panel 2: IC distribution
    ax2.hist(ic_series.dropna(), bins=50, color="#1f77b4",
             alpha=0.7, edgecolor="white")
    mean_ic = ic_series.mean()
    std_ic = ic_series.std()
    icir = mean_ic / std_ic if std_ic > 0 else 0
    t_stat = mean_ic / (std_ic / np.sqrt(len(ic_series))) if std_ic > 0 else 0

    stats_text = (
        f"Mean IC: {mean_ic:.4f}\n"
        f"Std IC: {std_ic:.4f}\n"
        f"ICIR: {icir:.3f}\n"
        f"t-stat: {t_stat:.2f}\n"
        f"% Positive: {(ic_series > 0).mean():.1%}"
    )
    ax2.text(0.98, 0.95, stats_text, transform=ax2.transAxes,
             fontsize=11, verticalalignment="top", horizontalalignment="right",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax2.axvline(mean_ic, color="#d62728", linestyle="--", linewidth=2)
    ax2.set_xlabel("IC")
    ax2.set_ylabel("Frequency")
    ax2.set_title("IC Distribution")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_monthly_returns_heatmap(
    returns: pd.Series,
    title: str = "Monthly Returns Heatmap",
    figsize: tuple = (14, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Monthly returns heatmap (rows=years, columns=months)."""
    monthly = returns.resample("M").apply(lambda x: (1 + x).prod() - 1)
    table = pd.DataFrame({
        "Year": monthly.index.year,
        "Month": monthly.index.month,
        "Return": monthly.values,
    })
    pivot = table.pivot(index="Year", columns="Month", values="Return")
    pivot.columns = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ][:len(pivot.columns)]

    fig, ax = plt.subplots(figsize=figsize)
    import seaborn as sns
    sns.heatmap(
        pivot * 100, annot=True, fmt=".1f", cmap="RdYlGn",
        center=0, ax=ax, linewidths=0.5,
        cbar_kws={"label": "Return (%)"},
    )
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel("Year")
    ax.set_xlabel("Month")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
