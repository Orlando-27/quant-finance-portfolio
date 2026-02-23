"""
Publication-Quality Visualizations
====================================

Generates 8 professional dark-themed figures for the momentum & mean
reversion multi-asset strategy. Consistent style with the full
quantitative finance portfolio.

Figures:
    1. Cumulative Returns: Strategy vs Benchmark
    2. Regime Detection Timeline
    3. Signal Decomposition: Momentum vs Mean Reversion
    4. Rolling Performance Metrics (Sharpe, Volatility)
    5. Drawdown Analysis
    6. Asset Class Attribution
    7. Turnover & Transaction Cost Analysis
    8. Signal Heatmap Across Assets

Author: Jose Orlando Bobadilla Fuentes, CQF, MSc AI
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from typing import Dict, Optional
import os
import warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------
# Global style configuration -- consistent across all portfolio projects
# -----------------------------------------------------------------------
COLORS = {
    "bg": "#0a0a0a",
    "panel": "#1a1a2e",
    "grid": "#2a2a3e",
    "text": "#e0e0e0",
    "accent1": "#00d4ff",   # Cyan
    "accent2": "#ff6b6b",   # Coral
    "accent3": "#51cf66",   # Green
    "accent4": "#ffd43b",   # Gold
    "accent5": "#845ef7",   # Purple
    "accent6": "#ff922b",   # Orange
    "momentum": "#00d4ff",  # Cyan for momentum
    "mr": "#ff6b6b",        # Coral for mean reversion
    "composite": "#51cf66", # Green for blended
    "benchmark": "#666666", # Gray for benchmark
}

WATERMARK = "J. Bobadilla | CQF"


def _setup_style():
    """Configure global matplotlib style for dark theme."""
    plt.rcParams.update({
        "figure.facecolor": COLORS["bg"],
        "axes.facecolor": COLORS["panel"],
        "axes.edgecolor": COLORS["grid"],
        "axes.labelcolor": COLORS["text"],
        "axes.grid": True,
        "grid.color": COLORS["grid"],
        "grid.alpha": 0.3,
        "text.color": COLORS["text"],
        "xtick.color": COLORS["text"],
        "ytick.color": COLORS["text"],
        "font.family": "sans-serif",
        "font.size": 10,
        "legend.facecolor": COLORS["panel"],
        "legend.edgecolor": COLORS["grid"],
    })


def _add_watermark(fig):
    """Add professional watermark to figure."""
    fig.text(
        0.99, 0.01, WATERMARK,
        fontsize=8, color="#444444",
        ha="right", va="bottom",
        style="italic", alpha=0.7,
    )


def plot_cumulative_returns(backtest: Dict, output_dir: str):
    """
    Figure 1: Cumulative Returns -- Strategy vs Benchmark.

    Shows wealth evolution of the blended strategy (net of costs),
    gross strategy, and equal-weight benchmark.
    """
    _setup_style()
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1],
                              gridspec_kw={"hspace": 0.05})

    # Upper panel: cumulative returns
    ax = axes[0]
    cum_net = backtest["cumulative_returns"]
    cum_gross = backtest["cumulative_gross"]
    cum_bm = backtest["benchmark_cumulative"]

    ax.plot(cum_net.index, cum_net.values, color=COLORS["composite"],
            linewidth=2.0, label="Strategy (Net)")
    ax.plot(cum_gross.index, cum_gross.values, color=COLORS["momentum"],
            linewidth=1.0, alpha=0.5, linestyle="--", label="Strategy (Gross)")
    ax.plot(cum_bm.index, cum_bm.values, color=COLORS["benchmark"],
            linewidth=1.5, label="Benchmark (EW)")

    ax.fill_between(cum_net.index, cum_net.values, cum_bm.values,
                     where=cum_net.values > cum_bm.values,
                     alpha=0.1, color=COLORS["accent3"])
    ax.fill_between(cum_net.index, cum_net.values, cum_bm.values,
                     where=cum_net.values <= cum_bm.values,
                     alpha=0.1, color=COLORS["accent2"])

    m = backtest["metrics"]
    info_text = (
        f"Ann. Return: {m['annualized_return']:.1%}  |  "
        f"Sharpe: {m['sharpe_ratio']:.2f}  |  "
        f"Max DD: {m['max_drawdown']:.1%}  |  "
        f"Calmar: {m['calmar_ratio']:.2f}"
    )
    ax.text(0.02, 0.95, info_text, transform=ax.transAxes,
            fontsize=9, color=COLORS["accent4"],
            bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS["bg"], alpha=0.8))

    ax.set_ylabel("Cumulative Return")
    ax.set_title("Momentum & Mean Reversion: Cumulative Performance",
                  fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xticklabels([])

    # Lower panel: rolling outperformance
    ax2 = axes[1]
    excess = backtest["portfolio_returns"] - backtest["benchmark_returns"]
    rolling_excess = excess.rolling(63).mean() * 252
    ax2.fill_between(rolling_excess.index, 0, rolling_excess.values,
                      where=rolling_excess.values > 0,
                      color=COLORS["accent3"], alpha=0.4, label="Outperformance")
    ax2.fill_between(rolling_excess.index, 0, rolling_excess.values,
                      where=rolling_excess.values <= 0,
                      color=COLORS["accent2"], alpha=0.4, label="Underperformance")
    ax2.axhline(0, color=COLORS["text"], linewidth=0.5, alpha=0.5)
    ax2.set_ylabel("Rolling Excess\n(Ann., 3M)")
    ax2.set_xlabel("Date")
    ax2.legend(loc="lower left", fontsize=8)

    _add_watermark(fig)
    plt.savefig(os.path.join(output_dir, "01_cumulative_returns.png"),
                dpi=300, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()


def plot_regime_timeline(regime_components: pd.DataFrame, output_dir: str):
    """
    Figure 2: Regime Detection Timeline.

    Shows the evolution of volatility, dispersion, and autocorrelation
    regimes over time, along with the composite momentum alpha.
    """
    _setup_style()
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True,
                              gridspec_kw={"hspace": 0.08})

    panels = [
        ("volatility", "Volatility Regime", COLORS["accent2"]),
        ("dispersion", "Dispersion Regime", COLORS["accent4"]),
        ("autocorrelation", "Autocorrelation Regime", COLORS["accent5"]),
        ("composite_mom_alpha", "Composite Momentum Alpha", COLORS["momentum"]),
    ]

    for ax, (col, title, color) in zip(axes, panels):
        if col not in regime_components.columns:
            continue
        data = regime_components[col].dropna()
        ax.fill_between(data.index, 0, data.values, alpha=0.4, color=color)
        ax.plot(data.index, data.values, color=color, linewidth=1.0)
        ax.axhline(0.5, color=COLORS["text"], linewidth=0.5, linestyle="--", alpha=0.3)
        ax.set_ylabel(title, fontsize=9)
        ax.set_ylim(0, 1)

        # Shade regimes
        if col == "composite_mom_alpha":
            ax.axhspan(0.65, 1.0, alpha=0.05, color=COLORS["momentum"])
            ax.axhspan(0.0, 0.35, alpha=0.05, color=COLORS["mr"])
            ax.text(0.01, 0.92, "MOMENTUM", transform=ax.transAxes,
                    fontsize=7, color=COLORS["momentum"], alpha=0.6)
            ax.text(0.01, 0.05, "MEAN REVERSION", transform=ax.transAxes,
                    fontsize=7, color=COLORS["mr"], alpha=0.6)

    axes[0].set_title("Market Regime Detection Timeline",
                       fontsize=14, fontweight="bold", pad=15)
    axes[-1].set_xlabel("Date")

    _add_watermark(fig)
    plt.savefig(os.path.join(output_dir, "02_regime_timeline.png"),
                dpi=300, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()


def plot_signal_decomposition(
    mom_signal: pd.DataFrame,
    mr_signal: pd.DataFrame,
    composite: pd.DataFrame,
    output_dir: str,
):
    """
    Figure 3: Signal Decomposition -- Momentum vs Mean Reversion.

    Shows cross-sectional average signals over time for each component
    and the blended composite.
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=(14, 6))

    # Average signal across assets
    avg_mom = mom_signal.mean(axis=1).rolling(21).mean()
    avg_mr = mr_signal.mean(axis=1).rolling(21).mean()
    avg_comp = composite.mean(axis=1).rolling(21).mean()

    ax.plot(avg_mom.index, avg_mom.values, color=COLORS["momentum"],
            linewidth=1.5, label="Momentum Signal", alpha=0.8)
    ax.plot(avg_mr.index, avg_mr.values, color=COLORS["mr"],
            linewidth=1.5, label="Mean Reversion Signal", alpha=0.8)
    ax.plot(avg_comp.index, avg_comp.values, color=COLORS["composite"],
            linewidth=2.0, label="Composite (Blended)", alpha=0.9)

    ax.axhline(0, color=COLORS["text"], linewidth=0.5, alpha=0.3)
    ax.fill_between(avg_comp.index, 0, avg_comp.values,
                     where=avg_comp.values > 0,
                     alpha=0.1, color=COLORS["accent3"])
    ax.fill_between(avg_comp.index, 0, avg_comp.values,
                     where=avg_comp.values <= 0,
                     alpha=0.1, color=COLORS["accent2"])

    ax.set_title("Signal Decomposition: Momentum vs Mean Reversion (21D MA)",
                  fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Average Signal Strength")
    ax.set_xlabel("Date")
    ax.legend(loc="upper right", fontsize=9)

    _add_watermark(fig)
    plt.savefig(os.path.join(output_dir, "03_signal_decomposition.png"),
                dpi=300, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()


def plot_rolling_metrics(backtest: Dict, output_dir: str):
    """
    Figure 4: Rolling Performance Metrics.

    Shows rolling 1-year Sharpe ratio and rolling annualized volatility.
    """
    _setup_style()
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                              gridspec_kw={"hspace": 0.08})

    net_ret = backtest["portfolio_returns"]
    window = 252

    # Rolling Sharpe
    ax = axes[0]
    rolling_mean = net_ret.rolling(window).mean() * 252
    rolling_std = net_ret.rolling(window).std() * np.sqrt(252)
    rolling_sharpe = (rolling_mean - 0.02) / rolling_std

    ax.plot(rolling_sharpe.index, rolling_sharpe.values,
            color=COLORS["accent1"], linewidth=1.5)
    ax.axhline(0, color=COLORS["text"], linewidth=0.5, alpha=0.3)
    ax.axhline(1.0, color=COLORS["accent3"], linewidth=0.5, linestyle="--", alpha=0.5)
    ax.axhline(-1.0, color=COLORS["accent2"], linewidth=0.5, linestyle="--", alpha=0.5)
    ax.fill_between(rolling_sharpe.index, 0, rolling_sharpe.values,
                     where=rolling_sharpe.values > 0,
                     alpha=0.15, color=COLORS["accent3"])
    ax.fill_between(rolling_sharpe.index, 0, rolling_sharpe.values,
                     where=rolling_sharpe.values <= 0,
                     alpha=0.15, color=COLORS["accent2"])
    ax.set_ylabel("Rolling Sharpe (1Y)")
    ax.set_title("Rolling Performance Metrics",
                  fontsize=14, fontweight="bold", pad=15)

    # Rolling Volatility
    ax2 = axes[1]
    ax2.plot(rolling_std.index, rolling_std.values * 100,
             color=COLORS["accent4"], linewidth=1.5, label="Strategy")
    bm_vol = backtest["benchmark_returns"].rolling(window).std() * np.sqrt(252) * 100
    ax2.plot(bm_vol.index, bm_vol.values, color=COLORS["benchmark"],
             linewidth=1.0, label="Benchmark")
    ax2.axhline(10.0, color=COLORS["accent5"], linewidth=0.5, linestyle="--",
                alpha=0.5, label="Vol Target (10%)")
    ax2.set_ylabel("Annualized Vol (%)")
    ax2.set_xlabel("Date")
    ax2.legend(loc="upper right", fontsize=8)

    _add_watermark(fig)
    plt.savefig(os.path.join(output_dir, "04_rolling_metrics.png"),
                dpi=300, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()


def plot_drawdown_analysis(backtest: Dict, output_dir: str):
    """
    Figure 5: Drawdown Analysis.

    Shows underwater chart for the strategy and benchmark, highlighting
    the top-5 worst drawdown periods.
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=(14, 5))

    cum = backtest["cumulative_returns"]
    peak = cum.cummax()
    dd = (cum - peak) / peak * 100  # Percentage

    cum_bm = backtest["benchmark_cumulative"]
    peak_bm = cum_bm.cummax()
    dd_bm = (cum_bm - peak_bm) / peak_bm * 100

    ax.fill_between(dd.index, 0, dd.values, color=COLORS["accent2"],
                     alpha=0.4, label="Strategy DD")
    ax.plot(dd.index, dd.values, color=COLORS["accent2"], linewidth=1.0)
    ax.plot(dd_bm.index, dd_bm.values, color=COLORS["benchmark"],
            linewidth=1.0, alpha=0.6, label="Benchmark DD")

    ax.set_title("Underwater Chart: Drawdown Analysis",
                  fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("Date")
    ax.legend(loc="lower left", fontsize=9)

    max_dd = backtest["metrics"]["max_drawdown"]
    ax.text(0.98, 0.05,
            f"Max Drawdown: {max_dd:.1%}",
            transform=ax.transAxes, fontsize=10, color=COLORS["accent2"],
            ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS["bg"], alpha=0.8))

    _add_watermark(fig)
    plt.savefig(os.path.join(output_dir, "05_drawdown_analysis.png"),
                dpi=300, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()


def plot_asset_class_attribution(
    attribution: pd.DataFrame,
    output_dir: str,
):
    """
    Figure 6: Asset Class Return Attribution.

    Horizontal bar chart showing annualized return and risk contribution
    by asset class.
    """
    _setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    palette = [COLORS["accent1"], COLORS["accent2"],
               COLORS["accent3"], COLORS["accent4"]]

    # Return attribution
    ax = axes[0]
    classes = attribution.index.tolist()
    returns_pct = attribution["ann_return"].values * 100
    colors = [palette[i % len(palette)] for i in range(len(classes))]
    bars = ax.barh(classes, returns_pct, color=colors, alpha=0.8, edgecolor="none")
    ax.axvline(0, color=COLORS["text"], linewidth=0.5)
    ax.set_xlabel("Annualized Return Contribution (%)")
    ax.set_title("Return Attribution", fontsize=12, fontweight="bold")

    for bar, val in zip(bars, returns_pct):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=9, color=COLORS["text"])

    # Exposure attribution
    ax2 = axes[1]
    exposure = attribution["avg_gross_exposure"].values
    bars2 = ax2.barh(classes, exposure, color=colors, alpha=0.8, edgecolor="none")
    ax2.set_xlabel("Average Gross Exposure")
    ax2.set_title("Exposure by Asset Class", fontsize=12, fontweight="bold")

    for bar, val in zip(bars2, exposure):
        ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=9, color=COLORS["text"])

    fig.suptitle("Performance Attribution by Asset Class",
                  fontsize=14, fontweight="bold", y=1.02)

    _add_watermark(fig)
    plt.savefig(os.path.join(output_dir, "06_asset_class_attribution.png"),
                dpi=300, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()


def plot_turnover_analysis(backtest: Dict, output_dir: str):
    """
    Figure 7: Turnover & Transaction Cost Analysis.

    Shows rolling turnover and cumulative transaction costs over time.
    """
    _setup_style()
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                              gridspec_kw={"hspace": 0.08})

    turnover = backtest["turnover"]

    # Rolling turnover
    ax = axes[0]
    rolling_to = turnover.rolling(21).mean() * 252  # Annualized
    ax.fill_between(rolling_to.index, 0, rolling_to.values,
                     color=COLORS["accent5"], alpha=0.3)
    ax.plot(rolling_to.index, rolling_to.values, color=COLORS["accent5"],
            linewidth=1.5)
    ax.set_ylabel("Ann. Turnover (21D MA)")
    ax.set_title("Turnover & Transaction Cost Analysis",
                  fontsize=14, fontweight="bold", pad=15)

    avg_to = backtest["metrics"]["avg_annual_turnover"]
    ax.axhline(avg_to, color=COLORS["accent4"], linewidth=1.0,
               linestyle="--", alpha=0.5, label=f"Avg: {avg_to:.1f}x")
    ax.legend(loc="upper right", fontsize=9)

    # Cumulative cost drag
    ax2 = axes[1]
    gross = backtest["gross_returns"]
    net = backtest["portfolio_returns"]
    cum_cost = ((1 + gross).cumprod() - (1 + net).cumprod()) * 100
    ax2.fill_between(cum_cost.index, 0, cum_cost.values,
                      color=COLORS["accent6"], alpha=0.3)
    ax2.plot(cum_cost.index, cum_cost.values, color=COLORS["accent6"],
             linewidth=1.5)
    ax2.set_ylabel("Cumulative TC Drag (%)")
    ax2.set_xlabel("Date")

    total_tc = backtest["metrics"]["total_transaction_costs"]
    ax2.text(0.98, 0.85,
             f"Total TC Drag: {total_tc:.2%}",
             transform=ax2.transAxes, fontsize=10, color=COLORS["accent6"],
             ha="right",
             bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS["bg"], alpha=0.8))

    _add_watermark(fig)
    plt.savefig(os.path.join(output_dir, "07_turnover_analysis.png"),
                dpi=300, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()


def plot_signal_heatmap(weights: pd.DataFrame, output_dir: str):
    """
    Figure 8: Signal Heatmap Across Assets.

    Shows a monthly-sampled heatmap of portfolio weights, color-coded
    from red (short) to green (long).
    """
    _setup_style()

    # Resample to monthly for readability
    monthly_w = weights.resample("M").last()
    # Take last 60 months max
    monthly_w = monthly_w.tail(60)

    fig, ax = plt.subplots(figsize=(14, 7))

    data = monthly_w.values.T
    im = ax.imshow(
        data,
        aspect="auto",
        cmap="RdYlGn",
        vmin=-0.2,
        vmax=0.2,
        interpolation="nearest",
    )

    ax.set_yticks(range(len(monthly_w.columns)))
    ax.set_yticklabels(monthly_w.columns, fontsize=8)

    # X-axis: show every 6th month
    n_months = len(monthly_w)
    step = max(n_months // 10, 1)
    tick_pos = list(range(0, n_months, step))
    tick_labels = [monthly_w.index[i].strftime("%Y-%m") for i in tick_pos]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)

    ax.set_title("Portfolio Weight Heatmap (Monthly)",
                  fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Date")
    ax.set_ylabel("Asset")

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Weight", fontsize=9)

    _add_watermark(fig)
    plt.savefig(os.path.join(output_dir, "08_signal_heatmap.png"),
                dpi=300, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()


def generate_all_figures(project_dir: str):
    """
    Generate all 8 publication-quality figures.

    This is the main entry point for visualization. It runs the full
    strategy pipeline and produces all figures.

    Parameters
    ----------
    project_dir : str
        Root directory of the project (e.g., ~/quant-finance-portfolio/13-...).
    """
    import sys
    sys.path.insert(0, project_dir)

    from src.data_generator import generate_multi_asset_data, get_asset_class_map
    from src.momentum import TimeSeriesMomentum, CrossSectionalMomentum
    from src.mean_reversion import CompositeMeanReversion
    from src.regime import RegimeDetector
    from src.portfolio import PortfolioConstructor
    from src.backtesting import BacktestEngine

    output_dir = os.path.join(project_dir, "outputs", "figures")
    os.makedirs(output_dir, exist_ok=True)

    print("  Generating synthetic multi-asset data...")
    prices, returns, metadata = generate_multi_asset_data(n_years=15, seed=42)
    class_map = get_asset_class_map(metadata)

    print("  Computing momentum signals...")
    tsmom = TimeSeriesMomentum(lookback_days=252, vol_target=0.10)
    csmom = CrossSectionalMomentum(lookback_months=12, skip_months=1)
    mom_ts = tsmom.compute_signal(returns)
    mom_cs = csmom.compute_signal(returns)
    # Blend TSMOM and CS-MOM equally
    mom_signal = 0.5 * mom_ts.fillna(0) + 0.5 * mom_cs.fillna(0)

    print("  Computing mean-reversion signals...")
    mr = CompositeMeanReversion()
    mr_signal = mr.compute_signal(prices, returns)

    print("  Detecting regimes...")
    regime_det = RegimeDetector()
    regime_alpha, regime_components = regime_det.compute_regime_scores(returns)

    print("  Constructing portfolio...")
    pc = PortfolioConstructor(vol_target=0.10, rebalance_freq=21)
    weights = pc.construct_portfolio(
        mom_signal=mom_signal.fillna(0),
        mr_signal=mr_signal.fillna(0),
        regime_alpha=regime_alpha.fillna(0.5),
        returns=returns,
        asset_class_map=class_map,
    )

    print("  Running backtest...")
    engine = BacktestEngine(transaction_cost_bps=10, slippage_bps=5)
    bt = engine.run_backtest(weights, returns)

    # Composite signal for decomposition plot
    composite = pc.blend_signals(
        mom_signal.fillna(0), mr_signal.fillna(0), regime_alpha.fillna(0.5)
    )

    # Attribution
    attribution = engine.attribution_by_class(weights, returns, class_map)

    print("  Generating figures...")
    plot_cumulative_returns(bt, output_dir)
    print("    [1/8] Cumulative Returns")
    plot_regime_timeline(regime_components, output_dir)
    print("    [2/8] Regime Timeline")
    plot_signal_decomposition(mom_signal, mr_signal, composite, output_dir)
    print("    [3/8] Signal Decomposition")
    plot_rolling_metrics(bt, output_dir)
    print("    [4/8] Rolling Metrics")
    plot_drawdown_analysis(bt, output_dir)
    print("    [5/8] Drawdown Analysis")
    plot_asset_class_attribution(attribution, output_dir)
    print("    [6/8] Asset Class Attribution")
    plot_turnover_analysis(bt, output_dir)
    print("    [7/8] Turnover Analysis")
    plot_signal_heatmap(weights, output_dir)
    print("    [8/8] Signal Heatmap")

    n_figs = len([f for f in os.listdir(output_dir) if f.endswith(".png")])
    print(f"  DONE: {n_figs} figures saved to outputs/figures/")

    # Print strategy metrics summary
    m = bt["metrics"]
    print("\n  === STRATEGY METRICS ===")
    print(f"  Annualized Return:   {m['annualized_return']:.2%}")
    print(f"  Annualized Vol:      {m['annualized_volatility']:.2%}")
    print(f"  Sharpe Ratio:        {m['sharpe_ratio']:.3f}")
    print(f"  Sortino Ratio:       {m['sortino_ratio']:.3f}")
    print(f"  Max Drawdown:        {m['max_drawdown']:.2%}")
    print(f"  Calmar Ratio:        {m['calmar_ratio']:.3f}")
    print(f"  Hit Rate:            {m['hit_rate']:.1%}")
    print(f"  Profit Factor:       {m['profit_factor']:.2f}")
    print(f"  Information Ratio:   {m['information_ratio']:.3f}")
    print(f"  Avg Annual Turnover: {m['avg_annual_turnover']:.1f}x")
