"""
Publication-Quality Pairs Trading Visualizations
=================================================

Generates 8 professional figures covering cointegration analysis,
spread dynamics, OU calibration, Kalman filtering, trading signals,
and backtest performance.

Figure Catalog:
    1. Price Series & Normalized Spread (pair overview)
    2. Engle-Granger Cointegration Diagnostics (residuals + ADF)
    3. Ornstein-Uhlenbeck Calibration (fitted OU + half-life + simulation)
    4. Kalman Filter Adaptive Hedge Ratio (time-varying beta)
    5. Z-Score Trading Signals (signals overlay on spread)
    6. Trade P&L Distribution (histogram + win/loss analysis)
    7. Strategy Cumulative Returns (equity curve + drawdown)
    8. Pair Selection Heatmap (cointegration p-value matrix)

Author: Jose Orlando Bobadilla Fuentes, CQF, MSc AI
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from typing import Dict, Optional
import os

# -- Professional dark style --
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "axes.grid": True,
    "grid.color": "#21262d",
    "grid.alpha": 0.6,
    "text.color": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.facecolor": "#161b22",
    "legend.edgecolor": "#30363d",
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.facecolor": "#0d1117",
})

COLORS = ["#58a6ff", "#f0883e", "#3fb950", "#bc8cff",
          "#f778ba", "#79c0ff", "#d2a8ff", "#ffa657"]


def _save(fig, path, name):
    fpath = os.path.join(path, "outputs", "figures", name)
    fig.savefig(fpath, dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"    [FIG] {name}")


def plot_pair_overview(pa, pb, spread, pair_name, save_path):
    """Fig 1: Price series and normalized spread."""
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.2)

    ax1 = fig.add_subplot(gs[0])
    ax1b = ax1.twinx()
    ax1.plot(pa.index, pa, color=COLORS[0], linewidth=1.2, label=pair_name[0])
    ax1b.plot(pb.index, pb, color=COLORS[1], linewidth=1.2, label=pair_name[1])
    ax1.set_ylabel(f"{pair_name[0]} Price", color=COLORS[0])
    ax1b.set_ylabel(f"{pair_name[1]} Price", color=COLORS[1])
    ax1.set_title(f"Pair: {pair_name[0]} / {pair_name[1]} -- Price Series",
                   fontsize=14, fontweight="bold")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    ax2 = fig.add_subplot(gs[1])
    ax2.plot(spread.index, spread, color=COLORS[2], linewidth=1)
    ax2.axhline(spread.mean(), color=COLORS[3], linestyle="--", linewidth=0.8,
                label=f"Mean: {spread.mean():.4f}")
    ax2.axhline(spread.mean() + 2 * spread.std(), color="#f85149",
                linestyle=":", linewidth=0.7, alpha=0.7, label="+/- 2 std")
    ax2.axhline(spread.mean() - 2 * spread.std(), color="#f85149",
                linestyle=":", linewidth=0.7, alpha=0.7)
    ax2.set_title("Log-Price Spread (Cointegrating Residual)", fontweight="bold")
    ax2.set_ylabel("Spread")
    ax2.legend(loc="upper right")

    _save(fig, save_path, "01_pair_overview.png")


def plot_cointegration_diagnostics(eg_results, save_path):
    """Fig 2: Engle-Granger cointegration diagnostics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    resid = eg_results["residuals"]

    # Residual time series
    axes[0, 0].plot(resid.index, resid, color=COLORS[0], linewidth=0.8)
    axes[0, 0].axhline(0, color="#8b949e", linewidth=0.5)
    axes[0, 0].set_title("Cointegrating Residuals", fontweight="bold")
    axes[0, 0].set_ylabel("Residual")

    # Residual histogram
    axes[0, 1].hist(resid.dropna(), bins=50, color=COLORS[1], alpha=0.7,
                     edgecolor="#30363d", density=True)
    axes[0, 1].set_title("Residual Distribution", fontweight="bold")
    axes[0, 1].set_xlabel("Residual Value")

    # ACF of residuals
    from statsmodels.tsa.stattools import acf
    acf_vals = acf(resid.dropna(), nlags=30)
    axes[1, 0].bar(range(len(acf_vals)), acf_vals, color=COLORS[2], alpha=0.8)
    ci = 1.96 / np.sqrt(len(resid.dropna()))
    axes[1, 0].axhline(ci, color="#f85149", linestyle="--", linewidth=0.7)
    axes[1, 0].axhline(-ci, color="#f85149", linestyle="--", linewidth=0.7)
    axes[1, 0].set_title("Residual Autocorrelation (ACF)", fontweight="bold")
    axes[1, 0].set_xlabel("Lag")

    # Test summary box
    axes[1, 1].axis("off")
    summary_text = (
        f"Engle-Granger Cointegration Test\n"
        f"{'=' * 35}\n"
        f"ADF Statistic: {eg_results['adf_stat']:.4f}\n"
        f"ADF p-value:   {eg_results['adf_pvalue']:.6f}\n"
        f"Hedge Ratio:   {eg_results['hedge_ratio']:.4f}\n"
        f"Intercept:     {eg_results['intercept']:.4f}\n"
        f"Cointegrated:  {eg_results['cointegrated']}\n"
        f"{'=' * 35}\n"
        f"Critical Values:\n"
    )
    for k, v in eg_results.get("critical_values", {}).items():
        summary_text += f"  {k}: {v:.4f}\n"

    axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                     fontsize=10, verticalalignment="top", color="#c9d1d9",
                     fontfamily="monospace",
                     bbox=dict(boxstyle="round,pad=0.5", facecolor="#21262d",
                               edgecolor="#30363d"))

    fig.suptitle("Cointegration Diagnostics", fontsize=14, fontweight="bold",
                  y=1.01)
    _save(fig, save_path, "02_cointegration_diagnostics.png")


def plot_ou_calibration(ou, spread, save_path):
    """Fig 3: OU process calibration and simulation."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Actual vs OU fit
    axes[0].plot(spread.index, spread.values, color=COLORS[0],
                 linewidth=0.8, label="Actual Spread", alpha=0.8)
    axes[0].axhline(ou.theta, color=COLORS[1], linestyle="--",
                     linewidth=1.2, label=f"theta = {ou.theta:.4f}")
    stat = ou.stationary_distribution()
    axes[0].axhline(ou.theta + 2 * stat["std"], color="#f85149",
                     linestyle=":", linewidth=0.7, alpha=0.6)
    axes[0].axhline(ou.theta - 2 * stat["std"], color="#f85149",
                     linestyle=":", linewidth=0.7, alpha=0.6)
    axes[0].set_title("Spread with OU Parameters", fontweight="bold")
    axes[0].legend(fontsize=8)

    # Monte Carlo fan chart
    paths = ou.simulate(spread.values[-1], n_steps=60, n_paths=500)
    pctiles = [5, 25, 50, 75, 95]
    pct_vals = np.percentile(paths, pctiles, axis=1)
    x_sim = range(paths.shape[0])

    axes[1].fill_between(x_sim, pct_vals[0], pct_vals[4],
                          color=COLORS[2], alpha=0.15, label="5-95% CI")
    axes[1].fill_between(x_sim, pct_vals[1], pct_vals[3],
                          color=COLORS[2], alpha=0.3, label="25-75% CI")
    axes[1].plot(x_sim, pct_vals[2], color=COLORS[2], linewidth=1.5,
                 label="Median")
    axes[1].axhline(ou.theta, color=COLORS[1], linestyle="--", linewidth=1)
    axes[1].set_title("60-Day OU Simulation (Fan Chart)", fontweight="bold")
    axes[1].set_xlabel("Days Forward")
    axes[1].legend(fontsize=8)

    # Parameter summary
    axes[2].axis("off")
    params_text = (
        f"Ornstein-Uhlenbeck Parameters\n"
        f"{'=' * 32}\n"
        f"kappa (mean-rev speed): {ou.kappa:.4f}\n"
        f"theta (long-run mean):  {ou.theta:.4f}\n"
        f"sigma (volatility):     {ou.sigma:.4f}\n"
        f"{'=' * 32}\n"
        f"Half-life: {ou.half_life * 252:.1f} days\n"
        f"Stationary std: {stat['std']:.4f}\n"
        f"{'=' * 32}"
    )
    axes[2].text(0.1, 0.85, params_text, transform=axes[2].transAxes,
                  fontsize=11, verticalalignment="top", color="#c9d1d9",
                  fontfamily="monospace",
                  bbox=dict(boxstyle="round,pad=0.5", facecolor="#21262d",
                            edgecolor="#30363d"))

    fig.suptitle("Ornstein-Uhlenbeck Process Calibration", fontsize=14,
                  fontweight="bold", y=1.02)
    _save(fig, save_path, "03_ou_calibration.png")


def plot_kalman_hedge_ratio(kf_result, save_path):
    """Fig 4: Kalman filter adaptive hedge ratio."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[2, 1])

    betas = kf_result["betas"]
    spread = kf_result["spreads"]

    # Hedge ratio evolution
    axes[0].plot(betas.index, betas["hedge_ratio"], color=COLORS[0],
                 linewidth=1.2, label="Kalman Hedge Ratio")
    axes[0].axhline(betas["hedge_ratio"].mean(), color=COLORS[1],
                     linestyle="--", linewidth=0.8,
                     label=f"Mean: {betas['hedge_ratio'].mean():.4f}")
    axes[0].fill_between(
        betas.index,
        betas["hedge_ratio"] - betas["hedge_ratio"].rolling(60).std(),
        betas["hedge_ratio"] + betas["hedge_ratio"].rolling(60).std(),
        color=COLORS[0], alpha=0.15
    )
    axes[0].set_title("Time-Varying Hedge Ratio (Kalman Filter)",
                       fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Hedge Ratio (beta)")
    axes[0].legend()

    # Kalman spread
    axes[1].plot(spread.index, spread, color=COLORS[2], linewidth=0.8)
    axes[1].axhline(0, color="#8b949e", linewidth=0.5)
    mu = spread.rolling(60).mean()
    sigma = spread.rolling(60).std()
    axes[1].fill_between(spread.index, mu - 2 * sigma, mu + 2 * sigma,
                          color=COLORS[2], alpha=0.15, label="+/- 2 std")
    axes[1].set_title("Kalman-Filtered Spread", fontweight="bold")
    axes[1].set_ylabel("Spread")
    axes[1].legend()

    _save(fig, save_path, "04_kalman_hedge_ratio.png")


def plot_trading_signals(signals_df, pair_name, save_path):
    """Fig 5: Z-score trading signals overlaid on spread."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[1, 1])

    z = signals_df["zscore"]
    pos = signals_df["position"]

    # Z-score with entry/exit bands
    axes[0].plot(z.index, z, color=COLORS[0], linewidth=0.8, alpha=0.8)
    axes[0].axhline(2.0, color="#f85149", linestyle="--", linewidth=0.8,
                     label="Entry (+/-2.0)")
    axes[0].axhline(-2.0, color="#f85149", linestyle="--", linewidth=0.8)
    axes[0].axhline(0.5, color=COLORS[2], linestyle=":", linewidth=0.7,
                     label="Exit (+/-0.5)")
    axes[0].axhline(-0.5, color=COLORS[2], linestyle=":", linewidth=0.7)
    axes[0].axhline(0, color="#8b949e", linewidth=0.5)
    axes[0].fill_between(z.index, 0, z, where=pos > 0, color=COLORS[2],
                          alpha=0.2, label="Long Spread")
    axes[0].fill_between(z.index, 0, z, where=pos < 0, color="#f85149",
                          alpha=0.2, label="Short Spread")
    axes[0].set_title(f"Z-Score Trading Signals: {pair_name[0]}/{pair_name[1]}",
                       fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Z-Score")
    axes[0].legend(loc="upper right", ncol=2, fontsize=8)

    # Position timeline
    axes[1].fill_between(pos.index, 0, pos, where=pos > 0,
                          color=COLORS[2], alpha=0.6, label="Long")
    axes[1].fill_between(pos.index, 0, pos, where=pos < 0,
                          color="#f85149", alpha=0.6, label="Short")
    axes[1].set_title("Position", fontweight="bold")
    axes[1].set_ylabel("Position")
    axes[1].set_yticks([-1, 0, 1])
    axes[1].set_yticklabels(["Short", "Flat", "Long"])
    axes[1].legend()

    _save(fig, save_path, "05_trading_signals.png")


def plot_trade_pnl_distribution(strategy_result, trade_stats, save_path):
    """Fig 6: Trade P&L distribution and win/loss analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    r = strategy_result["net_return"].dropna()

    # Daily return histogram
    axes[0].hist(r[r > 0], bins=40, color=COLORS[2], alpha=0.7,
                  label="Wins", edgecolor="#30363d")
    axes[0].hist(r[r <= 0], bins=40, color="#f85149", alpha=0.7,
                  label="Losses", edgecolor="#30363d")
    axes[0].axvline(r.mean(), color=COLORS[3], linestyle="--", linewidth=1.2,
                     label=f"Mean: {r.mean()*100:.3f}%")
    axes[0].set_title("Daily Return Distribution", fontweight="bold")
    axes[0].set_xlabel("Daily Return")
    axes[0].legend()

    # Trade statistics box
    axes[1].axis("off")
    ts = trade_stats
    stats_text = (
        f"Trade Statistics\n"
        f"{'=' * 30}\n"
        f"Total Trades:    {ts.get('n_trades', 0)}\n"
        f"  Long Trades:   {ts.get('n_long', 0)}\n"
        f"  Short Trades:  {ts.get('n_short', 0)}\n"
        f"{'=' * 30}\n"
        f"Win Rate:        {ts.get('win_rate', 0)*100:.1f}%\n"
        f"Avg Win:         {ts.get('avg_win', 0)*100:.3f}%\n"
        f"Avg Loss:        {ts.get('avg_loss', 0)*100:.3f}%\n"
        f"Profit Factor:   {ts.get('profit_factor', 0):.2f}\n"
        f"{'=' * 30}\n"
        f"Avg Holding:     {ts.get('avg_holding_days', 0):.1f} days\n"
        f"Stop Losses:     {ts.get('n_stop_losses', 0)}"
        f" ({ts.get('stop_loss_pct', 0):.1f}%)\n"
        f"{'=' * 30}"
    )
    axes[1].text(0.15, 0.85, stats_text, transform=axes[1].transAxes,
                  fontsize=11, verticalalignment="top", color="#c9d1d9",
                  fontfamily="monospace",
                  bbox=dict(boxstyle="round,pad=0.5", facecolor="#21262d",
                            edgecolor="#30363d"))

    fig.suptitle("Trade Performance Analysis", fontsize=14,
                  fontweight="bold", y=1.02)
    _save(fig, save_path, "06_trade_pnl_distribution.png")


def plot_equity_curve(backtest_result, perf_summary, save_path):
    """Fig 7: Strategy equity curve with drawdown."""
    fig = plt.figure(figsize=(14, 9))
    gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1], hspace=0.2)

    cum = backtest_result["cumulative_return"]
    r = backtest_result["portfolio_return"]
    dd = cum / cum.cummax() - 1

    # Equity curve
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(cum.index, cum, color=COLORS[0], linewidth=1.5)
    ax1.axhline(1.0, color="#8b949e", linestyle="--", linewidth=0.5)
    ax1.set_title("Pairs Trading Strategy -- Equity Curve (Net of TC)",
                   fontsize=14, fontweight="bold")
    ax1.set_ylabel("Cumulative Wealth")

    # Performance annotation
    ps = perf_summary
    ann_text = (
        f"Ann.Ret: {ps['Ann. Return']*100:.1f}% | "
        f"Ann.Vol: {ps['Ann. Volatility']*100:.1f}% | "
        f"Sharpe: {ps['Sharpe Ratio']:.2f} | "
        f"MaxDD: {ps['Max Drawdown']*100:.1f}%"
    )
    ax1.text(0.02, 0.05, ann_text, transform=ax1.transAxes, fontsize=9,
             color="#c9d1d9", bbox=dict(boxstyle="round,pad=0.3",
                                        facecolor="#21262d",
                                        edgecolor="#30363d", alpha=0.9))

    # Drawdown
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.fill_between(dd.index, 0, dd, color="#f85149", alpha=0.5)
    ax2.set_ylabel("Drawdown")
    ax2.set_title("Underwater Curve", fontweight="bold", fontsize=11)

    # Active pairs
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.fill_between(backtest_result.index,
                      0, backtest_result["n_active_pairs"],
                      color=COLORS[2], alpha=0.5)
    ax3.set_ylabel("# Pairs")
    ax3.set_title("Active Pairs per Period", fontweight="bold", fontsize=11)
    ax3.set_xlabel("Date")

    _save(fig, save_path, "07_equity_curve.png")


def plot_pair_selection_heatmap(pair_scores, tickers, save_path):
    """Fig 8: Cointegration p-value heatmap for all pairs."""
    n = len(tickers)
    pval_matrix = pd.DataFrame(np.nan, index=tickers, columns=tickers)

    for _, row in pair_scores.iterrows():
        t1, t2 = row["pair"]
        pval_matrix.loc[t1, t2] = row["adf_pvalue"]
        pval_matrix.loc[t2, t1] = row["adf_pvalue"]

    np.fill_diagonal(pval_matrix.values, 0.0)

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(pval_matrix, dtype=bool), k=1)

    cmap = sns.color_palette("RdYlGn_r", as_cmap=True)
    sns.heatmap(pval_matrix.astype(float), mask=mask, annot=True,
                fmt=".3f", cmap=cmap, vmin=0, vmax=0.2, ax=ax,
                square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8, "label": "ADF p-value"},
                annot_kws={"size": 8, "color": "#c9d1d9"})

    ax.set_title("Cointegration P-Value Matrix (Engle-Granger ADF)",
                  fontsize=14, fontweight="bold", pad=15)

    _save(fig, save_path, "08_pair_selection_heatmap.png")


def generate_all_figures(project_path):
    """
    Master function: generates all 8 publication-quality figures.

    Pipeline:
    1. Generate synthetic cointegrated pairs universe
    2. Run pair selection and cointegration tests
    3. Calibrate OU process
    4. Fit Kalman filter
    5. Generate trading signals
    6. Run walk-forward backtest
    7. Produce all visualizations
    """
    import sys
    sys.path.insert(0, project_path)

    from src.cointegration import EngleGranger, JohansenTest
    from src.pair_selection import PairSelector
    from src.ornstein_uhlenbeck import OrnsteinUhlenbeck
    from src.kalman_filter import KalmanHedgeRatio
    from src.strategy import PairsTradingStrategy
    from src.backtesting import PairsBacktester

    print("\n  Generating Project 12 visualizations...")
    print("  " + "-" * 50)

    # -- Step 1: Generate synthetic cointegrated universe --
    print("  [1/8] Generating synthetic cointegrated universe...")
    rng = np.random.RandomState(42)
    T = 1260  # ~5 years daily
    dates = pd.bdate_range("2019-01-02", periods=T)

    # Create 10 stocks with embedded cointegration structure
    # Group 1: 3 cointegrated stocks (common stochastic trend)
    trend1 = np.cumsum(rng.normal(0.0003, 0.01, T))
    s1 = np.exp(trend1 + rng.normal(0, 0.02, T).cumsum() * 0.1 + np.log(50))
    s2 = np.exp(0.8 * trend1 + rng.normal(0, 0.02, T).cumsum() * 0.1 + np.log(40))
    s3 = np.exp(1.2 * trend1 + rng.normal(0, 0.02, T).cumsum() * 0.1 + np.log(60))

    # Group 2: 2 cointegrated stocks
    trend2 = np.cumsum(rng.normal(0.0002, 0.012, T))
    s4 = np.exp(trend2 + rng.normal(0, 0.015, T).cumsum() * 0.08 + np.log(100))
    s5 = np.exp(0.9 * trend2 + rng.normal(0, 0.015, T).cumsum() * 0.08 + np.log(90))

    # Group 3: 5 independent stocks (not cointegrated)
    s6 = np.exp(np.cumsum(rng.normal(0.0004, 0.018, T)) + np.log(70))
    s7 = np.exp(np.cumsum(rng.normal(0.0001, 0.022, T)) + np.log(30))
    s8 = np.exp(np.cumsum(rng.normal(0.0003, 0.015, T)) + np.log(55))
    s9 = np.exp(np.cumsum(rng.normal(-0.0001, 0.020, T)) + np.log(45))
    s10 = np.exp(np.cumsum(rng.normal(0.0002, 0.016, T)) + np.log(80))

    tickers = ["ALPHA", "BETA", "GAMMA", "DELTA", "EPSILON",
               "ZETA", "ETA", "THETA", "IOTA", "KAPPA"]
    prices = pd.DataFrame(
        np.column_stack([s1, s2, s3, s4, s5, s6, s7, s8, s9, s10]),
        index=dates, columns=tickers
    )

    # -- Step 2: Pair selection --
    print("  [2/8] Running pair selection...")
    selector = PairSelector(method="cointegration", top_k=5)
    selected = selector.select(prices)
    print(f"    Selected {len(selected)} tradeable pairs")

    if not selected:
        print("    WARNING: No pairs found, using forced pair for plots")
        selected = [{"pair": ("ALPHA", "BETA"), "hedge_ratio": 0.8,
                      "half_life": 15, "adf_stat": -3.5, "adf_pvalue": 0.01,
                      "tradeable": True}]

    best_pair = selected[0]
    t1, t2 = best_pair["pair"]
    print(f"    Best pair: {t1}/{t2}")

    pa = prices[t1]
    pb = prices[t2]

    # -- Step 3: Engle-Granger cointegration --
    print("  [3/8] Running Engle-Granger cointegration test...")
    eg = EngleGranger(significance=0.05)
    eg_res = eg.test(np.log(pa), np.log(pb))
    print(eg.get_summary())
    spread = eg_res["residuals"]

    # -- Figure 1: Pair overview --
    plot_pair_overview(pa, pb, spread, (t1, t2), project_path)

    # -- Figure 2: Cointegration diagnostics --
    plot_cointegration_diagnostics(eg_res, project_path)

    # -- Step 4: OU calibration --
    print("  [4/8] Calibrating Ornstein-Uhlenbeck process...")
    ou = OrnsteinUhlenbeck(dt=1.0 / 252)
    ou_ols = ou.fit_ols(spread.values)
    ou_mle = ou.fit_mle(spread.values)
    print(f"    OLS: kappa={ou_ols['kappa']:.4f}, "
          f"HL={ou_ols['half_life_days']:.1f}d")
    print(f"    MLE: kappa={ou_mle['kappa']:.4f}, "
          f"HL={ou_mle['half_life_days']:.1f}d")

    # -- Figure 3: OU calibration --
    plot_ou_calibration(ou, spread, project_path)

    # -- Step 5: Kalman filter --
    print("  [5/8] Fitting Kalman filter...")
    kf = KalmanHedgeRatio(delta=1e-4, observation_noise=1e-3)
    kf_res = kf.filter(np.log(pa), np.log(pb))
    print(f"    Final hedge ratio: {kf_res['final_hedge_ratio']:.4f}")

    # -- Figure 4: Kalman hedge ratio --
    plot_kalman_hedge_ratio(kf_res, project_path)

    # -- Step 6: Trading signals --
    print("  [6/8] Generating trading signals...")
    strat = PairsTradingStrategy(z_entry=2.0, z_exit=0.5, z_stop=4.0,
                                  lookback=60)
    signals = strat.generate_signals(spread)
    strat_result = strat.compute_returns(signals, pa, pb,
                                          eg_res["hedge_ratio"], 10.0)
    trade_stats = strat.trade_statistics()
    print(f"    Trades: {trade_stats.get('n_trades', 0)}, "
          f"Win rate: {trade_stats.get('win_rate', 0)*100:.1f}%")

    # -- Figure 5: Trading signals --
    plot_trading_signals(signals, (t1, t2), project_path)

    # -- Figure 6: Trade P&L --
    plot_trade_pnl_distribution(strat_result, trade_stats, project_path)

    # -- Step 7: Walk-forward backtest --
    print("  [7/8] Running walk-forward backtest...")
    bt = PairsBacktester(
        formation_period=252, trading_period=126,
        n_pairs=3, z_entry=2.0, z_exit=0.5, z_stop=4.0,
        transaction_cost_bps=10, hedge_method="ols"
    )
    bt_result = bt.run(prices)
    perf = bt.performance_summary()

    print("\n  BACKTEST PERFORMANCE SUMMARY")
    print("  " + "=" * 50)
    for k, v in perf.items():
        if isinstance(v, float):
            print(f"  {k:<22}: {v:.4f}")
        else:
            print(f"  {k:<22}: {v}")
    print("  " + "=" * 50)

    # -- Figure 7: Equity curve --
    if bt_result is not None and not bt_result.empty:
        plot_equity_curve(bt_result, perf, project_path)
    else:
        print("    [SKIP] Equity curve (no backtest data)")

    # -- Figure 8: Pair selection heatmap --
    print("  [8/8] Plotting pair selection heatmap...")
    if selector.pair_scores is not None and len(selector.pair_scores) > 0:
        plot_pair_selection_heatmap(selector.pair_scores, tickers, project_path)
    else:
        print("    [SKIP] Heatmap (no pair scores)")

    # -- Johansen test (supplementary output) --
    print("\n  Supplementary: Johansen test on cointegrated group...")
    joh = JohansenTest(det_order=0, k_ar_diff=1)
    joh_res = joh.test(np.log(prices[["ALPHA", "BETA", "GAMMA"]]))
    print(joh.get_summary())

    print("\n  " + "=" * 50)
    print("  All 8 figures generated successfully.")
    print("  " + "=" * 50)
