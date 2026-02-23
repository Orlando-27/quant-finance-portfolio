"""
Publication-Quality Factor Investing Visualizations
====================================================

Generates 8 professional figures covering factor analytics, risk decomposition,
ML model performance, regime analysis, and portfolio backtest comparison.

Figure Catalog:
    1. Factor Cumulative Returns (with drawdown subplot)
    2. Factor Correlation Heatmap (rolling and full-sample)
    3. Fama-MacBeth Risk Premia (bar chart with confidence intervals)
    4. Barra Risk Decomposition (stacked bar: factor vs specific)
    5. ML Factor Timing Performance (predicted vs actual scatter + IC)
    6. Regime Detection (factor returns colored by HMM state)
    7. Portfolio Strategy Comparison (cumulative wealth curves)
    8. Strategy Risk-Return Scatter (efficient frontier in factor space)

Author: Jose Orlando Bobadilla Fuentes, CQF, MSc AI
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from typing import Dict, Optional
import os

# -- Professional style configuration --
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

# Professional color palette
COLORS = ["#58a6ff", "#f0883e", "#3fb950", "#bc8cff",
          "#f778ba", "#79c0ff", "#d2a8ff", "#ffa657"]


def _save(fig, path: str, name: str) -> None:
    """Save figure and close."""
    fpath = os.path.join(path, "outputs", "figures", name)
    fig.savefig(fpath, dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"    [FIG] {name}")


def plot_factor_cumulative_returns(factor_returns: pd.DataFrame,
                                    save_path: str) -> None:
    """
    Figure 1: Factor cumulative returns with drawdown subplot.

    Top panel: Cumulative wealth (log scale optional) for each factor.
    Bottom panel: Drawdown from peak for each factor.
    """
    fig = plt.figure(figsize=(14, 9))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.15)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    cum = (1 + factor_returns).cumprod()
    dd = cum / cum.cummax() - 1

    for i, col in enumerate(factor_returns.columns):
        c = COLORS[i % len(COLORS)]
        ax1.plot(cum.index, cum[col], color=c, linewidth=1.3,
                 label=col, alpha=0.9)
        ax2.fill_between(dd.index, 0, dd[col], color=c, alpha=0.3)
        ax2.plot(dd.index, dd[col], color=c, linewidth=0.7, alpha=0.7)

    ax1.set_title("Factor Cumulative Returns", fontsize=14, fontweight="bold",
                  pad=10)
    ax1.set_ylabel("Cumulative Wealth ($1 invested)")
    ax1.legend(loc="upper left", ncol=3, framealpha=0.8)
    ax1.axhline(1.0, color="#8b949e", linestyle="--", linewidth=0.5)

    ax2.set_title("Factor Drawdowns", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Date")
    plt.setp(ax1.get_xticklabels(), visible=False)

    _save(fig, save_path, "01_factor_cumulative_returns.png")


def plot_factor_correlation_heatmap(factor_returns: pd.DataFrame,
                                     save_path: str) -> None:
    """
    Figure 2: Factor correlation matrix -- full sample and rolling.

    Left panel: Full-sample Pearson correlation heatmap.
    Right panel: 36-month rolling average absolute correlation.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    corr = factor_returns.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", center=0,
                cmap="RdBu_r", vmin=-1, vmax=1, ax=axes[0],
                square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8, "label": "Correlation"},
                annot_kws={"size": 9, "color": "#c9d1d9"})
    axes[0].set_title("Full-Sample Factor Correlation", fontsize=12,
                       fontweight="bold")

    # Rolling average absolute correlation
    roll_corr = factor_returns.rolling(36).corr()
    K = len(factor_returns.columns)
    avg_abs_corr = pd.Series(index=factor_returns.index[35:], dtype=float)
    for t in factor_returns.index[35:]:
        try:
            c = roll_corr.loc[t]
            upper = c.values[np.triu_indices(K, k=1)]
            avg_abs_corr[t] = np.abs(upper).mean()
        except (KeyError, ValueError):
            pass
    avg_abs_corr = avg_abs_corr.dropna()

    axes[1].plot(avg_abs_corr.index, avg_abs_corr.values,
                 color=COLORS[0], linewidth=1.5)
    axes[1].axhline(avg_abs_corr.mean(), color=COLORS[1],
                     linestyle="--", linewidth=1, alpha=0.7,
                     label=f"Mean: {avg_abs_corr.mean():.3f}")
    axes[1].set_title("Rolling 36M Avg |Correlation|", fontsize=12,
                       fontweight="bold")
    axes[1].set_ylabel("Average |Correlation|")
    axes[1].legend()
    axes[1].set_ylim(0, None)

    fig.suptitle("Factor Correlation Analysis", fontsize=14,
                  fontweight="bold", y=1.02)
    _save(fig, save_path, "02_factor_correlation_heatmap.png")


def plot_fama_macbeth_premia(risk_premia: pd.DataFrame,
                              save_path: str) -> None:
    """
    Figure 3: Fama-MacBeth estimated risk premia with confidence intervals.

    Bar chart of annualized risk premia with Shanken-corrected 95% CI.
    Factors significant at 5% level highlighted with distinct color.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    premia = risk_premia["Ann. Premium (x12)"]
    se = risk_premia["Std Error (Shanken)"] * 12
    pvals = risk_premia["p-value"]
    factors = risk_premia.index

    x = np.arange(len(factors))
    colors_bar = [COLORS[0] if p < 0.05 else "#8b949e" for p in pvals]

    bars = ax.bar(x, premia, color=colors_bar, alpha=0.85, width=0.6,
                  edgecolor="#30363d", linewidth=0.5)
    ax.errorbar(x, premia, yerr=1.96 * se, fmt="none",
                ecolor="#c9d1d9", elinewidth=1.5, capsize=4, capthick=1.5)

    ax.axhline(0, color="#8b949e", linestyle="-", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(factors, rotation=30, ha="right")
    ax.set_ylabel("Annualized Risk Premium (%)")
    ax.set_title("Fama-MacBeth Factor Risk Premia (Shanken-Corrected 95% CI)",
                  fontsize=13, fontweight="bold")

    # Significance legend
    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor=COLORS[0], label="Significant (p < 0.05)"),
                  Patch(facecolor="#8b949e", label="Not Significant")]
    ax.legend(handles=legend_els, loc="upper right")

    # Annotate t-stats
    for i, (val, t) in enumerate(zip(premia, risk_premia["t-statistic"])):
        y_off = 0.002 if val >= 0 else -0.004
        ax.text(i, val + y_off, f"t={t:.2f}", ha="center", va="bottom",
                fontsize=8, color="#c9d1d9")

    _save(fig, save_path, "03_fama_macbeth_risk_premia.png")


def plot_risk_decomposition(risk_data: Dict, factor_contrib: pd.DataFrame,
                             save_path: str) -> None:
    """
    Figure 4: Barra-style portfolio risk decomposition.

    Left: Pie/donut chart of factor vs specific risk.
    Right: Bar chart of individual factor risk contributions.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Factor vs Specific risk donut
    sizes = [risk_data["pct_factor"], risk_data["pct_specific"]]
    labels_pie = ["Factor Risk", "Specific Risk"]
    pie_colors = [COLORS[0], COLORS[1]]
    wedges, texts, autotexts = axes[0].pie(
        sizes, labels=labels_pie, colors=pie_colors, autopct="%1.1f%%",
        startangle=90, pctdistance=0.75,
        wedgeprops=dict(width=0.35, edgecolor="#0d1117", linewidth=2)
    )
    for t in texts + autotexts:
        t.set_color("#c9d1d9")
    axes[0].set_title(
        f"Risk Decomposition (Total: {risk_data['total_risk']*100:.1f}%)",
        fontsize=12, fontweight="bold"
    )

    # Right: Factor risk contributions
    rc = factor_contrib["Pct Contribution"]
    x = np.arange(len(rc))
    bar_colors = [COLORS[i % len(COLORS)] for i in range(len(rc))]
    axes[1].barh(x, rc, color=bar_colors, alpha=0.85, height=0.6,
                 edgecolor="#30363d")
    axes[1].set_yticks(x)
    axes[1].set_yticklabels(rc.index)
    axes[1].set_xlabel("% Contribution to Factor Risk")
    axes[1].set_title("Factor Risk Contributions", fontsize=12,
                       fontweight="bold")

    for i, v in enumerate(rc):
        axes[1].text(v + 0.5, i, f"{v:.1f}%", va="center",
                     fontsize=9, color="#c9d1d9")

    fig.suptitle("Barra-Style Portfolio Risk Decomposition", fontsize=14,
                  fontweight="bold", y=1.02)
    _save(fig, save_path, "04_risk_decomposition.png")


def plot_ml_timing_performance(cv_results: Dict, save_path: str) -> None:
    """
    Figure 5: ML factor timing model evaluation.

    2x2 grid: predicted vs actual scatter for each model,
    with regression line, R-squared, IC, and RMSE annotations.
    """
    n_models = len(cv_results)
    fig, axes = plt.subplots(1, max(n_models, 2), figsize=(7 * n_models, 6))
    if n_models == 1:
        axes = [axes]

    for idx, (name, res) in enumerate(cv_results.items()):
        ax = axes[idx]
        pred = res["predictions"]
        actual = res["actuals"]

        ax.scatter(actual, pred, alpha=0.4, s=15, color=COLORS[idx],
                   edgecolors="none")

        # Regression line
        z = np.polyfit(actual, pred, 1)
        p = np.poly1d(z)
        x_line = np.linspace(actual.min(), actual.max(), 100)
        ax.plot(x_line, p(x_line), color=COLORS[3], linewidth=1.5,
                linestyle="--", alpha=0.8)

        # 45-degree line
        lims = [min(actual.min(), pred.min()), max(actual.max(), pred.max())]
        ax.plot(lims, lims, color="#8b949e", linewidth=0.8, linestyle=":",
                alpha=0.5)

        # Annotations
        text = (f"R² = {res['r2']:.4f}\n"
                f"IC = {res['ic']:.4f}\n"
                f"RMSE = {res['rmse']:.6f}")
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,
                verticalalignment="top", color="#c9d1d9",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#21262d",
                          edgecolor="#30363d", alpha=0.9))

        ax.set_xlabel("Actual Factor Return")
        ax.set_ylabel("Predicted Factor Return")
        ax.set_title(f"{name}", fontsize=12, fontweight="bold")

    fig.suptitle("ML Factor Timing: Walk-Forward Cross-Validation",
                  fontsize=14, fontweight="bold", y=1.02)
    _save(fig, save_path, "05_ml_timing_performance.png")


def plot_regime_analysis(factor_returns: pd.DataFrame,
                          states: pd.Series, save_path: str) -> None:
    """
    Figure 6: HMM regime detection visualization.

    Top: Market factor return colored by regime state.
    Bottom: Regime probability timeline with state-conditional statistics.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[2, 1])

    # Use first factor (MKT or equivalent) for visualization
    target_col = factor_returns.columns[0]
    r = factor_returns[target_col]
    common = r.index.intersection(states.index)
    r = r.loc[common]
    s = states.loc[common]

    n_states = s.nunique()
    state_colors = [COLORS[0], COLORS[1], COLORS[2]][:n_states]
    state_labels = [f"State {i}" for i in range(n_states)]

    # Top: Returns colored by regime
    for state in range(n_states):
        mask = s == state
        axes[0].bar(r.index[mask], r.values[mask], color=state_colors[state],
                    alpha=0.7, width=25, label=state_labels[state])

    axes[0].axhline(0, color="#8b949e", linewidth=0.5)
    axes[0].set_ylabel(f"{target_col} Return")
    axes[0].set_title("Factor Returns by Market Regime (HMM)",
                       fontsize=13, fontweight="bold")
    axes[0].legend(loc="upper right")

    # Bottom: Regime timeline
    for state in range(n_states):
        mask = s == state
        axes[1].fill_between(s.index, 0, 1, where=mask.values,
                             color=state_colors[state], alpha=0.5,
                             label=state_labels[state])

    # Annotate state statistics
    for state in range(n_states):
        mask = s == state
        sub = factor_returns.loc[common][mask.values]
        pct = mask.sum() / len(mask) * 100
        ann_r = sub.mean().mean() * 12 * 100
        ann_v = sub.std().mean() * np.sqrt(12) * 100
        text = f"S{state}: {pct:.0f}% time, ret={ann_r:.1f}%, vol={ann_v:.1f}%"
        y_pos = 0.85 - state * 0.15
        axes[1].text(0.01, y_pos, text, transform=axes[1].transAxes,
                     fontsize=9, color=state_colors[state], fontweight="bold")

    axes[1].set_ylabel("Regime")
    axes[1].set_xlabel("Date")
    axes[1].set_yticks([])
    axes[1].set_title("Regime Timeline", fontsize=11, fontweight="bold")

    _save(fig, save_path, "06_regime_analysis.png")


def plot_strategy_comparison(backtest_results: Dict[str, pd.DataFrame],
                              save_path: str) -> None:
    """
    Figure 7: Cumulative wealth comparison across all portfolio strategies.

    Shows cumulative return paths for each strategy with shaded drawdown
    periods for the best-performing strategy.
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    for i, (name, df) in enumerate(backtest_results.items()):
        cum = df["cumulative_return"]
        ax.plot(cum.index, cum.values, color=COLORS[i % len(COLORS)],
                linewidth=1.5, label=name.replace("_", " ").title(),
                alpha=0.9)

    ax.axhline(1.0, color="#8b949e", linestyle="--", linewidth=0.5)
    ax.set_ylabel("Cumulative Wealth ($1 invested)")
    ax.set_xlabel("Date")
    ax.set_title("Factor Strategy Performance Comparison (Net of TC)",
                  fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", ncol=2, framealpha=0.8)

    _save(fig, save_path, "07_strategy_comparison.png")


def plot_risk_return_scatter(summary: pd.DataFrame,
                              save_path: str) -> None:
    """
    Figure 8: Risk-return scatter plot of all strategies.

    Each strategy plotted as a point in (volatility, return) space.
    Annotated with Sharpe ratio isolines and strategy labels.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    vols = summary["Ann. Volatility"] * 100
    rets = summary["Ann. Return"] * 100
    sharpes = summary["Sharpe Ratio"]

    for i, (idx, row) in enumerate(summary.iterrows()):
        ax.scatter(vols.loc[idx], rets.loc[idx], s=120,
                   color=COLORS[i % len(COLORS)], edgecolors="#c9d1d9",
                   linewidth=1, zorder=5, alpha=0.9)
        label = idx.replace("_", " ").title()
        ax.annotate(f"{label}\nSR={sharpes.loc[idx]:.2f}",
                    (vols.loc[idx], rets.loc[idx]),
                    textcoords="offset points", xytext=(12, 8),
                    fontsize=8, color="#c9d1d9",
                    arrowprops=dict(arrowstyle="-", color="#8b949e",
                                    lw=0.5))

    # Sharpe ratio isolines
    vol_range = np.linspace(0, vols.max() * 1.3, 100)
    for sr in [0.5, 1.0, 1.5]:
        ret_line = sr * vol_range + 2.0  # rf = 2%
        ax.plot(vol_range, ret_line, color="#8b949e", linestyle=":",
                linewidth=0.7, alpha=0.4)
        ax.text(vol_range[-1], ret_line[-1], f"SR={sr:.1f}",
                fontsize=7, color="#8b949e", alpha=0.6)

    ax.set_xlabel("Annualized Volatility (%)")
    ax.set_ylabel("Annualized Return (%)")
    ax.set_title("Factor Strategy Risk-Return Profile",
                  fontsize=14, fontweight="bold")

    _save(fig, save_path, "08_risk_return_scatter.png")


def generate_all_figures(project_path: str) -> None:
    """
    Master function: generates all 8 publication-quality figures.

    Orchestrates the full pipeline:
    1. Generate synthetic universe and replicate factors
    2. Run Fama-MacBeth regression
    3. Fit Barra risk model
    4. Train ML timing models
    5. Detect regimes via HMM
    6. Construct and optimize factor portfolios
    7. Run walk-forward backtests
    8. Produce all visualizations
    """
    import sys
    sys.path.insert(0, project_path)

    from src.factors import FamaFrenchReplicator
    from src.cross_sectional import FamaMacBeth
    from src.risk_model import BarraRiskModel
    from src.ml_timing import FeatureEngineering, FactorTimingML, RegimeDetector
    from src.portfolio import FactorPortfolio
    from src.backtesting import FactorBacktester

    print("\n  Generating Project 11 visualizations...")
    print("  " + "-" * 50)

    # -- Step 1: Generate synthetic universe and replicate factors --
    print("  [1/8] Generating synthetic universe & replicating factors...")
    ffr = FamaFrenchReplicator(n_stocks=500, n_periods=240, seed=42)
    returns, chars, market_cap = ffr.generate_universe()
    replicated = ffr.replicate_factors()
    true_factors = ffr.factor_returns

    # Use replicated factors for downstream analysis
    factors = replicated.copy()

    # -- Figure 1: Cumulative returns --
    print("  [2/8] Plotting factor cumulative returns...")
    plot_factor_cumulative_returns(factors, project_path)

    # -- Figure 2: Correlation heatmap --
    print("  [3/8] Plotting correlation analysis...")
    plot_factor_correlation_heatmap(factors, project_path)

    # -- Step 2: Fama-MacBeth regression --
    print("  [4/8] Running Fama-MacBeth regression...")
    fmb = FamaMacBeth(rolling_window=60, use_shanken=True)
    fmb.estimate_betas(returns, factors)
    fmb.cross_sectional_regression(returns, factors)
    print(fmb.get_summary())

    # -- Figure 3: Risk premia --
    plot_fama_macbeth_premia(fmb.risk_premia, project_path)

    # -- Step 3: Barra risk model --
    print("  [5/8] Fitting Barra risk model...")
    brm = BarraRiskModel(n_factors=factors.shape[1])
    brm.fit(returns, factors, window=60)
    ew = np.ones(returns.shape[1]) / returns.shape[1]
    risk_data = brm.portfolio_risk(ew)
    factor_contrib = brm.factor_risk_contribution(ew)

    # -- Figure 4: Risk decomposition --
    plot_risk_decomposition(risk_data, factor_contrib, project_path)

    # -- Step 4: ML factor timing --
    print("  [6/8] Training ML factor timing models...")
    fe = FeatureEngineering(lookback_windows=[3, 6, 12])
    feat = fe.build_features(factors)
    target = factors.iloc[:, 0].shift(-1).dropna()  # predict MKT next period
    target.name = "MKT_next"

    ml = FactorTimingML(model_type="ensemble", n_splits=5)
    cv_results = ml.walk_forward_cv(feat, target)

    for name, res in cv_results.items():
        print(f"    {name}: RMSE={res['rmse']:.6f}, "
              f"R2={res['r2']:.4f}, IC={res['ic']:.4f}")

    # -- Figure 5: ML performance --
    plot_ml_timing_performance(cv_results, project_path)

    # -- Step 5: Regime detection --
    print("  [7/8] Detecting market regimes (HMM)...")
    rd = RegimeDetector(n_states=2, n_iter=200)
    states = rd.fit_predict(factors)
    state_stats = rd.get_state_statistics(factors)

    # -- Figure 6: Regime analysis --
    plot_regime_analysis(factors, states, project_path)

    # -- Step 6: Backtesting all strategies --
    print("  [8/8] Running walk-forward backtests...")
    bt = FactorBacktester(
        factors, rebalance_frequency=1,
        transaction_cost_bps=10, lookback_window=60
    )
    bt.run_all_strategies()

    # -- Figure 7: Strategy comparison --
    plot_strategy_comparison(bt.results, project_path)

    # -- Performance summary --
    perf = bt.performance_summary()
    print("\n  BACKTEST PERFORMANCE SUMMARY")
    print("  " + "=" * 60)
    print(perf.to_string(float_format="{:.4f}".format))

    # -- Figure 8: Risk-return scatter --
    plot_risk_return_scatter(perf, project_path)

    print("\n  " + "=" * 50)
    print("  All 8 figures generated successfully.")
    print("  " + "=" * 50)
