"""
Publication-quality visualizations for VaR & CVaR Risk Engine.

Figures generated:
    01_var_methods_comparison.png    - VaR estimates by methodology
    02_return_distribution.png       - Empirical vs fitted distribution
    03_var_backtesting.png           - VaR exceedances timeline
    04_cvar_tail_analysis.png        - Expected Shortfall tail decomposition
    05_risk_contribution.png         - Component VaR attribution
    06_var_surface.png               - VaR as f(confidence, horizon)

Author: Jose Orlando Bobadilla Fuentes, CQF
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, t as t_dist

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

def _synthetic_returns(n=2000):
    np.random.seed(42)
    # Fat-tailed returns with clustering
    base = 0.0003 + 0.012 * np.random.randn(n)
    # Add volatility clustering
    vol = np.ones(n) * 0.012
    for i in range(1, n):
        vol[i] = 0.94 * vol[i-1] + 0.06 * abs(base[i-1])
    returns = 0.0003 + vol * np.random.randn(n)
    # Add a few tail events
    returns[500] = -0.045
    returns[1200] = -0.038
    returns[1800] = -0.052
    return returns


def plot_var_comparison(proj=None):
    fig, ax = plt.subplots(figsize=(11, 6))
    rets = _synthetic_returns()
    alpha = 0.99
    # Historical
    var_hist = -np.percentile(rets, (1 - alpha) * 100)
    # Parametric (Normal)
    var_param = -(rets.mean() + norm.ppf(1 - alpha) * rets.std())
    # Parametric (t)
    df_t = 5
    var_t = -(rets.mean() + t_dist.ppf(1 - alpha, df_t) * rets.std())
    # Cornish-Fisher
    z = norm.ppf(1 - alpha)
    s = -0.3  # negative skew
    k = 4.5   # excess kurtosis
    z_cf = z + (z**2 - 1)*s/6 + (z**3 - 3*z)*k/24 - (2*z**3 - 5*z)*s**2/36
    var_cf = -(rets.mean() + z_cf * rets.std())

    methods = ["Historical\nSimulation", "Parametric\n(Normal)", "Parametric\n(Student-t)", "Cornish-\nFisher"]
    vars_ = [var_hist, var_param, var_t, var_cf]
    bars = ax.bar(methods, [v*100 for v in vars_], color=[TEAL, CORAL, GOLD, NAVY],
                  edgecolor="white", lw=1.5, width=0.6)
    for bar, val in zip(bars, vars_):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val*100:.2f}%", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("VaR (% of Portfolio)")
    ax.set_title(f"Value-at-Risk Comparison at {alpha*100:.0f}% Confidence Level")
    _wm(fig)
    return _sv(fig, proj, "01_var_methods_comparison.png")


def plot_return_distribution(proj=None):
    fig, ax = plt.subplots(figsize=(11, 6))
    rets = _synthetic_returns()
    ax.hist(rets*100, bins=80, density=True, alpha=0.6, color=TEAL,
            edgecolor="white", label="Empirical")
    x = np.linspace(rets.min()*100, rets.max()*100, 300)
    ax.plot(x, norm.pdf(x, rets.mean()*100, rets.std()*100), color=CORAL,
            lw=2.5, label="Normal Fit")
    ax.plot(x, t_dist.pdf(x, 5, rets.mean()*100, rets.std()*100), color=NAVY,
            lw=2.5, ls="--", label="Student-t (df=5)")
    var_99 = np.percentile(rets, 1) * 100
    ax.axvline(var_99, color=CORAL, ls=":", lw=2, label=f"VaR 99% = {var_99:.2f}%")
    ax.fill_betweenx([0, 0.5], rets.min()*100, var_99, alpha=0.15, color=CORAL)
    ax.set_xlabel("Daily Return (%)")
    ax.set_ylabel("Density")
    ax.set_title("Return Distribution: Empirical vs Fitted")
    ax.legend()
    _wm(fig)
    return _sv(fig, proj, "02_return_distribution.png")


def plot_var_backtesting(proj=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), height_ratios=[3, 1])
    rets = _synthetic_returns()
    window = 250
    var_series = []
    for i in range(window, len(rets)):
        hist = rets[i-window:i]
        var_series.append(-np.percentile(hist, 1))
    var_arr = np.array(var_series)
    actual = rets[window:]
    t = np.arange(len(actual))
    exceedances = actual < -var_arr

    ax1.plot(t, actual*100, color=TEAL, lw=0.8, alpha=0.7, label="Daily Returns")
    ax1.plot(t, -var_arr*100, color=CORAL, lw=1.5, label="VaR 99% (Rolling 250d)")
    ax1.scatter(t[exceedances], actual[exceedances]*100, color=CORAL, s=40,
                zorder=5, label=f"Exceedances ({exceedances.sum()})")
    ax1.set_ylabel("Return (%)")
    ax1.set_title("VaR Backtesting: Exceedance Analysis")
    ax1.legend()

    cum_exc = np.cumsum(exceedances)
    expected = np.arange(1, len(actual)+1) * 0.01
    ax2.plot(t, cum_exc, color=NAVY, lw=2, label="Actual Exceedances")
    ax2.plot(t, expected, color=GOLD, ls="--", lw=2, label="Expected (1%)")
    ax2.set_xlabel("Trading Days")
    ax2.set_ylabel("Cumulative Violations")
    ax2.legend()
    fig.tight_layout()
    _wm(fig)
    return _sv(fig, proj, "03_var_backtesting.png")


def plot_cvar_tail(proj=None):
    fig, ax = plt.subplots(figsize=(11, 6))
    rets = _synthetic_returns()
    x = np.sort(rets) * 100
    var_99 = np.percentile(rets, 1) * 100
    cvar_99 = rets[rets <= np.percentile(rets, 1)].mean() * 100

    ax.hist(x, bins=80, density=True, alpha=0.5, color=TEAL, edgecolor="white")
    ax.axvline(var_99, color=CORAL, ls="--", lw=2.5, label=f"VaR 99% = {var_99:.2f}%")
    ax.axvline(cvar_99, color=NAVY, ls="-", lw=2.5, label=f"CVaR 99% = {cvar_99:.2f}%")
    mask = x <= var_99
    ax.fill_between(x[mask], 0, norm.pdf(x[mask], rets.mean()*100, rets.std()*100),
                    alpha=0.3, color=CORAL, label="Tail Region (ES)")
    ax.set_xlabel("Daily Return (%)")
    ax.set_ylabel("Density")
    ax.set_title("VaR vs CVaR (Expected Shortfall): Tail Risk Analysis")
    ax.legend()
    _wm(fig)
    return _sv(fig, proj, "04_cvar_tail_analysis.png")


def plot_risk_contribution(proj=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    assets = ["US Equity", "Intl Equity", "EM Equity", "Bonds", "Commod"]
    weights = [0.30, 0.15, 0.10, 0.30, 0.15]
    marginal_var = [0.022, 0.025, 0.035, 0.005, 0.028]
    comp_var = [w * m for w, m in zip(weights, marginal_var)]
    total_var = sum(comp_var)

    colors_list = [TEAL, CORAL, GOLD, NAVY, SLATE]
    ax1.barh(assets, [c/total_var*100 for c in comp_var], color=colors_list,
             edgecolor="white", lw=1.5)
    ax1.set_xlabel("Contribution to Portfolio VaR (%)")
    ax1.set_title("Component VaR Attribution")

    ax2.pie([c/total_var for c in comp_var], labels=assets, colors=colors_list,
            autopct="%1.1f%%", startangle=140, pctdistance=0.85)
    ax2.set_title("Risk Budget Allocation")
    fig.suptitle(f"Risk Decomposition (Total VaR = {total_var*100:.2f}%)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _wm(fig)
    return _sv(fig, proj, "05_risk_contribution.png")


def plot_var_surface(proj=None):
    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection="3d")
    conf = np.linspace(0.90, 0.995, 40)
    horizon = np.linspace(1, 30, 40)
    CC, HH = np.meshgrid(conf, horizon)
    daily_vol = 0.015
    VaR = norm.ppf(CC) * daily_vol * np.sqrt(HH) * 100
    surf = ax.plot_surface(CC*100, HH, VaR, cmap="RdYlBu_r", alpha=0.85,
                           edgecolor="none")
    ax.set_xlabel("Confidence (%)")
    ax.set_ylabel("Horizon (days)")
    ax.set_zlabel("VaR (%)")
    ax.set_title("VaR Surface: f(Confidence, Horizon)")
    fig.colorbar(surf, shrink=0.5, aspect=10)
    ax.view_init(elev=25, azim=-50)
    _wm(fig)
    return _sv(fig, proj, "06_var_surface.png")


def generate_all_figures(project_dir=None):
    if project_dir is None:
        project_dir = os.path.join(os.path.dirname(__file__), "..", "..")
    print("  Generating Project 04 figures...")
    files = [
        plot_var_comparison(project_dir),
        plot_return_distribution(project_dir),
        plot_var_backtesting(project_dir),
        plot_cvar_tail(project_dir),
        plot_risk_contribution(project_dir),
        plot_var_surface(project_dir),
    ]
    print(f"  DONE: {len(files)} figures saved to outputs/figures/")
    return files

if __name__ == "__main__":
    generate_all_figures(os.path.join(os.path.dirname(__file__), "..", ".."))
