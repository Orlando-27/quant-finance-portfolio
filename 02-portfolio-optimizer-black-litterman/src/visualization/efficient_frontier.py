"""
Publication-quality visualizations for Portfolio Optimization.

Figures generated:
    01_efficient_frontier.png       - MV frontier with tangency portfolio
    02_black_litterman_shift.png    - Prior vs posterior expected returns
    03_weight_allocation.png        - Asset allocation comparison bar chart
    04_risk_decomposition.png       - Contribution to portfolio risk
    05_rolling_sharpe.png           - Rolling Sharpe ratio over time
    06_correlation_heatmap.png      - Asset correlation matrix

Author: Jose Orlando Bobadilla Fuentes, CQF
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyArrowPatch
from scipy.optimize import minimize

NAVY   = "#1a1a2e"
TEAL   = "#16697a"
CORAL  = "#db6400"
GOLD   = "#c5a880"
SLATE  = "#4a4e69"
COLORS = [NAVY, TEAL, CORAL, GOLD, SLATE, "#2d6a4f", "#e07a5f", "#3d405b"]

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
    "font.size": 11, "axes.titlesize": 14, "axes.labelsize": 12,
    "figure.dpi": 150, "savefig.dpi": 300,
})


def _wm(fig):
    fig.text(0.99, 0.01, "J. Bobadilla | CQF", fontsize=7,
             color="gray", alpha=0.5, ha="right", va="bottom")


def _sv(fig, proj, name):
    path = os.path.join(proj, "outputs", "figures", name)
    fig.savefig(path); plt.close(fig); return path


def _frontier_data():
    """Synthetic 6-asset universe with realistic parameters."""
    np.random.seed(42)
    n = 6
    names = ["US Equity", "Intl Equity", "EM Equity",
             "US Bonds", "Corp Bonds", "Commodities"]
    mu = np.array([0.10, 0.09, 0.12, 0.04, 0.05, 0.07])
    vols = np.array([0.16, 0.18, 0.24, 0.05, 0.07, 0.20])
    corr = np.array([
        [1.0, 0.8, 0.7, -0.1, 0.1, 0.2],
        [0.8, 1.0, 0.8, -0.1, 0.1, 0.3],
        [0.7, 0.8, 1.0,  0.0, 0.1, 0.4],
        [-0.1,-0.1, 0.0, 1.0, 0.8,-0.1],
        [0.1, 0.1, 0.1,  0.8, 1.0, 0.0],
        [0.2, 0.3, 0.4, -0.1, 0.0, 1.0],
    ])
    cov = np.outer(vols, vols) * corr
    return names, mu, cov, vols


def _optimize_portfolio(mu, cov, target_ret):
    n = len(mu)
    def vol(w): return np.sqrt(w @ cov @ w)
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w: w @ mu - target_ret}]
    bounds = [(0, 1)] * n
    w0 = np.ones(n) / n
    res = minimize(vol, w0, method="SLSQP", bounds=bounds, constraints=cons)
    return res.x if res.success else w0


# Figure 01
def plot_efficient_frontier(proj=None):
    names, mu, cov, vols = _frontier_data()
    fig, ax = plt.subplots(figsize=(11, 7))
    targets = np.linspace(mu.min(), mu.max(), 80)
    port_vols, port_rets = [], []
    for t in targets:
        w = _optimize_portfolio(mu, cov, t)
        port_vols.append(np.sqrt(w @ cov @ w))
        port_rets.append(w @ mu)
    ax.plot(np.array(port_vols)*100, np.array(port_rets)*100,
            color=NAVY, lw=2.5, label="Efficient Frontier")
    # Individual assets
    for i, name in enumerate(names):
        ax.scatter(vols[i]*100, mu[i]*100, s=100, zorder=5,
                   color=COLORS[i], edgecolor="white", lw=1.5)
        ax.annotate(name, (vols[i]*100, mu[i]*100), fontsize=9,
                    xytext=(8, 5), textcoords="offset points")
    # Tangency
    rf = 0.03
    sharpes = [(mu[i] - rf) / vols[i] for i in range(len(mu))]
    best_sr = np.argmax([(r - rf) / v for r, v in zip(port_rets, port_vols)])
    ax.scatter(port_vols[best_sr]*100, port_rets[best_sr]*100, s=200,
               marker="*", color=CORAL, zorder=6, label="Tangency Portfolio")
    # CML
    x_cml = np.linspace(0, max(port_vols)*100*1.1, 100)
    sr_tang = (port_rets[best_sr] - rf) / port_vols[best_sr]
    ax.plot(x_cml, rf*100 + sr_tang * x_cml, color=GOLD, ls="--", lw=1.5,
            label=f"CML (Sharpe={sr_tang:.2f})")
    ax.set_xlabel("Annualized Volatility (%)")
    ax.set_ylabel("Annualized Return (%)")
    ax.set_title("Mean-Variance Efficient Frontier with Tangency Portfolio")
    ax.legend(loc="upper left")
    _wm(fig)
    return _sv(fig, proj, "01_efficient_frontier.png")


# Figure 02
def plot_bl_shift(proj=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    names, mu, _, _ = _frontier_data()
    # Simulated BL posterior (views shift EM up, Bonds down)
    mu_post = mu.copy()
    mu_post[2] += 0.03   # EM bullish view
    mu_post[3] -= 0.01   # Bonds bearish
    mu_post[5] += 0.02   # Commodities up
    y = np.arange(len(names))
    ax.barh(y - 0.15, mu*100, 0.3, color=TEAL, label="CAPM Equilibrium (Prior)")
    ax.barh(y + 0.15, mu_post*100, 0.3, color=CORAL, label="BL Posterior")
    for i in range(len(names)):
        diff = (mu_post[i] - mu[i]) * 100
        if abs(diff) > 0.1:
            ax.annotate(f"{diff:+.1f}%", (max(mu[i], mu_post[i])*100 + 0.3, y[i]),
                       fontsize=9, fontweight="bold", color=NAVY)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("Expected Return (%)")
    ax.set_title("Black-Litterman: Prior vs Posterior Expected Returns")
    ax.legend()
    _wm(fig)
    return _sv(fig, proj, "02_black_litterman_shift.png")


# Figure 03
def plot_weight_allocation(proj=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    names = ["US Eq", "Intl Eq", "EM Eq", "US Bond", "Corp Bond", "Commod"]
    w_mv = [0.25, 0.15, 0.05, 0.30, 0.15, 0.10]
    w_bl = [0.20, 0.12, 0.18, 0.22, 0.10, 0.18]
    w_rp = [0.12, 0.10, 0.08, 0.35, 0.25, 0.10]
    x = np.arange(len(names))
    w = 0.25
    ax.bar(x - w, w_mv, w, color=TEAL, label="Mean-Variance")
    ax.bar(x, w_bl, w, color=CORAL, label="Black-Litterman")
    ax.bar(x + w, w_rp, w, color=GOLD, label="Risk Parity")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel("Portfolio Weight")
    ax.set_title("Optimal Weight Allocation: MV vs BL vs Risk Parity")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    _wm(fig)
    return _sv(fig, proj, "03_weight_allocation.png")


# Figure 04
def plot_risk_decomposition(proj=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    names = ["US Eq", "Intl Eq", "EM Eq", "US Bond", "Corp Bond", "Commod"]
    weights = [0.20, 0.12, 0.18, 0.22, 0.10, 0.18]
    risk_contrib = [0.28, 0.14, 0.32, 0.04, 0.03, 0.19]
    ax1.pie(risk_contrib, labels=names, colors=COLORS[:6], autopct="%1.1f%%",
            startangle=140, pctdistance=0.85)
    ax1.set_title("Risk Contribution by Asset")
    x = np.arange(len(names))
    ax2.bar(x, weights, 0.4, color=TEAL, label="Weight", alpha=0.7)
    ax2.bar(x + 0.4, risk_contrib, 0.4, color=CORAL, label="Risk Contrib", alpha=0.7)
    ax2.set_xticks(x + 0.2)
    ax2.set_xticklabels(names, rotation=15)
    ax2.set_title("Weight vs Risk Contribution")
    ax2.legend()
    fig.suptitle("Portfolio Risk Decomposition", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    _wm(fig)
    return _sv(fig, proj, "04_risk_decomposition.png")


# Figure 05
def plot_rolling_sharpe(proj=None):
    fig, ax = plt.subplots(figsize=(11, 5))
    np.random.seed(123)
    days = 756
    rets = 0.0004 + 0.01 * np.random.randn(days)
    window = 63
    roll_mean = np.convolve(rets, np.ones(window)/window, mode="valid")
    roll_std = np.array([rets[i:i+window].std() for i in range(len(rets)-window+1)])
    roll_sharpe = (roll_mean / roll_std) * np.sqrt(252)
    t = np.arange(len(roll_sharpe))
    ax.plot(t, roll_sharpe, color=NAVY, lw=1.5)
    ax.fill_between(t, roll_sharpe, where=roll_sharpe > 0, alpha=0.2, color=TEAL)
    ax.fill_between(t, roll_sharpe, where=roll_sharpe < 0, alpha=0.2, color=CORAL)
    ax.axhline(0, color="gray", lw=0.8)
    ax.axhline(np.mean(roll_sharpe), color=GOLD, ls="--", lw=1.5,
               label=f"Mean = {np.mean(roll_sharpe):.2f}")
    ax.set_xlabel("Trading Days")
    ax.set_ylabel("Rolling Sharpe (63d)")
    ax.set_title("Rolling Sharpe Ratio: 3-Month Window")
    ax.legend()
    _wm(fig)
    return _sv(fig, proj, "05_rolling_sharpe.png")


# Figure 06
def plot_correlation_heatmap(proj=None):
    names, _, cov, vols = _frontier_data()
    corr = cov / np.outer(vols, vols)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center",
                    fontsize=10, color="white" if abs(corr[i,j]) > 0.5 else "black")
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    ax.set_title("Asset Correlation Matrix")
    fig.colorbar(im, ax=ax, shrink=0.8)
    _wm(fig)
    return _sv(fig, proj, "06_correlation_heatmap.png")


def generate_all_figures(project_dir=None):
    if project_dir is None:
        project_dir = os.path.join(os.path.dirname(__file__), "..", "..")
    print("  Generating Project 02 figures...")
    files = [
        plot_efficient_frontier(project_dir),
        plot_bl_shift(project_dir),
        plot_weight_allocation(project_dir),
        plot_risk_decomposition(project_dir),
        plot_rolling_sharpe(project_dir),
        plot_correlation_heatmap(project_dir),
    ]
    print(f"  DONE: {len(files)} figures saved to outputs/figures/")
    return files


if __name__ == "__main__":
    generate_all_figures(os.path.join(os.path.dirname(__file__), "..", ".."))
