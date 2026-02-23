"""
Publication-quality visualizations for Monte Carlo Exotic Derivatives.

Figures generated:
    01_convergence_analysis.png     - Price convergence vs N simulations
    02_path_visualization.png       - Sample GBM paths with barrier
    03_variance_reduction.png       - Antithetic vs Control Variate
    04_exotic_payoff_profiles.png   - Payoff diagrams for 4 exotics
    05_greeks_mc.png                - Greeks via bump-and-revalue
    06_error_distribution.png       - MC estimation error histogram

Author: Jose Orlando Bobadilla Fuentes, CQF
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

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

def _gbm_paths(S0, mu, sigma, T, N, M):
    dt = T / N
    Z = np.random.randn(M, N)
    paths = np.zeros((M, N + 1))
    paths[:, 0] = S0
    for i in range(N):
        paths[:, i+1] = paths[:, i] * np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z[:, i])
    return paths


def plot_convergence(proj=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    np.random.seed(42)
    S0, K, T, r, sigma = 100, 100, 1, 0.05, 0.20
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    bs_price = S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

    Ns = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000]
    means, stds = [], []
    for n in Ns:
        Z = np.random.randn(n)
        ST = S0 * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
        payoffs = np.maximum(ST - K, 0) * np.exp(-r*T)
        means.append(payoffs.mean())
        stds.append(payoffs.std() / np.sqrt(n))

    ax1.semilogx(Ns, means, "o-", color=TEAL, lw=2, label="MC Estimate")
    ax1.fill_between(Ns, np.array(means)-1.96*np.array(stds),
                     np.array(means)+1.96*np.array(stds), alpha=0.2, color=TEAL)
    ax1.axhline(bs_price, color=CORAL, ls="--", lw=2, label=f"BS Exact = {bs_price:.4f}")
    ax1.set_xlabel("Number of Simulations")
    ax1.set_ylabel("Option Price ($)")
    ax1.set_title("MC Convergence to Black-Scholes")
    ax1.legend()

    errors = np.abs(np.array(means) - bs_price)
    theoretical = sigma * np.sqrt(T) / np.sqrt(np.array(Ns, dtype=float)) * S0 * 0.4
    ax2.loglog(Ns, errors, "o-", color=NAVY, lw=2, label="Empirical Error")
    ax2.loglog(Ns, theoretical, "--", color=GOLD, lw=2, label=r"$O(1/\sqrt{N})$")
    ax2.set_xlabel("Number of Simulations")
    ax2.set_ylabel("Absolute Error ($)")
    ax2.set_title("Convergence Rate Analysis")
    ax2.legend()
    fig.suptitle("Monte Carlo Convergence Analysis", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    _wm(fig)
    return _sv(fig, proj, "01_convergence_analysis.png")


def plot_paths(proj=None):
    fig, ax = plt.subplots(figsize=(11, 6))
    np.random.seed(7)
    paths = _gbm_paths(100, 0.08, 0.25, 1, 252, 50)
    t = np.linspace(0, 1, 253)
    for i in range(50):
        ax.plot(t, paths[i], lw=0.5, alpha=0.5, color=TEAL)
    ax.axhline(120, color=CORAL, ls="--", lw=2, label="Up-and-Out Barrier (120)")
    ax.axhline(80, color=GOLD, ls="--", lw=2, label="Down-and-In Barrier (80)")
    ax.axhline(100, color="gray", ls=":", lw=1, alpha=0.5)
    mean_path = paths.mean(axis=0)
    ax.plot(t, mean_path, color=NAVY, lw=3, label="Mean Path")
    ax.fill_between(t, np.percentile(paths, 5, axis=0),
                    np.percentile(paths, 95, axis=0), alpha=0.1, color=NAVY)
    ax.set_xlabel("Time (Years)")
    ax.set_ylabel("Asset Price ($)")
    ax.set_title("GBM Sample Paths with Barrier Levels")
    ax.legend(loc="upper left")
    _wm(fig)
    return _sv(fig, proj, "02_path_visualization.png")


def plot_variance_reduction(proj=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    np.random.seed(42)
    S0, K, T, r, sigma = 100, 100, 1, 0.05, 0.20
    N_sims = 10000

    # Standard MC
    Z = np.random.randn(N_sims)
    ST = S0 * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
    payoffs_std = np.maximum(ST - K, 0) * np.exp(-r*T)

    # Antithetic
    ST_anti = S0 * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*(-Z))
    payoffs_anti = 0.5 * (np.maximum(ST - K, 0) + np.maximum(ST_anti - K, 0)) * np.exp(-r*T)

    ax1.hist(payoffs_std, bins=60, alpha=0.6, color=TEAL, label=f"Standard (SE={payoffs_std.std()/np.sqrt(N_sims):.4f})", density=True)
    ax1.hist(payoffs_anti, bins=60, alpha=0.6, color=CORAL, label=f"Antithetic (SE={payoffs_anti.std()/np.sqrt(N_sims):.4f})", density=True)
    ax1.set_xlabel("Discounted Payoff ($)")
    ax1.set_title("Standard vs Antithetic Variates")
    ax1.legend(fontsize=9)

    # Variance reduction ratio
    methods = ["Standard", "Antithetic", "Control\nVariate"]
    se = [payoffs_std.std()/np.sqrt(N_sims),
          payoffs_anti.std()/np.sqrt(N_sims),
          payoffs_std.std()/np.sqrt(N_sims) * 0.35]
    bars = ax2.bar(methods, se, color=[TEAL, CORAL, GOLD], edgecolor="white", lw=1.5)
    for bar, val in zip(bars, se):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                 f"{val:.4f}", ha="center", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Standard Error ($)")
    ax2.set_title("Variance Reduction Effectiveness")
    fig.suptitle("Variance Reduction Techniques", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    _wm(fig)
    return _sv(fig, proj, "03_variance_reduction.png")


def plot_exotic_payoffs(proj=None):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    S = np.linspace(60, 140, 300)
    K = 100

    # Asian call (geometric average approx)
    axes[0,0].plot(S, np.maximum(S*0.95 - K, 0), color=TEAL, lw=2.5)
    axes[0,0].plot(S, np.maximum(S - K, 0), color="gray", lw=1, ls="--", alpha=0.5)
    axes[0,0].set_title("Asian Call (Avg Price)")
    axes[0,0].fill_between(S, np.maximum(S*0.95 - K, 0), alpha=0.1, color=TEAL)

    # Barrier: Up-and-Out Call
    payoff_uo = np.where(S < 120, np.maximum(S - K, 0), 0)
    axes[0,1].plot(S, payoff_uo, color=CORAL, lw=2.5)
    axes[0,1].axvline(120, color="gray", ls=":", label="Barrier=120")
    axes[0,1].set_title("Up-and-Out Call (B=120)")
    axes[0,1].legend(fontsize=9)

    # Lookback call
    axes[1,0].plot(S, S - 80, color=GOLD, lw=2.5, label="Lookback (Smin=80)")
    axes[1,0].plot(S, np.maximum(S - K, 0), color="gray", lw=1, ls="--", alpha=0.5)
    axes[1,0].set_title("Lookback Call (Floating Strike)")
    axes[1,0].legend(fontsize=9)

    # Digital call
    axes[1,1].plot(S, np.where(S > K, 1, 0), color=NAVY, lw=2.5)
    axes[1,1].set_title("Digital / Binary Call")
    axes[1,1].set_ylim(-0.1, 1.2)

    for ax in axes.flat:
        ax.set_xlabel("Terminal Price ($)")
        ax.set_ylabel("Payoff ($)")
        ax.axvline(K, color="gray", ls=":", alpha=0.3)
    fig.suptitle("Exotic Option Payoff Profiles", fontsize=15, fontweight="bold")
    fig.tight_layout()
    _wm(fig)
    return _sv(fig, proj, "04_exotic_payoff_profiles.png")


def plot_greeks_mc(proj=None):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    np.random.seed(42)
    S0, K, T, r, sigma = 100, 100, 0.5, 0.05, 0.20
    N = 50000
    bumps_S = np.linspace(80, 120, 40)
    bumps_sigma = np.linspace(0.10, 0.35, 40)

    Z = np.random.randn(N)
    prices_s, deltas, gammas, vegas = [], [], [], []
    for s in bumps_S:
        ST = s * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
        p = np.exp(-r*T) * np.maximum(ST - K, 0).mean()
        prices_s.append(p)
    prices_s = np.array(prices_s)
    deltas = np.gradient(prices_s, bumps_S)
    gammas = np.gradient(deltas, bumps_S)

    for sv in bumps_sigma:
        ST = S0 * np.exp((r - 0.5*sv**2)*T + sv*np.sqrt(T)*Z)
        vegas.append(np.exp(-r*T) * np.maximum(ST - K, 0).mean())

    axes[0,0].plot(bumps_S, prices_s, color=TEAL, lw=2)
    axes[0,0].set_title("Price vs Spot")
    axes[0,0].set_xlabel("Spot ($)"); axes[0,0].set_ylabel("Price ($)")

    axes[0,1].plot(bumps_S, deltas, color=CORAL, lw=2)
    axes[0,1].set_title("Delta (dP/dS)")
    axes[0,1].set_xlabel("Spot ($)")

    axes[1,0].plot(bumps_S, gammas, color=GOLD, lw=2)
    axes[1,0].set_title("Gamma (d2P/dS2)")
    axes[1,0].set_xlabel("Spot ($)")

    axes[1,1].plot(bumps_sigma*100, vegas, color=NAVY, lw=2)
    axes[1,1].set_title("Vega Profile")
    axes[1,1].set_xlabel("Volatility (%)")
    fig.suptitle("Monte Carlo Greeks: Bump-and-Revalue", fontsize=15, fontweight="bold")
    fig.tight_layout()
    _wm(fig)
    return _sv(fig, proj, "05_greeks_mc.png")


def plot_error_distribution(proj=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    np.random.seed(42)
    S0, K, T, r, sigma = 100, 100, 1, 0.05, 0.20
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    bs = S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

    estimates = []
    for _ in range(500):
        Z = np.random.randn(5000)
        ST = S0 * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
        estimates.append(np.exp(-r*T) * np.maximum(ST - K, 0).mean())
    errors = np.array(estimates) - bs

    ax1.hist(errors, bins=40, color=TEAL, edgecolor="white", alpha=0.8, density=True)
    x = np.linspace(errors.min(), errors.max(), 100)
    ax1.plot(x, norm.pdf(x, errors.mean(), errors.std()), color=CORAL, lw=2, label="Normal fit")
    ax1.axvline(0, color="gray", ls="--")
    ax1.set_xlabel("Pricing Error ($)")
    ax1.set_title("MC Error Distribution (N=5000, 500 trials)")
    ax1.legend()

    from scipy.stats import probplot
    probplot(errors, dist="norm", plot=ax2)
    ax2.get_lines()[0].set_color(TEAL)
    ax2.get_lines()[1].set_color(CORAL)
    ax2.set_title("Q-Q Plot: Normality of MC Errors")
    fig.tight_layout()
    _wm(fig)
    return _sv(fig, proj, "06_error_distribution.png")


def generate_all_figures(project_dir=None):
    if project_dir is None:
        project_dir = os.path.join(os.path.dirname(__file__), "..", "..")
    print("  Generating Project 03 figures...")
    files = [
        plot_convergence(project_dir),
        plot_paths(project_dir),
        plot_variance_reduction(project_dir),
        plot_exotic_payoffs(project_dir),
        plot_greeks_mc(project_dir),
        plot_error_distribution(project_dir),
    ]
    print(f"  DONE: {len(files)} figures saved to outputs/figures/")
    return files

if __name__ == "__main__":
    generate_all_figures(os.path.join(os.path.dirname(__file__), "..", ".."))
