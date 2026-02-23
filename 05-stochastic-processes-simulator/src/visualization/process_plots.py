"""
Publication-quality visualizations for Stochastic Processes Simulator.

Figures generated:
    01_gbm_paths.png               - GBM sample paths with analytics
    02_ou_mean_reversion.png       - OU process mean-reversion demo
    03_cir_positivity.png          - CIR process with positivity constraint
    04_process_comparison.png      - Side-by-side: ABM vs GBM vs OU vs CIR
    05_distribution_evolution.png  - Time-evolving density for GBM
    06_ou_stationary.png           - OU stationary distribution convergence

Author: Jose Orlando Bobadilla Fuentes, CQF
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, lognorm

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


def plot_gbm_paths(proj=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    np.random.seed(42)
    S0, mu, sigma, T, N, M = 100, 0.08, 0.20, 2, 500, 200
    dt = T / N
    paths = np.zeros((M, N+1)); paths[:, 0] = S0
    for i in range(N):
        Z = np.random.randn(M)
        paths[:, i+1] = paths[:, i] * np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z)
    t = np.linspace(0, T, N+1)
    for i in range(30):
        ax1.plot(t, paths[i], lw=0.5, alpha=0.4, color=TEAL)
    ax1.plot(t, S0*np.exp(mu*t), color=CORAL, lw=2.5, label=r"$E[S_t] = S_0 e^{\mu t}$")
    ax1.plot(t, np.median(paths, axis=0), color=NAVY, lw=2, ls="--", label="Median")
    ax1.fill_between(t, np.percentile(paths, 5, axis=0),
                     np.percentile(paths, 95, axis=0), alpha=0.1, color=NAVY)
    ax1.set_xlabel("Time (years)"); ax1.set_ylabel("Price ($)")
    ax1.set_title("GBM Sample Paths"); ax1.legend()

    terminal = paths[:, -1]
    ax2.hist(terminal, bins=50, density=True, alpha=0.6, color=TEAL, edgecolor="white")
    x = np.linspace(terminal.min(), terminal.max(), 200)
    mean_log = np.log(S0) + (mu - 0.5*sigma**2)*T
    std_log = sigma*np.sqrt(T)
    ax2.plot(x, lognorm.pdf(x, std_log, scale=np.exp(mean_log)), color=CORAL, lw=2.5,
             label="Lognormal (theoretical)")
    ax2.set_xlabel("Terminal Price ($)"); ax2.set_title("Terminal Distribution")
    ax2.legend()
    fig.suptitle("Geometric Brownian Motion", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    _wm(fig)
    return _sv(fig, proj, "01_gbm_paths.png")


def plot_ou_mean_reversion(proj=None):
    fig, ax = plt.subplots(figsize=(11, 6))
    np.random.seed(7)
    kappa, theta, sigma = 5.0, 0.05, 0.01
    T, N = 3, 750
    dt = T / N
    t = np.linspace(0, T, N+1)
    for i, x0 in enumerate([0.02, 0.05, 0.08, 0.10]):
        path = np.zeros(N+1); path[0] = x0
        for j in range(N):
            path[j+1] = path[j] + kappa*(theta - path[j])*dt + sigma*np.sqrt(dt)*np.random.randn()
        ax.plot(t, path*100, color=COLORS[i], lw=2, label=f"$X_0$ = {x0*100:.0f}%")
    ax.axhline(theta*100, color="gray", ls="--", lw=2, label=r"$\theta$ = " + f"{theta*100:.0f}%")
    band = sigma / np.sqrt(2*kappa) * 100
    ax.fill_between(t, (theta - band)*100, (theta + band)*100, alpha=0.1, color="gray")
    ax.set_xlabel("Time (years)"); ax.set_ylabel("Rate (%)")
    ax.set_title(r"Ornstein-Uhlenbeck: Mean Reversion ($\kappa$=5, $\theta$=5%)")
    ax.legend()
    _wm(fig)
    return _sv(fig, proj, "02_ou_mean_reversion.png")


def plot_cir_positivity(proj=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    np.random.seed(42)
    kappa, theta, sigma = 3.0, 0.04, 0.05
    T, N, M = 5, 1250, 100
    dt = T / N
    feller = 2*kappa*theta / sigma**2
    paths = np.zeros((M, N+1)); paths[:, 0] = 0.04
    t = np.linspace(0, T, N+1)
    for i in range(N):
        Z = np.random.randn(M)
        drift = kappa * (theta - paths[:, i]) * dt
        diff = sigma * np.sqrt(np.maximum(paths[:, i], 0) * dt) * Z
        paths[:, i+1] = np.maximum(paths[:, i] + drift + diff, 0)

    for i in range(20):
        ax1.plot(t, paths[i]*100, lw=0.6, alpha=0.5, color=TEAL)
    ax1.axhline(theta*100, color=CORAL, ls="--", lw=2, label=r"$\theta$")
    ax1.fill_between(t, np.percentile(paths, 5, axis=0)*100,
                     np.percentile(paths, 95, axis=0)*100, alpha=0.1, color=NAVY)
    ax1.set_xlabel("Time (years)"); ax1.set_ylabel("Rate (%)")
    ax1.set_title(f"CIR Process (Feller = {feller:.2f})")
    ax1.legend()

    ax2.hist(paths[:, -1]*100, bins=40, density=True, alpha=0.6, color=TEAL, edgecolor="white")
    ax2.axvline(theta*100, color=CORAL, ls="--", lw=2)
    ax2.set_xlabel("Terminal Rate (%)"); ax2.set_title("CIR Stationary Distribution")
    fig.tight_layout()
    _wm(fig)
    return _sv(fig, proj, "03_cir_positivity.png")


def plot_process_comparison(proj=None):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    np.random.seed(42)
    T, N = 2, 500; dt = T / N; t = np.linspace(0, T, N+1)
    Z = np.random.randn(N)

    # ABM
    abm = np.zeros(N+1); abm[0] = 100
    for i in range(N): abm[i+1] = abm[i] + 0.05*dt + 2*np.sqrt(dt)*Z[i]
    axes[0,0].plot(t, abm, color=TEAL, lw=2)
    axes[0,0].set_title("Arithmetic BM: dX = mu*dt + sigma*dW")

    # GBM
    gbm = np.zeros(N+1); gbm[0] = 100
    for i in range(N): gbm[i+1] = gbm[i] * np.exp(0.03*dt + 0.2*np.sqrt(dt)*Z[i])
    axes[0,1].plot(t, gbm, color=CORAL, lw=2)
    axes[0,1].set_title("Geometric BM: dS = S(mu*dt + sigma*dW)")

    # OU
    ou = np.zeros(N+1); ou[0] = 0.08
    for i in range(N): ou[i+1] = ou[i] + 3*(0.05 - ou[i])*dt + 0.01*np.sqrt(dt)*Z[i]
    axes[1,0].plot(t, ou*100, color=GOLD, lw=2)
    axes[1,0].axhline(5, color="gray", ls="--", alpha=0.5)
    axes[1,0].set_title(r"Ornstein-Uhlenbeck: dX = $\kappa$($\theta$-X)dt + $\sigma$dW")

    # CIR
    cir = np.zeros(N+1); cir[0] = 0.04
    for i in range(N):
        cir[i+1] = max(cir[i] + 2*(0.04 - cir[i])*dt + 0.03*np.sqrt(max(cir[i],0)*dt)*Z[i], 0)
    axes[1,1].plot(t, cir*100, color=NAVY, lw=2)
    axes[1,1].axhline(4, color="gray", ls="--", alpha=0.5)
    axes[1,1].set_title(r"CIR: dX = $\kappa$($\theta$-X)dt + $\sigma\sqrt{X}$dW")

    for ax in axes.flat:
        ax.set_xlabel("Time (years)")
    fig.suptitle("Canonical Stochastic Processes in Quantitative Finance",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    _wm(fig)
    return _sv(fig, proj, "04_process_comparison.png")


def plot_distribution_evolution(proj=None):
    fig, ax = plt.subplots(figsize=(11, 6))
    S0, mu, sigma = 100, 0.08, 0.20
    x = np.linspace(20, 300, 500)
    for i, T in enumerate([0.25, 0.5, 1.0, 2.0, 5.0]):
        mean_log = np.log(S0) + (mu - 0.5*sigma**2)*T
        std_log = sigma*np.sqrt(T)
        pdf = lognorm.pdf(x, std_log, scale=np.exp(mean_log))
        ax.plot(x, pdf, color=COLORS[i], lw=2, label=f"T = {T}y")
        ax.fill_between(x, pdf, alpha=0.05, color=COLORS[i])
    ax.axvline(S0, color="gray", ls=":", alpha=0.5)
    ax.set_xlabel("Asset Price ($)"); ax.set_ylabel("Density")
    ax.set_title("GBM: Evolution of Price Distribution Over Time")
    ax.legend()
    _wm(fig)
    return _sv(fig, proj, "05_distribution_evolution.png")


def plot_ou_stationary_distribution(proj=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    np.random.seed(42)
    kappa, theta, sigma = 5.0, 0.05, 0.01
    T, N, M = 10, 2500, 500
    dt = T / N
    paths = np.zeros((M, N+1)); paths[:, 0] = np.random.uniform(0.01, 0.10, M)
    for i in range(N):
        paths[:, i+1] = paths[:, i] + kappa*(theta - paths[:, i])*dt + sigma*np.sqrt(dt)*np.random.randn(M)

    # Time snapshots
    for i, snap in enumerate([0, 250, 1000, 2500]):
        ax1.hist(paths[:, snap]*100, bins=30, density=True, alpha=0.5,
                 color=COLORS[i], label=f"t={snap*dt:.1f}y")
    stat_std = sigma / np.sqrt(2*kappa) * 100
    x = np.linspace(2, 8, 200)
    ax1.plot(x, norm.pdf(x, theta*100, stat_std), color="black", lw=2, ls="--",
             label="Stationary")
    ax1.set_xlabel("Rate (%)"); ax1.set_title("Convergence to Stationary Distribution")
    ax1.legend(fontsize=8)

    # Variance over time
    empirical_var = paths.var(axis=0)
    theoretical_var = (sigma**2 / (2*kappa)) * (1 - np.exp(-2*kappa*np.linspace(0, T, N+1)))
    ax2.plot(np.linspace(0, T, N+1), empirical_var*1e4, color=TEAL, lw=2, label="Empirical")
    ax2.plot(np.linspace(0, T, N+1), theoretical_var*1e4, color=CORAL, lw=2, ls="--",
             label="Theoretical")
    ax2.axhline((sigma**2 / (2*kappa))*1e4, color="gray", ls=":", alpha=0.5)
    ax2.set_xlabel("Time (years)"); ax2.set_ylabel("Variance (x1e4)")
    ax2.set_title("Variance Convergence to Stationary Level")
    ax2.legend()
    fig.tight_layout()
    _wm(fig)
    return _sv(fig, proj, "06_ou_stationary.png")


def generate_all_figures(project_dir=None):
    if project_dir is None:
        project_dir = os.path.join(os.path.dirname(__file__), "..", "..")
    print("  Generating Project 05 figures...")
    files = [
        plot_gbm_paths(project_dir),
        plot_ou_mean_reversion(project_dir),
        plot_cir_positivity(project_dir),
        plot_process_comparison(project_dir),
        plot_distribution_evolution(project_dir),
        plot_ou_stationary_distribution(project_dir),
    ]
    print(f"  DONE: {len(files)} figures saved to outputs/figures/")
    return files

if __name__ == "__main__":
    generate_all_figures(os.path.join(os.path.dirname(__file__), "..", ".."))
