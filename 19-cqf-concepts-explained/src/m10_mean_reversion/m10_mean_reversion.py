#!/usr/bin/env python3
"""
M10 — Mean Reversion: Ornstein-Uhlenbeck Process
=================================================
Module 2 of 9 | CQF Concepts Explained

Theory
------
The Ornstein-Uhlenbeck (OU) process:

    dX_t = kappa * (theta - X_t) * dt + sigma * dW_t

Parameters:
    kappa  : speed of mean reversion  (kappa > 0)
    theta  : long-run mean (level to which X reverts)
    sigma  : instantaneous volatility

Exact discretization (no discretization error):
    X_{t+dt} = X_t * e^(-kappa*dt)
              + theta * (1 - e^(-kappa*dt))
              + sigma * sqrt[(1 - e^(-2*kappa*dt)) / (2*kappa)] * Z

    where Z ~ N(0,1)

Stationary distribution (t -> inf):
    X_inf ~ N(theta,  sigma^2 / (2*kappa))

Half-life of reversion:
    t_{1/2} = ln(2) / kappa

Conditional moments:
    E[X_t | X_0]   = theta + (X_0 - theta) * e^(-kappa*t)
    Var[X_t | X_0] = sigma^2/(2*kappa) * (1 - e^(-2*kappa*t))

Applications:
    - Vasicek short-rate model (interest rates)
    - Pairs trading (spread mean reversion)
    - Commodity prices (supply/demand equilibrium)
    - Volatility modeling (SABR, Heston mean-reverting vol)

Calibration from discrete observations {X_0, X_1, ..., X_n} with step dt:
    OLS regression: X_{t+dt} = a + b * X_t + epsilon
    kappa = -ln(b) / dt
    theta = a / (1 - b)
    sigma = std(epsilon) * sqrt(2*kappa / (1 - b^2))

Author : Jose O. Bobadilla | CQF
Project: 19-cqf-concepts-explained
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm
from scipy.optimize import minimize_scalar

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
DARK   = "#0a0a0a"
PANEL  = "#111111"
WHITE  = "#e8e8e8"
GREY   = "#444444"
BLUE   = "#4a9eff"
GREEN  = "#2ecc71"
YELLOW = "#f1c40f"
RED    = "#e74c3c"
ORANGE = "#e67e22"
PURPLE = "#9b59b6"

plt.rcParams.update({
    "figure.facecolor": DARK, "axes.facecolor": PANEL,
    "axes.edgecolor": GREY,   "axes.labelcolor": WHITE,
    "xtick.color": WHITE,     "ytick.color": WHITE,
    "text.color": WHITE,      "grid.color": GREY,
    "grid.alpha": 0.35,       "grid.linestyle": "--",
    "font.family": "monospace",
})

OUT_DIR = os.path.expanduser(
    "~/quant-finance-portfolio/19-cqf-concepts-explained/outputs/figures"
)
os.makedirs(OUT_DIR, exist_ok=True)
WATERMARK = "Jose O. Bobadilla | CQF"

def watermark(ax):
    ax.text(0.98, 0.02, WATERMARK, transform=ax.transAxes,
            fontsize=6, color=GREY, ha="right", va="bottom",
            style="italic", alpha=0.8)

# ---------------------------------------------------------------------------
# OU simulation (exact discretization)
# ---------------------------------------------------------------------------
def simulate_ou_exact(X0, kappa, theta, sigma, T, n_steps, n_paths, rng):
    """
    Exact simulation of OU process — no discretization bias.
    Returns array of shape (n_paths, n_steps+1).
    """
    dt   = T / n_steps
    e_k  = np.exp(-kappa * dt)
    std  = sigma * np.sqrt((1 - np.exp(-2*kappa*dt)) / (2*kappa))

    Z = rng.standard_normal((n_paths, n_steps))
    X = np.empty((n_paths, n_steps + 1))
    X[:, 0] = X0
    for i in range(n_steps):
        X[:, i+1] = X[:, i] * e_k + theta * (1 - e_k) + std * Z[:, i]
    return X

def ou_halflife(kappa):
    return np.log(2) / kappa

def ou_stationary_std(kappa, sigma):
    return sigma / np.sqrt(2 * kappa)

def ou_conditional_mean(X0, kappa, theta, t):
    return theta + (X0 - theta) * np.exp(-kappa * t)

def ou_conditional_var(kappa, sigma, t):
    return (sigma**2 / (2*kappa)) * (1 - np.exp(-2*kappa*t))

# ---------------------------------------------------------------------------
# OLS calibration
# ---------------------------------------------------------------------------
def calibrate_ou_ols(X, dt):
    """
    Calibrate OU from a single observed path via OLS.
    X : 1-D array of observations spaced dt apart.
    Returns (kappa, theta, sigma).
    """
    X_t  = X[:-1]
    X_t1 = X[1:]
    n    = len(X_t)
    # OLS: X_{t+1} = a + b * X_t
    b = (n*np.dot(X_t, X_t1) - X_t.sum()*X_t1.sum()) / \
        (n*np.dot(X_t, X_t)  - X_t.sum()**2)
    a = (X_t1.sum() - b * X_t.sum()) / n
    residuals = X_t1 - (a + b * X_t)
    s_eps = residuals.std(ddof=2)

    kappa_hat = -np.log(np.clip(b, 1e-10, 1 - 1e-10)) / dt
    theta_hat = a / (1 - b)
    sigma_hat = s_eps * np.sqrt(2*kappa_hat / (1 - b**2))
    return kappa_hat, theta_hat, sigma_hat


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
kappa = 2.0     # speed of mean reversion  (half-life ~ 0.35Y)
theta = 0.05    # long-run mean (e.g. short rate = 5%)
sigma = 0.02    # volatility
X0    = 0.10    # starting value (above mean — will revert)
T     = 5.0     # years
N_STEPS = 1260  # 5Y * 252
N_PATHS = 2000
SEED = 42
dt = T / N_STEPS

hl = ou_halflife(kappa)
stat_std = ou_stationary_std(kappa, sigma)
t_grid = np.linspace(0, T, N_STEPS + 1)

rng = np.random.default_rng(SEED)
paths = simulate_ou_exact(X0, kappa, theta, sigma, T, N_STEPS, N_PATHS, rng)

print(f"[M10] OU Parameters: kappa={kappa}, theta={theta:.2%}, "
      f"sigma={sigma:.2%}, X0={X0:.2%}")
print(f"      Half-life = {hl:.4f}Y  |  Stationary std = {stat_std:.4f}")


# ===========================================================================
# FIGURE 1 — Path Fan, Conditional Moments, Stationary Distribution
# ===========================================================================
print("[M10] Figure 1: Paths and conditional moments ...")
t0 = time.perf_counter()

fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=DARK)
fig.suptitle(
    "M10 — Ornstein-Uhlenbeck Process\n"
    f"dX = kappa*(theta - X)*dt + sigma*dW  |  kappa={kappa}, "
    f"theta={theta:.2%}, sigma={sigma:.2%}, X0={X0:.2%}",
    color=WHITE, fontsize=10
)

# (0) Path fan
ax = axes[0]
n_show = 60
for k in range(n_show):
    ax.plot(t_grid, paths[k] * 100, color=BLUE, alpha=0.12, lw=0.6)
# Mean path
sim_mean = paths.mean(axis=0)
sim_p5   = np.percentile(paths, 5, axis=0)
sim_p95  = np.percentile(paths, 95, axis=0)
theo_mean = ou_conditional_mean(X0, kappa, theta, t_grid)
theo_var  = ou_conditional_var(kappa, sigma, t_grid)
theo_std  = np.sqrt(theo_var)

ax.plot(t_grid, theo_mean * 100,  color=YELLOW, lw=2.5,
        label=r"$E[X_t|X_0]$ theoretical")
ax.plot(t_grid, sim_mean  * 100,  color=WHITE,  lw=1.5, linestyle="--",
        label="Simulated mean")
ax.fill_between(t_grid, (theo_mean-1.96*theo_std)*100,
                         (theo_mean+1.96*theo_std)*100,
                color=BLUE, alpha=0.12, label="95% confidence band")
ax.axhline(theta * 100, color=GREEN, lw=1.5, linestyle=":",
           label=f"Long-run mean theta = {theta:.2%}")
ax.set_xlabel("Time (years)"); ax.set_ylabel("X_t (%)")
ax.set_title(f"OU Path Fan  (n={N_PATHS:,})\n"
             f"Half-life = {hl:.3f}Y  ({hl*12:.1f} months)",
             color=WHITE, fontsize=9)
ax.legend(fontsize=6); ax.grid(True); watermark(ax)

# (1) Conditional variance over time: approaches stationary variance
ax = axes[1]
stat_var = sigma**2 / (2*kappa)
ax.plot(t_grid, theo_var * 1e4,  color=YELLOW, lw=2.5,
        label=r"$Var[X_t|X_0]$ theoretical (x $10^{-4}$)")
ax.plot(t_grid, paths.var(axis=0) * 1e4, color=WHITE, lw=1.5, linestyle="--",
        label="Simulated variance (x $10^{-4}$)")
ax.axhline(stat_var * 1e4, color=RED, lw=2, linestyle=":",
           label=r"Stationary $\sigma^2/(2\kappa)$"
                 f" = {stat_var*1e4:.4f} x$10^{{-4}}$")
ax.axvline(hl, color=ORANGE, lw=1.5, linestyle="--",
           label=f"Half-life = {hl:.3f}Y")
ax.set_xlabel("Time (years)"); ax.set_ylabel(r"Variance (x $10^{-4}$)")
ax.set_title("Conditional Variance Convergence\nto Stationary Variance",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (2) Stationary distribution: simulated terminal vs N(theta, sigma^2/2kappa)
ax = axes[2]
X_terminal = paths[:, -1]
x_range = np.linspace(X_terminal.min(), X_terminal.max(), 400)
pdf_stat = norm.pdf(x_range, theta, stat_std)
ax.hist(X_terminal * 100, bins=60, density=True, color=GREEN, alpha=0.55,
        edgecolor="none", label=f"Simulated X_T  (T={T}Y)")
ax.plot(x_range * 100, pdf_stat / 100, color=YELLOW, lw=2.5,
        label=f"N(theta, sigma^2/2kappa)\n= N({theta:.3f}, {stat_var:.6f})")
ax.axvline(theta * 100,      color=YELLOW, lw=2,   linestyle=":",
           label=f"theta = {theta:.2%}")
ax.axvline(X_terminal.mean()*100, color=WHITE, lw=1.5, linestyle="--",
           label=f"Sim. mean = {X_terminal.mean():.4f}")
ax.set_xlabel("X_T (%)"); ax.set_ylabel("Density")
ax.set_title("Stationary Distribution\n"
             r"$X_\infty \sim N(\theta,\; \sigma^2/2\kappa)$",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

plt.tight_layout()
p1 = os.path.join(OUT_DIR, "m10_01_ou_paths.png")
plt.savefig(p1, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p1}  ({time.perf_counter()-t0:.1f}s)")


# ===========================================================================
# FIGURE 2 — Speed of Reversion: kappa comparison
# ===========================================================================
print("[M10] Figure 2: Kappa sensitivity ...")
t0 = time.perf_counter()

kappas = [0.5, 1.0, 2.0, 5.0, 10.0]
colors_k = [PURPLE, BLUE, GREEN, YELLOW, ORANGE]
T_k   = 3.0
ns_k  = 756    # 3Y * 252
t_k   = np.linspace(0, T_k, ns_k + 1)
rng_k = np.random.default_rng(SEED + 1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=DARK)
fig.suptitle("M10 — Effect of Mean-Reversion Speed kappa",
             color=WHITE, fontsize=11)

ax = axes[0]   # representative single-path per kappa
for kap, col in zip(kappas, colors_k):
    hl_k = ou_halflife(kap)
    path_k = simulate_ou_exact(X0, kap, theta, sigma, T_k, ns_k, 1, rng_k)[0]
    ax.plot(t_k, path_k * 100, color=col, lw=1.5,
            label=f"kappa={kap:4.1f}  (t_half={hl_k:.3f}Y)")
ax.axhline(theta * 100, color=WHITE, lw=1.5, linestyle=":",
           label=f"Long-run mean = {theta:.2%}")
ax.axhline(X0 * 100,    color=RED,   lw=1,   linestyle="--",
           label=f"X0 = {X0:.2%}", alpha=0.6)
ax.set_xlabel("Time (years)"); ax.set_ylabel("X_t (%)")
ax.set_title("Single-Path Realization per kappa\n(same Brownian driver)",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

ax = axes[1]   # conditional mean decay for each kappa
t_decay = np.linspace(0, T_k, 500)
for kap, col in zip(kappas, colors_k):
    cond_m = ou_conditional_mean(X0, kap, theta, t_decay)
    ax.plot(t_decay, (cond_m - theta) * 100, color=col, lw=2,
            label=f"kappa={kap:.1f}")
ax.axhline(0, color=WHITE, lw=1, linestyle=":", alpha=0.7,
           label="theta (target)")
ax.set_xlabel("Time (years)")
ax.set_ylabel("E[X_t | X_0] - theta  (%)")
ax.set_title("Conditional Mean Decay\n"
             r"$E[X_t|X_0] - \theta = (X_0-\theta)e^{-\kappa t}$",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

plt.tight_layout()
p2 = os.path.join(OUT_DIR, "m10_02_kappa_sensitivity.png")
plt.savefig(p2, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p2}  ({time.perf_counter()-t0:.1f}s)")


# ===========================================================================
# FIGURE 3 — OLS Calibration: Estimation vs True Parameters
# ===========================================================================
print("[M10] Figure 3: OLS calibration study ...")
t0 = time.perf_counter()

# Simulate many paths and calibrate each — study estimation distribution
N_CAL   = 1000
T_cal   = 2.0
ns_cal  = 504    # 2Y * 252
dt_cal  = T_cal / ns_cal
rng_cal = np.random.default_rng(SEED + 2)

paths_cal = simulate_ou_exact(X0, kappa, theta, sigma, T_cal,
                               ns_cal, N_CAL, rng_cal)

kappa_hats = np.zeros(N_CAL)
theta_hats = np.zeros(N_CAL)
sigma_hats = np.zeros(N_CAL)

for k in range(N_CAL):
    try:
        kh, th, sh = calibrate_ou_ols(paths_cal[k], dt_cal)
        kappa_hats[k] = kh
        theta_hats[k] = th
        sigma_hats[k] = sh
    except Exception:
        kappa_hats[k] = np.nan
        theta_hats[k] = np.nan
        sigma_hats[k] = np.nan

# Filter outliers (OLS can blow up for nearly non-stationary paths)
mask = (
    (kappa_hats > 0) & (kappa_hats < 20) &
    (np.abs(theta_hats) < 0.5)  &
    (sigma_hats > 0)   & (sigma_hats < 0.20)
)
kh = kappa_hats[mask]
th = theta_hats[mask]
sh = sigma_hats[mask]

fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=DARK)
fig.suptitle(
    f"M10 — OLS Calibration Study  (N={N_CAL} paths, T={T_cal}Y, dt=1/252)\n"
    f"True: kappa={kappa}, theta={theta:.2%}, sigma={sigma:.2%}",
    color=WHITE, fontsize=10
)

param_data = [
    (kh,   kappa, "kappa (speed)", BLUE,   "kappa"),
    (th*100, theta*100, "theta (%) (long-run mean)", GREEN,  "theta (%)"),
    (sh*100, sigma*100, "sigma (%) (volatility)",    ORANGE, "sigma (%)"),
]

for ax, (data, true_val, label, col, xlabel) in zip(axes, param_data):
    ax.hist(data, bins=50, density=True, color=col, alpha=0.6,
            edgecolor="none", label=f"OLS estimates  (n={mask.sum()})")
    ax.axvline(true_val, color=YELLOW, lw=2.5, linestyle="--",
               label=f"True value = {true_val:.4f}")
    ax.axvline(data.mean(), color=WHITE, lw=2, linestyle=":",
               label=f"Mean estimate = {data.mean():.4f}")
    bias = data.mean() - true_val
    rmse = np.sqrt(((data - true_val)**2).mean())
    ax.set_xlabel(xlabel); ax.set_ylabel("Density")
    ax.set_title(f"{label}\nBias = {bias:+.4f}  |  RMSE = {rmse:.4f}",
                 color=WHITE, fontsize=9)
    ax.legend(fontsize=7); ax.grid(True); watermark(ax)

plt.tight_layout()
p3 = os.path.join(OUT_DIR, "m10_03_ols_calibration.png")
plt.savefig(p3, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p3}  ({time.perf_counter()-t0:.1f}s)")

print()
print("=" * 65)
print("  MODULE 10 COMPLETE -- 3 figures saved")
print("  Key Concepts:")
print("  [1] OU SDE: dX = kappa*(theta-X)*dt + sigma*dW")
print("  [2] Exact discretization: no accumulation of error")
print(f"  [3] Half-life = ln(2)/kappa = {hl:.4f}Y")
print(f"  [4] Stationary distribution: N(theta, sigma^2/2kappa)")
print(f"       = N({theta:.4f}, {stat_std**2:.8f})")
print(f"       std = {stat_std:.6f}  ({stat_std*100:.4f} bps * 100)")
print("  [5] OLS calibration: X_{t+dt} = a + b*X_t + eps")
print("  [6] Vasicek model = OU applied to short rate")
print("=" * 65)
