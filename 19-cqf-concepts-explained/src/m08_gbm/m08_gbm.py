#!/usr/bin/env python3
"""
M08 — GBM Simulation: Euler-Maruyama and Exact Scheme
======================================================
Module 2 of 9 | CQF Concepts Explained

Theory
------
Geometric Brownian Motion (GBM) is the canonical model for equity prices:

    dS_t = mu * S_t * dt + sigma * S_t * dW_t

Exact solution (via Ito's Lemma on ln S):

    S_t = S_0 * exp[(mu - sigma^2/2)*t + sigma*W_t]

The term (mu - sigma^2/2) is the *geometric* drift — smaller than the
arithmetic drift mu by exactly sigma^2/2. This Ito correction is the
central insight: the expected log-return is not mu but mu - sigma^2/2.

Numerical Schemes
-----------------
Euler-Maruyama (strong order 0.5, weak order 1.0):
    S_{t+dt} = S_t (1 + mu*dt + sigma*sqrt(dt)*Z),  Z ~ N(0,1)

Milstein (strong order 1.0 for GBM = Exact for GBM):
    S_{t+dt} = S_t * exp[(mu - sigma^2/2)*dt + sigma*sqrt(dt)*Z]
    (coincides with the exact solution step-by-step)

Convergence Study
-----------------
Strong error: E[|S_T^{scheme} - S_T^{exact}|] ~ C * dt^p
    Euler-Maruyama: p = 0.5
    Milstein / Exact: p = 1.0  (for GBM)

Weak error: |E[f(S_T^{scheme})] - E[f(S_T^{exact})]| ~ C * dt^p
    Euler-Maruyama: p = 1.0
    Milstein / Exact: p = 2.0

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
from scipy.stats import norm, lognorm

# ---------------------------------------------------------------------------
# Global style
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
# Parameters
# ---------------------------------------------------------------------------
S0    = 100.0    # initial price
mu    = 0.08     # annual drift (arithmetic)
sigma = 0.20     # annual volatility
T     = 1.0      # horizon (years)
N_PATHS = 2000
SEED    = 42

# ---------------------------------------------------------------------------
# Core simulation functions
# ---------------------------------------------------------------------------

def simulate_gbm_exact(S0, mu, sigma, T, n_steps, n_paths, rng):
    """Exact GBM simulation using the analytical log-normal transition."""
    dt = T / n_steps
    Z  = rng.standard_normal((n_paths, n_steps))
    # Incremental exact steps (no accumulated error)
    log_increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(log_increments, axis=1)], axis=1
    )
    return S0 * np.exp(log_paths)                          # shape (n_paths, n_steps+1)


def simulate_gbm_euler(S0, mu, sigma, T, n_steps, n_paths, rng):
    """Euler-Maruyama discretization of GBM."""
    dt  = T / n_steps
    sqdt = np.sqrt(dt)
    Z   = rng.standard_normal((n_paths, n_steps))
    S   = np.empty((n_paths, n_steps + 1))
    S[:, 0] = S0
    for i in range(n_steps):
        S[:, i+1] = S[:, i] * (1.0 + mu * dt + sigma * sqdt * Z[:, i])
        S[:, i+1] = np.maximum(S[:, i+1], 1e-12)          # absorbing floor
    return S


def simulate_gbm_milstein(S0, mu, sigma, T, n_steps, n_paths, rng):
    """
    Milstein scheme for GBM.
    For GBM b(S)=sigma*S => b'(S)=sigma, so the Milstein correction becomes:
        S_{t+dt} = S_t [1 + mu*dt + sigma*dW + 0.5*sigma^2*(dW^2 - dt)]
    which equals the exact step, so strong order = 1.0.
    """
    dt   = T / n_steps
    sqdt = np.sqrt(dt)
    Z    = rng.standard_normal((n_paths, n_steps))
    dW   = sqdt * Z
    S    = np.empty((n_paths, n_steps + 1))
    S[:, 0] = S0
    for i in range(n_steps):
        dw = dW[:, i]
        S[:, i+1] = S[:, i] * (
            1.0 + mu * dt + sigma * dw + 0.5 * sigma**2 * (dw**2 - dt)
        )
        S[:, i+1] = np.maximum(S[:, i+1], 1e-12)
    return S


# ===========================================================================
# FIGURE 1 — Path Comparison: Exact vs Euler vs Milstein
# ===========================================================================
print("[M08] Figure 1: Path comparison ...")
t0 = time.perf_counter()

n_steps_plot = 252
rng = np.random.default_rng(SEED)

# Use same Brownian increments for a fair visual comparison
Z_shared = rng.standard_normal((N_PATHS, n_steps_plot))
dt_plot   = T / n_steps_plot
sqdt_plot = np.sqrt(dt_plot)
t_grid    = np.linspace(0, T, n_steps_plot + 1)

# Exact
exact_paths = simulate_gbm_exact(S0, mu, sigma, T, n_steps_plot, N_PATHS,
                                  np.random.default_rng(SEED))

# Euler  — same Z_shared
euler_paths = np.empty((N_PATHS, n_steps_plot + 1))
euler_paths[:, 0] = S0
for i in range(n_steps_plot):
    euler_paths[:, i+1] = euler_paths[:, i] * (
        1 + mu * dt_plot + sigma * sqdt_plot * Z_shared[:, i]
    )
    euler_paths[:, i+1] = np.maximum(euler_paths[:, i+1], 1e-12)

# Milstein — same Z_shared
mils_paths = np.empty((N_PATHS, n_steps_plot + 1))
mils_paths[:, 0] = S0
for i in range(n_steps_plot):
    dw = sqdt_plot * Z_shared[:, i]
    mils_paths[:, i+1] = mils_paths[:, i] * (
        1 + mu * dt_plot + sigma * dw + 0.5 * sigma**2 * (dw**2 - dt_plot)
    )
    mils_paths[:, i+1] = np.maximum(mils_paths[:, i+1], 1e-12)

fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=DARK)
fig.suptitle(
    "M08 — GBM Simulation: Exact vs Euler-Maruyama vs Milstein\n"
    r"$dS_t = \mu S_t\,dt + \sigma S_t\,dW_t$"
    f"    mu={mu:.0%}, sigma={sigma:.0%}, T={T}Y, {N_PATHS} paths",
    color=WHITE, fontsize=10, y=1.01
)

n_show = 50
datasets = [
    (exact_paths, BLUE,   "Exact (log-normal)", r"$S_T = S_0\,e^{(\mu-\sigma^2/2)t+\sigma W_t}$"),
    (euler_paths, ORANGE, "Euler-Maruyama",     r"$S_{t+\Delta t}=S_t(1+\mu\Delta t+\sigma\sqrt{\Delta t}\,Z)$"),
    (mils_paths,  GREEN,  "Milstein",           r"Euler + $\frac{1}{2}\sigma^2(\Delta W^2-\Delta t)$ correction"),
]

for ax, (paths, col, label, formula) in zip(axes, datasets):
    # Fan of paths
    for k in range(n_show):
        ax.plot(t_grid, paths[k], color=col, alpha=0.12, lw=0.6)
    # Mean path
    mean_path = paths.mean(axis=0)
    ax.plot(t_grid, mean_path, color=WHITE, lw=2.0, label="Simulated mean")
    # Theoretical mean
    ax.plot(t_grid, S0 * np.exp(mu * t_grid), color=YELLOW, lw=1.5,
            linestyle="--", label=r"$E[S_t]=S_0 e^{\mu t}$")
    # 5th / 95th percentiles
    p5  = np.percentile(paths, 5,  axis=0)
    p95 = np.percentile(paths, 95, axis=0)
    ax.fill_between(t_grid, p5, p95, color=col, alpha=0.08,
                    label="5th–95th pct")
    ax.set_title(f"{label}\n{formula}", color=WHITE, fontsize=8)
    ax.set_xlabel("Time (years)"); ax.set_ylabel("$S_t$")
    ax.legend(fontsize=6); ax.grid(True)
    watermark(ax)

plt.tight_layout()
path1 = os.path.join(OUT_DIR, "m08_01_path_comparison.png")
plt.savefig(path1, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {path1}  ({time.perf_counter()-t0:.1f}s)")


# ===========================================================================
# FIGURE 2 — Strong Convergence Study: Euler vs Milstein
# ===========================================================================
print("[M08] Figure 2: Strong convergence study ...")
t0 = time.perf_counter()

# True terminal value at T using exact simulation with finest grid
N_CONV    = 500        # paths for convergence
N_REF     = 4096       # reference steps (very fine)
dt_steps  = np.array([4, 8, 16, 32, 64, 128, 256, 512])

rng_ref = np.random.default_rng(SEED + 1)
# Generate reference Brownian motion on fine grid, then subsample
Z_fine = rng_ref.standard_normal((N_CONV, N_REF))
W_fine = np.cumsum(np.sqrt(T / N_REF) * Z_fine, axis=1)   # (N_CONV, N_REF)

# Exact terminal value S_T using fine-grid W_T
W_T   = W_fine[:, -1]
S_exact_term = S0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * W_T)

strong_euler = []
strong_mils  = []

for n_steps_c in dt_steps:
    dt_c   = T / n_steps_c
    sqdt_c = np.sqrt(dt_c)
    step_size = N_REF // n_steps_c   # subsample factor

    # Subsample Brownian increments from fine grid
    dW_coarse = np.zeros((N_CONV, n_steps_c))
    for j in range(n_steps_c):
        idx_start = j * step_size
        idx_end   = (j + 1) * step_size
        # Sum fine increments to get coarse dW_j
        dW_coarse[:, j] = (
            W_fine[:, idx_end - 1] - (W_fine[:, idx_start - 1] if idx_start > 0
                                       else np.zeros(N_CONV))
        )

    # Euler terminal
    S_e = np.full(N_CONV, S0)
    for i in range(n_steps_c):
        S_e = S_e * (1 + mu * dt_c + sigma * dW_coarse[:, i])
        S_e = np.maximum(S_e, 1e-12)
    strong_euler.append(np.mean(np.abs(S_e - S_exact_term)))

    # Milstein terminal
    S_m = np.full(N_CONV, S0)
    for i in range(n_steps_c):
        dw = dW_coarse[:, i]
        S_m = S_m * (1 + mu * dt_c + sigma * dw + 0.5 * sigma**2 * (dw**2 - dt_c))
        S_m = np.maximum(S_m, 1e-12)
    strong_mils.append(np.mean(np.abs(S_m - S_exact_term)))

strong_euler = np.array(strong_euler)
strong_mils  = np.array(strong_mils)
dt_vals = T / dt_steps

# Fit log-log slopes
p_euler = np.polyfit(np.log(dt_vals), np.log(strong_euler), 1)[0]
p_mils  = np.polyfit(np.log(dt_vals), np.log(strong_mils),  1)[0]

# Reference lines
ref_half = dt_vals**0.5 * (strong_euler[0] / dt_vals[0]**0.5)
ref_one  = dt_vals**1.0 * (strong_mils[0]  / dt_vals[0]**1.0)

# --- Weak convergence: E[S_T] for each scheme vs exact E[S_T] = S0*exp(mu*T)
E_exact = S0 * np.exp(mu * T)
weak_euler = []
weak_mils  = []
N_WEAK = 2000
for n_steps_c in dt_steps:
    dt_c = T / n_steps_c
    rng_w = np.random.default_rng(SEED + 2)
    Z_w   = rng_w.standard_normal((N_WEAK, n_steps_c))
    sqdt_c = np.sqrt(dt_c)

    # Euler
    S_e = np.full(N_WEAK, S0)
    for i in range(n_steps_c):
        S_e = S_e * (1 + mu * dt_c + sigma * sqdt_c * Z_w[:, i])
        S_e = np.maximum(S_e, 1e-12)
    weak_euler.append(abs(S_e.mean() - E_exact))

    # Milstein
    S_m = np.full(N_WEAK, S0)
    for i in range(n_steps_c):
        dw = sqdt_c * Z_w[:, i]
        S_m = S_m * (1 + mu * dt_c + sigma * dw + 0.5 * sigma**2 * (dw**2 - dt_c))
        S_m = np.maximum(S_m, 1e-12)
    weak_mils.append(abs(S_m.mean() - E_exact))

weak_euler = np.array(weak_euler)
weak_mils  = np.array(weak_mils)
pw_euler = np.polyfit(np.log(dt_vals), np.log(weak_euler + 1e-15), 1)[0]
pw_mils  = np.polyfit(np.log(dt_vals), np.log(weak_mils  + 1e-15), 1)[0]

fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=DARK)
fig.suptitle("M08 — Convergence Analysis: Strong and Weak Errors",
             color=WHITE, fontsize=11)

# Strong convergence
ax = axes[0]
ax.loglog(dt_vals, strong_euler, "o-", color=ORANGE, lw=2, ms=6,
          label=f"Euler-Maruyama  (slope={p_euler:.2f}, theory=0.50)")
ax.loglog(dt_vals, strong_mils,  "s-", color=GREEN,  lw=2, ms=6,
          label=f"Milstein        (slope={p_mils:.2f}, theory=1.00)")
ax.loglog(dt_vals, ref_half, "--", color=ORANGE, alpha=0.5, lw=1,
          label=r"$O(\Delta t^{0.5})$ reference")
ax.loglog(dt_vals, ref_one,  "--", color=GREEN,  alpha=0.5, lw=1,
          label=r"$O(\Delta t^{1.0})$ reference")
ax.set_xlabel(r"Step size $\Delta t$"); ax.set_ylabel(r"$E[|S_T^{scheme} - S_T^{exact}|]$")
ax.set_title("Strong Convergence\n"
             r"$E[|S_T^{\mathrm{scheme}}-S_T^{\mathrm{exact}}|]\sim C\,\Delta t^p$",
             color=WHITE, fontsize=9)
ax.legend(fontsize=8); ax.grid(True); watermark(ax)

# Weak convergence
ax = axes[1]
ax.loglog(dt_vals, weak_euler, "o-", color=ORANGE, lw=2, ms=6,
          label=f"Euler-Maruyama  (slope={pw_euler:.2f}, theory=1.00)")
ax.loglog(dt_vals, weak_mils,  "s-", color=GREEN,  lw=2, ms=6,
          label=f"Milstein        (slope={pw_mils:.2f}, theory=2.00)")
ax.set_xlabel(r"Step size $\Delta t$")
ax.set_ylabel(r"$|E[S_T^{scheme}] - E[S_T^{exact}]|$")
ax.set_title("Weak Convergence\n"
             r"$|E[S_T^{\mathrm{scheme}}] - E[S_T^{\mathrm{exact}}]|\sim C\,\Delta t^p$",
             color=WHITE, fontsize=9)
ax.legend(fontsize=8); ax.grid(True); watermark(ax)

plt.tight_layout()
path2 = os.path.join(OUT_DIR, "m08_02_convergence.png")
plt.savefig(path2, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {path2}  ({time.perf_counter()-t0:.1f}s)")


# ===========================================================================
# FIGURE 3 — Path Statistics, Ito Correction, Terminal Distribution
# ===========================================================================
print("[M08] Figure 3: Path statistics and terminal distribution ...")
t0 = time.perf_counter()

n_steps_stat = 252
rng3 = np.random.default_rng(SEED + 3)
paths_exact = simulate_gbm_exact(S0, mu, sigma, T, n_steps_stat, N_PATHS, rng3)
t_grid_s = np.linspace(0, T, n_steps_stat + 1)
dt_s = T / n_steps_stat

fig = plt.figure(figsize=(16, 10), facecolor=DARK)
fig.suptitle(
    "M08 — GBM Path Statistics, Ito Correction & Terminal Distribution",
    color=WHITE, fontsize=11
)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.35)

# (0,0) Simulated mean vs theoretical E[S_t] and Var[S_t]
ax = fig.add_subplot(gs[0, 0])
sim_mean = paths_exact.mean(axis=0)
sim_std  = paths_exact.std(axis=0)
theo_mean = S0 * np.exp(mu * t_grid_s)
theo_var  = S0**2 * np.exp(2*mu*t_grid_s) * (np.exp(sigma**2 * t_grid_s) - 1)
theo_std  = np.sqrt(theo_var)
ax.plot(t_grid_s, theo_mean, color=YELLOW, lw=2, label=r"$E[S_t]=S_0 e^{\mu t}$")
ax.plot(t_grid_s, sim_mean,  color=WHITE,  lw=1.5, linestyle="--",
        label="Simulated mean")
ax.fill_between(t_grid_s, theo_mean - theo_std, theo_mean + theo_std,
                color=BLUE, alpha=0.15, label=r"$\pm 1\sigma$ theoretical")
ax.fill_between(t_grid_s, sim_mean - sim_std, sim_mean + sim_std,
                color=GREEN, alpha=0.10, label=r"$\pm 1\sigma$ simulated")
ax.set_xlabel("Time (years)"); ax.set_ylabel("$S_t$")
ax.set_title("Empirical vs Theoretical Moments", color=WHITE, fontsize=9)
ax.legend(fontsize=6); ax.grid(True); watermark(ax)

# (0,1) Ito correction: arithmetic vs geometric drift across sigma range
ax = fig.add_subplot(gs[0, 1])
sigmas_range = np.linspace(0.05, 0.80, 300)
arith_drift  = mu * np.ones_like(sigmas_range)
geom_drift   = mu - 0.5 * sigmas_range**2
ax.plot(sigmas_range * 100, arith_drift * 100, color=YELLOW, lw=2.5,
        label=f"Arithmetic drift = mu = {mu:.0%}")
ax.plot(sigmas_range * 100, geom_drift  * 100, color=GREEN,  lw=2.5,
        label=r"Geometric drift = $\mu - \sigma^2/2$")
ax.fill_between(sigmas_range * 100, geom_drift * 100, arith_drift * 100,
                color=RED, alpha=0.15, label=r"Ito correction = $\sigma^2/2$")
ax.axvline(sigma * 100, color=WHITE, lw=1.5, linestyle="--",
           label=f"Current sigma = {sigma:.0%}")
ax.set_xlabel("Volatility sigma (%)"); ax.set_ylabel("Annual drift (%)")
ax.set_title(r"Ito Correction: $\mu$ vs $\mu - \sigma^2/2$", color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (0,2) Log-return distribution: simulated vs N(mu - sigma^2/2, sigma^2)*dt
ax = fig.add_subplot(gs[0, 2])
log_returns = np.diff(np.log(paths_exact[:800]), axis=1).flatten()
x_lr = np.linspace(log_returns.min(), log_returns.max(), 400)
mu_lr  = (mu - 0.5 * sigma**2) * dt_s
std_lr = sigma * np.sqrt(dt_s)
pdf_lr = norm.pdf(x_lr, mu_lr, std_lr)
ax.hist(log_returns * 100, bins=80, density=True, color=GREEN, alpha=0.5,
        edgecolor="none", label="Simulated log-returns")
ax.plot(x_lr * 100, pdf_lr / 100, color=YELLOW, lw=2.5,
        label=r"$N\left((\mu-\frac{\sigma^2}{2})\Delta t,\;\sigma^2\Delta t\right)$")
ax.set_xlabel("Daily Log-Return (%)"); ax.set_ylabel("Density")
ax.set_title("Log-Returns are Gaussian\n(consequence of Ito's Lemma)", color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (1,0) Terminal distribution S_T: simulated histogram vs log-normal PDF
ax = fig.add_subplot(gs[1, 0])
S_T = paths_exact[:, -1]
log_mean = np.log(S0) + (mu - 0.5 * sigma**2) * T
log_std  = sigma * np.sqrt(T)
x_st = np.linspace(S_T.min(), S_T.max(), 500)
pdf_st = lognorm.pdf(x_st, s=log_std, scale=np.exp(log_mean))
ax.hist(S_T, bins=60, density=True, color=BLUE, alpha=0.5, edgecolor="none",
        label=f"Simulated $S_T$  (n={N_PATHS:,})")
ax.plot(x_st, pdf_st, color=YELLOW, lw=2.5,
        label=r"Log-Normal PDF")
ax.axvline(S_T.mean(), color=WHITE,  lw=1.5, linestyle="--",
           label=f"Sim. mean = {S_T.mean():.2f}")
ax.axvline(S0 * np.exp(mu * T), color=YELLOW, lw=1.5, linestyle=":",
           label=f"Theo. mean = {S0*np.exp(mu*T):.2f}")
ax.set_xlabel("$S_T$"); ax.set_ylabel("Density")
ax.set_title(r"Terminal Distribution of $S_T$" "\n"
             r"$\ln S_T \sim N(\ln S_0 + (\mu-\frac{\sigma^2}{2})T,\;\sigma^2 T)$",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (1,1) Euler error vs n_steps for a single path realisation
ax = fig.add_subplot(gs[1, 1])
n_steps_range = [4, 8, 16, 32, 64, 128, 252]
rng_err = np.random.default_rng(SEED + 10)
N_ERR   = 1000
mae_euler = []
for ns in n_steps_range:
    dt_e = T / ns
    sqdt_e = np.sqrt(dt_e)
    # Use same fine-grid Brownian — subsample from 252
    ratio = max(1, 252 // ns)
    Z_e   = rng_err.standard_normal((N_ERR, ns))
    # Exact using same Z
    S_ex_t = S0 * np.exp(
        np.cumsum((mu - 0.5*sigma**2)*dt_e + sigma*sqdt_e*Z_e, axis=1)
    )[:, -1]
    S_eu_t = np.full(N_ERR, S0)
    for i in range(ns):
        S_eu_t = S_eu_t * (1 + mu*dt_e + sigma*sqdt_e*Z_e[:, i])
        S_eu_t = np.maximum(S_eu_t, 1e-12)
    mae_euler.append(np.mean(np.abs(S_eu_t - S_ex_t)))

ax.loglog(T / np.array(n_steps_range), mae_euler, "o-", color=ORANGE, lw=2, ms=7,
          label="Euler MAE")
ref_e = (T / np.array(n_steps_range))**0.5 * (mae_euler[0] / (T/n_steps_range[0])**0.5)
ax.loglog(T / np.array(n_steps_range), ref_e, "--", color=ORANGE, alpha=0.5,
          label=r"$O(\Delta t^{0.5})$ reference")
ax.set_xlabel(r"Step size $\Delta t$"); ax.set_ylabel("MAE($S_T$)")
ax.set_title("Euler Strong Error vs Step Size\n(single-path paired comparison)",
             color=WHITE, fontsize=9)
ax.legend(fontsize=8); ax.grid(True); watermark(ax)

# (1,2) Summary table: theoretical vs simulated moments
ax = fig.add_subplot(gs[1, 2])
ax.axis("off")
E_theo  = S0 * np.exp(mu * T)
V_theo  = S0**2 * np.exp(2*mu*T) * (np.exp(sigma**2 * T) - 1)
Sd_theo = np.sqrt(V_theo)
E_sim   = S_T.mean()
Sd_sim  = S_T.std()
med_theo = np.exp(log_mean)                         # median of log-normal
med_sim  = np.median(S_T)
skew_theo = (np.exp(sigma**2*T) + 2) * np.sqrt(np.exp(sigma**2*T) - 1)

rows = [
    ["Statistic",              "Theoretical",      "Simulated"],
    ["Mean E[S_T]",            f"{E_theo:.4f}",    f"{E_sim:.4f}"],
    ["Std Dev",                f"{Sd_theo:.4f}",   f"{Sd_sim:.4f}"],
    ["Median",                 f"{med_theo:.4f}",  f"{med_sim:.4f}"],
    ["Geom. drift",            f"{mu-sigma**2/2:.4f}", "—"],
    ["Ito correction sigma^2/2", f"{sigma**2/2:.4f}", "—"],
    ["Skewness (theoretical)", f"{skew_theo:.4f}", "—"],
    ["Paths / Steps",          f"{N_PATHS:,}",     f"{n_steps_stat:,}"],
]
colors_row = [[PANEL]*3] + [[DARK if i % 2 == 0 else PANEL]*3
                             for i in range(len(rows)-1)]
tbl = ax.table(cellText=rows, cellLoc="center", loc="center",
               cellColours=colors_row)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1.0, 1.6)
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor(GREY)
    cell.set_text_props(color=WHITE if r == 0 else WHITE)
    if r == 0:
        cell.set_text_props(weight="bold", color=YELLOW)
ax.set_title("Moment Verification Summary", color=WHITE, fontsize=9, pad=8)

plt.tight_layout()
path3 = os.path.join(OUT_DIR, "m08_03_path_statistics.png")
plt.savefig(path3, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {path3}  ({time.perf_counter()-t0:.1f}s)")

# ===========================================================================
# Summary
# ===========================================================================
print()
print("=" * 65)
print("  MODULE 08 COMPLETE -- 3 figures saved")
print("  Key Concepts:")
print("  [1] GBM SDE:  dS = mu*S*dt + sigma*S*dW")
print("  [2] Exact:    S_t = S0*exp[(mu-sigma^2/2)*t + sigma*W_t]")
print("  [3] Euler-Maruyama:  strong order 0.5 | weak order 1.0")
print("  [4] Milstein = Exact for GBM (strong order 1.0)")
print(f"  [5] Ito correction: geom. drift = {mu:.2%} - {sigma**2/2:.4f} = {mu-sigma**2/2:.4f}")
print(f"  [6] Terminal distribution: Log-Normal,  E[S_T] = {S0*np.exp(mu*T):.4f}")
print("=" * 65)
