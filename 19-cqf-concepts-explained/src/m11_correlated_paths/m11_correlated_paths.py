#!/usr/bin/env python3
"""
M11 — Correlated Paths via Cholesky Decomposition
==================================================
Module 2 of 9 | CQF Concepts Explained

Theory
------
To simulate n correlated Brownian motions W_1, ..., W_n with
correlation matrix Rho (n x n, symmetric positive definite):

Step 1 — Cholesky factorization:
    Rho = L @ L.T       (L lower triangular)

Step 2 — Generate independent standard normals:
    Z ~ N(0, I_n)       (n independent draws)

Step 3 — Produce correlated increments:
    dW = L @ Z          =>  Cov(dW) = L @ I @ L.T = Rho

Multi-asset GBM under correlated Brownian motions:
    dS_i = mu_i * S_i * dt + sigma_i * S_i * dW_i,  i=1,...,n

Exact step for asset i:
    S_i(t+dt) = S_i(t) * exp[(mu_i - sigma_i^2/2)*dt + sigma_i*sqrt(dt)*(L@Z)_i]

Key properties:
    - Correlation of log-returns: Corr(d lnS_i, d lnS_j) = Rho_ij
    - Diversification benefit: portfolio vol < weighted avg of individual vols
    - Cholesky requires Rho to be positive (semi-)definite

Neareast PD correction (Higham, 2002):
    If Rho is not PD (e.g. from empirical estimates), project to nearest PD
    matrix via eigenvalue clipping: set negative eigenvalues to epsilon > 0.

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
from scipy.stats import pearsonr

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
TEAL   = "#1abc9c"

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
# Helpers
# ---------------------------------------------------------------------------
def nearest_pd(A, eps=1e-8):
    """Project A to nearest positive definite matrix (eigenvalue clipping)."""
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.maximum(eigvals, eps)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T

def safe_cholesky(Rho, eps=1e-8):
    """Cholesky with automatic nearest-PD fallback."""
    try:
        return np.linalg.cholesky(Rho)
    except np.linalg.LinAlgError:
        return np.linalg.cholesky(nearest_pd(Rho, eps))

def simulate_correlated_gbm(S0_vec, mu_vec, sigma_vec, Rho, T, n_steps, n_paths, rng):
    """
    Multi-asset GBM with correlated Brownian motions via Cholesky.
    Returns paths of shape (n_assets, n_paths, n_steps+1).
    """
    n = len(S0_vec)
    L = safe_cholesky(Rho)
    dt = T / n_steps
    sqdt = np.sqrt(dt)

    # Independent standard normals: (n_paths, n_steps, n_assets)
    Z = rng.standard_normal((n_paths, n_steps, n))

    # Correlated increments: apply L to last axis
    dW = (L @ Z.reshape(-1, n).T).T.reshape(n_paths, n_steps, n)

    # Build paths
    paths = np.empty((n, n_paths, n_steps + 1))
    for i in range(n):
        paths[i, :, 0] = S0_vec[i]
        log_drift = (mu_vec[i] - 0.5 * sigma_vec[i]**2) * dt
        log_inc   = log_drift + sigma_vec[i] * sqdt * dW[:, :, i]
        paths[i, :, :] = S0_vec[i] * np.exp(
            np.concatenate([np.zeros((n_paths, 1)), np.cumsum(log_inc, axis=1)], axis=1)
        )
    return paths

def empirical_corr(paths, n_steps):
    """Compute empirical correlation matrix of log-returns."""
    n_assets = paths.shape[0]
    log_rets = np.diff(np.log(paths), axis=2)          # (n_assets, n_paths, n_steps)
    # Stack all path log-returns per asset
    lr_flat = log_rets.reshape(n_assets, -1)            # (n_assets, n_paths*n_steps)
    return np.corrcoef(lr_flat)

# ---------------------------------------------------------------------------
# Parameters — 4-asset portfolio
# ---------------------------------------------------------------------------
ASSETS   = ["Equity", "Bonds", "Commodities", "FX"]
S0_vec   = np.array([100., 100., 100., 100.])
mu_vec   = np.array([0.10, 0.04, 0.06, 0.02])
sigma_vec= np.array([0.20, 0.06, 0.25, 0.10])

# Target correlation matrix
Rho = np.array([
    [ 1.00, -0.30,  0.40,  0.10],
    [-0.30,  1.00, -0.10, -0.05],
    [ 0.40, -0.10,  1.00,  0.20],
    [ 0.10, -0.05,  0.20,  1.00],
])

T       = 1.0
N_STEPS = 252
N_PATHS = 5000
SEED    = 42
dt      = T / N_STEPS
t_grid  = np.linspace(0, T, N_STEPS + 1)

# Cholesky factor
L = safe_cholesky(Rho)

rng = np.random.default_rng(SEED)
paths = simulate_correlated_gbm(S0_vec, mu_vec, sigma_vec, Rho,
                                 T, N_STEPS, N_PATHS, rng)


# ===========================================================================
# FIGURE 1 — Cholesky illustration + Path fan per asset
# ===========================================================================
print("[M11] Figure 1: Cholesky factor and asset paths ...")
t0 = time.perf_counter()

fig = plt.figure(figsize=(16, 9), facecolor=DARK)
fig.suptitle(
    "M11 — Correlated GBM via Cholesky Decomposition\n"
    "4-Asset Portfolio: Equity / Bonds / Commodities / FX",
    color=WHITE, fontsize=11
)
gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.40, wspace=0.35)

colors_a = [BLUE, GREEN, YELLOW, ORANGE]

# Heatmap: target Rho
ax = fig.add_subplot(gs[0, 0])
im = ax.imshow(Rho, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
for i in range(4):
    for j in range(4):
        ax.text(j, i, f"{Rho[i,j]:.2f}", ha="center", va="center",
                color="black", fontsize=9, fontweight="bold")
ax.set_xticks(range(4)); ax.set_xticklabels(ASSETS, fontsize=7, rotation=20)
ax.set_yticks(range(4)); ax.set_yticklabels(ASSETS, fontsize=7)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_title("Target Correlation Matrix Rho", color=WHITE, fontsize=9)
watermark(ax)

# Heatmap: Cholesky L
ax = fig.add_subplot(gs[0, 1])
im2 = ax.imshow(L, cmap="Blues", aspect="auto")
for i in range(4):
    for j in range(4):
        ax.text(j, i, f"{L[i,j]:.3f}", ha="center", va="center",
                color="white" if L[i,j] < L.max()*0.5 else "black",
                fontsize=8)
ax.set_xticks(range(4)); ax.set_xticklabels([f"Z_{k+1}" for k in range(4)], fontsize=7)
ax.set_yticks(range(4)); ax.set_yticklabels([f"dW_{k+1}" for k in range(4)], fontsize=7)
plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
ax.set_title("Cholesky Factor L\nRho = L @ L.T", color=WHITE, fontsize=9)
watermark(ax)

# Heatmap: empirical correlation from simulation
Rho_emp = empirical_corr(paths, N_STEPS)
ax = fig.add_subplot(gs[0, 2])
im3 = ax.imshow(Rho_emp, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
for i in range(4):
    for j in range(4):
        ax.text(j, i, f"{Rho_emp[i,j]:.3f}", ha="center", va="center",
                color="black", fontsize=9, fontweight="bold")
ax.set_xticks(range(4)); ax.set_xticklabels(ASSETS, fontsize=7, rotation=20)
ax.set_yticks(range(4)); ax.set_yticklabels(ASSETS, fontsize=7)
plt.colorbar(im3, ax=ax, fraction=0.046, pad=0.04)
ax.set_title(f"Empirical Correlation\n(N={N_PATHS:,} paths)", color=WHITE, fontsize=9)
watermark(ax)

# Error heatmap: |Rho_target - Rho_empirical|
ax = fig.add_subplot(gs[0, 3])
err = np.abs(Rho - Rho_emp)
im4 = ax.imshow(err, cmap="Reds", vmin=0, vmax=0.05, aspect="auto")
for i in range(4):
    for j in range(4):
        ax.text(j, i, f"{err[i,j]:.4f}", ha="center", va="center",
                color="white", fontsize=8)
ax.set_xticks(range(4)); ax.set_xticklabels(ASSETS, fontsize=7, rotation=20)
ax.set_yticks(range(4)); ax.set_yticklabels(ASSETS, fontsize=7)
plt.colorbar(im4, ax=ax, fraction=0.046, pad=0.04)
ax.set_title("|Target - Empirical| Error\n(should be < 0.02)", color=WHITE, fontsize=9)
watermark(ax)

# Asset path fans (bottom row)
n_show = 40
for idx, (name, col) in enumerate(zip(ASSETS, colors_a)):
    ax = fig.add_subplot(gs[1, idx])
    for k in range(n_show):
        ax.plot(t_grid, paths[idx, k], color=col, alpha=0.12, lw=0.7)
    mean_p = paths[idx].mean(axis=0)
    p5  = np.percentile(paths[idx], 5,  axis=0)
    p95 = np.percentile(paths[idx], 95, axis=0)
    ax.plot(t_grid, mean_p, color=WHITE, lw=2, label="Mean")
    ax.fill_between(t_grid, p5, p95, color=col, alpha=0.10, label="5th-95th")
    theo_mean = S0_vec[idx] * np.exp(mu_vec[idx] * t_grid)
    ax.plot(t_grid, theo_mean, color=YELLOW, lw=1.5, linestyle="--",
            label=f"E[S]  mu={mu_vec[idx]:.0%}")
    ax.set_xlabel("Time (years)"); ax.set_ylabel("Price")
    ax.set_title(f"{name}\nmu={mu_vec[idx]:.0%}, sigma={sigma_vec[idx]:.0%}",
                 color=col, fontsize=9)
    ax.legend(fontsize=6); ax.grid(True); watermark(ax)

plt.tight_layout()
p1 = os.path.join(OUT_DIR, "m11_01_cholesky_paths.png")
plt.savefig(p1, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p1}  ({time.perf_counter()-t0:.1f}s)")


# ===========================================================================
# FIGURE 2 — Scatter plots of log-returns (verify correlations)
# ===========================================================================
print("[M11] Figure 2: Log-return scatter plots ...")
t0 = time.perf_counter()

# Use single-step log-returns across all paths
log_rets_1step = np.log(paths[:, :, 1] / paths[:, :, 0])   # (4, N_PATHS)

pairs = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
pair_labels = [(ASSETS[i], ASSETS[j]) for i, j in pairs]

fig, axes = plt.subplots(2, 3, figsize=(14, 8), facecolor=DARK)
fig.suptitle("M11 — Log-Return Scatter Plots: Verifying Pairwise Correlations",
             color=WHITE, fontsize=11)

for ax, (i, j), (la, lb) in zip(axes.flat, pairs, pair_labels):
    xi = log_rets_1step[i] * 100
    xj = log_rets_1step[j] * 100
    r_emp, _ = pearsonr(xi, xj)
    r_tgt    = Rho[i, j]
    ax.scatter(xi, xj, alpha=0.12, s=4, color=colors_a[i],
               rasterized=True)
    # OLS line
    coef = np.polyfit(xi, xj, 1)
    x_line = np.linspace(xi.min(), xi.max(), 100)
    ax.plot(x_line, np.polyval(coef, x_line), color=YELLOW, lw=1.5,
            label=f"OLS slope={coef[0]:.3f}")
    ax.set_xlabel(f"{la} log-ret (%)", fontsize=8)
    ax.set_ylabel(f"{lb} log-ret (%)", fontsize=8)
    ax.set_title(f"{la} vs {lb}\n"
                 f"Target rho={r_tgt:.2f}  |  Empirical rho={r_emp:.4f}",
                 color=WHITE, fontsize=8)
    ax.legend(fontsize=7); ax.grid(True); watermark(ax)

plt.tight_layout()
p2 = os.path.join(OUT_DIR, "m11_02_scatter_correlations.png")
plt.savefig(p2, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p2}  ({time.perf_counter()-t0:.1f}s)")


# ===========================================================================
# FIGURE 3 — Diversification: Portfolio Vol vs Correlation
# ===========================================================================
print("[M11] Figure 3: Diversification benefit ...")
t0 = time.perf_counter()

# 2-asset case: vary correlation rho_12 and show portfolio vol
w = np.array([0.5, 0.5])
s1, s2 = 0.20, 0.30
rho_range = np.linspace(-1, 1, 300)
port_vol = np.sqrt(w[0]**2*s1**2 + w[1]**2*s2**2
                   + 2*w[0]*w[1]*s1*s2*rho_range)
weighted_avg = w[0]*s1 + w[1]*s2

# Multi-asset: simulate portfolio terminal wealth for various correlation levels
rho_levels = [-0.8, -0.5, 0.0, 0.5, 0.8]
colors_rho  = [PURPLE, BLUE, GREEN, YELLOW, RED]
n_assets_port = 4
T_port   = 1.0
ns_port  = 252
N_P_port = 3000
w_port   = np.ones(n_assets_port) / n_assets_port   # equal weight

fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=DARK)
fig.suptitle("M11 — Diversification Benefit and Correlation Effect",
             color=WHITE, fontsize=11)

# (0) Analytical: 2-asset portfolio vol vs rho
ax = axes[0]
ax.plot(rho_range, port_vol * 100, color=BLUE, lw=2.5,
        label="Portfolio vol (50/50)")
ax.axhline(weighted_avg * 100, color=RED, lw=2, linestyle="--",
           label=f"Weighted avg vol = {weighted_avg:.0%}")
ax.axhline(abs(s1-s2)*0.5*100 if False else min(s1,s2)*100,
           color=GREY, lw=1, linestyle=":", alpha=0.6)
ax.fill_between(rho_range, port_vol*100, weighted_avg*100,
                where=port_vol*100 < weighted_avg*100,
                color=GREEN, alpha=0.15, label="Diversification benefit")
ax.scatter([0], [np.sqrt(0.5**2*(s1**2+s2**2))*100], color=YELLOW,
           s=80, zorder=5, label=f"rho=0: {np.sqrt(0.5**2*(s1**2+s2**2))*100:.2f}%")
ax.set_xlabel("Correlation rho_12"); ax.set_ylabel("Portfolio Vol (%)")
ax.set_title("2-Asset Portfolio Vol vs Correlation\n"
             f"sigma_1={s1:.0%}, sigma_2={s2:.0%}, w=(50%,50%)",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (1) Terminal wealth distribution per correlation level (4 equal-weight assets)
ax = axes[1]
mu_p = np.array([0.08]*n_assets_port)
sg_p = np.array([0.20]*n_assets_port)
S0_p = np.ones(n_assets_port) * 100.0

for rho_lv, col in zip(rho_levels, colors_rho):
    Rho_lv = np.full((n_assets_port, n_assets_port), rho_lv)
    np.fill_diagonal(Rho_lv, 1.0)
    try:
        L_lv = safe_cholesky(Rho_lv)
    except Exception:
        continue
    rng_p = np.random.default_rng(SEED + int(rho_lv*100) + 200)
    pths  = simulate_correlated_gbm(S0_p, mu_p, sg_p, Rho_lv,
                                     T_port, ns_port, N_P_port, rng_p)
    # Equal-weight portfolio value
    port_val = (w_port[:, None] * pths[:, :, -1]).sum(axis=0)
    port_ret  = port_val / 100 - 1          # return
    ax.hist(port_ret*100, bins=60, density=True, color=col,
            alpha=0.40, edgecolor="none",
            label=f"rho={rho_lv:+.1f}  std={port_ret.std()*100:.2f}%")

ax.set_xlabel("Portfolio Return (%)"); ax.set_ylabel("Density")
ax.set_title("Terminal Portfolio Return Distribution\n"
             f"4 Assets, Equal Weight, sigma=20% each, T={T_port}Y",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (2) Empirical convergence: corr estimate error vs n_paths
ax = axes[2]
n_paths_range = [100, 250, 500, 1000, 2000, 5000]
mae_corr = []
for np_ in n_paths_range:
    rng_cv = np.random.default_rng(SEED + 99)
    pths_cv = simulate_correlated_gbm(S0_vec, mu_vec, sigma_vec, Rho,
                                       T, N_STEPS, np_, rng_cv)
    Rho_cv = empirical_corr(pths_cv, N_STEPS)
    # MAE of off-diagonal elements
    mask_od = ~np.eye(4, dtype=bool)
    mae_corr.append(np.abs(Rho_cv[mask_od] - Rho[mask_od]).mean())

ax.loglog(n_paths_range, mae_corr, "o-", color=ORANGE, lw=2, ms=7,
          label="MAE of empirical correlation")
ref = np.array(n_paths_range, dtype=float)**(-0.5)
ref *= mae_corr[0] / ref[0]
ax.loglog(n_paths_range, ref, "--", color=GREY, lw=1.5,
          label=r"$O(N^{-1/2})$ reference")
ax.set_xlabel("Number of Paths N"); ax.set_ylabel("MAE (empirical vs target corr)")
ax.set_title("Correlation Estimation Error vs N\n"
             r"Converges at $O(N^{-1/2})$ as expected",
             color=WHITE, fontsize=9)
ax.legend(fontsize=8); ax.grid(True); watermark(ax)

plt.tight_layout()
p3 = os.path.join(OUT_DIR, "m11_03_diversification.png")
plt.savefig(p3, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p3}  ({time.perf_counter()-t0:.1f}s)")

print()
print("=" * 65)
print("  MODULE 11 COMPLETE -- 3 figures saved")
print("  Key Concepts:")
print("  [1] Cholesky: Rho = L @ L.T  =>  dW = L @ Z  =>  Cov(dW) = Rho")
print("  [2] Multi-asset GBM: exact log-normal steps per asset")
print("  [3] Empirical corr converges to target at O(N^-0.5)")
print("  [4] Diversification: port vol < weighted avg vol  (rho < 1)")
print(f"  [5] Target Rho verified: max error = {np.abs(Rho - empirical_corr(paths, N_STEPS)).max():.4f}")
print("  [6] Nearest-PD projection handles non-PD empirical matrices")
print("=" * 65)
