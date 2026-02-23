#!/usr/bin/env python3
"""
M17 — Local Volatility: Dupire's Formula
=========================================
Module 4 of 9 | CQF Concepts Explained

Theory
------
Black-Scholes assumes constant sigma. To match a full volatility
surface IV(K,T), Dupire (1994) derived the unique local volatility
function sigma_loc(S,t) such that the SDE:

    dS_t = r*S_t*dt + sigma_loc(S_t, t)*S_t*dW_t

reprices all European options consistently with the market surface.

Dupire's formula (in terms of market call prices C(K,T)):

    sigma_loc^2(K,T) = 2 * [dC/dT + r*K*dC/dK] / [K^2 * d2C/dK2]

Equivalently, in terms of implied vol sigma_I(K,T):

    sigma_loc^2 = sigma_I^2 * N / D

where:
    N = 1 + 2*T*sigma_I*(d(sigma_I)/dT + r*K*(d(sigma_I)/dK))
    D = (1 - K*d1*d(sigma_I)/dK / sigma_I * sqrt(T))^2
        + T*K^2*(d2(sigma_I)/dK2 - d1*(d(sigma_I)/dK)^2*sqrt(T)/sigma_I)

Key insight:
    Local vol is the conditional expectation of instantaneous variance:
    sigma_loc^2(K,T) = E[sigma_t^2 | S_T = K]

    This is why local vol perfectly fits the market surface but
    produces incorrect forward smile dynamics (smiles flatten forward).

Calibration workflow:
    1. Observe market prices C_mkt(K_i, T_j)
    2. Extract IV surface sigma_I(K, T) via root-finding
    3. Apply Dupire's formula using finite differences on the surface
    4. Simulate paths using the calibrated sigma_loc(S, t)
    5. Verify: MC prices match market prices

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
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import gaussian_filter

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
# BS engine
# ---------------------------------------------------------------------------
def bs_call(S, K, T, r, sigma):
    T = np.maximum(T, 1e-10)
    sigma = np.maximum(sigma, 1e-10)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def bs_vega(S, K, T, r, sigma):
    T = np.maximum(T, 1e-10)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)

def implied_vol(S, K, T, r, C_mkt, sigma0=0.25):
    sigma = sigma0
    for _ in range(100):
        p = bs_call(S, K, T, r, sigma)
        v = bs_vega(S, K, T, r, sigma)
        if v < 1e-12: break
        sigma -= (p - C_mkt) / v
        sigma = np.clip(sigma, 1e-4, 5.0)
        if abs(bs_call(S, K, T, r, sigma) - C_mkt) < 1e-9:
            break
    return sigma

# ---------------------------------------------------------------------------
# Build synthetic IV surface (equity skew)
# ---------------------------------------------------------------------------
S0 = 100.0
r  = 0.05

K_nodes = np.linspace(70, 130, 25)
T_nodes = np.array([0.08, 0.17, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0])

def synth_iv(K, T, S=S0):
    """Synthetic equity skew IV surface."""
    lm   = np.log(K / (S * np.exp(r*T)))    # log-moneyness
    atm  = 0.20 - 0.015*T                   # term structure: slight contango
    skew = 0.10 * np.exp(-0.5*T)            # skew decays with T
    curv = 0.04 * np.exp(-0.3*T)            # curvature decays
    return np.maximum(atm - skew*lm + curv*lm**2, 0.02)

IV_grid = np.array([[synth_iv(k, t) for k in K_nodes] for t in T_nodes])

# Spline interpolation over the surface
spline_iv = RectBivariateSpline(T_nodes, K_nodes, IV_grid, kx=3, ky=3)

print(f"[M17] IV surface built: {len(T_nodes)} expiries x {len(K_nodes)} strikes")
print(f"      ATM IV range: {IV_grid[:, len(K_nodes)//2].min():.2%} - "
      f"{IV_grid[:, len(K_nodes)//2].max():.2%}")


# ---------------------------------------------------------------------------
# Dupire local vol from IV surface via finite differences
# ---------------------------------------------------------------------------
def dupire_local_vol(K_arr, T_arr, S=S0):
    """
    Compute local vol surface using Dupire's formula via finite differences
    on the call price surface C(K,T).
    Returns sigma_loc(T, K) on the grid.
    """
    eps_K = 0.5   # finite difference step in K
    eps_T = 1/252 # finite difference step in T (1 trading day)

    lv = np.zeros((len(T_arr), len(K_arr)))

    for i, T in enumerate(T_arr):
        for j, K in enumerate(K_arr):
            # Call prices for finite differences
            iv_c   = float(spline_iv(T,        K,       grid=False))
            iv_ku  = float(spline_iv(T,        K+eps_K, grid=False))
            iv_kd  = float(spline_iv(T,        K-eps_K, grid=False))
            iv_tu  = float(spline_iv(T+eps_T,  K,       grid=False))

            C_c  = bs_call(S, K,        T,        r, max(iv_c,  0.01))
            C_ku = bs_call(S, K+eps_K,  T,        r, max(iv_ku, 0.01))
            C_kd = bs_call(S, K-eps_K,  T,        r, max(iv_kd, 0.01))
            C_tu = bs_call(S, K,        T+eps_T,  r, max(iv_tu, 0.01))

            # Derivatives
            dC_dT  = (C_tu - C_c)  / eps_T
            dC_dK  = (C_ku - C_kd) / (2*eps_K)
            d2C_dK2= (C_ku - 2*C_c + C_kd) / eps_K**2

            numerator   = 2*(dC_dT + r*K*dC_dK)
            denominator = K**2 * d2C_dK2

            if denominator > 1e-10 and numerator > 0:
                lv[i, j] = np.sqrt(numerator / denominator)
            else:
                lv[i, j] = iv_c   # fallback to IV

    # Smooth to remove finite-difference noise
    lv = gaussian_filter(lv, sigma=0.8)
    return np.clip(lv, 0.01, 0.80)

K_lv = np.linspace(75, 125, 20)
T_lv = np.array([0.25, 0.50, 0.75, 1.0, 1.5, 2.0])

print("[M17] Computing Dupire local vol surface ...")
t0_lv = time.perf_counter()
LV_grid = dupire_local_vol(K_lv, T_lv)
print(f"      Done in {time.perf_counter()-t0_lv:.1f}s")


# ===========================================================================
# FIGURE 1 — IV surface vs Local Vol surface (side by side)
# ===========================================================================
print("[M17] Figure 1: IV vs Local Vol surface ...")
t0 = time.perf_counter()

K_plot = np.linspace(75, 125, 30)
T_plot = np.linspace(0.10, 2.0, 30)
KK, TT = np.meshgrid(K_plot, T_plot)
IV_plot = np.vectorize(lambda k, t: float(spline_iv(t, k, grid=False)))(KK, TT)
IV_plot = np.clip(IV_plot, 0.01, 0.80)

# Interpolate LV onto same grid
from scipy.interpolate import RegularGridInterpolator
lv_interp = RegularGridInterpolator(
    (T_lv, K_lv), LV_grid, method="linear",
    bounds_error=False, fill_value=None
)
pts = np.column_stack([TT.ravel(), KK.ravel()])
LV_plot = lv_interp(pts).reshape(KK.shape)
LV_plot = np.clip(LV_plot, 0.01, 0.80)

fig = plt.figure(figsize=(14, 6), facecolor=DARK)
fig.suptitle(
    "M17 — Implied Volatility vs Dupire Local Volatility Surface\n"
    f"S={S0}, r={r:.0%}  |  Equity skew model",
    color=WHITE, fontsize=11
)

for idx, (ZZ, title, cmap) in enumerate([
    (IV_plot*100, "Implied Volatility Surface  sigma_I(K,T)", "RdYlGn_r"),
    (LV_plot*100, "Dupire Local Volatility  sigma_loc(K,T)", "RdYlBu_r"),
]):
    ax = fig.add_subplot(1, 2, idx+1, projection="3d")
    ax.set_facecolor(PANEL)
    surf = ax.plot_surface(KK, TT, ZZ, cmap=cmap, alpha=0.88, linewidth=0)
    ax.set_xlabel("Strike K", labelpad=5, color=WHITE)
    ax.set_ylabel("Expiry T", labelpad=5, color=WHITE)
    ax.set_zlabel("Vol (%)",  labelpad=5, color=WHITE)
    ax.set_title(title, color=WHITE, fontsize=9, pad=8)
    ax.tick_params(colors=WHITE, labelsize=6)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor(GREY)
    ax.yaxis.pane.set_edgecolor(GREY)
    ax.zaxis.pane.set_edgecolor(GREY)
    fig.colorbar(surf, ax=ax, shrink=0.45, aspect=10, pad=0.1)

plt.tight_layout()
p1 = os.path.join(OUT_DIR, "m17_01_iv_vs_localvol.png")
plt.savefig(p1, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p1}  ({time.perf_counter()-t0:.1f}s)")


# ===========================================================================
# FIGURE 2 — Cross-sections: IV vs LV for fixed T and fixed K
# ===========================================================================
print("[M17] Figure 2: IV vs LV cross-sections ...")
t0 = time.perf_counter()

fig, axes = plt.subplots(2, 3, figsize=(16, 9), facecolor=DARK)
fig.suptitle(
    "M17 — Implied Vol vs Local Vol: Cross-Sections by Expiry and Strike",
    color=WHITE, fontsize=11
)

colors_t = [BLUE, GREEN, YELLOW, ORANGE, PURPLE, RED]
T_cross  = [0.25, 0.50, 1.0, 2.0]
colors_tc = [BLUE, GREEN, YELLOW, ORANGE]

# Top row: smile slices for fixed T
for idx, (t_c, col) in enumerate(zip(T_cross, colors_tc)):
    ax = axes[0, idx % 3] if idx < 3 else axes[1, 0]
    iv_slice = np.array([float(spline_iv(t_c, k, grid=False)) for k in K_plot])
    iv_slice = np.clip(iv_slice, 0.01, 0.80)
    # LV at this T
    lv_slice = lv_interp(np.column_stack([
        np.full(len(K_lv), t_c), K_lv
    ]))
    ax.plot(K_plot, iv_slice*100, color=col, lw=2.5,
            label=f"IV  T={t_c}Y")
    ax.plot(K_lv, lv_slice*100, color=col, lw=2, linestyle="--",
            label=f"LV  T={t_c}Y")
    ax.axvline(S0, color=WHITE, lw=1, linestyle=":", alpha=0.5)
    ax.set_xlabel("Strike K"); ax.set_ylabel("Vol (%)")
    ax.set_title(f"Smile Slice at T={t_c}Y\n"
                 "LV flatter than IV (Derman rule: LV ~ 2*IV - ATM_IV)",
                 color=WHITE, fontsize=8)
    ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# Bottom-left: term structure for ATM
ax = axes[1, 0]
ax.clear()
T_ts = np.linspace(0.10, 2.0, 100)
iv_ts = np.array([float(spline_iv(t, S0, grid=False)) for t in T_ts])
lv_ts = lv_interp(np.column_stack([T_lv, np.full(len(T_lv), S0)]))
ax.plot(T_ts, iv_ts*100,   color=YELLOW, lw=2.5, label="ATM IV term structure")
ax.plot(T_lv, lv_ts*100,   color=ORANGE, lw=2, linestyle="--",
        label="ATM LV term structure")
ax.set_xlabel("Expiry T"); ax.set_ylabel("Vol (%)")
ax.set_title("ATM Term Structure: IV vs LV\n"
             "LV flatter in T: forward vol information",
             color=WHITE, fontsize=8)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# Bottom-mid: IV - LV difference heatmap
ax = axes[1, 1]
diff_surf = IV_plot - LV_plot
im = ax.imshow(diff_surf*100, aspect="auto", origin="lower",
               extent=[K_plot[0], K_plot[-1], T_plot[0], T_plot[-1]],
               cmap="RdBu_r", vmin=-5, vmax=5)
plt.colorbar(im, ax=ax, label="IV - LV (%)")
ax.set_xlabel("Strike K"); ax.set_ylabel("Expiry T")
ax.set_title("IV - LV Difference (%)\n"
             "IV > LV for OTM options (Derman's rule)",
             color=WHITE, fontsize=8)
ax.axvline(S0, color=WHITE, lw=1.5, linestyle="--", alpha=0.8)
watermark(ax)

# Bottom-right: Derman's approximation: LV ~ 2*IV_K - IV_ATM
ax = axes[1, 2]
T_derm = 1.0
iv_derm    = np.array([float(spline_iv(T_derm, k, grid=False)) for k in K_plot])
iv_atm_d   = float(spline_iv(T_derm, S0, grid=False))
lv_derman  = 2*iv_derm - iv_atm_d      # Derman approximation
lv_exact   = lv_interp(np.column_stack([
    np.full(len(K_lv), T_derm), K_lv
]))
ax.plot(K_plot, iv_derm*100,   color=BLUE,   lw=2, label="IV(K)")
ax.plot(K_plot, lv_derman*100, color=YELLOW, lw=2, linestyle="--",
        label="Derman approx: 2*IV(K)-IV(ATM)")
ax.plot(K_lv,   lv_exact*100,  color=GREEN,  lw=2, linestyle=":",
        label="Dupire LV (exact)")
ax.axvline(S0, color=WHITE, lw=1, linestyle=":", alpha=0.5)
ax.set_xlabel("Strike K"); ax.set_ylabel("Vol (%)  T=1Y")
ax.set_title("Derman's Rule: LV ~ 2*IV(K) - IV(ATM)\n"
             "Valid near ATM for small skew",
             color=WHITE, fontsize=8)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

plt.tight_layout()
p2 = os.path.join(OUT_DIR, "m17_02_crosssections.png")
plt.savefig(p2, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p2}  ({time.perf_counter()-t0:.1f}s)")


# ===========================================================================
# FIGURE 3 — MC simulation under local vol + repricing verification
# ===========================================================================
print("[M17] Figure 3: MC local vol simulation and repricing ...")
t0 = time.perf_counter()

N_PATHS = 3000
N_STEPS = 252
T_sim   = 1.0
dt      = T_sim / N_STEPS
rng     = np.random.default_rng(42)
Z       = rng.standard_normal((N_PATHS, N_STEPS))

# Euler-Maruyama under local vol
S_lv = np.full(N_PATHS, S0)
t_path = 0.0
for i in range(N_STEPS):
    t_path += dt
    # Local vol at current S values (clamp K to grid range)
    K_curr = np.clip(S_lv, K_lv[0], K_lv[-1])
    T_curr = np.clip(t_path, T_lv[0], T_lv[-1])
    pts_i  = np.column_stack([np.full(N_PATHS, T_curr), K_curr])
    lv_i   = np.clip(lv_interp(pts_i), 0.01, 0.80)
    S_lv   = S_lv * (1 + r*dt + lv_i * np.sqrt(dt) * Z[:, i])
    S_lv   = np.maximum(S_lv, 0.01)

# Reprice calls from MC vs BS with market IV
K_test  = np.array([85, 90, 95, 100, 105, 110, 115])
mc_call = np.array([np.exp(-r*T_sim)*np.maximum(S_lv-k,0).mean() for k in K_test])
iv_test = np.array([float(spline_iv(T_sim, k, grid=False)) for k in K_test])
bs_call_test = np.array([bs_call(S0, k, T_sim, r, iv) for k, iv in zip(K_test, iv_test)])

# GBM paths for comparison (flat vol = ATM IV)
S_gbm = S0 * np.exp((r - 0.5*0.20**2)*T_sim + 0.20*np.sqrt(T_sim)*rng.standard_normal(N_PATHS))
gbm_call = np.array([np.exp(-r*T_sim)*np.maximum(S_gbm-k,0).mean() for k in K_test])

fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=DARK)
fig.suptitle(
    f"M17 — Local Vol MC Simulation: Repricing Verification  (N={N_PATHS:,} paths, T={T_sim}Y)",
    color=WHITE, fontsize=11
)

# (0) Terminal distribution: LV vs GBM
ax = axes[0]
ax.hist(S_lv,  bins=60, density=True, color=BLUE,   alpha=0.55,
        edgecolor="none", label=f"Local Vol MC  (std={S_lv.std():.2f})")
ax.hist(S_gbm, bins=60, density=True, color=GREEN,  alpha=0.55,
        edgecolor="none", label=f"GBM flat 20%  (std={S_gbm.std():.2f})")
ax.axvline(S_lv.mean(),  color=BLUE,   lw=2, linestyle="--",
           label=f"LV mean = {S_lv.mean():.2f}")
ax.axvline(S_gbm.mean(), color=GREEN,  lw=2, linestyle="--",
           label=f"GBM mean = {S_gbm.mean():.2f}")
ax.set_xlabel("S_T"); ax.set_ylabel("Density")
ax.set_title("Terminal Distribution\nLV vs GBM (flat sigma)",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (1) Repricing: MC call prices vs BS market prices
ax = axes[1]
ax.plot(K_test, bs_call_test, "o-", color=YELLOW, lw=2.5, ms=8,
        label="BS market price (IV smile)")
ax.plot(K_test, mc_call,      "s-", color=BLUE,   lw=2, ms=7,
        label="Local Vol MC price")
ax.plot(K_test, gbm_call,     "^-", color=GREEN,  lw=2, ms=7,
        label="GBM flat 20% MC price")
for k, mc, bs in zip(K_test, mc_call, bs_call_test):
    ax.annotate(f"{abs(mc-bs):.3f}", (k, min(mc, bs)),
                textcoords="offset points", xytext=(0, -14),
                fontsize=6, color=WHITE, ha="center")
ax.set_xlabel("Strike K"); ax.set_ylabel("Call Price")
ax.set_title("Repricing: LV MC vs BS Market\n"
             "(numbers show |error|)",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (2) Relative error %
ax = axes[2]
rel_lv  = (mc_call - bs_call_test) / bs_call_test * 100
rel_gbm = (gbm_call - bs_call_test) / bs_call_test * 100
x = np.arange(len(K_test))
ax.bar(x-0.2, rel_lv,  width=0.35, color=BLUE,  alpha=0.75, label="Local Vol MC error %")
ax.bar(x+0.2, rel_gbm, width=0.35, color=GREEN, alpha=0.75, label="GBM flat error %")
ax.axhline(0, color=WHITE, lw=1, linestyle="--")
ax.set_xticks(x); ax.set_xticklabels([str(k) for k in K_test])
ax.set_xlabel("Strike K"); ax.set_ylabel("Relative error (%)")
ax.set_title("Repricing Error (%)\nLV much smaller for skewed strikes",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True, axis="y"); watermark(ax)

plt.tight_layout()
p3 = os.path.join(OUT_DIR, "m17_03_mc_repricing.png")
plt.savefig(p3, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p3}  ({time.perf_counter()-t0:.1f}s)")

print()
mae_lv  = np.abs(mc_call - bs_call_test).mean()
mae_gbm = np.abs(gbm_call - bs_call_test).mean()
print("=" * 65)
print("  MODULE 17 COMPLETE -- 3 figures saved")
print("  Key Concepts:")
print("  [1] Dupire: sigma_loc^2 = 2*(dC/dT + r*K*dC/dK) / (K^2*d2C/dK2)")
print("  [2] LV matches all European market prices by construction")
print("  [3] LV < IV for OTM options (Derman: LV ~ 2*IV(K)-IV(ATM))")
print("  [4] LV forward smiles flatten (limitation vs stochastic vol)")
print("  [5] Spline interpolation required for smooth finite differences")
print(f"  [6] MC repricing: LV MAE = {mae_lv:.4f}  |  GBM MAE = {mae_gbm:.4f}")
print("=" * 65)
