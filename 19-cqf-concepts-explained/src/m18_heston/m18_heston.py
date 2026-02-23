#!/usr/bin/env python3
"""
M18 — Heston Stochastic Volatility Model
=========================================
Module 4 of 9 | CQF Concepts Explained

Theory
------
The Heston (1993) model extends GBM by letting volatility follow
a mean-reverting square-root (CIR) process:

    dS_t = mu * S_t * dt + sqrt(v_t) * S_t * dW_S
    dv_t = kappa*(theta - v_t)*dt + xi*sqrt(v_t)*dW_v

    Corr(dW_S, dW_v) = rho * dt

Parameters:
    v0    : initial variance (spot vol^2)
    kappa : mean-reversion speed of variance
    theta : long-run variance (long-run vol = sqrt(theta))
    xi    : vol-of-vol (volatility of variance)
    rho   : correlation (typically negative for equities: leverage effect)

Feller condition (ensures v_t > 0 a.s.):
    2*kappa*theta > xi^2

Semi-analytic pricing (Heston, 1993):
    Uses characteristic function of log(S_T):
    phi(u) = exp(C(u,T)*theta + D(u,T)*v0 + i*u*ln(F))

    where C and D satisfy Riccati ODEs, and the call price is:
    C = S*P1 - K*e^{-rT}*P2

    with P1, P2 obtained via Fourier inversion of phi.

Key advantages over local vol:
    - Generates realistic forward smiles (smile is sticky)
    - Negative rho produces put skew naturally
    - Mean-reverting vol is more realistic than deterministic paths

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
from scipy.optimize import minimize
from scipy.integrate import quad

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
# Heston characteristic function (Gatheral formulation — numerically stable)
# ---------------------------------------------------------------------------
def heston_cf(u, S, v0, kappa, theta, xi, rho, r, T):
    """
    Heston characteristic function phi(u) using Gatheral's form
    (avoids discontinuities in complex logarithm).
    """
    i   = 1j
    lnS = np.log(S)
    d   = np.sqrt((rho*xi*i*u - kappa)**2 + xi**2*(i*u + u**2))
    g   = (kappa - rho*xi*i*u - d) / (kappa - rho*xi*i*u + d)
    exp_dT = np.exp(-d*T)

    C = r*i*u*T + (kappa*theta/xi**2)*(
        (kappa - rho*xi*i*u - d)*T - 2*np.log((1 - g*exp_dT)/(1 - g))
    )
    D = ((kappa - rho*xi*i*u - d)/xi**2) * ((1 - exp_dT)/(1 - g*exp_dT))

    return np.exp(C + D*v0 + i*u*lnS)

# ---------------------------------------------------------------------------
# Heston call price via Fourier inversion (Lewis 2001 — single integral)
# ---------------------------------------------------------------------------
def _heston_P(j, S, K, T, r, v0, kappa, theta, xi, rho):
    """
    Heston (1993) risk-neutral probabilities P1 and P2.
    j=1: uses u=0.5, b=kappa-rho*xi
    j=2: uses u=-0.5, b=kappa
    """
    lnS = np.log(S)
    lnK = np.log(K)
    u_j = 0.5 if j == 1 else -0.5
    b_j = (kappa - rho*xi) if j == 1 else kappa

    def integrand(phi):
        i   = 1j
        d   = np.sqrt((rho*xi*i*phi - b_j)**2
                      - xi**2*(2*u_j*i*phi - phi**2))
        g   = (b_j - rho*xi*i*phi + d) / (b_j - rho*xi*i*phi - d)
        exp_dT = np.exp(d*T)
        # Avoid division by zero
        denom = 1 - g*exp_dT
        if abs(denom) < 1e-14:
            return 0.0
        C = (r*i*phi*T
             + kappa*theta/xi**2 * (
                 (b_j - rho*xi*i*phi + d)*T
                 - 2*np.log((1 - g*exp_dT)/(1 - g))
             ))
        D = (b_j - rho*xi*i*phi + d)/xi**2 * (1 - exp_dT)/denom
        f = np.exp(C + D*v0 + i*phi*lnS)
        return np.real(np.exp(-i*phi*lnK) * f / (i*phi))

    val, _ = quad(integrand, 1e-6, 200, limit=300,
                  epsabs=1e-7, epsrel=1e-7)
    return 0.5 + val/np.pi

def heston_call(S, K, T, r, v0, kappa, theta, xi, rho,
                n_integration=128, u_max=100.0):
    """
    Heston (1993) European call: C = S*P1 - K*e^{-rT}*P2
    """
    P1 = _heston_P(1, S, K, T, r, v0, kappa, theta, xi, rho)
    P2 = _heston_P(2, S, K, T, r, v0, kappa, theta, xi, rho)
    call = S*P1 - K*np.exp(-r*T)*P2
    return max(call, max(S - K*np.exp(-r*T), 0.0))

# ---------------------------------------------------------------------------
# BS helpers
# ---------------------------------------------------------------------------
def bs_call(S, K, T, r, sigma):
    T = max(T, 1e-10); sigma = max(sigma, 1e-10)
    d1 = (np.log(S/K) + (r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def bs_vega(S, K, T, r, sigma):
    T = max(T, 1e-10)
    d1 = (np.log(S/K) + (r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    return S*norm.pdf(d1)*np.sqrt(T)

def implied_vol_from_price(S, K, T, r, C_mkt, sigma0=0.25):
    sigma = sigma0
    for _ in range(100):
        p = bs_call(S, K, T, r, sigma)
        v = bs_vega(S, K, T, r, sigma)
        if v < 1e-12: break
        sigma -= (p - C_mkt)/v
        sigma  = np.clip(sigma, 1e-4, 5.0)
        if abs(bs_call(S, K, T, r, sigma) - C_mkt) < 1e-9: break
    return sigma

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
S0    = 100.0
r     = 0.05
v0    = 0.04      # initial variance  (vol = 20%)
kappa = 2.0       # mean-reversion speed
theta = 0.04      # long-run variance (vol = 20%)
xi    = 0.30      # vol-of-vol
rho   = -0.70     # leverage correlation

feller = 2*kappa*theta / xi**2
print(f"[M18] Heston: kappa={kappa}, theta={theta}, xi={xi}, rho={rho}, v0={v0}")
print(f"      Feller condition 2*kappa*theta/xi^2 = {feller:.4f}  "
      f"({'satisfied' if feller > 1 else 'VIOLATED'})")
print(f"      Initial vol = {np.sqrt(v0):.2%}  |  LR vol = {np.sqrt(theta):.2%}")


# ===========================================================================
# FIGURE 1 — Stochastic vol paths: S and v simultaneously
# ===========================================================================
print("[M18] Figure 1: Heston path simulation ...")
t0 = time.perf_counter()

N_PATHS = 1000
N_STEPS = 252
T_sim   = 2.0
dt      = T_sim / N_STEPS
t_grid  = np.linspace(0, T_sim, N_STEPS+1)
rng     = np.random.default_rng(42)

# Correlated Brownian increments
Z1 = rng.standard_normal((N_PATHS, N_STEPS))
Z2 = rng.standard_normal((N_PATHS, N_STEPS))
ZS = Z1
Zv = rho*Z1 + np.sqrt(1-rho**2)*Z2

# Euler-Maruyama with full truncation for variance (Andersen, 2008)
S_h = np.zeros((N_PATHS, N_STEPS+1)); S_h[:, 0] = S0
v_h = np.zeros((N_PATHS, N_STEPS+1)); v_h[:, 0] = v0

for i in range(N_STEPS):
    v_pos = np.maximum(v_h[:, i], 0)
    sv    = np.sqrt(v_pos)
    v_h[:, i+1] = v_h[:, i] + kappa*(theta - v_pos)*dt + xi*sv*np.sqrt(dt)*Zv[:, i]
    S_h[:, i+1] = S_h[:, i] * np.exp((r - 0.5*v_pos)*dt + sv*np.sqrt(dt)*ZS[:, i])

v_h = np.maximum(v_h, 0)  # full truncation

fig, axes = plt.subplots(2, 2, figsize=(14, 8), facecolor=DARK)
fig.suptitle(
    "M18 — Heston Stochastic Volatility: Path Simulation\n"
    f"kappa={kappa}, theta={theta:.2f} (LR vol={np.sqrt(theta):.0%}), "
    f"xi={xi}, rho={rho}, v0={v0} (spot vol={np.sqrt(v0):.0%})",
    color=WHITE, fontsize=10
)

n_show = 50
# (0,0) Price paths
ax = axes[0, 0]
for k in range(n_show):
    ax.plot(t_grid, S_h[k], color=BLUE, alpha=0.10, lw=0.6)
ax.plot(t_grid, S_h.mean(axis=0), color=WHITE, lw=2, label="Mean S_t")
ax.plot(t_grid, np.percentile(S_h, 5,  axis=0), color=GREEN, lw=1.5,
        linestyle="--", label="5th pct")
ax.plot(t_grid, np.percentile(S_h, 95, axis=0), color=GREEN, lw=1.5,
        linestyle="--", label="95th pct")
ax.plot(t_grid, S0*np.exp(r*t_grid), color=YELLOW, lw=1.5,
        linestyle=":", label=r"$S_0 e^{rt}$")
ax.set_xlabel("Time (years)"); ax.set_ylabel("S_t")
ax.set_title("Price Paths", color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (0,1) Variance paths
ax = axes[0, 1]
for k in range(n_show):
    ax.plot(t_grid, np.sqrt(v_h[k])*100, color=ORANGE, alpha=0.10, lw=0.6)
ax.plot(t_grid, np.sqrt(v_h.mean(axis=0))*100, color=WHITE, lw=2,
        label="Mean sqrt(v_t)")
ax.axhline(np.sqrt(theta)*100, color=YELLOW, lw=2, linestyle="--",
           label=f"LR vol = {np.sqrt(theta):.0%}")
ax.axhline(np.sqrt(v0)*100,    color=RED,    lw=1.5, linestyle=":",
           label=f"Spot vol = {np.sqrt(v0):.0%}")
ax.set_xlabel("Time (years)"); ax.set_ylabel("Instantaneous Vol (%)")
ax.set_title("Stochastic Volatility Paths sqrt(v_t)", color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (1,0) Correlation: S_T vs v_T scatter (leverage effect)
ax = axes[1, 0]
S_ret = np.log(S_h[:, -1] / S0)
v_fin = np.sqrt(v_h[:, -1])
ax.scatter(S_ret*100, v_fin*100, s=4, color=BLUE, alpha=0.30,
           rasterized=True)
corr_sv = np.corrcoef(S_ret, v_fin)[0, 1]
ax.set_xlabel("Log-return S_T / S_0 (%)"); ax.set_ylabel("Terminal vol sqrt(v_T) (%)")
ax.set_title(f"Leverage Effect: S_T vs vol_T\n"
             f"Empirical corr = {corr_sv:.3f}  (rho={rho})",
             color=WHITE, fontsize=9)
ax.grid(True); watermark(ax)

# (1,1) Realized vol distribution vs stationary
ax = axes[1, 1]
vol_paths_flat = np.sqrt(v_h[:, N_STEPS//2:]).flatten() * 100
ax.hist(vol_paths_flat, bins=60, density=True, color=ORANGE, alpha=0.6,
        edgecolor="none", label="Simulated sqrt(v_t)")
ax.axvline(np.sqrt(theta)*100, color=YELLOW, lw=2.5, linestyle="--",
           label=f"LR vol = {np.sqrt(theta):.0%}")
ax.axvline(np.sqrt(v0)*100,    color=RED,    lw=2,   linestyle=":",
           label=f"Spot vol = {np.sqrt(v0):.0%}")
ax.set_xlabel("Instantaneous Vol (%)"); ax.set_ylabel("Density")
ax.set_title("Stationary Vol Distribution\n"
             "(variance ~ non-central chi-squared)",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

plt.tight_layout()
p1 = os.path.join(OUT_DIR, "m18_01_heston_paths.png")
plt.savefig(p1, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p1}  ({time.perf_counter()-t0:.1f}s)")


# ===========================================================================
# FIGURE 2 — Implied Vol Smile generated by Heston
# ===========================================================================
print("[M18] Figure 2: Heston implied vol smile ...")
t0 = time.perf_counter()

K_range = np.linspace(75, 125, 21)
T_expiries = [0.25, 0.50, 1.0, 2.0]
colors_t   = [BLUE, GREEN, YELLOW, ORANGE]

# Effect of rho on smile shape (fixed T=1Y)
rho_vals   = [-0.80, -0.50, 0.0, 0.50]
colors_rho = [PURPLE, BLUE, GREEN, ORANGE]

# Effect of xi on smile curvature
xi_vals    = [0.10, 0.20, 0.30, 0.50]
colors_xi  = [BLUE, GREEN, YELLOW, RED]

fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=DARK)
fig.suptitle(
    "M18 — Heston Implied Volatility Smile\n"
    "Semi-analytic pricing via Fourier inversion",
    color=WHITE, fontsize=11
)

# (0) Smile for multiple expiries
ax = axes[0]
for T_exp, col in zip(T_expiries, colors_t):
    ivs = []
    for k in K_range:
        c = heston_call(S0, k, T_exp, r, v0, kappa, theta, xi, rho)
        iv = implied_vol_from_price(S0, k, T_exp, r, c)
        ivs.append(iv)
    ax.plot(K_range, np.array(ivs)*100, color=col, lw=2,
            label=f"T={T_exp}Y")
ax.axvline(S0, color=WHITE, lw=1, linestyle=":", alpha=0.6, label=f"ATM K={S0}")
ax.set_xlabel("Strike K"); ax.set_ylabel("Implied Vol (%)")
ax.set_title("Heston Smile: Multiple Expiries\n"
             "Skew decreases with T (term structure)",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (1) Effect of rho (skew driver)
ax = axes[1]
T_fix = 1.0
for rho_v, col in zip(rho_vals, colors_rho):
    ivs = []
    for k in K_range:
        c = heston_call(S0, k, T_fix, r, v0, kappa, theta, xi, rho_v)
        iv = implied_vol_from_price(S0, k, T_fix, r, c)
        ivs.append(iv)
    ax.plot(K_range, np.array(ivs)*100, color=col, lw=2,
            label=f"rho={rho_v:+.2f}")
ax.axvline(S0, color=WHITE, lw=1, linestyle=":", alpha=0.6)
ax.set_xlabel("Strike K"); ax.set_ylabel("Implied Vol (%)")
ax.set_title(f"Effect of rho on Skew  (T={T_fix}Y)\n"
             "rho < 0: put skew  |  rho > 0: call skew",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (2) Effect of xi (smile curvature / convexity driver)
ax = axes[2]
for xi_v, col in zip(xi_vals, colors_xi):
    ivs = []
    for k in K_range:
        c = heston_call(S0, k, T_fix, r, v0, kappa, theta, xi_v, rho)
        iv = implied_vol_from_price(S0, k, T_fix, r, c)
        ivs.append(iv)
    ax.plot(K_range, np.array(ivs)*100, color=col, lw=2,
            label=f"xi={xi_v:.2f}")
ax.axvline(S0, color=WHITE, lw=1, linestyle=":", alpha=0.6)
ax.set_xlabel("Strike K"); ax.set_ylabel("Implied Vol (%)")
ax.set_title(f"Effect of xi (vol-of-vol) on Curvature  (T={T_fix}Y)\n"
             "Higher xi => more convex smile",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

plt.tight_layout()
p2 = os.path.join(OUT_DIR, "m18_02_heston_smile.png")
plt.savefig(p2, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p2}  ({time.perf_counter()-t0:.1f}s)")


# ===========================================================================
# FIGURE 3 — Parameter sensitivity + Feller + MC vs Analytic verification
# ===========================================================================
print("[M18] Figure 3: Parameter sensitivity and MC verification ...")
t0 = time.perf_counter()

# MC vs analytic call prices for verification
K_test   = np.array([85, 90, 95, 100, 105, 110, 115])
T_verify = 1.0
analytic = np.array([heston_call(S0, k, T_verify, r, v0, kappa, theta, xi, rho)
                     for k in K_test])

# MC prices using end-of-sim paths (T_sim=2Y, use T=1Y slice)
step_1y  = N_STEPS // 2     # index for T=1Y (out of 2Y sim)
S_1y     = S_h[:, step_1y]
mc_prices = np.array([np.exp(-r*T_verify)*np.maximum(S_1y-k,0).mean()
                      for k in K_test])

# Feller condition study
xi_feller = np.linspace(0.05, 0.80, 200)
feller_vals = 2*kappa*theta / xi_feller**2

# kappa sensitivity: smile at different kappa values
kappa_vals  = [0.5, 1.0, 2.0, 5.0]
colors_kap  = [PURPLE, BLUE, GREEN, YELLOW]

fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=DARK)
fig.suptitle("M18 — Heston: Feller Condition, kappa Sensitivity, MC Verification",
             color=WHITE, fontsize=11)

# (0) Feller condition boundary
ax = axes[0]
ax.plot(xi_feller*100, feller_vals, color=YELLOW, lw=2.5,
        label=r"Feller ratio = $2\kappa\theta / \xi^2$")
ax.axhline(1.0, color=RED, lw=2, linestyle="--",
           label="Feller boundary = 1")
ax.fill_between(xi_feller*100, feller_vals, 1,
                where=feller_vals > 1, color=GREEN, alpha=0.12,
                label="Feller satisfied (v_t > 0 a.s.)")
ax.fill_between(xi_feller*100, feller_vals, 1,
                where=feller_vals < 1, color=RED, alpha=0.12,
                label="Feller violated (v_t can hit 0)")
ax.axvline(xi*100, color=WHITE, lw=1.5, linestyle=":",
           label=f"Current xi={xi:.2f}")
ax.set_xlabel("Vol-of-vol xi (%)"); ax.set_ylabel("Feller ratio")
ax.set_title(f"Feller Condition: 2*kappa*theta / xi^2\n"
             f"kappa={kappa}, theta={theta}  |  Current ratio={feller:.3f}",
             color=WHITE, fontsize=9)
ax.legend(fontsize=6); ax.grid(True); watermark(ax)

# (1) kappa effect on smile (speed of mean reversion)
ax = axes[1]
for kap_v, col in zip(kappa_vals, colors_kap):
    ivs = []
    for k in K_range:
        c  = heston_call(S0, k, 1.0, r, v0, kap_v, theta, xi, rho)
        iv = implied_vol_from_price(S0, k, 1.0, r, c)
        ivs.append(iv)
    ax.plot(K_range, np.array(ivs)*100, color=col, lw=2,
            label=f"kappa={kap_v}  (HL={np.log(2)/kap_v:.2f}Y)")
ax.axvline(S0, color=WHITE, lw=1, linestyle=":", alpha=0.6)
ax.set_xlabel("Strike K"); ax.set_ylabel("Implied Vol (%)")
ax.set_title("Effect of kappa on Smile  (T=1Y)\n"
             "Higher kappa => faster reversion => flatter smile",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (2) MC vs analytic verification
ax = axes[2]
ax.plot(K_test, analytic,  "o-", color=YELLOW, lw=2.5, ms=8,
        label="Heston analytic (Fourier)")
ax.plot(K_test, mc_prices, "s-", color=BLUE,   lw=2,   ms=7,
        label=f"Heston MC  (N={N_PATHS:,})")
for k, mc, an in zip(K_test, mc_prices, analytic):
    ax.annotate(f"{abs(mc-an):.3f}", (k, min(mc,an)),
                textcoords="offset points", xytext=(0,-14),
                fontsize=6, color=WHITE, ha="center")
mae = np.abs(mc_prices - analytic).mean()
ax.set_xlabel("Strike K"); ax.set_ylabel("Call Price")
ax.set_title(f"MC vs Analytic Verification\n"
             f"MAE = {mae:.4f}  (numbers = |error| per strike)",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

plt.tight_layout()
p3 = os.path.join(OUT_DIR, "m18_03_heston_verification.png")
plt.savefig(p3, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p3}  ({time.perf_counter()-t0:.1f}s)")

print()
atm_price = heston_call(S0, S0, 1.0, r, v0, kappa, theta, xi, rho)
atm_iv    = implied_vol_from_price(S0, S0, 1.0, r, atm_price)
print("=" * 65)
print("  MODULE 18 COMPLETE -- 3 figures saved")
print("  Key Concepts:")
print("  [1] Heston SDE: dv=kappa*(theta-v)*dt + xi*sqrt(v)*dW_v")
print("  [2] Feller: 2*kappa*theta > xi^2 => v_t > 0 always")
print("  [3] rho < 0: leverage => put skew  |  xi: curvature")
print("  [4] Semi-analytic via characteristic function + Fourier")
print("  [5] kappa higher => faster mean reversion => flatter smile")
print(f"  [6] Feller ratio = {feller:.4f}  ({'OK' if feller>1 else 'VIOLATED'})")
print(f"      ATM Heston price = {atm_price:.6f}  "
      f"|  ATM Heston IV = {atm_iv:.4%}")
print(f"      MC vs Analytic MAE = {mae:.4f}")
print("=" * 65)
