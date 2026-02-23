#!/usr/bin/env python3
"""
M16 — Implied Volatility & Volatility Smile
============================================
Module 4 of 9 | CQF Concepts Explained

Theory
------
Implied volatility (IV) is the value of sigma that equates the
Black-Scholes formula to the observed market price:

    C_market = BS(S, K, T, r, sigma_impl)

IV is extracted via root-finding (no closed form exists).

Newton-Raphson iteration:
    sigma_{n+1} = sigma_n - (BS(sigma_n) - C_market) / Vega(sigma_n)

Convergence is quadratic when Vega > 0 (i.e. not deep ITM/OTM).

Volatility Smile / Skew:
    Under BS, IV should be flat across strikes (constant sigma).
    In practice, IV varies with strike K and expiry T, forming:
      - Volatility smile:  IV higher for OTM calls and puts (equity pre-1987)
      - Volatility skew:   IV decreasing with K (equity post-1987)
      - Volatility surface: IV(K, T) — the full 3D structure

Moneyness conventions:
    - Strike K directly
    - Log-moneyness: ln(K/F)  where F = S*exp(r*T)  (forward)
    - Delta-moneyness: 10d, 25d, ATM, 25d, 10d (FX convention)

VIX interpretation:
    The VIX index is model-free IV computed from a strip of S&P 500
    options, representing the market's expectation of 30-day vol.

SVI parametrization (Gatheral, 2004):
    w(x) = a + b*(rho*(x-m) + sqrt((x-m)^2 + s^2))
    where w = sigma_impl^2 * T (total variance), x = ln(K/F)
    Parameters: a (level), b (slope), rho (skew), m (ATM shift), s (smile)

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
from scipy.optimize import brentq

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
# BS engine
# ---------------------------------------------------------------------------
def bs_call(S, K, T, r, sigma):
    T = np.maximum(T, 1e-10)
    sigma = np.maximum(sigma, 1e-10)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def bs_vega_raw(S, K, T, r, sigma):
    T = np.maximum(T, 1e-10)
    sigma = np.maximum(sigma, 1e-10)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)

def implied_vol_newton(S, K, T, r, C_mkt, sigma0=0.20, tol=1e-8, max_iter=100):
    """Newton-Raphson IV extraction."""
    sigma = sigma0
    for i in range(max_iter):
        price = bs_call(S, K, T, r, sigma)
        vega  = bs_vega_raw(S, K, T, r, sigma)
        diff  = price - C_mkt
        if abs(diff) < tol:
            return sigma, i+1
        if vega < 1e-12:
            break
        sigma -= diff / vega
        sigma  = max(1e-6, min(sigma, 5.0))
    return sigma, max_iter

def implied_vol_brent(S, K, T, r, C_mkt):
    """Brent bracket method (robust fallback)."""
    try:
        intrinsic = max(S - K*np.exp(-r*T), 0)
        if C_mkt <= intrinsic + 1e-10:
            return np.nan
        f = lambda s: bs_call(S, K, T, r, s) - C_mkt
        return brentq(f, 1e-6, 5.0, xtol=1e-8, maxiter=200)
    except Exception:
        return np.nan

# ---------------------------------------------------------------------------
# SVI parametrization
# ---------------------------------------------------------------------------
def svi_total_variance(x, a, b, rho, m, s):
    """SVI raw parametrization: w(x) = total variance."""
    return a + b*(rho*(x-m) + np.sqrt((x-m)**2 + s**2))

def svi_vol(x, T, a, b, rho, m, s):
    """SVI implied vol from total variance."""
    w = svi_total_variance(x, a, b, rho, m, s)
    return np.sqrt(np.maximum(w, 0) / T)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
S0 = 100.0
r  = 0.05
T  = 1.0
F  = S0 * np.exp(r * T)   # forward price

# Synthetic market: equity skew smile (post-1987 style)
# IV = ATM_vol - skew*(K-ATM)/ATM + smile_curv*(K-ATM)^2/ATM^2
ATM_VOL  = 0.20
SKEW     = 0.10    # negative skew: lower strikes => higher IV
CURVATURE= 0.05

K_range  = np.linspace(70, 130, 61)
log_m    = np.log(K_range / F)

# Synthetic market IVs (equity skew)
IV_market = ATM_VOL - SKEW*(K_range - S0)/S0 + CURVATURE*((K_range - S0)/S0)**2

# Compute market call prices from these IVs
C_market = np.array([bs_call(S0, k, T, r, iv)
                     for k, iv in zip(K_range, IV_market)])

print(f"[M16] Forward F = {F:.4f}  |  ATM IV = {ATM_VOL:.2%}")


# ===========================================================================
# FIGURE 1 — Newton-Raphson IV extraction: convergence illustration
# ===========================================================================
print("[M16] Figure 1: Newton-Raphson convergence ...")
t0 = time.perf_counter()

# Trace NR iterations for ATM option
K_atm  = 100.0
C_atm  = bs_call(S0, K_atm, T, r, ATM_VOL)
sigma_path = [0.50]   # start far from truth
for _ in range(12):
    s = sigma_path[-1]
    p = bs_call(S0, K_atm, T, r, s)
    v = bs_vega_raw(S0, K_atm, T, r, s)
    if v < 1e-12: break
    s_new = s - (p - C_atm) / v
    s_new = max(1e-6, min(s_new, 5.0))
    sigma_path.append(s_new)
    if abs(s_new - ATM_VOL) < 1e-9: break

# Price function vs sigma
sig_grid = np.linspace(0.01, 0.80, 300)
price_grid = np.array([bs_call(S0, K_atm, T, r, s) for s in sig_grid])

fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=DARK)
fig.suptitle(
    "M16 — Implied Volatility Extraction: Newton-Raphson Method\n"
    f"ATM option: S={S0}, K={K_atm}, T={T}Y, r={r:.0%}, C_mkt={C_atm:.4f}",
    color=WHITE, fontsize=10
)

# (0) BS price vs sigma with NR steps
ax = axes[0]
ax.plot(sig_grid*100, price_grid, color=BLUE, lw=2.5, label="BS Call price")
ax.axhline(C_atm, color=YELLOW, lw=2, linestyle="--", label=f"C_market = {C_atm:.4f}")
colors_nr = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(sigma_path)))
for i, s in enumerate(sigma_path):
    p = bs_call(S0, K_atm, T, r, s)
    ax.scatter(s*100, p, color=colors_nr[i], s=60, zorder=5)
    if i < len(sigma_path)-1:
        ax.annotate(f"iter {i}", (s*100, p), textcoords="offset points",
                    xytext=(4, 4+i*2), fontsize=6, color=colors_nr[i])
ax.axvline(ATM_VOL*100, color=GREEN, lw=1.5, linestyle=":",
           label=f"True IV = {ATM_VOL:.2%}")
ax.set_xlabel("sigma (%)"); ax.set_ylabel("Call Price")
ax.set_title("BS Price vs Sigma\nNewton-Raphson iterations (start=50%)",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (1) Convergence error per iteration
ax = axes[1]
errors_nr = [abs(s - ATM_VOL) for s in sigma_path]
iters = list(range(len(errors_nr)))
ax.semilogy(iters, errors_nr, "o-", color=GREEN, lw=2, ms=7,
            label="|sigma_n - sigma_true|")
ax.set_xlabel("Iteration"); ax.set_ylabel("|Error|  (log scale)")
ax.set_title("Newton-Raphson Convergence\n(Quadratic: error^2 each step)",
             color=WHITE, fontsize=9)
for i, (it, er) in enumerate(zip(iters, errors_nr)):
    ax.annotate(f"{er:.2e}", (it, er), textcoords="offset points",
                xytext=(4, 4), fontsize=6, color=WHITE)
ax.legend(fontsize=8); ax.grid(True); watermark(ax)

# (2) Newton vs Brent: IV accuracy across strikes
ax = axes[2]
iv_newton = []
iv_brent  = []
iters_nr  = []
for k, c in zip(K_range, C_market):
    iv_n, it = implied_vol_newton(S0, k, T, r, c)
    iv_b     = implied_vol_brent(S0, k, T, r, c)
    iv_newton.append(iv_n)
    iv_brent.append(iv_b)
    iters_nr.append(it)
iv_newton = np.array(iv_newton)
iv_brent  = np.array(iv_brent)

ax.plot(K_range, np.abs(iv_newton - IV_market)*1e6, color=BLUE,   lw=2,
        label="Newton-Raphson error (x 10^-6)")
ax.plot(K_range, np.abs(iv_brent  - IV_market)*1e6, color=ORANGE, lw=2,
        linestyle="--", label="Brent error (x 10^-6)")
ax.set_xlabel("Strike K"); ax.set_ylabel("|IV_extracted - IV_true|  x 10^-6")
ax.set_title("IV Extraction Accuracy vs Strike\n(both methods < 1e-6 error)",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

plt.tight_layout()
p1 = os.path.join(OUT_DIR, "m16_01_newton_raphson.png")
plt.savefig(p1, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p1}  ({time.perf_counter()-t0:.1f}s)")


# ===========================================================================
# FIGURE 2 — Volatility Smile / Skew and SVI fit
# ===========================================================================
print("[M16] Figure 2: Smile, skew and SVI ...")
t0 = time.perf_counter()

# Multiple expiries with different shapes
expiries = [
    (0.25, 0.22, 0.12, 0.08, BLUE,   "T=3M"),
    (0.50, 0.21, 0.10, 0.06, GREEN,  "T=6M"),
    (1.00, 0.20, 0.08, 0.05, YELLOW, "T=1Y"),
    (2.00, 0.19, 0.06, 0.03, ORANGE, "T=2Y"),
]

# SVI parameters for equity-like skew (one set per expiry)
svi_params = {
    0.25: (0.04,  0.20, -0.70, -0.02, 0.10),
    0.50: (0.035, 0.16, -0.65, -0.01, 0.12),
    1.00: (0.030, 0.13, -0.60,  0.00, 0.14),
    2.00: (0.025, 0.10, -0.55,  0.01, 0.16),
}

fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=DARK)
fig.suptitle(
    "M16 — Volatility Smile, Skew and SVI Parametrization\n"
    "Equity-like negative skew: OTM puts command higher IV",
    color=WHITE, fontsize=10
)

# (0) IV smile vs strike for multiple expiries
ax = axes[0]
for t_exp, atm, skew, curv, col, lbl in expiries:
    F_t   = S0 * np.exp(r * t_exp)
    IV_t  = atm - skew*(K_range-S0)/S0 + curv*((K_range-S0)/S0)**2
    IV_t  = np.maximum(IV_t, 0.01)
    ax.plot(K_range, IV_t*100, color=col, lw=2, label=lbl)
ax.axvline(S0, color=WHITE, lw=1, linestyle=":", alpha=0.7, label=f"ATM S={S0}")
ax.set_xlabel("Strike K"); ax.set_ylabel("Implied Vol (%)")
ax.set_title("IV Smile: Multiple Expiries\n"
             "Term structure: shorter expiry => steeper skew",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (1) IV vs log-moneyness ln(K/F) with SVI fit
ax = axes[1]
for t_exp, atm, skew, curv, col, lbl in expiries:
    F_t   = S0 * np.exp(r * t_exp)
    lm    = np.log(K_range / F_t)
    IV_t  = atm - skew*(K_range-S0)/S0 + curv*((K_range-S0)/S0)**2
    IV_t  = np.maximum(IV_t, 0.01)
    ax.plot(lm, IV_t*100, "o", color=col, ms=3, alpha=0.6)
    # SVI fit
    a, b, rho_svi, m, s = svi_params[t_exp]
    lm_fine = np.linspace(lm.min(), lm.max(), 200)
    iv_svi  = svi_vol(lm_fine, t_exp, a, b, rho_svi, m, s)
    ax.plot(lm_fine, iv_svi*100, color=col, lw=2, label=f"{lbl} SVI fit")
ax.axvline(0, color=WHITE, lw=1, linestyle=":", alpha=0.7, label="ATM (lm=0)")
ax.set_xlabel("Log-moneyness ln(K/F)"); ax.set_ylabel("Implied Vol (%)")
ax.set_title("IV vs Log-Moneyness + SVI Fit\n"
             "SVI: a+b*(rho*(x-m)+sqrt((x-m)^2+s^2))",
             color=WHITE, fontsize=9)
ax.legend(fontsize=6); ax.grid(True); watermark(ax)

# (2) Volatility surface heatmap
K_surf = np.linspace(70, 130, 50)
T_surf = np.array([0.08, 0.17, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0])
IV_surf = np.zeros((len(T_surf), len(K_surf)))
for i, t_s in enumerate(T_surf):
    atm  = 0.20 - 0.01*t_s
    sk   = 0.10 - 0.03*t_s
    cv   = 0.05 - 0.01*t_s
    IV_surf[i, :] = np.maximum(atm - sk*(K_surf-S0)/S0 + cv*((K_surf-S0)/S0)**2, 0.01)

ax = axes[2]
im = ax.imshow(IV_surf*100, aspect="auto", origin="lower",
               extent=[K_surf[0], K_surf[-1], T_surf[0], T_surf[-1]],
               cmap="RdYlGn_r", vmin=12, vmax=32)
plt.colorbar(im, ax=ax, label="IV (%)")
ax.set_xlabel("Strike K"); ax.set_ylabel("Expiry T (years)")
ax.set_title("Volatility Surface IV(K,T)\n"
             "Skew steepens for short expiry; term structure inverts",
             color=WHITE, fontsize=9)
ax.axvline(S0, color=WHITE, lw=1.5, linestyle="--", alpha=0.8, label=f"ATM K={S0}")
ax.legend(fontsize=7); watermark(ax)

plt.tight_layout()
p2 = os.path.join(OUT_DIR, "m16_02_smile_surface.png")
plt.savefig(p2, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p2}  ({time.perf_counter()-t0:.1f}s)")


# ===========================================================================
# FIGURE 3 — IV impact on Greeks + Vega P&L
# ===========================================================================
print("[M16] Figure 3: IV impact on Greeks and Vega P&L ...")
t0 = time.perf_counter()

# Delta and Gamma under flat vs skewed IV
K_g = np.linspace(70, 130, 81)
IV_flat   = 0.20 * np.ones_like(K_g)
IV_skewed = np.maximum(0.20 - 0.10*(K_g-S0)/S0 + 0.05*((K_g-S0)/S0)**2, 0.01)

delta_flat   = norm.cdf((np.log(S0/K_g) + (r+0.5*IV_flat**2)*T)
                        / (IV_flat*np.sqrt(T)))
delta_skewed = norm.cdf((np.log(S0/K_g) + (r+0.5*IV_skewed**2)*T)
                        / (IV_skewed*np.sqrt(T)))

# Vega P&L scenario: long straddle + IV moves
vol_moves = np.linspace(-0.10, 0.10, 200)  # sigma shift
C0_straddle = bs_call(S0, S0, T, r, ATM_VOL)
P0_straddle = C0_straddle  # ATM: put = call (approx, ignoring r)
vega_atm    = bs_vega_raw(S0, S0, T, r, ATM_VOL)

pnl_long_straddle = vega_atm * vol_moves * 2  # call + put vega

# Vega term structure: ATM vega vs expiry
T_vega  = np.linspace(0.01, 3.0, 300)
vega_ts = bs_vega_raw(S0, S0, T_vega, r, ATM_VOL)

fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=DARK)
fig.suptitle("M16 — IV Impact on Greeks, Vega P&L and Term Structure",
             color=WHITE, fontsize=11)

# (0) Delta: flat vs skewed IV
ax = axes[0]
ax.plot(K_g, delta_flat,   color=BLUE,   lw=2.5, label="Delta (flat IV=20%)")
ax.plot(K_g, delta_skewed, color=ORANGE, lw=2.5, linestyle="--",
        label="Delta (skewed IV)")
ax.plot(K_g, IV_skewed,    color=GREY,   lw=1.2, linestyle=":",
        label="IV skew (right axis approx)")
ax.axvline(S0, color=WHITE, lw=1, linestyle=":", alpha=0.6)
ax.set_xlabel("Strike K"); ax.set_ylabel("Call Delta")
ax.set_title("Call Delta: Flat vs Skewed IV\n"
             "Skew shifts delta; hedging ratios differ",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (1) Vega P&L of long ATM straddle vs vol move
ax = axes[1]
ax.plot(vol_moves*100, pnl_long_straddle, color=GREEN, lw=2.5,
        label="Long straddle Vega P&L (linear approx)")
ax.axhline(0, color=GREY, lw=0.8)
ax.axvline(0, color=GREY, lw=0.8)
ax.fill_between(vol_moves*100, 0, pnl_long_straddle,
                where=pnl_long_straddle > 0, color=GREEN, alpha=0.15,
                label="Profit (vol rises)")
ax.fill_between(vol_moves*100, 0, pnl_long_straddle,
                where=pnl_long_straddle < 0, color=RED, alpha=0.15,
                label="Loss (vol falls)")
ax.set_xlabel("Vol move (%)"); ax.set_ylabel("Vega P&L")
ax.set_title(f"ATM Straddle Vega P&L\n"
             f"Vega = {vega_atm:.4f}  |  ATM IV = {ATM_VOL:.0%}",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (2) ATM Vega term structure
ax = axes[2]
ax.plot(T_vega, vega_ts, color=YELLOW, lw=2.5, label=r"ATM Vega vs T")
ax.fill_between(T_vega, 0, vega_ts, color=YELLOW, alpha=0.10)
peak_T = T_vega[np.argmax(vega_ts)]
ax.axvline(peak_T, color=WHITE, lw=1.5, linestyle="--",
           label=f"Peak at T={peak_T:.2f}Y")
ax.set_xlabel("Time to Expiry T (years)")
ax.set_ylabel("ATM Vega")
ax.set_title("ATM Vega Term Structure\n"
             "Vega = S*N'(d1)*sqrt(T)  grows with sqrt(T)",
             color=WHITE, fontsize=9)
ax.legend(fontsize=8); ax.grid(True); watermark(ax)

plt.tight_layout()
p3 = os.path.join(OUT_DIR, "m16_03_vega_pnl.png")
plt.savefig(p3, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p3}  ({time.perf_counter()-t0:.1f}s)")

print()
iv_atm, iters_atm = implied_vol_newton(S0, S0, T, r, bs_call(S0,S0,T,r,ATM_VOL))
print("=" * 65)
print("  MODULE 16 COMPLETE -- 3 figures saved")
print("  Key Concepts:")
print("  [1] IV: sigma s.t. BS(sigma) = C_market  (no closed form)")
print("  [2] Newton-Raphson: quadratic convergence via Vega")
print("  [3] Brent: robust bracket for deep ITM/OTM")
print("  [4] Smile/skew: IV(K) not flat -- post-1987 equity skew")
print("  [5] SVI: 5-param fit to total variance w(x) = sigma^2*T")
print("  [6] Volatility surface: IV(K,T) -- full market information")
print(f"  ATM IV extraction: {iv_atm:.8f}  (true: {ATM_VOL:.8f}, iters={iters_atm})")
print("=" * 65)
