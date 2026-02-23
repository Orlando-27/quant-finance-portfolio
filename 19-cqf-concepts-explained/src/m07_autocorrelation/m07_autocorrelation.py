#!/usr/bin/env python3
# =============================================================================
# MODULE 07: SERIAL CORRELATION AND AUTOCORRELATION
# =============================================================================
# Author      : Jose Orlando Bobadilla Fuentes | CQF | MSc AI
# Institution : Colombian Pension Fund — Investment Division
# Project     : 19 - CQF Concepts Explained
# Output      : outputs/figures/m07_*.png
# Run         : python src/m07_autocorrelation/m07_autocorrelation.py
# =============================================================================
"""
SERIAL CORRELATION AND AUTOCORRELATION
========================================

THEORETICAL FOUNDATIONS
------------------------

1. AUTOCOVARIANCE AND AUTOCORRELATION FUNCTION (ACF)
   For a weakly stationary process {X_t}, the autocovariance at lag h is:
       gamma(h) = Cov(X_t, X_{t+h}) = E[(X_t - mu)(X_{t+h} - mu)]

   The autocorrelation function (ACF):
       rho(h) = gamma(h) / gamma(0) = Cov(X_t, X_{t+h}) / Var(X_t)

   Properties: rho(0) = 1, rho(h) = rho(-h), |rho(h)| <= 1

   Sample ACF (empirical estimate):
       rho_hat(h) = [sum_{t=1}^{T-h} (x_t - x_bar)(x_{t+h} - x_bar)] /
                    [sum_{t=1}^{T} (x_t - x_bar)^2]

2. PARTIAL AUTOCORRELATION FUNCTION (PACF)
   The PACF at lag h measures the correlation between X_t and X_{t+h}
   after removing the linear dependence on X_{t+1}, ..., X_{t+h-1}:
       phi_{hh} = Corr(X_t - P_{h-1}X_t, X_{t+h} - P_{h-1}X_{t+h})

   where P_{h-1} denotes projection onto {X_{t+1},...,X_{t+h-1}}.

   Computed via the Yule-Walker equations:
       [rho(0)  rho(1) ... rho(h-1)] [phi_1]   [rho(1)]
       [rho(1)  rho(0) ... rho(h-2)] [phi_2] = [rho(2)]
       [  ...                      ] [ ... ]   [  ... ]
       [rho(h-1)...    rho(0)      ] [phi_h]   [rho(h)]

3. MODEL IDENTIFICATION (Box-Jenkins)
   ACF and PACF patterns identify ARMA(p,q) structure:

   | Process  | ACF                        | PACF                       |
   |----------|----------------------------|----------------------------|
   | AR(p)    | Decays exponentially/sinusoidal | Cuts off after lag p  |
   | MA(q)    | Cuts off after lag q        | Decays exponentially      |
   | ARMA(p,q)| Decays after lag q         | Decays after lag p        |
   | White Noise| All near zero            | All near zero             |

4. LJUNG-BOX Q-STATISTIC
   Tests H0: rho(1) = rho(2) = ... = rho(m) = 0 (no autocorrelation).
       Q(m) = T(T+2) * sum_{h=1}^{m} rho_hat(h)^2 / (T-h) ~ chi^2(m)

   Box-Pierce (simpler): Q_BP(m) = T * sum_{h=1}^{m} rho_hat(h)^2

5. FINANCIAL IMPLICATIONS
   - Log returns: ACF ~ 0 (consistent with weak-form EMH)
   - Absolute/squared returns: significant ACF (volatility clustering)
   - Order flow, volume: strong positive autocorrelation
   - Bid-ask bounce: negative lag-1 autocorrelation in tick data

References:
    Box, G.E.P. and Jenkins, G.M. (1970). Time Series Analysis:
        Forecasting and Control. Holden-Day.
    Ljung, G.M. and Box, G.E.P. (1978). On a measure of lack of fit
        in time series models. Biometrika, 65(2), 297-303.
"""

import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

DARK   = "#0a0a0a"; DARK2  = "#111111"; DARK3  = "#1a1a1a"
ACCENT = "#00d4ff"; GREEN  = "#00ff88"; RED    = "#ff4444"
YELLOW = "#ffd700"; ORANGE = "#ff8c00"; WHITE  = "#e0e0e0"; MUTED  = "#888888"
WATERMARK = "Jose O. Bobadilla | CQF | MSc AI"

OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "outputs", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": DARK, "axes.facecolor": DARK2,
    "axes.edgecolor": MUTED, "axes.labelcolor": WHITE,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": WHITE, "grid.color": "#2a2a2a",
    "grid.linestyle": "--", "grid.alpha": 0.4,
    "font.family": "monospace", "font.size": 9,
    "legend.facecolor": DARK3, "legend.edgecolor": MUTED,
})

def watermark(ax):
    ax.text(0.99, 0.01, WATERMARK, transform=ax.transAxes,
            fontsize=7, color=MUTED, ha="right", va="bottom", alpha=0.6)

def plot_acf_pacf(ax_acf, ax_pacf, series, max_lags=30, title="", color=ACCENT):
    """Plot ACF and PACF with confidence bands."""
    n = len(series)
    conf = 1.96 / np.sqrt(n)

    acf_vals  = acf(series,  nlags=max_lags, fft=True)
    pacf_vals = pacf(series, nlags=max_lags, method="ywm")
    lags = np.arange(0, max_lags + 1)

    # ACF
    ax_acf.bar(lags[1:], acf_vals[1:], color=color, alpha=0.7, width=0.6)
    ax_acf.axhline( conf, color=YELLOW, lw=1.5, linestyle="--", label=f"95% CI ±{conf:.3f}")
    ax_acf.axhline(-conf, color=YELLOW, lw=1.5, linestyle="--")
    ax_acf.axhline(0, color=WHITE, lw=0.5)
    ax_acf.set_xlabel("Lag"); ax_acf.set_ylabel("ACF")
    ax_acf.set_title(f"ACF — {title}", color=WHITE)
    ax_acf.legend(fontsize=7); ax_acf.grid(True); watermark(ax_acf)

    # PACF
    ax_pacf.bar(lags[1:], pacf_vals[1:], color=color, alpha=0.7, width=0.6)
    ax_pacf.axhline( conf, color=YELLOW, lw=1.5, linestyle="--")
    ax_pacf.axhline(-conf, color=YELLOW, lw=1.5, linestyle="--")
    ax_pacf.axhline(0, color=WHITE, lw=0.5)
    ax_pacf.set_xlabel("Lag"); ax_pacf.set_ylabel("PACF")
    ax_pacf.set_title(f"PACF — {title}", color=WHITE)
    ax_pacf.grid(True); watermark(ax_pacf)

# =============================================================================
# DATA GENERATION
# =============================================================================
np.random.seed(42)
N = 1000

eps = np.random.normal(0, 1, N + 10)

# White noise
wn = eps[:N]

# AR(2): X_t = 0.6*X_{t-1} - 0.3*X_{t-2} + eps
ar2 = np.zeros(N)
for t in range(2, N):
    ar2[t] = 0.6*ar2[t-1] - 0.3*ar2[t-2] + eps[t]

# MA(3): X_t = eps_t + 0.5*eps_{t-1} - 0.3*eps_{t-2} + 0.2*eps_{t-3}
ma3 = np.zeros(N)
for t in range(3, N):
    ma3[t] = eps[t] + 0.5*eps[t-1] - 0.3*eps[t-2] + 0.2*eps[t-3]

# ARMA(1,1): X_t = 0.7*X_{t-1} + eps_t + 0.4*eps_{t-1}
arma11 = np.zeros(N)
for t in range(1, N):
    arma11[t] = 0.7*arma11[t-1] + eps[t] + 0.4*eps[t-1]

# GARCH-like financial returns
def garch_r(n):
    r  = np.zeros(n); s2 = np.zeros(n)
    z  = np.random.randn(n)
    s2[0] = 1e-4
    for t in range(1, n):
        s2[t] = 1e-6 + 0.09*r[t-1]**2 + 0.90*s2[t-1]
        r[t]  = np.sqrt(s2[t]) * z[t]
    return r

fin_returns = garch_r(N)

# =============================================================================
# LJUNG-BOX TESTS
# =============================================================================
print("=" * 65)
print("  MODULE 07: SERIAL CORRELATION AND AUTOCORRELATION")
print("=" * 65)

series_dict = {
    "White Noise":    wn,
    "AR(2)":          ar2,
    "MA(3)":          ma3,
    "ARMA(1,1)":      arma11,
    "Fin. Returns":   fin_returns,
    "Abs. Returns":   np.abs(fin_returns),
    "Sqd. Returns":   fin_returns**2,
}

print(f"\n  {'Series':<18} {'LB Q(10)':>10} {'p-value':>10} {'Reject H0?':>12}")
print("  " + "-" * 52)
for name, s in series_dict.items():
    lb = acorr_ljungbox(s, lags=[10], return_df=True)
    q_stat = lb["lb_stat"].values[0]
    p_val  = lb["lb_pvalue"].values[0]
    reject = "YES (autocorr)" if p_val < 0.05 else "NO"
    print(f"  {name:<18} {q_stat:>10.3f} {p_val:>10.4f} {reject:>12}")

# =============================================================================
# FIGURE 1: ACF/PACF for Four Processes
# =============================================================================
fig, axes = plt.subplots(4, 2, figsize=(16, 18))
fig.patch.set_facecolor(DARK)
fig.suptitle("M07 — ACF & PACF: Model Identification",
             color=WHITE, fontsize=13, fontweight="bold")

processes = [
    (wn,     "White Noise",  MUTED),
    (ar2,    "AR(2): phi1=0.6, phi2=-0.3", GREEN),
    (ma3,    "MA(3): theta1=0.5, theta2=-0.3, theta3=0.2", ACCENT),
    (arma11, "ARMA(1,1): phi=0.7, theta=0.4", ORANGE),
]
for i, (s, title, col) in enumerate(processes):
    plot_acf_pacf(axes[i, 0], axes[i, 1], s, max_lags=25, title=title, color=col)

plt.tight_layout()
path = os.path.join(OUT_DIR, "m07_01_acf_pacf_processes.png")
plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"\n[OK] {path}")

# =============================================================================
# FIGURE 2: Financial Returns ACF — Stylized Facts
# =============================================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.patch.set_facecolor(DARK)
fig.suptitle("M07 — Financial Returns: ACF Stylized Facts",
             color=WHITE, fontsize=13, fontweight="bold")

# Returns time series
ax = axes[0, 0]
ax.plot(fin_returns * 100, color=ACCENT, lw=0.6, alpha=0.8)
ax.set_title("Daily Returns (%)", color=WHITE)
ax.set_xlabel("Day"); ax.set_ylabel("Return (%)")
ax.grid(True); watermark(ax)

# ACF of returns
ax = axes[0, 1]
n = len(fin_returns)
conf = 1.96 / np.sqrt(n)
acf_r = acf(fin_returns, nlags=30, fft=True)
ax.bar(range(1, 31), acf_r[1:], color=ACCENT, alpha=0.7, width=0.6)
ax.axhline( conf, color=YELLOW, lw=1.5, linestyle="--", label="95% CI")
ax.axhline(-conf, color=YELLOW, lw=1.5, linestyle="--")
ax.axhline(0, color=WHITE, lw=0.5)
ax.set_title("ACF of Returns\n(near zero — EMH)", color=WHITE)
ax.set_xlabel("Lag"); ax.set_ylabel("ACF")
ax.legend(fontsize=8); ax.grid(True); watermark(ax)

# ACF of absolute returns
ax = axes[0, 2]
acf_abs = acf(np.abs(fin_returns), nlags=30, fft=True)
ax.bar(range(1, 31), acf_abs[1:], color=ORANGE, alpha=0.7, width=0.6)
ax.axhline( conf, color=YELLOW, lw=1.5, linestyle="--", label="95% CI")
ax.axhline(-conf, color=YELLOW, lw=1.5, linestyle="--")
ax.axhline(0, color=WHITE, lw=0.5)
ax.set_title("ACF of |Returns|\n(strong — vol clustering)", color=WHITE)
ax.set_xlabel("Lag"); ax.set_ylabel("ACF")
ax.legend(fontsize=8); ax.grid(True); watermark(ax)

# ACF of squared returns
ax = axes[1, 0]
acf_sq = acf(fin_returns**2, nlags=30, fft=True)
ax.bar(range(1, 31), acf_sq[1:], color=RED, alpha=0.7, width=0.6)
ax.axhline( conf, color=YELLOW, lw=1.5, linestyle="--", label="95% CI")
ax.axhline(-conf, color=YELLOW, lw=1.5, linestyle="--")
ax.axhline(0, color=WHITE, lw=0.5)
ax.set_title("ACF of Returns^2\n(ARCH effects confirmed)", color=WHITE)
ax.set_xlabel("Lag"); ax.set_ylabel("ACF")
ax.legend(fontsize=8); ax.grid(True); watermark(ax)

# Ljung-Box p-values across lags
ax = axes[1, 1]
lags_lb = range(1, 31)
lb_r   = acorr_ljungbox(fin_returns,           lags=lags_lb, return_df=True)
lb_abs = acorr_ljungbox(np.abs(fin_returns),   lags=lags_lb, return_df=True)
lb_sq  = acorr_ljungbox(fin_returns**2,         lags=lags_lb, return_df=True)
ax.plot(lags_lb, lb_r["lb_pvalue"],   color=ACCENT,  lw=2, label="Returns")
ax.plot(lags_lb, lb_abs["lb_pvalue"], color=ORANGE,  lw=2, label="|Returns|")
ax.plot(lags_lb, lb_sq["lb_pvalue"],  color=RED,     lw=2, label="Returns^2")
ax.axhline(0.05, color=YELLOW, lw=1.5, linestyle="--", label="5% significance")
ax.set_xlabel("Lag m"); ax.set_ylabel("Ljung-Box p-value")
ax.set_title("Ljung-Box p-values across lags", color=WHITE)
ax.legend(fontsize=8); ax.grid(True); watermark(ax)

# Scatter: r_t vs r_{t-1}
ax = axes[1, 2]
ax.scatter(fin_returns[:-1]*100, fin_returns[1:]*100,
           color=ACCENT, s=3, alpha=0.3)
slope, intercept, r_val, p_val, _ = stats.linregress(fin_returns[:-1], fin_returns[1:])
x_fit = np.linspace(fin_returns.min(), fin_returns.max(), 100)
ax.plot(x_fit*100, (slope*x_fit + intercept)*100, color=YELLOW, lw=2,
        label=f"OLS: rho={r_val:.4f}, p={p_val:.3f}")
ax.set_xlabel("r_t (%)"); ax.set_ylabel("r_{t+1} (%)")
ax.set_title("Scatter r_t vs r_{t+1}\n(no linear predictability)", color=WHITE)
ax.legend(fontsize=8); ax.grid(True); watermark(ax)

plt.tight_layout()
path = os.path.join(OUT_DIR, "m07_02_financial_acf.png")
plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"[OK] {path}")

# =============================================================================
# FIGURE 3: ACF Theoretical Patterns Reference Card
# =============================================================================
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
fig.patch.set_facecolor(DARK)
fig.suptitle("M07 — ACF/PACF Pattern Reference Card (Box-Jenkins Identification)",
             color=WHITE, fontsize=13, fontweight="bold")

ref_processes = [
    (wn,     "White Noise\nACF: all~0\nPACF: all~0",   MUTED),
    (ar2,    "AR(2)\nACF: exp decay\nPACF: cuts at 2",  GREEN),
    (ma3,    "MA(3)\nACF: cuts at 3\nPACF: exp decay",  ACCENT),
    (arma11, "ARMA(1,1)\nACF: decay q+\nPACF: decay p+",ORANGE),
]
for i, (s, title, col) in enumerate(ref_processes):
    acf_v  = acf(s,  nlags=20, fft=True)
    pacf_v = pacf(s, nlags=20, method="ywm")
    conf_  = 1.96 / np.sqrt(len(s))
    lags_  = range(1, 21)

    ax = axes[0, i]
    ax.bar(lags_, acf_v[1:], color=col, alpha=0.7, width=0.6)
    ax.axhline( conf_, color=YELLOW, lw=1, linestyle="--")
    ax.axhline(-conf_, color=YELLOW, lw=1, linestyle="--")
    ax.axhline(0, color=WHITE, lw=0.4)
    ax.set_title(f"ACF\n{title}", color=WHITE, fontsize=8)
    ax.grid(True); watermark(ax)

    ax = axes[1, i]
    ax.bar(lags_, pacf_v[1:], color=col, alpha=0.7, width=0.6)
    ax.axhline( conf_, color=YELLOW, lw=1, linestyle="--")
    ax.axhline(-conf_, color=YELLOW, lw=1, linestyle="--")
    ax.axhline(0, color=WHITE, lw=0.4)
    ax.set_title(f"PACF — {title.split(chr(10))[0]}", color=WHITE, fontsize=8)
    ax.grid(True); watermark(ax)

plt.tight_layout()
path = os.path.join(OUT_DIR, "m07_03_reference_card.png")
plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"[OK] {path}")

print("\n" + "=" * 65)
print("  MODULE 07 COMPLETE — 3 figures saved")
print("  Key Concepts:")
print("  [1] ACF: linear correlation at lag h")
print("  [2] PACF: direct effect after removing intermediate lags")
print("  [3] AR(p): ACF decays, PACF cuts at p")
print("  [4] MA(q): ACF cuts at q, PACF decays")
print("  [5] Ljung-Box: formal test for no autocorrelation")
print("  [6] Financial: ACF(r)~0, ACF(|r|)>>0 (volatility memory)")
print("=" * 65)
