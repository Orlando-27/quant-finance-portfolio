#!/usr/bin/env python3
# =============================================================================
# MODULE 05: ROLLING VOLATILITY AND EWMA (RISKMETRICS)
# =============================================================================
# Author      : Jose Orlando Bobadilla Fuentes | CQF | MSc AI
# Institution : Colombian Pension Fund — Investment Division
# Project     : 19 - CQF Concepts Explained
# Output      : outputs/figures/m05_*.png
# Run         : python src/m05_volatility_ewma/m05_volatility_ewma.py
# =============================================================================
"""
ROLLING VOLATILITY AND EWMA (RISKMETRICS)
==========================================

THEORETICAL FOUNDATIONS
------------------------

1. HISTORICAL ROLLING VOLATILITY
   The simplest volatility estimate uses a fixed window of m observations:

       sigma_t^2 = (1/(m-1)) * sum_{i=1}^{m} (r_{t-i} - r_bar)^2

   where r_bar = (1/m) * sum r_{t-i}  (sample mean, often set to 0 for daily)

   Properties:
   - All observations receive equal weight 1/m
   - Observations older than m days receive zero weight (ghost effect)
   - Larger window -> smoother but more lagged
   - Smaller window -> more responsive but noisier

2. EXPONENTIALLY WEIGHTED MOVING AVERAGE (EWMA / RISKMETRICS)
   J.P. Morgan's RiskMetrics (1994) proposed the EWMA estimator:

       sigma_t^2 = lambda * sigma_{t-1}^2 + (1 - lambda) * r_{t-1}^2

   This is a recursive formula where:
   - lambda in (0, 1) is the decay factor
   - (1 - lambda) is the weight on the most recent squared return
   - Expanding the recursion: sigma_t^2 = (1-lambda) * sum_{i=0}^{inf} lambda^i * r_{t-1-i}^2

   Weight on observation i periods ago: w_i = (1 - lambda) * lambda^i
   This forms a geometric series summing to 1: sum_{i=0}^{inf} w_i = 1

   RiskMetrics recommended values:
   - Daily data:   lambda = 0.94
   - Monthly data: lambda = 0.97

   Effective number of observations (half-life):
       t_{1/2} = -ln(2) / ln(lambda)

   For lambda = 0.94: t_{1/2} = -ln(2)/ln(0.94) ~ 11.2 days

3. OPTIMAL LAMBDA SELECTION
   Minimize the quasi-likelihood loss function:

       L(lambda) = sum_t [ln(sigma_t^2(lambda)) + r_t^2 / sigma_t^2(lambda)]

   This is the negative log-likelihood under Gaussian innovations.

4. VOLATILITY CONE
   The volatility cone shows the distribution of realized volatility
   across different measurement windows and horizons, providing context
   for whether current volatility is historically high or low.

References:
    J.P. Morgan (1994). RiskMetrics Technical Document. 4th Edition.
    Nelson, D.B. (1992). Filtering and forecasting with misspecified ARCH
        models I: Getting the right variance with the wrong model.
        Journal of Econometrics, 52(1-2), 61-90.
"""

import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy import stats

# =============================================================================
# CONFIGURATION
# =============================================================================
DARK   = "#0a0a0a"
DARK2  = "#111111"
DARK3  = "#1a1a1a"
ACCENT = "#00d4ff"
GREEN  = "#00ff88"
RED    = "#ff4444"
YELLOW = "#ffd700"
ORANGE = "#ff8c00"
PURPLE = "#bb86fc"
WHITE  = "#e0e0e0"
MUTED  = "#888888"
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

# =============================================================================
# DATA GENERATION — GARCH(1,1) with regime change
# =============================================================================
np.random.seed(42)
N = 1500

def garch_returns(n, omega=1e-6, alpha=0.09, beta=0.90, regime_change=True):
    r  = np.zeros(n)
    s2 = np.zeros(n)
    s2[0] = omega / (1 - alpha - beta)
    for t in range(1, n):
        # Regime change: double volatility in middle third
        alpha_t = alpha * 2 if regime_change and n//3 < t < 2*n//3 else alpha
        s2[t] = omega + alpha_t * r[t-1]**2 + beta * s2[t-1]
        r[t]  = np.sqrt(s2[t]) * np.random.randn()
    return r, np.sqrt(s2)

returns, true_vol = garch_returns(N)
dates = pd.date_range("2018-01-01", periods=N, freq="B")
r = pd.Series(returns, index=dates)

# =============================================================================
# VOLATILITY ESTIMATORS
# =============================================================================

def rolling_vol(r, window):
    """Equal-weighted rolling volatility, annualised."""
    return r.rolling(window).std() * np.sqrt(252) * 100

def ewma_vol(r, lam):
    """
    EWMA volatility estimator (RiskMetrics).
    sigma_t^2 = lambda * sigma_{t-1}^2 + (1-lambda) * r_{t-1}^2
    """
    r_arr = r.values
    n     = len(r_arr)
    s2    = np.zeros(n)
    s2[0] = r_arr[0]**2
    for t in range(1, n):
        s2[t] = lam * s2[t-1] + (1 - lam) * r_arr[t-1]**2
    return pd.Series(np.sqrt(s2) * np.sqrt(252) * 100, index=r.index)

def ewma_loss(lam, r):
    """Quasi-likelihood loss for lambda optimisation."""
    r_arr = r.values
    n     = len(r_arr)
    s2    = np.zeros(n)
    s2[0] = r_arr[0]**2
    for t in range(1, n):
        s2[t] = lam * s2[t-1] + (1 - lam) * r_arr[t-1]**2
    s2 = np.maximum(s2, 1e-12)
    return float(np.mean(np.log(s2[1:]) + r_arr[1:]**2 / s2[1:]))

# Compute estimators
vol_10  = rolling_vol(r, 10)
vol_21  = rolling_vol(r, 21)
vol_63  = rolling_vol(r, 63)
ewma_94 = ewma_vol(r, 0.94)
ewma_97 = ewma_vol(r, 0.97)

# Optimal lambda
result = minimize_scalar(ewma_loss, bounds=(0.80, 0.99), method="bounded", args=(r,))
lam_opt = result.x
ewma_opt = ewma_vol(r, lam_opt)
half_life = -np.log(2) / np.log(lam_opt)

print("=" * 60)
print("  MODULE 05: ROLLING VOLATILITY AND EWMA")
print("=" * 60)
print(f"\n  Optimal lambda      : {lam_opt:.4f}")
print(f"  Half-life           : {half_life:.1f} trading days")
print(f"  lambda=0.94 HL      : {-np.log(2)/np.log(0.94):.1f} trading days")
print(f"  lambda=0.97 HL      : {-np.log(2)/np.log(0.97):.1f} trading days")
print(f"  Quasi-likelihood opt: {result.fun:.6f}")
print(f"  Current vol (EWMA94): {ewma_94.dropna().iloc[-1]:.2f}%")
print(f"  Current vol (21d):    {vol_21.dropna().iloc[-1]:.2f}%")

# =============================================================================
# FIGURE 1: Volatility Estimators Comparison
# =============================================================================
fig, axes = plt.subplots(3, 1, figsize=(14, 12))
fig.patch.set_facecolor(DARK)
fig.suptitle("M05 — Rolling Volatility & EWMA (RiskMetrics)",
             color=WHITE, fontsize=13, fontweight="bold")

# Returns
ax = axes[0]
ax.plot(dates, r * 100, color=ACCENT, lw=0.6, alpha=0.8)
ax.fill_between(dates, r*100, 0, where=r>0, color=GREEN, alpha=0.1)
ax.fill_between(dates, r*100, 0, where=r<0, color=RED, alpha=0.1)
ax.axvspan(dates[N//3], dates[2*N//3], color=ORANGE, alpha=0.08,
           label="High-vol regime")
ax.set_ylabel("Daily Return (%)")
ax.set_title("Return Series with Regime Change", color=WHITE)
ax.legend(fontsize=8); ax.grid(True); watermark(ax)

# Rolling volatility comparison
ax = axes[1]
ax.plot(dates, vol_10,  color=RED,    lw=1.0, alpha=0.8, label="Rolling 10d")
ax.plot(dates, vol_21,  color=ACCENT, lw=1.2, alpha=0.9, label="Rolling 21d")
ax.plot(dates, vol_63,  color=GREEN,  lw=1.5, alpha=0.9, label="Rolling 63d")
ax.plot(dates, pd.Series(true_vol * np.sqrt(252) * 100, index=dates),
        color=WHITE, lw=0.8, linestyle=":", alpha=0.5, label="True GARCH vol")
ax.axvspan(dates[N//3], dates[2*N//3], color=ORANGE, alpha=0.06)
ax.set_ylabel("Ann. Volatility (%)")
ax.set_title("Equal-Weighted Rolling Volatility: Window Size Effect", color=WHITE)
ax.legend(fontsize=8); ax.grid(True); watermark(ax)

# EWMA comparison
ax = axes[2]
ax.plot(dates, ewma_94, color=YELLOW,  lw=1.5, label=f"EWMA lambda=0.94 (HL=11d)")
ax.plot(dates, ewma_97, color=ORANGE,  lw=1.5, label=f"EWMA lambda=0.97 (HL=23d)")
ax.plot(dates, ewma_opt,color=GREEN,   lw=1.5, label=f"EWMA lambda={lam_opt:.3f} (HL={half_life:.0f}d) [optimal]")
ax.plot(dates, vol_21,  color=ACCENT,  lw=1.0, linestyle="--", alpha=0.6, label="Rolling 21d (reference)")
ax.axvspan(dates[N//3], dates[2*N//3], color=ORANGE, alpha=0.06)
ax.set_ylabel("Ann. Volatility (%)")
ax.set_title("EWMA vs Rolling: Responsiveness to Regime Change", color=WHITE)
ax.legend(fontsize=8); ax.grid(True); watermark(ax)

plt.tight_layout()
path = os.path.join(OUT_DIR, "m05_01_volatility_comparison.png")
plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"\n[OK] {path}")

# =============================================================================
# FIGURE 2: EWMA Weights & Optimal Lambda
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor(DARK)
fig.suptitle("M05 — EWMA: Weight Decay & Optimal Lambda",
             color=WHITE, fontsize=13, fontweight="bold")

# Weight decay
ax = axes[0]
lags = np.arange(0, 60)
for lam, col, lbl in [(0.90, RED, "lambda=0.90"),
                       (0.94, YELLOW, "lambda=0.94 (RiskMetrics)"),
                       (0.97, GREEN, "lambda=0.97"),
                       (lam_opt, ACCENT, f"lambda={lam_opt:.3f} (optimal)")]:
    weights = (1 - lam) * lam**lags
    ax.plot(lags, weights * 100, color=col, lw=2, label=lbl)
ax.set_xlabel("Lag (days)")
ax.set_ylabel("Weight (%)")
ax.set_title("EWMA Weights: (1-lambda)*lambda^i", color=WHITE)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# Cumulative weights
ax = axes[1]
for lam, col, lbl in [(0.90, RED, "0.90"), (0.94, YELLOW, "0.94"),
                       (0.97, GREEN, "0.97"), (lam_opt, ACCENT, f"{lam_opt:.3f}")]:
    cum_w = np.cumsum((1 - lam) * lam**lags)
    ax.plot(lags, cum_w * 100, color=col, lw=2, label=f"lambda={lbl}")
ax.axhline(50, color=WHITE, lw=0.8, linestyle="--", alpha=0.5, label="50% (half-life)")
ax.axhline(95, color=MUTED, lw=0.8, linestyle="--", alpha=0.5, label="95%")
ax.set_xlabel("Lag (days)")
ax.set_ylabel("Cumulative Weight (%)")
ax.set_title("Cumulative Weights: Effective Memory", color=WHITE)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# Lambda optimization landscape
ax = axes[2]
lambdas = np.linspace(0.80, 0.99, 100)
losses  = [ewma_loss(lam, r) for lam in lambdas]
ax.plot(lambdas, losses, color=ACCENT, lw=2)
ax.axvline(lam_opt, color=YELLOW, lw=2, linestyle="--",
           label=f"Optimal lambda={lam_opt:.4f}")
ax.axvline(0.94, color=RED, lw=1.5, linestyle=":",
           label="RiskMetrics lambda=0.94")
ax.set_xlabel("Lambda")
ax.set_ylabel("Quasi-Likelihood Loss")
ax.set_title("Lambda Optimization Landscape", color=WHITE)
ax.legend(fontsize=8); ax.grid(True); watermark(ax)

plt.tight_layout()
path = os.path.join(OUT_DIR, "m05_02_ewma_weights_optimization.png")
plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"[OK] {path}")

# =============================================================================
# FIGURE 3: Volatility Cone
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor(DARK)
fig.suptitle("M05 — Volatility Cone: Historical Context",
             color=WHITE, fontsize=13, fontweight="bold")

windows = [5, 10, 21, 42, 63, 126]
percentiles = [5, 25, 50, 75, 95]
cone = {p: [] for p in percentiles}
current_vols = []

for w in windows:
    rv = r.rolling(w).std().dropna() * np.sqrt(252) * 100
    for p in percentiles:
        cone[p].append(np.percentile(rv, p))
    current_vols.append(rv.iloc[-1])

ax = axes[0]
colors_cone = [RED, ORANGE, GREEN, ORANGE, RED]
labels_cone = ["5th", "25th", "50th", "75th", "95th"]
for p, col, lbl in zip(percentiles, colors_cone, labels_cone):
    ax.plot(windows, cone[p], color=col, lw=1.5, marker="o", ms=5, label=f"{lbl} pct")
ax.fill_between(windows, cone[5], cone[95], color=ACCENT, alpha=0.08)
ax.fill_between(windows, cone[25], cone[75], color=ACCENT, alpha=0.12)
ax.plot(windows, current_vols, color=WHITE, lw=2.5, marker="D", ms=7,
        linestyle="--", label="Current realized vol")
ax.set_xlabel("Measurement Window (days)")
ax.set_ylabel("Annualised Volatility (%)")
ax.set_title("Volatility Cone: Current vs Historical Percentiles", color=WHITE)
ax.legend(fontsize=8); ax.grid(True); watermark(ax)

# Ghost effect demonstration
ax = axes[1]
# Show how dropping a large return causes a jump in rolling vol
r_demo = pd.Series(np.random.normal(0, 0.01, 100))
r_demo.iloc[30] = 0.08  # large shock
roll = r_demo.rolling(21).std() * np.sqrt(252) * 100
ewma_d = ewma_vol(r_demo, 0.94)
ax.plot(r_demo.index, roll,    color=RED,    lw=2, label="Rolling 21d (ghost effect)")
ax.plot(r_demo.index, ewma_d,  color=ACCENT, lw=2, label="EWMA 0.94 (smooth decay)")
ax.axvline(30, color=YELLOW, lw=1.5, linestyle="--", label="Large shock")
ax.axvline(51, color=RED, lw=1, linestyle=":", alpha=0.7, label="Shock exits window (+21d)")
ax.set_xlabel("Day")
ax.set_ylabel("Ann. Volatility (%)")
ax.set_title("Ghost Effect: Rolling vs EWMA Response to Shock", color=WHITE)
ax.legend(fontsize=8); ax.grid(True); watermark(ax)

plt.tight_layout()
path = os.path.join(OUT_DIR, "m05_03_volatility_cone.png")
plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"[OK] {path}")

print("\n" + "=" * 60)
print("  MODULE 05 COMPLETE — 3 figures saved")
print("  Key Concepts:")
print(f"  [1] Rolling vol: equal weights, ghost effect")
print(f"  [2] EWMA: geometric decay weights, sum=1")
print(f"  [3] RiskMetrics: lambda=0.94 (daily), HL=11d")
print(f"  [4] Optimal lambda={lam_opt:.4f}, HL={half_life:.1f}d")
print(f"  [5] Volatility cone: current vol in historical context")
print("=" * 60)
