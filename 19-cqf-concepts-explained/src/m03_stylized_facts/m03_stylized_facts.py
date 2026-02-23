#!/usr/bin/env python3
# =============================================================================
# MODULE 03: STYLIZED FACTS OF FINANCIAL MARKETS
# =============================================================================
# Author      : Jose Orlando Bobadilla Fuentes | CQF | MSc AI
# Institution : Colombian Pension Fund — Investment Division
# Project     : 19 - CQF Concepts Explained
# Output      : outputs/figures/m03_*.png
# Run         : python src/m03_stylized_facts/m03_stylized_facts.py
# =============================================================================
"""
STYLIZED FACTS OF FINANCIAL MARKETS
=====================================

THEORETICAL FOUNDATIONS
------------------------
Cont (2001) identified a set of statistical properties shared across asset
classes, geographies and time periods — the so-called "stylized facts":

1. HEAVY TAILS (Leptokurtosis)
   Return distributions exhibit excess kurtosis k > 3. The standardized
   4th central moment:
       kurtosis = E[(r - mu)^4] / sigma^4
   For a Gaussian, kurtosis = 3. Empirical returns show kurtosis 4-20+,
   meaning extreme events occur far more frequently than Gaussian models predict.
   Mathematically, the tail decay follows a power law:
       P(|r| > x) ~ x^{-alpha},  alpha in [2, 5]
   rather than the Gaussian exp(-x^2/2) decay.

2. NEGATIVE SKEWNESS (Leverage Effect)
   Equity return distributions show negative skewness:
       skewness = E[(r - mu)^3] / sigma^3 < 0
   This reflects the asymmetric response to shocks: bad news increases
   volatility more than good news (Black 1976, Christie 1982).

3. VOLATILITY CLUSTERING (ARCH Effects)
   While returns are approximately uncorrelated: Corr(r_t, r_{t+k}) ~ 0,
   absolute returns and squared returns exhibit long memory:
       Corr(|r_t|, |r_{t+k}|) > 0  for k up to months
   This is captured by GARCH(p,q) models:
       sigma_t^2 = omega + sum_i alpha_i * r_{t-i}^2 + sum_j beta_j * sigma_{t-j}^2

4. ABSENCE OF LINEAR AUTOCORRELATION
   The Efficient Market Hypothesis (weak form) implies:
       Corr(r_t, r_{t+k}) ~ 0 for k >= 1
   Tested via Ljung-Box Q-statistic:
       Q(m) = n(n+2) * sum_{k=1}^{m} rho_k^2 / (n-k) ~ chi^2(m)

5. AGGREGATIONAL GAUSSIANITY
   At longer horizons (monthly, quarterly), the central limit theorem
   drives distributions toward normality. At daily frequency, fat tails dominate.

References:
    Cont, R. (2001). Empirical properties of asset returns: stylized facts
        and statistical issues. Quantitative Finance, 1(2), 223-236.
    Mandelbrot, B. (1963). The variation of certain speculative prices.
        Journal of Business, 36(4), 394-419.
"""

import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from scipy import stats
from scipy.stats import norm, t as student_t, jarque_bera

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
# DATA GENERATION — Synthetic multi-asset returns
# =============================================================================
np.random.seed(42)
N = 2000  # trading days (~8 years)

def simulate_garch_returns(n, mu=0.0003, omega=1e-6, alpha=0.09, beta=0.90):
    """
    GARCH(1,1) simulation:
        r_t = mu + sigma_t * z_t,  z_t ~ N(0,1)
        sigma_t^2 = omega + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2

    Stationarity requires: alpha + beta < 1 (unconditional variance exists)
    Unconditional variance: sigma^2_inf = omega / (1 - alpha - beta)
    """
    sigma2 = np.zeros(n)
    r      = np.zeros(n)
    z      = np.random.standard_normal(n)
    sigma2[0] = omega / (1 - alpha - beta)  # start at unconditional variance
    r[0]      = mu + np.sqrt(sigma2[0]) * z[0]
    for t in range(1, n):
        sigma2[t] = omega + alpha * r[t-1]**2 + beta * sigma2[t-1]
        r[t]      = mu + np.sqrt(sigma2[t]) * z[t]
    return r, np.sqrt(sigma2)

returns_garch, vol_garch = simulate_garch_returns(N)
returns_normal = np.random.normal(0.0003, returns_garch.std(), N)

# Student-t returns (fat tails, df=4)
df_student = 4
returns_student = stats.t.rvs(df_student, loc=0.0003,
                               scale=returns_garch.std() * np.sqrt((df_student-2)/df_student),
                               size=N)

dates = pd.date_range("2016-01-01", periods=N, freq="B")
prices = 100 * np.exp(np.cumsum(returns_garch))

r = pd.Series(returns_garch, index=dates, name="GARCH Returns")
r_norm = pd.Series(returns_normal, index=dates, name="Normal Returns")

# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================
def compute_stylized_stats(returns):
    """Compute all stylized fact statistics."""
    r = np.array(returns)
    mu    = r.mean()
    sigma = r.std()
    skew  = stats.skew(r)
    kurt  = stats.kurtosis(r, fisher=True)  # excess kurtosis (Gaussian=0)
    jb_stat, jb_p = jarque_bera(r)
    # Ljung-Box at lag 10
    n = len(r)
    acf_r  = [pd.Series(r).autocorr(lag=k) for k in range(1, 11)]
    acf_r2 = [pd.Series(r**2).autocorr(lag=k) for k in range(1, 11)]
    q_r  = n*(n+2) * sum(rho**2/(n-k) for k, rho in enumerate(acf_r, 1))
    q_r2 = n*(n+2) * sum(rho**2/(n-k) for k, rho in enumerate(acf_r2, 1))
    # Tail index via Hill estimator
    abs_r = np.abs(r)
    k_tail = int(0.05 * n)
    sorted_r = np.sort(abs_r)[::-1]
    hill_alpha = 1 / np.mean(np.log(sorted_r[:k_tail] / sorted_r[k_tail]))
    return {
        "Mean (ann.)":    round(mu * 252 * 100, 2),
        "Vol (ann.)":     round(sigma * np.sqrt(252) * 100, 2),
        "Skewness":       round(skew, 4),
        "Excess Kurtosis":round(kurt, 4),
        "JB Statistic":   round(jb_stat, 2),
        "JB p-value":     f"{jb_p:.2e}",
        "LB-Q(10) r":     round(q_r, 2),
        "LB-Q(10) r^2":   round(q_r2, 2),
        "Hill Tail Index":round(hill_alpha, 3),
    }

stats_garch  = compute_stylized_stats(r)
stats_normal = compute_stylized_stats(r_norm)

print("=" * 60)
print("  MODULE 03: STYLIZED FACTS OF FINANCIAL MARKETS")
print("=" * 60)
print(f"\n{'Statistic':<22} {'GARCH Returns':>15} {'Normal Returns':>15}")
print("-" * 54)
for k in stats_garch:
    print(f"{k:<22} {str(stats_garch[k]):>15} {str(stats_normal[k]):>15}")

# =============================================================================
# FIGURE 1: RETURN DISTRIBUTION — Fat Tails vs Normal
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor(DARK)
fig.suptitle("M03 — Stylized Facts: Return Distribution Analysis",
             color=WHITE, fontsize=13, fontweight="bold", y=1.02)

ax = axes[0]
x = np.linspace(r.min(), r.max(), 500)
ax.hist(r * 100, bins=80, density=True, color=ACCENT, alpha=0.6,
        label="GARCH Returns", edgecolor="none")
mu_, sig_ = r.mean(), r.std()
ax.plot(x * 100, norm.pdf(x, mu_, sig_), color=YELLOW, lw=2, label="Gaussian fit")
ax.plot(x * 100, student_t.pdf(x, df_student, mu_, sig_ * np.sqrt((df_student-2)/df_student)),
        color=RED, lw=2, linestyle="--", label=f"Student-t (df={df_student})")
ax.set_xlabel("Daily Return (%)")
ax.set_ylabel("Density")
ax.set_title("Return Distribution vs Gaussian", color=WHITE)
ax.legend(fontsize=8)
ax.grid(True)
watermark(ax)

# Q-Q plot
ax = axes[1]
(osm, osr), (slope, intercept, _) = stats.probplot(r * 100, dist="norm")
ax.scatter(osm, osr, color=ACCENT, s=4, alpha=0.5, label="Empirical quantiles")
ax.plot(osm, slope * np.array(osm) + intercept, color=YELLOW, lw=2,
        label="Gaussian reference")
ax.set_xlabel("Theoretical Quantiles")
ax.set_ylabel("Sample Quantiles (%)")
ax.set_title("Q-Q Plot: Fat Tails Visible", color=WHITE)
ax.legend(fontsize=8)
ax.grid(True)
watermark(ax)

# Tail probability
ax = axes[2]
abs_r = np.sort(np.abs(r))[::-1]
n_ = len(abs_r)
emp_prob = np.arange(1, n_+1) / n_
x_gauss = np.linspace(abs_r.min(), abs_r.max(), 200)
gauss_prob = 2 * (1 - norm.cdf(x_gauss, 0, r.std()))
ax.loglog(abs_r * 100, emp_prob, color=ACCENT, lw=1.5, label="Empirical")
ax.loglog(x_gauss * 100, gauss_prob, color=YELLOW, lw=2, linestyle="--",
          label="Gaussian")
ax.set_xlabel("|Return| (%)")
ax.set_ylabel("P(|r| > x)")
ax.set_title("Tail Probability (log-log scale)", color=WHITE)
ax.legend(fontsize=8)
ax.grid(True)
watermark(ax)

plt.tight_layout()
path = os.path.join(OUT_DIR, "m03_01_return_distribution.png")
plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"\n[OK] {path}")

# =============================================================================
# FIGURE 2: VOLATILITY CLUSTERING & AUTOCORRELATION
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 9))
fig.patch.set_facecolor(DARK)
fig.suptitle("M03 — Stylized Facts: Volatility Clustering & Serial Correlation",
             color=WHITE, fontsize=13, fontweight="bold")

# Returns time series
ax = axes[0, 0]
ax.plot(dates, r * 100, color=ACCENT, lw=0.6, alpha=0.9)
ax.axhline(0, color=MUTED, lw=0.5)
ax.fill_between(dates, r * 100, 0, where=r > 0, color=GREEN, alpha=0.15)
ax.fill_between(dates, r * 100, 0, where=r < 0, color=RED, alpha=0.15)
ax.set_ylabel("Daily Return (%)")
ax.set_title("GARCH Return Series — Volatility Clustering", color=WHITE)
ax.grid(True)
watermark(ax)

# Rolling volatility
ax = axes[0, 1]
roll_vol = r.rolling(21).std() * np.sqrt(252) * 100
ax.plot(dates, roll_vol, color=ORANGE, lw=1.2, label="21d Rolling Vol (ann.)")
ax.plot(dates, pd.Series(vol_garch * np.sqrt(252) * 100, index=dates),
        color=YELLOW, lw=0.8, linestyle="--", alpha=0.7, label="GARCH sigma_t (ann.)")
ax.set_ylabel("Volatility (%)")
ax.set_title("Conditional Volatility: Clustering Confirmed", color=WHITE)
ax.legend(fontsize=8)
ax.grid(True)
watermark(ax)

# ACF of returns
ax = axes[1, 0]
lags = range(1, 31)
acf_r  = [r.autocorr(lag=k) for k in lags]
acf_r2 = [(r**2).autocorr(lag=k) for k in lags]
conf = 1.96 / np.sqrt(N)
ax.bar(lags, acf_r, color=ACCENT, alpha=0.7, label="ACF(r_t) — raw returns")
ax.axhline(conf, color=YELLOW, lw=1, linestyle="--", label=f"95% CI (+/-{conf:.3f})")
ax.axhline(-conf, color=YELLOW, lw=1, linestyle="--")
ax.axhline(0, color=WHITE, lw=0.5)
ax.set_xlabel("Lag (days)")
ax.set_ylabel("Autocorrelation")
ax.set_title("ACF of Returns — No Linear Predictability", color=WHITE)
ax.legend(fontsize=8)
ax.grid(True)
watermark(ax)

# ACF of squared returns
ax = axes[1, 1]
ax.bar(lags, acf_r2, color=RED, alpha=0.7, label="ACF(r_t^2) — squared returns")
ax.axhline(conf, color=YELLOW, lw=1, linestyle="--", label=f"95% CI (+/-{conf:.3f})")
ax.axhline(-conf, color=YELLOW, lw=1, linestyle="--")
ax.axhline(0, color=WHITE, lw=0.5)
ax.set_xlabel("Lag (days)")
ax.set_ylabel("Autocorrelation")
ax.set_title("ACF of Squared Returns — Long Memory in Volatility", color=WHITE)
ax.legend(fontsize=8)
ax.grid(True)
watermark(ax)

plt.tight_layout()
path = os.path.join(OUT_DIR, "m03_02_volatility_clustering.png")
plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"[OK] {path}")

# =============================================================================
# FIGURE 3: AGGREGATIONAL GAUSSIANITY
# =============================================================================
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
fig.patch.set_facecolor(DARK)
fig.suptitle("M03 — Stylized Fact 5: Aggregational Gaussianity",
             color=WHITE, fontsize=13, fontweight="bold")

horizons = [1, 5, 21, 63]
labels   = ["Daily (1d)", "Weekly (5d)", "Monthly (21d)", "Quarterly (63d)"]

r_series = pd.Series(returns_garch)
for ax, h, lbl in zip(axes, horizons, labels):
    r_h = r_series.rolling(h).sum().dropna()
    kurt_h = stats.kurtosis(r_h, fisher=True)
    skew_h = stats.skew(r_h)
    x = np.linspace(r_h.min(), r_h.max(), 300)
    ax.hist(r_h * 100, bins=50, density=True, color=ACCENT, alpha=0.6,
            edgecolor="none", label="Empirical")
    ax.plot(x * 100, norm.pdf(x, r_h.mean(), r_h.std()), color=YELLOW,
            lw=2, label="Gaussian")
    ax.set_title(f"{lbl}\nKurt={kurt_h:.2f} | Skew={skew_h:.2f}", color=WHITE)
    ax.set_xlabel("Return (%)")
    ax.legend(fontsize=7)
    ax.grid(True)
    watermark(ax)

plt.tight_layout()
path = os.path.join(OUT_DIR, "m03_03_aggregational_gaussianity.png")
plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"[OK] {path}")

print("\n" + "=" * 60)
print("  MODULE 03 COMPLETE — 3 figures saved")
print("  Stylized Facts Demonstrated:")
print("  [1] Heavy tails: excess kurtosis >> 0")
print("  [2] Negative skewness: leverage effect")
print("  [3] Volatility clustering: ACF(r^2) >> 0")
print("  [4] No linear autocorrelation: ACF(r) ~ 0")
print("  [5] Aggregational Gaussianity: kurtosis -> 3 at longer horizons")
print("=" * 60)
