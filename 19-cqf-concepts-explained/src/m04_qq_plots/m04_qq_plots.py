#!/usr/bin/env python3
# =============================================================================
# MODULE 04: Q-Q PLOTS AND PROBABILITY ANALYSIS
# =============================================================================
# Author      : Jose Orlando Bobadilla Fuentes | CQF | MSc AI
# Institution : Colombian Pension Fund — Investment Division
# Project     : 19 - CQF Concepts Explained
# Output      : outputs/figures/m04_*.png
# Run         : python src/m04_qq_plots/m04_qq_plots.py
# =============================================================================
"""
Q-Q PLOTS AND PROBABILITY ANALYSIS
====================================

THEORETICAL FOUNDATIONS
------------------------
A Quantile-Quantile (Q-Q) plot is a graphical diagnostic tool that compares
two probability distributions by plotting their quantiles against each other.

CONSTRUCTION
------------
Given n observations x_1, ..., x_n sorted in ascending order (order statistics),
the empirical quantile at position i is:

    q_i = x_(i)  (i-th order statistic)

The corresponding theoretical quantile from distribution F is:

    z_i = F^{-1}((i - 0.5) / n)   [Hazen plotting position]

Alternative plotting positions:
    - Weibull:  i / (n + 1)
    - Blom:     (i - 3/8) / (n + 1/4)
    - Hazen:    (i - 0.5) / n      [most common for finance]

If the data follows distribution F exactly, the Q-Q plot lies on the 45-degree
reference line y = x. Deviations reveal:

    HEAVY TAILS (S-curve):  points curve away from the line at both ends
    LIGHT TAILS:            points curve toward the line at both ends
    SKEWNESS:               asymmetric deviation — one tail longer than the other
    OUTLIERS:               isolated points far from the reference line

THEORETICAL QUANTILES FOR COMMON DISTRIBUTIONS
-----------------------------------------------
Normal:    z_i = mu + sigma * Phi^{-1}(p_i)
Student-t: z_i = mu + sigma * t_{nu}^{-1}(p_i)
Laplace:   z_i = mu - b * sign(p_i - 0.5) * ln(1 - 2|p_i - 0.5|)

ANDERSON-DARLING TEST
---------------------
A formal test for distributional fit based on the empirical CDF F_n(x):
    A^2 = -n - (1/n) * sum_{i=1}^{n} (2i-1)[ln(F(x_i)) + ln(1-F(x_{n+1-i}))]
Large A^2 rejects the null hypothesis of the assumed distribution.

References:
    Wilk, M.B. and Gnanadesikan, R. (1968). Probability plotting methods
        for the analysis of data. Biometrika, 55(1), 1-17.
    Anderson, T.W. and Darling, D.A. (1954). A test of goodness of fit.
        Journal of the American Statistical Association, 49(268), 765-769.
"""

import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm, t as student_t, laplace, anderson

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
# DATA GENERATION
# =============================================================================
np.random.seed(42)
N = 1000

# Four distributions to compare
data = {
    "Normal":    norm.rvs(loc=0, scale=0.01, size=N),
    "Student-t (df=4)": student_t.rvs(df=4, loc=0, scale=0.008, size=N),
    "Laplace":   laplace.rvs(loc=0, scale=0.007, size=N),
    "Skewed":    stats.skewnorm.rvs(a=-5, loc=0.001, scale=0.01, size=N),
}

# GARCH-like returns (realistic financial)
def garch_returns(n, omega=1e-6, alpha=0.09, beta=0.90):
    r = np.zeros(n); s2 = np.zeros(n)
    s2[0] = omega / (1 - alpha - beta)
    r[0]  = np.sqrt(s2[0]) * np.random.randn()
    for t in range(1, n):
        s2[t] = omega + alpha * r[t-1]**2 + beta * s2[t-1]
        r[t]  = np.sqrt(s2[t]) * np.random.randn()
    return r

financial = garch_returns(N)

# =============================================================================
# Q-Q PLOT CONSTRUCTION (manual, educational)
# =============================================================================
def qq_data(sample, dist, *dist_params):
    """
    Compute Q-Q plot data manually.
    Returns (theoretical_quantiles, empirical_quantiles, slope, intercept)
    """
    n = len(sample)
    sorted_sample = np.sort(sample)
    # Hazen plotting positions
    probs = (np.arange(1, n+1) - 0.5) / n
    theoretical = dist.ppf(probs, *dist_params)
    # Reference line: robust line through 1st and 3rd quartiles
    q25_t = dist.ppf(0.25, *dist_params)
    q75_t = dist.ppf(0.75, *dist_params)
    q25_e = np.percentile(sorted_sample, 25)
    q75_e = np.percentile(sorted_sample, 75)
    slope     = (q75_e - q25_e) / (q75_t - q25_t)
    intercept = q25_e - slope * q25_t
    return theoretical, sorted_sample, slope, intercept

# =============================================================================
# FIGURE 1: Q-Q Plots for Four Distributions vs Normal
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.patch.set_facecolor(DARK)
fig.suptitle("M04 — Q-Q Plots: Empirical vs Normal Distribution",
             color=WHITE, fontsize=13, fontweight="bold")

colors = [ACCENT, RED, GREEN, ORANGE]
for ax, (name, d), col in zip(axes.flat, data.items(), colors):
    mu_, sig_ = d.mean(), d.std()
    theor, emp, slope, intercept = qq_data(d, norm, mu_, sig_)
    ax.scatter(theor * 100, emp * 100, color=col, s=6, alpha=0.5, label="Data quantiles")
    x_line = np.array([theor.min(), theor.max()])
    ax.plot(x_line * 100, (slope * x_line + intercept) * 100,
            color=YELLOW, lw=2, label="Reference line (IQR)")
    # Annotate deviations
    kurt = stats.kurtosis(d, fisher=True)
    skw  = stats.skew(d)
    ax.set_xlabel("Theoretical Normal Quantiles (%)")
    ax.set_ylabel("Sample Quantiles (%)")
    ax.set_title(f"{name}\nExcess Kurt={kurt:.2f} | Skew={skw:.2f}", color=WHITE)
    ax.legend(fontsize=8)
    ax.grid(True)
    watermark(ax)

plt.tight_layout()
path = os.path.join(OUT_DIR, "m04_01_qq_four_distributions.png")
plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"[OK] {path}")

# =============================================================================
# FIGURE 2: Financial Returns — Normal vs Student-t Q-Q
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor(DARK)
fig.suptitle("M04 — Financial Returns: Q-Q vs Normal & Student-t",
             color=WHITE, fontsize=13, fontweight="bold")

mu_f, sig_f = financial.mean(), financial.std()

# vs Normal
ax = axes[0]
theor, emp, slope, intercept = qq_data(financial, norm, mu_f, sig_f)
ax.scatter(theor * 100, emp * 100, color=ACCENT, s=5, alpha=0.4, label="GARCH returns")
x_line = np.array([theor.min(), theor.max()])
ax.plot(x_line * 100, (slope * x_line + intercept) * 100, color=YELLOW, lw=2)
ax.set_xlabel("Normal Theoretical Quantiles (%)")
ax.set_ylabel("Sample Quantiles (%)")
ax.set_title("vs Normal\n(S-curve = fat tails)", color=WHITE)
ax.legend(fontsize=8); ax.grid(True); watermark(ax)

# vs Student-t (df=4)
ax = axes[1]
df_fit = 4
scale_t = sig_f * np.sqrt((df_fit - 2) / df_fit)
theor_t, emp_t, slope_t, int_t = qq_data(financial, student_t, df_fit, mu_f, scale_t)
ax.scatter(theor_t * 100, emp_t * 100, color=GREEN, s=5, alpha=0.4, label="GARCH returns")
x_line = np.array([theor_t.min(), theor_t.max()])
ax.plot(x_line * 100, (slope_t * x_line + int_t) * 100, color=YELLOW, lw=2)
ax.set_xlabel("Student-t (df=4) Quantiles (%)")
ax.set_ylabel("Sample Quantiles (%)")
ax.set_title("vs Student-t (df=4)\n(better tail fit)", color=WHITE)
ax.legend(fontsize=8); ax.grid(True); watermark(ax)

# PP plot (probability-probability)
ax = axes[2]
sorted_f  = np.sort(financial)
n_f       = len(sorted_f)
emp_probs = (np.arange(1, n_f+1) - 0.5) / n_f
theor_probs_norm = norm.cdf(sorted_f, mu_f, sig_f)
theor_probs_t    = student_t.cdf(sorted_f, df_fit, mu_f, scale_t)
ax.scatter(theor_probs_norm, emp_probs, color=ACCENT, s=4, alpha=0.4, label="vs Normal")
ax.scatter(theor_probs_t, emp_probs, color=GREEN, s=4, alpha=0.4, label="vs Student-t")
ax.plot([0, 1], [0, 1], color=YELLOW, lw=2, label="Perfect fit")
ax.set_xlabel("Theoretical CDF")
ax.set_ylabel("Empirical CDF")
ax.set_title("P-P Plot\n(deviations = poor fit)", color=WHITE)
ax.legend(fontsize=8); ax.grid(True); watermark(ax)

plt.tight_layout()
path = os.path.join(OUT_DIR, "m04_02_financial_returns_qq.png")
plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"[OK] {path}")

# =============================================================================
# FIGURE 3: Anderson-Darling Goodness-of-Fit Statistics
# =============================================================================
print("\nAnderson-Darling Tests on Financial Returns:")
print("-" * 50)

# Test normality
ad_result = anderson(financial, dist='norm')
print(f"  A^2 statistic : {ad_result.statistic:.4f}")
print(f"  Critical values (significance levels):")
for sl, cv in zip(ad_result.significance_level, ad_result.critical_values):
    reject = "REJECT" if ad_result.statistic > cv else "FAIL TO REJECT"
    print(f"    {sl}%: {cv:.3f}  -> {reject} normality")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor(DARK)
fig.suptitle("M04 — Anderson-Darling Test & Tail Probability",
             color=WHITE, fontsize=13, fontweight="bold")

# AD test visualization
ax = axes[0]
dfs = [3, 4, 5, 6, 8, 10, 15, 20]
ad_stats = []
for df_ in dfs:
    sc_ = sig_f * np.sqrt((df_ - 2) / df_)
    sample_t = student_t.rvs(df_, mu_f, sc_, size=N)
    ad_ = anderson(sample_t, dist='norm')
    ad_stats.append(ad_.statistic)

ax.plot(dfs, ad_stats, color=ACCENT, lw=2, marker="o", ms=6, label="A^2 (Student-t vs Normal)")
ax.axhline(ad_result.critical_values[2], color=YELLOW, lw=1.5, linestyle="--",
           label=f"Critical value 5% = {ad_result.critical_values[2]:.3f}")
ax.axhline(ad_result.statistic, color=RED, lw=1.5, linestyle="-.",
           label=f"GARCH A^2 = {ad_result.statistic:.3f}")
ax.set_xlabel("Degrees of Freedom (Student-t)")
ax.set_ylabel("Anderson-Darling A^2")
ax.set_title("AD Statistic vs Degrees of Freedom", color=WHITE)
ax.legend(fontsize=8); ax.grid(True); watermark(ax)

# Tail probability comparison
ax = axes[1]
x_tail = np.linspace(1, 5, 200)
p_normal  = 2 * (1 - norm.cdf(x_tail))
p_t4      = 2 * (1 - student_t.cdf(x_tail, df=4))
p_t10     = 2 * (1 - student_t.cdf(x_tail, df=10))

ax.semilogy(x_tail, p_normal, color=YELLOW, lw=2, label="Normal")
ax.semilogy(x_tail, p_t4,    color=RED,    lw=2, label="Student-t (df=4)")
ax.semilogy(x_tail, p_t10,   color=GREEN,  lw=2, label="Student-t (df=10)")
ax.set_xlabel("Standard Deviations from Mean")
ax.set_ylabel("P(|Z| > x)  [log scale]")
ax.set_title("Tail Probability: Normal vs Student-t", color=WHITE)
ax.legend(fontsize=8); ax.grid(True); watermark(ax)

plt.tight_layout()
path = os.path.join(OUT_DIR, "m04_03_anderson_darling.png")
plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"\n[OK] {path}")

print("\n" + "=" * 60)
print("  MODULE 04 COMPLETE — 3 figures saved")
print("  Key Concepts:")
print("  [1] Q-Q plot construction: Hazen plotting positions")
print("  [2] S-curve pattern = fat tails vs Normal")
print("  [3] Student-t provides better tail fit (df=4)")
print("  [4] Anderson-Darling formally rejects Normality")
print("  [5] P-P plot as complement to Q-Q")
print("=" * 60)
