#!/usr/bin/env python3
# =============================================================================
# MODULE 06: STATIONARITY — UNIT ROOT TESTING
# =============================================================================
# Author      : Jose Orlando Bobadilla Fuentes | CQF | MSc AI
# Institution : Colombian Pension Fund — Investment Division
# Project     : 19 - CQF Concepts Explained
# Output      : outputs/figures/m06_*.png
# Run         : python src/m06_stationarity/m06_stationarity.py
# =============================================================================
"""
STATIONARITY AND UNIT ROOT TESTING
====================================

THEORETICAL FOUNDATIONS
------------------------

1. STRICT vs WEAK STATIONARITY
   A process {X_t} is strictly stationary if the joint distribution of
   (X_{t1}, ..., X_{tk}) equals that of (X_{t1+h}, ..., X_{tk+h}) for all h.

   Weak (covariance) stationarity requires only:
       E[X_t] = mu          (constant mean)
       Var[X_t] = sigma^2   (constant variance)
       Cov(X_t, X_{t+h}) = gamma(h)  (covariance depends only on lag h)

2. RANDOM WALK (UNIT ROOT PROCESS)
   The simplest non-stationary process:
       X_t = X_{t-1} + epsilon_t,  epsilon_t ~ WN(0, sigma^2)

   Properties:
       E[X_t] = X_0         (mean does not change)
       Var[X_t] = t*sigma^2 (variance grows without bound)
       Cov(X_t, X_s) = min(t,s) * sigma^2

   This is also called an I(1) process (integrated of order 1).
   First differences are stationary: delta_X_t = X_t - X_{t-1} ~ I(0)

3. AUGMENTED DICKEY-FULLER TEST (ADF)
   Tests H0: unit root (non-stationary) vs H1: stationary.
   The ADF regression:
       delta_X_t = alpha + beta*t + gamma*X_{t-1} + sum_{j=1}^{p} delta_j*delta_X_{t-j} + eps_t

   Test statistic: tau = gamma_hat / SE(gamma_hat)
   Under H0: gamma = 0 (unit root). Reject H0 if tau < critical value.
   Critical values (MacKinnon 1994) are more negative than standard t:
       1%: -3.43,  5%: -2.86,  10%: -2.57  (with constant, no trend)

4. KPSS TEST (Kwiatkowski-Phillips-Schmidt-Shin)
   Tests H0: stationarity vs H1: unit root (opposite of ADF).
   Based on the partial sum process S_t = sum_{i=1}^{t} epsilon_i:
       KPSS = (1/T^2) * sum_t S_t^2 / sigma^2_LR

   where sigma^2_LR is the long-run variance estimate.
   Reject H0 (stationarity) if KPSS > critical value.

5. COMPLEMENTARY TESTING STRATEGY
   Use ADF and KPSS together:
       ADF reject + KPSS not reject -> stationary (I(0))
       ADF not reject + KPSS reject -> non-stationary (I(1))
       Both reject -> structural break or other issue
       Neither reject -> inconclusive

References:
    Dickey, D.A. and Fuller, W.A. (1979). Distribution of the estimators
        for autoregressive time series with a unit root. JASA, 74, 427-431.
    Kwiatkowski, D. et al. (1992). Testing the null hypothesis of stationarity
        against the alternative of a unit root. Journal of Econometrics, 54, 159-178.
"""

import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.arima_process import arma_generate_sample

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

# =============================================================================
# DATA GENERATION
# =============================================================================
np.random.seed(42)
N = 1000

eps = np.random.normal(0, 0.01, N)

# Random walk (I(1) — non-stationary)
rw = np.cumsum(eps)

# Stationary AR(1) process: X_t = 0.7*X_{t-1} + eps
ar1 = np.zeros(N)
for t in range(1, N):
    ar1[t] = 0.7 * ar1[t-1] + eps[t]

# Stationary mean-reverting (OU-like): phi close to 1 but < 1
ou = np.zeros(N)
for t in range(1, N):
    ou[t] = 0.95 * ou[t-1] + eps[t]

# Actual financial price (log prices) and returns
log_prices = np.cumsum(np.random.normal(0.0003, 0.012, N))
log_returns = np.diff(log_prices)

dates = pd.date_range("2018-01-01", periods=N, freq="B")

# =============================================================================
# UNIT ROOT TESTS
# =============================================================================
def run_tests(series, name):
    """Run ADF and KPSS tests and print results."""
    # ADF
    adf_result = adfuller(series, autolag="AIC")
    adf_stat, adf_p, adf_lags = adf_result[0], adf_result[1], adf_result[2]
    adf_cv = adf_result[4]

    # KPSS
    kpss_result = kpss(series, regression="c", nlags="auto")
    kpss_stat, kpss_p, kpss_lags = kpss_result[0], kpss_result[1], kpss_result[2]
    kpss_cv = kpss_result[3]

    adf_reject  = adf_stat  < adf_cv["5%"]
    kpss_reject = kpss_stat > kpss_cv["5%"]

    if adf_reject and not kpss_reject:
        conclusion = "STATIONARY I(0)"
        col = "GREEN"
    elif not adf_reject and kpss_reject:
        conclusion = "NON-STATIONARY I(1)"
        col = "RED"
    else:
        conclusion = "INCONCLUSIVE"
        col = "YELLOW"

    print(f"\n  {name}")
    print(f"    ADF stat={adf_stat:.4f}  p={adf_p:.4f}  CV5%={adf_cv['5%']:.4f}  "
          f"Reject H0(unit root)={'YES' if adf_reject else 'NO'}")
    print(f"    KPSS stat={kpss_stat:.4f} p={kpss_p:.4f}  CV5%={kpss_cv['5%']:.4f}  "
          f"Reject H0(stationary)={'YES' if kpss_reject else 'NO'}")
    print(f"    => {conclusion}")
    return conclusion

print("=" * 65)
print("  MODULE 06: STATIONARITY — UNIT ROOT TESTING")
print("=" * 65)
c1 = run_tests(rw,          "Random Walk (I(1) expected)")
c2 = run_tests(ar1,         "AR(1) phi=0.70 (Stationary expected)")
c3 = run_tests(ou,          "Near-unit-root AR(1) phi=0.95")
c4 = run_tests(log_prices,  "Log Prices (I(1) expected)")
c5 = run_tests(log_returns, "Log Returns (I(0) expected)")

# =============================================================================
# FIGURE 1: Stationary vs Non-Stationary Processes
# =============================================================================
fig, axes = plt.subplots(3, 2, figsize=(16, 12))
fig.patch.set_facecolor(DARK)
fig.suptitle("M06 — Stationarity: Process Comparison",
             color=WHITE, fontsize=13, fontweight="bold")

series_list = [
    (rw,         "Random Walk I(1)\nVar grows with t", RED),
    (ar1,        "AR(1) phi=0.70\nStationary I(0)", GREEN),
    (ou,         "Near-unit-root phi=0.95\nPersistent but stationary", YELLOW),
    (log_prices, "Log Prices\nNon-stationary I(1)", RED),
    (log_returns,"Log Returns\nStationary I(0)", GREEN),
]

for i, (s, title, col) in enumerate(series_list):
    row, col_idx = divmod(i, 2)
    ax = axes[row, col_idx]
    ax.plot(range(len(s)), s, color=col, lw=0.8, alpha=0.9)
    # Rolling mean and std to show non-stationarity
    s_pd = pd.Series(s)
    roll_mean = s_pd.rolling(50).mean()
    roll_std  = s_pd.rolling(50).std()
    ax.plot(roll_mean.index, roll_mean, color=WHITE, lw=1.5,
            linestyle="--", label="Rolling mean (50d)")
    ax.fill_between(roll_std.index,
                    roll_mean - roll_std, roll_mean + roll_std,
                    color=ACCENT, alpha=0.1, label="±1 rolling std")
    ax.set_title(title, color=WHITE, fontsize=9)
    ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# Hide last subplot
axes[2, 1].set_visible(False)

plt.tight_layout()
path = os.path.join(OUT_DIR, "m06_01_stationarity_comparison.png")
plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"\n[OK] {path}")

# =============================================================================
# FIGURE 2: ADF Test Power & First-Difference Transformation
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.patch.set_facecolor(DARK)
fig.suptitle("M06 — ADF Test Power & First-Difference Transformation",
             color=WHITE, fontsize=13, fontweight="bold")

# ADF statistic as function of phi
ax = axes[0, 0]
phis = np.linspace(0.50, 1.02, 60)
adf_stats_phi = []
for phi in phis:
    x = np.zeros(200)
    e = np.random.normal(0, 0.01, 200)
    for t in range(1, 200):
        x[t] = phi * x[t-1] + e[t]
    try:
        stat = adfuller(x, autolag="AIC")[0]
    except:
        stat = np.nan
    adf_stats_phi.append(stat)

ax.plot(phis, adf_stats_phi, color=ACCENT, lw=2)
ax.axhline(-2.86, color=YELLOW, lw=1.5, linestyle="--", label="5% critical value (-2.86)")
ax.axhline(-3.43, color=RED,    lw=1.5, linestyle=":",  label="1% critical value (-3.43)")
ax.axvline(1.0,   color=WHITE,  lw=1,   linestyle="--", alpha=0.5, label="phi=1 (unit root)")
ax.set_xlabel("AR(1) coefficient phi")
ax.set_ylabel("ADF t-statistic")
ax.set_title("ADF Statistic vs AR Coefficient", color=WHITE)
ax.legend(fontsize=8); ax.grid(True); watermark(ax)

# First difference
ax = axes[0, 1]
rw_diff = np.diff(rw)
ax.plot(range(len(rw)),      rw,      color=RED,   lw=0.8, alpha=0.7, label="Random Walk X_t")
ax2 = ax.twinx()
ax2.plot(range(len(rw_diff)), rw_diff, color=GREEN, lw=0.8, alpha=0.8, label="First Diff dX_t")
ax2.tick_params(colors=MUTED)
ax2.yaxis.label.set_color(WHITE)
ax.set_ylabel("X_t", color=RED)
ax2.set_ylabel("dX_t", color=GREEN)
ax.set_title("First-Difference Transformation: I(1) -> I(0)", color=WHITE)
ax.legend(loc="upper left",  fontsize=8)
ax2.legend(loc="upper right", fontsize=8)
ax.grid(True); watermark(ax)

# Variance of random walk grows with t
ax = axes[1, 0]
t_vals = np.arange(1, 501)
theoretical_var = (0.01**2) * t_vals
n_sim = 200
rw_matrix = np.cumsum(np.random.normal(0, 0.01, (n_sim, 500)), axis=1)
emp_var = np.var(rw_matrix, axis=0)
ax.plot(t_vals, theoretical_var, color=YELLOW, lw=2, label="Theoretical: sigma^2 * t")
ax.plot(t_vals, emp_var,         color=ACCENT, lw=1.5, alpha=0.8, label="Empirical variance")
ax.set_xlabel("Time t")
ax.set_ylabel("Variance")
ax.set_title("Random Walk: Var(X_t) = sigma^2 * t", color=WHITE)
ax.legend(fontsize=8); ax.grid(True); watermark(ax)

# KPSS test statistic visualization
ax = axes[1, 1]
series_kpss = {"RW I(1)": rw, "AR(1) I(0)": ar1, "Log Prices": log_prices, "Returns": log_returns}
kpss_vals   = []
names_k     = []
for nm, s in series_kpss.items():
    try:
        kv = kpss(s, regression="c", nlags="auto")[0]
    except:
        kv = np.nan
    kpss_vals.append(kv)
    names_k.append(nm)

colors_k = [RED if v > 0.463 else GREEN for v in kpss_vals]
bars = ax.bar(names_k, kpss_vals, color=colors_k, alpha=0.8, edgecolor=MUTED)
ax.axhline(0.463, color=YELLOW, lw=2, linestyle="--", label="5% CV = 0.463")
ax.axhline(0.739, color=RED,    lw=1.5, linestyle=":", label="1% CV = 0.739")
ax.set_ylabel("KPSS Statistic")
ax.set_title("KPSS Test: Reject Stationarity if > CV", color=WHITE)
ax.legend(fontsize=8); ax.grid(True); watermark(ax)

plt.tight_layout()
path = os.path.join(OUT_DIR, "m06_02_unit_root_tests.png")
plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"[OK] {path}")

# =============================================================================
# FIGURE 3: Summary Table Visualization
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 5))
fig.patch.set_facecolor(DARK)
ax.set_facecolor(DARK)
ax.set_title("M06 — Unit Root Test Results Summary", color=WHITE, fontsize=13, fontweight="bold")
ax.axis("off")

rows = [
    ["Series",         "ADF stat", "ADF p-val", "Reject H0\n(unit root)", "KPSS stat", "Reject H0\n(stationary)", "Conclusion"],
]
series_results = [
    ("Random Walk",    rw),
    ("AR(1) phi=0.70", ar1),
    ("phi=0.95 (OU)",  ou),
    ("Log Prices",     log_prices),
    ("Log Returns",    log_returns),
]
for nm, s in series_results:
    adf_r = adfuller(s, autolag="AIC")
    try:
        kpss_r = kpss(s, regression="c", nlags="auto")
        kstat, kp = round(kpss_r[0], 3), round(kpss_r[1], 3)
        k_rej = "YES" if kstat > 0.463 else "NO"
    except:
        kstat, kp, k_rej = "N/A", "N/A", "N/A"
    a_rej = "YES" if adf_r[0] < adf_r[4]["5%"] else "NO"
    if a_rej == "YES" and k_rej == "NO":
        concl = "STATIONARY"
    elif a_rej == "NO" and k_rej == "YES":
        concl = "NON-STATIONARY"
    else:
        concl = "MIXED"
    rows.append([nm, round(adf_r[0],3), round(adf_r[1],4), a_rej, kstat, k_rej, concl])

table = ax.table(cellText=rows[1:], colLabels=rows[0],
                 loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.8)
for (r_idx, c_idx), cell in table.get_celld().items():
    cell.set_facecolor(DARK3)
    cell.set_edgecolor(MUTED)
    cell.set_text_props(color=WHITE)
    if r_idx == 0:
        cell.set_facecolor("#1a3a4a")
        cell.set_text_props(color=ACCENT, fontweight="bold")
    if c_idx == 6 and r_idx > 0:
        txt = cell.get_text().get_text()
        cell.set_facecolor("#003300" if txt == "STATIONARY" else
                          "#330000" if txt == "NON-STATIONARY" else "#332200")

watermark(ax)
plt.tight_layout()
path = os.path.join(OUT_DIR, "m06_03_test_summary.png")
plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"[OK] {path}")

print("\n" + "=" * 65)
print("  MODULE 06 COMPLETE — 3 figures saved")
print("  Key Concepts:")
print("  [1] Stationarity: constant mean, variance, autocovariance")
print("  [2] Random walk I(1): Var grows with t, must difference")
print("  [3] ADF: H0=unit root, reject if tau < critical value")
print("  [4] KPSS: H0=stationary, complementary to ADF")
print("  [5] Log prices I(1) -> log returns I(0) via differencing")
print("=" * 65)
