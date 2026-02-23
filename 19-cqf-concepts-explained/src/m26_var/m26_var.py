#!/usr/bin/env python3
"""
M26 — Value at Risk: Historical, Parametric & Monte Carlo
==========================================================
Module 26 | CQF Concepts Explained
Group 5   | Risk Management

Theory
------
Value at Risk (VaR) at confidence level alpha and horizon h:
    VaR_{alpha,h} = -inf{x : P(L > x) <= 1 - alpha}
    equivalently: P(L > VaR) = 1 - alpha

where L is the loss (negative P&L) over horizon h.

Three canonical approaches:

1. Parametric (Variance-Covariance) VaR
----------------------------------------
Assume P&L ~ N(mu, sigma^2):
    VaR_param = -(mu - z_alpha * sigma)   z_alpha = N^{-1}(alpha)
For a portfolio with weights w, covariance matrix Sigma:
    sigma_p^2 = w^T Sigma w
    VaR = z_alpha * sqrt(w^T Sigma w) * N  (zero-mean assumption)

Pros: fast, closed-form, easy to decompose (component VaR).
Cons: normality assumption underestimates fat tails.

2. Historical Simulation VaR
------------------------------
Apply today's portfolio weights to the last T historical return
scenarios. VaR = empirical (1-alpha)-quantile of the P&L distribution.
    VaR_hist = -P&L_{(floor((1-alpha)*T))}  (sorted ascending)

Pros: no distributional assumption, captures empirical fat tails and
      skewness, correlations preserved by construction.
Cons: limited by historical window, slow to adapt to regime changes.

3. Monte Carlo VaR
-------------------
Simulate N scenarios of risk factor changes:
    Delta RF ~ N(0, Sigma_RF)   or any calibrated model
Full revaluation of portfolio under each scenario.
VaR = empirical quantile of the simulated P&L distribution.

Pros: handles nonlinear instruments, any distributional model.
Cons: computationally expensive, model risk in simulation engine.

Expected Shortfall (CVaR)
--------------------------
ES_{alpha} = E[L | L > VaR_{alpha}]
           = (1/(1-alpha)) * integral_{alpha}^1 VaR_u du

ES is coherent (satisfies subadditivity), VaR is not.
Under normality: ES = sigma * phi(z_alpha) / (1-alpha) - mu

Scaling Rules
-------------
Square-root-of-time rule (i.i.d. returns):
    VaR_h = VaR_1 * sqrt(h)   (h = holding period in days)
Valid under normality and independence. Breaks down for fat tails
and autocorrelated returns.

Backtesting (Basel III)
------------------------
Kupiec (1995) Proportion of Failures (POF) test:
    H0: p_failure = 1 - alpha
    LR_POF = -2*ln[(1-p*)^(T-x) * p*^x] + 2*ln[(1-x/T)^(T-x) * (x/T)^x]
    ~ chi^2(1) under H0

Basel traffic light: Green <5 exceptions/250days, Yellow 5-9, Red >=10.

References
----------
- Jorion, P. (2006). Value at Risk. 3rd ed. McGraw-Hill.
- Basel Committee (2019). Minimum capital requirements for market risk.
  (FRTB — Fundamental Review of the Trading Book)
- Kupiec, P. (1995). Techniques for verifying the accuracy of risk
  measurement models. Journal of Derivatives, 3(2), 73-84.
- Christoffersen, P. (1998). Evaluating interval forecasts.
  International Economic Review, 39(4), 841-862.
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm, t as student_t, chi2

# ── Styling ──────────────────────────────────────────────────────────────────
DARK   = "#0a0a0a";  PANEL  = "#111111"; GRID   = "#1e1e1e"
WHITE  = "#e8e8e8";  BLUE   = "#4a9eff"; GREEN  = "#00d4aa"
ORANGE = "#ff8c42";  RED    = "#ff4757"; PURPLE = "#a855f7"
YELLOW = "#ffd700"

WATERMARK = "Jose O. Bobadilla | CQF"
OUT_DIR   = os.path.expanduser(
    "~/quant-finance-portfolio/19-cqf-concepts-explained/outputs"
)
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": DARK,   "axes.facecolor":   PANEL,
    "axes.edgecolor":   GRID,   "axes.labelcolor":  WHITE,
    "axes.titlecolor":  WHITE,  "xtick.color":      WHITE,
    "ytick.color":      WHITE,  "text.color":       WHITE,
    "grid.color":       GRID,   "grid.linewidth":   0.6,
    "legend.facecolor": PANEL,  "legend.edgecolor": GRID,
    "font.family":      "monospace",
    "axes.spines.top":  False,  "axes.spines.right": False,
})

def watermark(ax):
    ax.text(0.99, 0.02, WATERMARK, transform=ax.transAxes,
            fontsize=7, color=WHITE, alpha=0.35, ha="right", va="bottom",
            fontstyle="italic")

# ============================================================
# SECTION 1 — RETURN SIMULATION (synthetic multi-asset)
# ============================================================

def simulate_returns(n_assets, n_days, mu_ann, vol_ann, corr_mat, seed=42):
    """
    Simulate daily log returns for a multi-asset portfolio.
    Returns (n_days, n_assets) array.
    """
    rng      = np.random.default_rng(seed)
    mu_daily = mu_ann / 252
    vol_daily = vol_ann / np.sqrt(252)
    L = np.linalg.cholesky(corr_mat)
    Z = rng.standard_normal((n_days, n_assets))
    W = Z @ L.T
    return mu_daily + vol_daily * W

# ── Portfolio parameters ──────────────────────────────────────────────────
N_ASSETS = 4
NAMES    = ["Equity", "EM Bond", "USD/COP FX", "Commodity"]
MU_ANN   = np.array([0.08, 0.05, 0.02, 0.04])
VOL_ANN  = np.array([0.20, 0.10, 0.12, 0.25])
CORR     = np.array([
    [1.00,  0.30,  0.20,  0.40],
    [0.30,  1.00, -0.10,  0.15],
    [0.20, -0.10,  1.00,  0.10],
    [0.40,  0.15,  0.10,  1.00],
])
WEIGHTS  = np.array([0.40, 0.30, 0.15, 0.15])
NOTIONAL = 10_000_000   # USD 10M portfolio

N_HIST = 1000           # historical window (trading days)
rng_returns = simulate_returns(N_ASSETS, N_HIST, MU_ANN, VOL_ANN, CORR)

# Portfolio daily P&L (approximate: w^T * r * N)
pnl_hist = rng_returns @ WEIGHTS * NOTIONAL

# ============================================================
# SECTION 2 — THREE VaR METHODS
# ============================================================

ALPHA = 0.99   # 99% confidence level

# --- Parametric VaR ---
SIGMA_MAT = np.diag(VOL_ANN / np.sqrt(252)) @ CORR @ np.diag(VOL_ANN / np.sqrt(252))
sigma_p   = np.sqrt(WEIGHTS @ SIGMA_MAT @ WEIGHTS)
mu_p_d    = np.sum(WEIGHTS * MU_ANN / 252)
z_alpha   = norm.ppf(ALPHA)
var_param = (z_alpha * sigma_p - mu_p_d) * NOTIONAL
es_param  = ((norm.pdf(z_alpha) / (1 - ALPHA)) * sigma_p - mu_p_d) * NOTIONAL

# --- Historical VaR ---
losses_hist = -np.sort(pnl_hist)          # sorted ascending losses
var_hist    = np.percentile(-pnl_hist, ALPHA * 100)
es_hist     = losses_hist[losses_hist > var_hist].mean()

# --- Monte Carlo VaR ---
N_MC = 50_000
rng_mc = np.random.default_rng(99)
L_chol = np.linalg.cholesky(SIGMA_MAT)
Z_mc   = rng_mc.standard_normal((N_MC, N_ASSETS))
r_mc   = Z_mc @ L_chol.T + MU_ANN / 252
pnl_mc = r_mc @ WEIGHTS * NOTIONAL
var_mc = np.percentile(-pnl_mc, ALPHA * 100)
es_mc  = (-pnl_mc)[-pnl_mc > var_mc].mean()

# --- Component VaR ---
# CVaR_i = w_i * rho_{i,p} * sigma_p * z_alpha * N
cov_ip      = SIGMA_MAT @ WEIGHTS          # Cov(asset_i, portfolio)
rho_ip      = cov_ip / sigma_p
comp_var    = WEIGHTS * rho_ip * z_alpha * NOTIONAL

print("[M26] Portfolio VaR Summary (99%, 1-day, N=10M USD)")
print(f"      Parametric VaR  : USD {var_param:>12,.0f}  | ES: {es_param:>12,.0f}")
print(f"      Historical VaR  : USD {var_hist:>12,.0f}  | ES: {es_hist:>12,.0f}")
print(f"      Monte Carlo VaR : USD {var_mc:>12,.0f}  | ES: {es_mc:>12,.0f}")
print(f"      sigma_portfolio = {sigma_p*100:.4f}% daily")
print(f"\n      Component VaR (parametric, 99%):")
for i, (nm, cv) in enumerate(zip(NAMES, comp_var)):
    print(f"        {nm:15s}: USD {cv:>10,.0f}  ({cv/var_param*100:.1f}% of total)")

# ============================================================
# SECTION 3 — BACKTESTING
# ============================================================

def kupiec_pof_test(exceptions, n_obs, alpha):
    """
    Kupiec (1995) Proportion of Failures LR test.
    Returns (LR statistic, p-value, result string).
    """
    p_hat = exceptions / n_obs
    p0    = 1 - alpha
    if p_hat == 0:
        lr = -2 * n_obs * np.log(1 - p0)
    elif p_hat == 1:
        lr = -2 * n_obs * np.log(p0)
    else:
        lr = -2 * (n_obs*np.log(1-p0) + exceptions*np.log(p0/p_hat)
                   + (n_obs-exceptions)*np.log((1-p0)/(1-p_hat)))
    pval = 1 - chi2.cdf(lr, df=1)
    result = "PASS" if pval > 0.05 else "FAIL"
    return lr, pval, result

# Simulate out-of-sample P&L for backtesting (250 days)
rng_bt  = np.random.default_rng(7)
Z_bt    = rng_bt.standard_normal((250, N_ASSETS))
pnl_bt  = (Z_bt @ np.linalg.cholesky(SIGMA_MAT).T + MU_ANN/252) @ WEIGHTS * NOTIONAL
excep_param = np.sum(-pnl_bt > var_param)
excep_hist  = np.sum(-pnl_bt > var_hist)
excep_mc    = np.sum(-pnl_bt > var_mc)

lr_p, pv_p, res_p = kupiec_pof_test(excep_param, 250, ALPHA)
lr_h, pv_h, res_h = kupiec_pof_test(excep_hist,  250, ALPHA)
lr_m, pv_m, res_m = kupiec_pof_test(excep_mc,    250, ALPHA)

print(f"\n[M26] Kupiec Backtesting (250 days, expected exceptions: {250*(1-ALPHA):.1f})")
print(f"      Parametric: {excep_param} exceptions  LR={lr_p:.3f}  p={pv_p:.4f}  {res_p}")
print(f"      Historical: {excep_hist} exceptions  LR={lr_h:.3f}  p={pv_h:.4f}  {res_h}")
print(f"      Monte Carlo:{excep_mc} exceptions  LR={lr_m:.3f}  p={pv_m:.4f}  {res_m}")

# ============================================================
# FIGURE 1 — P&L distributions and three VaR estimates
# ============================================================
t0 = time.perf_counter()
print("\n[M26] Figure 1: P&L distributions and VaR comparison ...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor(DARK)
fig.suptitle("M26 — Value at Risk: Three Approaches\n"
             f"99% VaR | 1-day | N=USD 10M | alpha={ALPHA}",
             color=WHITE, fontsize=12, fontweight="bold")

# (a) Historical P&L distribution
ax = axes[0]
ax.hist(pnl_hist/1e3, bins=60, density=True, color=BLUE, alpha=0.6,
        label="Historical P&L")
x_norm = np.linspace(pnl_hist.min()/1e3, pnl_hist.max()/1e3, 300)
ax.plot(x_norm, norm.pdf(x_norm, mu_p_d*NOTIONAL/1e3,
                          sigma_p*NOTIONAL/1e3) * 1e3,
        color=ORANGE, lw=2, label="Normal fit")
ax.axvline(-var_hist/1e3, color=RED,    lw=2, linestyle="--",
           label=f"Hist VaR = {var_hist/1e3:.1f}k")
ax.axvline(-var_param/1e3, color=GREEN, lw=2, linestyle=":",
           label=f"Param VaR = {var_param/1e3:.1f}k")
ax.set_xlabel("Daily P&L (USD thousands)"); ax.set_ylabel("Density")
ax.set_title("Historical P&L Distribution\nvs Normal Approximation",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (b) Monte Carlo P&L distribution
ax = axes[1]
ax.hist(pnl_mc/1e3, bins=80, density=True, color=GREEN, alpha=0.5,
        label="MC P&L (50k scenarios)")
ax.axvline(-var_mc/1e3,    color=RED,    lw=2, linestyle="--",
           label=f"MC VaR = {var_mc/1e3:.1f}k")
ax.axvline(-es_mc/1e3,     color=ORANGE, lw=2, linestyle=":",
           label=f"MC ES = {es_mc/1e3:.1f}k")
# Shade tail
bins_mc = np.linspace(pnl_mc.min()/1e3, pnl_mc.max()/1e3, 200)
ax.fill_between(bins_mc[bins_mc < -var_mc/1e3],
                norm.pdf(bins_mc[bins_mc < -var_mc/1e3],
                         mu_p_d*NOTIONAL/1e3, sigma_p*NOTIONAL/1e3)*1e3,
                color=RED, alpha=0.25, label="Tail (1%)")
ax.set_xlabel("Daily P&L (USD thousands)"); ax.set_ylabel("Density")
ax.set_title("Monte Carlo P&L Distribution\nVaR and Expected Shortfall",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (c) VaR comparison across methods and confidence levels
ax = axes[2]
alphas = np.linspace(0.90, 0.999, 100)
var_p_a = [(norm.ppf(a)*sigma_p - mu_p_d)*NOTIONAL/1e3 for a in alphas]
var_h_a = [np.percentile(-pnl_hist, a*100)/1e3 for a in alphas]
var_m_a = [np.percentile(-pnl_mc,   a*100)/1e3 for a in alphas]
ax.plot(alphas*100, var_p_a, color=GREEN,  lw=2.5, label="Parametric (Normal)")
ax.plot(alphas*100, var_h_a, color=BLUE,   lw=2.5, label="Historical")
ax.plot(alphas*100, var_m_a, color=ORANGE, lw=2.5, label="Monte Carlo")
ax.axvline(ALPHA*100, color=WHITE, lw=1, linestyle=":", alpha=0.6,
           label=f"Base alpha={ALPHA*100:.0f}%")
ax.set_xlabel("Confidence level (%)"); ax.set_ylabel("VaR (USD thousands)")
ax.set_title("VaR vs Confidence Level\nParametric < Historical at high alpha",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

plt.tight_layout()
p1 = os.path.join(OUT_DIR, "m26_01_var_distributions.png")
plt.savefig(p1, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p1}  ({time.perf_counter()-t0:.1f}s)")

# ============================================================
# FIGURE 2 — Component VaR and time-scaling
# ============================================================
t0 = time.perf_counter()
print("[M26] Figure 2: Component VaR, ES, and time scaling ...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor(DARK)
fig.suptitle("M26 — Component VaR | ES vs VaR | Time Scaling",
             color=WHITE, fontsize=12, fontweight="bold")

colors_comp = [BLUE, GREEN, ORANGE, PURPLE]

# (a) Component VaR waterfall
ax = axes[0]
ax.bar(NAMES, comp_var/1e3, color=colors_comp, alpha=0.85)
ax.axhline(var_param/1e3, color=RED, lw=2, linestyle="--",
           label=f"Total VaR = {var_param/1e3:.1f}k")
for i, (nm, cv) in enumerate(zip(NAMES, comp_var)):
    ax.text(i, cv/1e3 + 0.5, f"{cv/var_param*100:.0f}%",
            ha="center", fontsize=8, color=WHITE)
ax.set_ylabel("Component VaR (USD thousands)")
ax.set_title("Component VaR Decomposition\nw_i * rho_{i,p} * sigma_p * z * N",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True, axis="y"); watermark(ax)

# (b) VaR vs ES across confidence levels
ax = axes[1]
es_p_a = [(norm.pdf(norm.ppf(a))/(1-a)*sigma_p - mu_p_d)*NOTIONAL/1e3
          for a in alphas]
es_h_a = [(-pnl_hist[-pnl_hist > np.percentile(-pnl_hist, a*100)]).mean()/1e3
          if np.sum(-pnl_hist > np.percentile(-pnl_hist, a*100)) > 0 else np.nan
          for a in alphas]
ax.plot(alphas*100, var_p_a, color=BLUE,   lw=2,   linestyle="--", label="VaR Parametric")
ax.plot(alphas*100, es_p_a,  color=BLUE,   lw=2.5, label="ES Parametric")
ax.plot(alphas*100, var_h_a, color=ORANGE, lw=2,   linestyle="--", label="VaR Historical")
ax.plot(alphas*100, es_h_a,  color=ORANGE, lw=2.5, label="ES Historical")
ax.fill_between(alphas*100, var_p_a, es_p_a, color=BLUE, alpha=0.12)
ax.set_xlabel("Confidence level (%)"); ax.set_ylabel("Risk measure (USD thousands)")
ax.set_title("VaR vs Expected Shortfall\nES >= VaR, ES is coherent",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (c) Time scaling: sqrt(h) rule vs empirical
ax = axes[2]
horizons = np.arange(1, 22)
var_scaled = var_param / 1e3 * np.sqrt(horizons)
# Empirical: block bootstrap (approx: resample pnl_hist in h-day blocks)
var_emp = []
for h in horizons:
    n_blocks = len(pnl_hist) // h
    block_pnl = [pnl_hist[i*h:(i+1)*h].sum() for i in range(n_blocks)]
    var_emp.append(np.percentile(-np.array(block_pnl), ALPHA*100)/1e3)

ax.plot(horizons, var_scaled, color=BLUE,   lw=2.5, label="sqrt(h) scaling")
ax.plot(horizons, var_emp,    color=ORANGE, lw=2.5, marker="o", ms=4,
        label="Empirical (block resample)")
ax.fill_between(horizons, var_scaled, var_emp,
                color=PURPLE, alpha=0.15, label="Scaling error")
ax.set_xlabel("Holding period h (days)"); ax.set_ylabel("VaR (USD thousands)")
ax.set_title("Time Scaling: sqrt(h) Rule\nVaR_h = VaR_1 * sqrt(h)  (i.i.d. assumption)",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

plt.tight_layout()
p2 = os.path.join(OUT_DIR, "m26_02_component_scaling.png")
plt.savefig(p2, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p2}  ({time.perf_counter()-t0:.1f}s)")

# ============================================================
# FIGURE 3 — Backtesting
# ============================================================
t0 = time.perf_counter()
print("[M26] Figure 3: Backtesting — exceptions and Kupiec test ...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor(DARK)
fig.suptitle("M26 — VaR Backtesting\n"
             "Kupiec POF Test | Basel Traffic Light | Exception Analysis",
             color=WHITE, fontsize=12, fontweight="bold")

days = np.arange(1, 251)
pnl_bt_k = pnl_bt / 1e3

# (a) Backtesting chart: P&L vs -VaR threshold
ax = axes[0]
excep_mask = -pnl_bt > var_param
ax.bar(days, pnl_bt_k, color=np.where(excep_mask, RED, GREEN),
       alpha=0.6, width=1.0)
ax.axhline(-var_param/1e3, color=RED, lw=2, linestyle="--",
           label=f"99% VaR = {var_param/1e3:.1f}k")
ax.scatter(days[excep_mask], pnl_bt_k[excep_mask], color=RED, s=40,
           zorder=5, label=f"Exceptions: {excep_param}")
ax.set_xlabel("Day"); ax.set_ylabel("P&L (USD thousands)")
ax.set_title(f"Parametric VaR Backtest (250 days)\n"
             f"{excep_param} exceptions  (expected: {250*(1-ALPHA):.1f})",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True, axis="y"); watermark(ax)

# (b) Kupiec test: critical region and LR statistic
ax = axes[1]
exc_range = np.arange(0, 25)
lr_arr    = [kupiec_pof_test(x, 250, ALPHA)[0] for x in exc_range]
pv_arr    = [kupiec_pof_test(x, 250, ALPHA)[1] for x in exc_range]
critical  = chi2.ppf(0.95, df=1)
colors_bar = [GREEN if lr < critical else RED for lr in lr_arr]
ax.bar(exc_range, lr_arr, color=colors_bar, alpha=0.8)
ax.axhline(critical, color=YELLOW, lw=2, linestyle="--",
           label=f"chi2(1) 95% = {critical:.2f}")
ax.scatter([excep_param], [lr_p], color=WHITE, s=80, zorder=5,
           label=f"Param: LR={lr_p:.2f} ({res_p})")
ax.set_xlabel("Number of exceptions"); ax.set_ylabel("Kupiec LR statistic")
ax.set_title("Kupiec POF Test — Critical Region\nGreen=PASS  Red=FAIL",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True, axis="y"); watermark(ax)

# (c) Basel traffic light zones
ax = axes[2]
# Expected number of exceptions at 99% VaR over 250 days ~ 2.5
zone_x = [0, 4, 9, 25]
zone_colors = [GREEN, YELLOW, RED]
zone_labels = ["Green zone\n(0-4)", "Yellow zone\n(5-9)", "Red zone\n(10+)"]
for i in range(3):
    ax.barh(0, zone_x[i+1]-zone_x[i], left=zone_x[i], height=0.4,
            color=zone_colors[i], alpha=0.7, label=zone_labels[i])
# Mark each model
for exc_, lbl_, col_ in [
    (excep_param, f"Parametric ({excep_param})", BLUE),
    (excep_hist,  f"Historical ({excep_hist})",  ORANGE),
    (excep_mc,    f"Monte Carlo ({excep_mc})",   PURPLE),
]:
    ax.scatter([exc_], [0], s=150, color=col_, zorder=5, label=lbl_)
ax.axvline(250*(1-ALPHA), color=WHITE, lw=1.5, linestyle=":",
           label=f"Expected = {250*(1-ALPHA):.1f}")
ax.set_xlim(0, 25); ax.set_ylim(-0.5, 0.5)
ax.set_xlabel("Number of exceptions in 250 trading days")
ax.set_yticks([]); ax.set_title(
    "Basel III Traffic Light\n250-day backtest, 99% VaR",
    color=WHITE, fontsize=9)
ax.legend(fontsize=7, loc="upper right"); ax.grid(True, axis="x"); watermark(ax)

plt.tight_layout()
p3 = os.path.join(OUT_DIR, "m26_03_backtesting.png")
plt.savefig(p3, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p3}  ({time.perf_counter()-t0:.1f}s)")

# ── Summary ──────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("  MODULE 26 COMPLETE -- 3 figures saved")
print("  Key Concepts:")
print("  [1] VaR_param = (z_alpha*sigma - mu) * N  (normality)")
print("  [2] VaR_hist  = empirical percentile of observed P&L")
print("  [3] VaR_MC    = percentile of full-revaluation MC P&L")
print("  [4] ES = E[L|L>VaR]  (coherent, ES >= VaR always)")
print("  [5] Component VaR: sum = total VaR (diversification)")
print("  [6] Kupiec: LR ~ chi2(1) under H0: p_fail = 1-alpha")
print(f"  99% 1-day VaR  (param): USD {var_param:>10,.0f}")
print(f"  99% 1-day VaR  (hist):  USD {var_hist:>10,.0f}")
print(f"  99% 1-day VaR  (MC):    USD {var_mc:>10,.0f}")
print(f"  99% 1-day ES   (param): USD {es_param:>10,.0f}")
print(f"  Backtesting: {excep_param} exceptions / 250 days  "
      f"[expected {250*(1-ALPHA):.1f}]  Kupiec: {res_p}")
print("=" * 65)
