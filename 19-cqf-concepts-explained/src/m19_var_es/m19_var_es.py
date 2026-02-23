#!/usr/bin/env python3
"""
M19 — Value at Risk & Expected Shortfall
=========================================
Module 5 of 9 | CQF Concepts Explained

Theory
------
Value at Risk (VaR) at confidence level alpha:
    VaR_alpha = -inf{x : P(L > x) <= 1 - alpha}
              = -quantile_{1-alpha}(L)   (for continuous L)

    Interpretation: with probability alpha, losses will not exceed VaR.
    Standard levels: alpha = 95%, 99%, 99.9%

Expected Shortfall (ES) / Conditional VaR (CVaR):
    ES_alpha = -E[L | L <= -VaR_alpha]
             = -1/(1-alpha) * integral_{-inf}^{-VaR} x * f_L(x) dx

    ES is the average loss in the worst (1-alpha) fraction of scenarios.
    ES is coherent (satisfies subadditivity) — VaR is NOT coherent.

Three estimation methods:

1. Historical Simulation (HS):
   Use past T returns directly; no distributional assumption.
   VaR = -percentile(returns, (1-alpha)*100)

2. Parametric (Variance-Covariance) method:
   Assume L ~ N(mu, sigma^2):
   VaR_alpha = -(mu + sigma * z_{1-alpha})
   ES_alpha  = -(mu - sigma * phi(z_{1-alpha})/(1-alpha))
   where z = N^{-1}(1-alpha), phi = standard normal PDF

3. Monte Carlo:
   Simulate P&L distribution; compute empirical quantiles.
   Handles fat tails, nonlinear positions (options), correlations.

Backtesting (Basel III):
   Count exceptions (L > VaR) over N days.
   Expected exceptions = N*(1-alpha)
   Traffic light: Green (< 5 exceptions/250d), Yellow, Red

Coherence: ES satisfies all 4 axioms (monotonicity, translation
invariance, positive homogeneity, subadditivity). VaR fails
subadditivity — diversification can appear to INCREASE VaR.

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
from scipy.stats import norm, t as t_dist

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
# Core risk measure functions
# ---------------------------------------------------------------------------
def historical_var_es(returns, alpha=0.99):
    """Historical simulation VaR and ES."""
    var = -np.percentile(returns, (1 - alpha)*100)
    es  = -returns[returns <= -var].mean()
    return var, es

def parametric_var_es(mu, sigma, alpha=0.99, dist="normal", df=5):
    """Parametric VaR/ES under Normal or Student-t."""
    if dist == "normal":
        z   = norm.ppf(1 - alpha)
        var = -(mu + sigma * z)
        es  = -(mu - sigma * norm.pdf(z) / (1 - alpha))
    else:  # Student-t
        z   = t_dist.ppf(1 - alpha, df)
        var = -(mu + sigma * z)
        # ES for Student-t
        es  = -(mu + sigma * (t_dist.pdf(z, df) / (1-alpha))
                            * (df + z**2) / (df - 1))
    return var, es

def mc_var_es(returns, alpha=0.99):
    """Empirical VaR/ES from simulated returns."""
    return historical_var_es(returns, alpha)

# ---------------------------------------------------------------------------
# Simulate synthetic portfolio returns
# ---------------------------------------------------------------------------
SEED = 42
rng  = np.random.default_rng(SEED)

# 3-year daily history (756 days)
N_HIST  = 756
mu_d    = 0.0003      # daily mean (~7.5% annualized)
sigma_d = 0.012       # daily vol (~19% annualized)

# Mix: 80% normal + 20% fat-tail events (Student-t, df=3)
n_norm  = int(0.80 * N_HIST)
n_tail  = N_HIST - n_norm
ret_hist = np.concatenate([
    rng.normal(mu_d, sigma_d, n_norm),
    t_dist.rvs(df=3, loc=mu_d, scale=sigma_d*0.7, size=n_tail,
               random_state=rng)
])
rng.shuffle(ret_hist)

# MC simulated returns (50k scenarios)
N_MC    = 50000
ret_mc  = rng.normal(mu_d, sigma_d, N_MC)

PORTFOLIO = 1_000_000   # USD 1M portfolio
ALPHA     = 0.99

hs_var,  hs_es  = historical_var_es(ret_hist, ALPHA)
par_var, par_es = parametric_var_es(mu_d, sigma_d, ALPHA, "normal")
par_t_var, par_t_es = parametric_var_es(mu_d, sigma_d, ALPHA, "student", df=5)
mc_var,  mc_es  = mc_var_es(ret_mc, ALPHA)

print(f"[M19] Portfolio = USD {PORTFOLIO:,}  |  alpha = {ALPHA:.0%}")
print(f"      HS  VaR = {hs_var:.4f}  ({hs_var*PORTFOLIO:,.0f})  "
      f"ES = {hs_es:.4f}  ({hs_es*PORTFOLIO:,.0f})")
print(f"      Par VaR = {par_var:.4f}  ({par_var*PORTFOLIO:,.0f})  "
      f"ES = {par_es:.4f}  ({par_es*PORTFOLIO:,.0f})")
print(f"      t-Par VaR= {par_t_var:.4f}  ({par_t_var*PORTFOLIO:,.0f})  "
      f"ES = {par_t_es:.4f}  ({par_t_es*PORTFOLIO:,.0f})")
print(f"      MC  VaR = {mc_var:.4f}  ({mc_var*PORTFOLIO:,.0f})  "
      f"ES = {mc_es:.4f}  ({mc_es*PORTFOLIO:,.0f})")


# ===========================================================================
# FIGURE 1 — VaR/ES illustration on return distribution
# ===========================================================================
print("[M19] Figure 1: VaR and ES illustration ...")
t0 = time.perf_counter()

fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=DARK)
fig.suptitle(
    f"M19 — Value at Risk & Expected Shortfall  (alpha={ALPHA:.0%})\n"
    f"Portfolio = USD {PORTFOLIO:,}  |  mu_d={mu_d:.4f}  sigma_d={sigma_d:.4f}",
    color=WHITE, fontsize=10
)

# (0) Historical distribution with VaR/ES marked
ax = axes[0]
n_bins = 60
counts, bins, _ = ax.hist(ret_hist*100, bins=n_bins, density=True,
                           color=BLUE, alpha=0.55, edgecolor="none",
                           label=f"Historical returns  (N={N_HIST})")
tail_mask = ret_hist <= -hs_var
ax.hist(ret_hist[tail_mask]*100, bins=n_bins, density=True,
        color=RED, alpha=0.80, edgecolor="none", label=f"Tail (worst {(1-ALPHA):.0%})")

ax.axvline(-hs_var*100,  color=YELLOW, lw=2.5, linestyle="--",
           label=f"HS VaR = {hs_var*100:.2f}%  (USD {hs_var*PORTFOLIO:,.0f})")
ax.axvline(-hs_es*100,   color=RED,    lw=2.5, linestyle=":",
           label=f"HS ES  = {hs_es*100:.2f}%  (USD {hs_es*PORTFOLIO:,.0f})")
ax.axvline(-par_var*100, color=GREEN,  lw=2,   linestyle="--",
           label=f"Par VaR = {par_var*100:.2f}%")

x_pdf = np.linspace(ret_hist.min()*100, ret_hist.max()*100, 300)
pdf_n = norm.pdf(x_pdf, mu_d*100, sigma_d*100)
ax.plot(x_pdf, pdf_n, color=GREEN, lw=1.5, linestyle="-", alpha=0.7,
        label="Normal PDF fit")

ax.set_xlabel("Daily Return (%)"); ax.set_ylabel("Density")
ax.set_title("Return Distribution\nVaR = quantile  |  ES = tail mean",
             color=WHITE, fontsize=9)
ax.legend(fontsize=6); ax.grid(True); watermark(ax)

# (1) VaR vs confidence level: HS, Normal, Student-t
ax = axes[1]
alphas   = np.linspace(0.90, 0.999, 200)
var_hs_a  = [-np.percentile(ret_hist, (1-a)*100) for a in alphas]
es_hs_a   = []
for a in alphas:
    v = -np.percentile(ret_hist, (1-a)*100)
    tail = ret_hist[ret_hist <= -v]
    es_hs_a.append(-tail.mean() if len(tail) > 0 else v)

var_n_a  = [parametric_var_es(mu_d, sigma_d, a, "normal")[0]  for a in alphas]
es_n_a   = [parametric_var_es(mu_d, sigma_d, a, "normal")[1]  for a in alphas]
var_t_a  = [parametric_var_es(mu_d, sigma_d, a, "student", 5)[0] for a in alphas]
es_t_a   = [parametric_var_es(mu_d, sigma_d, a, "student", 5)[1] for a in alphas]

ax.plot(alphas*100, np.array(var_hs_a)*100,  color=BLUE,   lw=2, label="HS VaR")
ax.plot(alphas*100, np.array(es_hs_a)*100,   color=BLUE,   lw=2, linestyle="--",
        label="HS ES")
ax.plot(alphas*100, np.array(var_n_a)*100,   color=GREEN,  lw=2, label="Normal VaR")
ax.plot(alphas*100, np.array(es_n_a)*100,    color=GREEN,  lw=2, linestyle="--",
        label="Normal ES")
ax.plot(alphas*100, np.array(var_t_a)*100,   color=ORANGE, lw=2, label="t(5) VaR")
ax.plot(alphas*100, np.array(es_t_a)*100,    color=ORANGE, lw=2, linestyle="--",
        label="t(5) ES")
ax.axvline(ALPHA*100, color=WHITE, lw=1.5, linestyle=":", alpha=0.6,
           label=f"alpha={ALPHA:.0%}")
ax.set_xlabel("Confidence Level alpha (%)"); ax.set_ylabel("Risk Measure (daily %)")
ax.set_title("VaR and ES vs Confidence Level\nFat tails diverge at high alpha",
             color=WHITE, fontsize=9)
ax.legend(fontsize=6); ax.grid(True); watermark(ax)

# (2) Subadditivity: VaR vs ES for two correlated portfolios
ax = axes[2]
rho_vals = np.linspace(-1, 1, 100)
s1, s2 = sigma_d, sigma_d * 1.5
z_99 = norm.ppf(1 - ALPHA)

# Individual VaRs (without correlation)
var1 = -z_99 * s1
var2 = -z_99 * s2
sum_var = var1 + var2   # VaR sum (upper bound)

# Portfolio VaR (50/50 weights)
port_var_rho = np.array([
    -z_99 * np.sqrt(0.25*s1**2 + 0.25*s2**2 + 0.5*rho*s1*s2)
    for rho in rho_vals
])

# Portfolio ES (Normal)
def port_es(rho):
    s_p = np.sqrt(0.25*s1**2 + 0.25*s2**2 + 0.5*rho*s1*s2)
    return norm.pdf(z_99)/(1-ALPHA) * s_p

port_es_rho = np.array([port_es(rho) for rho in rho_vals])
es1  = norm.pdf(z_99)/(1-ALPHA) * s1
es2  = norm.pdf(z_99)/(1-ALPHA) * s2

ax.plot(rho_vals, port_var_rho*100, color=RED,   lw=2.5,
        label="Portfolio VaR (50/50)")
ax.plot(rho_vals, port_es_rho*100,  color=GREEN, lw=2.5,
        label="Portfolio ES (50/50)")
ax.axhline(sum_var*100/2, color=RED,   lw=1.5, linestyle="--",
           label=f"VaR1+VaR2 / 2 = {sum_var*100/2:.3f}%")
ax.axhline((es1+es2)*100/2, color=GREEN, lw=1.5, linestyle="--",
           label=f"ES1+ES2 / 2 = {(es1+es2)*100/2:.3f}%")
ax.fill_between(rho_vals,
                port_var_rho*100, sum_var*100/2,
                where=port_var_rho*100 > sum_var*100/2,
                color=RED, alpha=0.15, label="VaR superadditive region")
ax.set_xlabel("Correlation rho"); ax.set_ylabel("Risk Measure (daily %)")
ax.set_title("Coherence: VaR fails Subadditivity\n"
             "ES always subadditive (diversification benefit)",
             color=WHITE, fontsize=9)
ax.legend(fontsize=6); ax.grid(True); watermark(ax)

plt.tight_layout()
p1 = os.path.join(OUT_DIR, "m19_01_var_es_illustration.png")
plt.savefig(p1, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p1}  ({time.perf_counter()-t0:.1f}s)")


# ===========================================================================
# FIGURE 2 — Backtesting: VaR exceptions over time
# ===========================================================================
print("[M19] Figure 2: Backtesting ...")
t0 = time.perf_counter()

# Rolling 1-year (252-day) window backtest
N_BT    = N_HIST
rng_bt  = np.random.default_rng(SEED + 1)
ret_bt  = np.concatenate([
    rng_bt.normal(mu_d, sigma_d, 550),
    # Stress period: higher vol
    rng_bt.normal(mu_d, sigma_d*2.5, 100),
    rng_bt.normal(mu_d, sigma_d, 106),
])[:N_BT]

dates = np.arange(N_BT)

# Rolling parametric 99% VaR (1-day, 1-year lookback)
window = 252
var_rolling = np.full(N_BT, np.nan)
for i in range(window, N_BT):
    w_rets = ret_bt[i-window:i]
    mu_w   = w_rets.mean()
    sig_w  = w_rets.std()
    var_rolling[i] = -mu_w - sig_w * norm.ppf(1 - ALPHA)

# Exceptions: actual loss > VaR
exceptions = (ret_bt < -var_rolling) & ~np.isnan(var_rolling)
n_except = exceptions.sum()
n_days   = (~np.isnan(var_rolling)).sum()
expect   = n_days * (1 - ALPHA)

fig, axes = plt.subplots(2, 2, figsize=(14, 8), facecolor=DARK)
fig.suptitle(
    f"M19 — VaR Backtesting  (alpha={ALPHA:.0%}, rolling {window}-day window)\n"
    f"Exceptions = {n_except}  |  Expected = {expect:.0f}  |  "
    f"Exception rate = {n_except/n_days:.2%}",
    color=WHITE, fontsize=10
)

# (0,0) Returns vs rolling VaR
ax = axes[0, 0]
ax.plot(dates, ret_bt*100, color=BLUE, lw=0.8, alpha=0.6, label="Daily return")
ax.plot(dates, -var_rolling*100, color=YELLOW, lw=1.5,
        linestyle="--", label=f"Rolling {ALPHA:.0%} VaR (negative)")
ax.scatter(dates[exceptions], ret_bt[exceptions]*100,
           color=RED, s=25, zorder=5, label=f"Exceptions (n={n_except})")
ax.axvspan(550, 650, color=RED, alpha=0.08, label="Stress period")
ax.set_xlabel("Day"); ax.set_ylabel("Return (%)")
ax.set_title("Returns vs Rolling VaR", color=WHITE, fontsize=9)
ax.legend(fontsize=6); ax.grid(True); watermark(ax)

# (0,1) Cumulative exceptions vs expected
ax = axes[0, 1]
cum_except = np.cumsum(exceptions.astype(int))
cum_expect = np.arange(N_BT) * (1-ALPHA)
ax.plot(dates, cum_except, color=RED,    lw=2, label="Cumulative exceptions")
ax.plot(dates, cum_expect, color=YELLOW, lw=2, linestyle="--",
        label=f"Expected (slope={1-ALPHA:.3f}/day)")
# Basel traffic light zones at T=252
ax.axhline(4,  color=GREEN,  lw=1, linestyle=":", alpha=0.7, label="Basel Green limit (4)")
ax.axhline(9,  color=ORANGE, lw=1, linestyle=":", alpha=0.7, label="Basel Yellow limit (9)")
ax.axhline(10, color=RED,    lw=1, linestyle=":", alpha=0.7, label="Basel Red limit (10)")
ax.set_xlabel("Day"); ax.set_ylabel("Cumulative exceptions")
ax.set_title("Cumulative Exceptions vs Expected\nBasel III traffic light system",
             color=WHITE, fontsize=9)
ax.legend(fontsize=6); ax.grid(True); watermark(ax)

# (1,0) Rolling VaR over time
ax = axes[1, 0]
ax.plot(dates, var_rolling*100, color=YELLOW, lw=1.5, label="Rolling 99% VaR (%)")
ax.axvspan(550, 650, color=RED, alpha=0.08, label="Stress period")
ax.set_xlabel("Day"); ax.set_ylabel("VaR (%)")
ax.set_title("Rolling VaR Term Structure\n"
             "(spikes during stress period)",
             color=WHITE, fontsize=9)
ax.legend(fontsize=6); ax.grid(True); watermark(ax)

# (1,1) Summary table
ax = axes[1, 1]
ax.axis("off")
rows = [
    ["Method",     "VaR 99%",      "ES 99%",
     "VaR USD",    "ES USD"],
    ["Historical", f"{hs_var:.4f}", f"{hs_es:.4f}",
     f"{hs_var*PORTFOLIO:,.0f}", f"{hs_es*PORTFOLIO:,.0f}"],
    ["Normal",     f"{par_var:.4f}", f"{par_es:.4f}",
     f"{par_var*PORTFOLIO:,.0f}", f"{par_es*PORTFOLIO:,.0f}"],
    ["Student-t5", f"{par_t_var:.4f}", f"{par_t_es:.4f}",
     f"{par_t_var*PORTFOLIO:,.0f}", f"{par_t_es*PORTFOLIO:,.0f}"],
    ["MC Normal",  f"{mc_var:.4f}", f"{mc_es:.4f}",
     f"{mc_var*PORTFOLIO:,.0f}", f"{mc_es*PORTFOLIO:,.0f}"],
    ["", "", "", "", ""],
    ["Backtest",   f"Exceptions={n_except}", f"Expected={expect:.0f}",
     f"Rate={n_except/n_days:.2%}", f"Days={n_days}"],
]
cols_row = [[PANEL]*5] + [
    [DARK if i%2==0 else PANEL]*5 for i in range(len(rows)-1)
]
tbl = ax.table(cellText=rows, cellLoc="center", loc="center",
               cellColours=cols_row)
tbl.auto_set_font_size(False); tbl.set_fontsize(7); tbl.scale(1.0, 1.55)
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor(GREY)
    cell.set_text_props(color=YELLOW if r==0 else WHITE,
                        weight="bold" if r==0 else "normal")
ax.set_title("Risk Measures Summary", color=WHITE, fontsize=9, pad=8)

plt.tight_layout()
p2 = os.path.join(OUT_DIR, "m19_02_backtesting.png")
plt.savefig(p2, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p2}  ({time.perf_counter()-t0:.1f}s)")


# ===========================================================================
# FIGURE 3 — Multi-asset portfolio VaR: MC with correlations
# ===========================================================================
print("[M19] Figure 3: Multi-asset portfolio VaR ...")
t0 = time.perf_counter()

# 4-asset portfolio
ASSETS  = ["Equity", "EM Bonds", "Commodities", "FX"]
weights = np.array([0.40, 0.30, 0.20, 0.10])
mu_vec  = np.array([0.0004, 0.0002, 0.0001, 0.00005])
sig_vec = np.array([0.012,  0.006,  0.018,  0.007])

Rho = np.array([
    [1.00, -0.20,  0.35,  0.15],
    [-0.20, 1.00, -0.10, -0.05],
    [0.35, -0.10,  1.00,  0.20],
    [0.15, -0.05,  0.20,  1.00],
])
Cov = np.diag(sig_vec) @ Rho @ np.diag(sig_vec)

# Individual VaRs
var_indiv = np.array([parametric_var_es(mu_vec[i], sig_vec[i], ALPHA)[0]
                      for i in range(4)])
es_indiv  = np.array([parametric_var_es(mu_vec[i], sig_vec[i], ALPHA)[1]
                      for i in range(4)])

# Portfolio parametric
mu_port   = weights @ mu_vec
sig_port  = np.sqrt(weights @ Cov @ weights)
var_port, es_port = parametric_var_es(mu_port, sig_port, ALPHA)

# Undiversified VaR
var_undiv = weights @ var_indiv
div_benefit = var_undiv - var_port

# MC portfolio VaR
L_chol = np.linalg.cholesky(Cov)
rng_ma = np.random.default_rng(SEED + 2)
Z_ma   = rng_ma.standard_normal((N_MC, 4))
R_ma   = (mu_vec + (L_chol @ Z_ma.T).T)          # (N_MC, 4) asset returns
R_port = R_ma @ weights                            # portfolio returns
var_mc_port, es_mc_port = historical_var_es(R_port, ALPHA)

fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=DARK)
fig.suptitle(
    "M19 — Multi-Asset Portfolio VaR: Diversification & Component VaR\n"
    f"4-Asset Portfolio  |  alpha={ALPHA:.0%}  |  Portfolio = USD {PORTFOLIO:,}",
    color=WHITE, fontsize=10
)

colors_a = [BLUE, GREEN, YELLOW, ORANGE]

# (0) Stacked bar: undiversified vs diversified VaR
ax = axes[0]
component_var = weights * var_indiv     # naive component
bar_x = np.arange(4)
ax.bar(bar_x, component_var*100, color=colors_a, alpha=0.75,
       edgecolor=GREY, label="Component VaR (undiversified)")
ax.axhline(var_port*100,   color=RED,   lw=2.5, linestyle="--",
           label=f"Portfolio VaR = {var_port*100:.3f}%")
ax.axhline(var_undiv*100,  color=WHITE, lw=2,   linestyle=":",
           label=f"Undiversified = {var_undiv*100:.3f}%")
ax.set_xticks(bar_x); ax.set_xticklabels(ASSETS)
ax.set_ylabel("Daily VaR (%)"); 
ax.set_title(f"Component VaR\nDiversification benefit = {div_benefit*100:.4f}%",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True, axis="y"); watermark(ax)

# (1) Portfolio return distribution
ax = axes[1]
ax.hist(R_port*100, bins=80, density=True, color=BLUE, alpha=0.55,
        edgecolor="none", label=f"MC portfolio returns  (N={N_MC:,})")
x_pdf2 = np.linspace(R_port.min()*100, R_port.max()*100, 300)
pdf_p  = norm.pdf(x_pdf2, mu_port*100, sig_port*100)
ax.plot(x_pdf2, pdf_p, color=GREEN, lw=2, label="Normal fit")
ax.axvline(-var_port*100,    color=YELLOW, lw=2.5, linestyle="--",
           label=f"Parametric VaR = {var_port*100:.3f}%")
ax.axvline(-var_mc_port*100, color=ORANGE, lw=2,   linestyle=":",
           label=f"MC VaR = {var_mc_port*100:.3f}%")
ax.axvline(-es_port*100,     color=RED,    lw=2.5, linestyle="--",
           label=f"ES = {es_port*100:.3f}%")
ax.set_xlabel("Daily Portfolio Return (%)"); ax.set_ylabel("Density")
ax.set_title("Portfolio Return Distribution\nParametric vs MC VaR",
             color=WHITE, fontsize=9)
ax.legend(fontsize=6); ax.grid(True); watermark(ax)

# (2) VaR scaling: square-root-of-time rule
ax = axes[2]
horizons = np.arange(1, 31)
var_sqrt  = var_port * np.sqrt(horizons)     # sqrt-of-time
es_sqrt   = es_port  * np.sqrt(horizons)
# MC multi-step (compound)
var_mc_h  = []
es_mc_h   = []
for h in horizons:
    R_h = np.zeros(N_MC)
    rng_h = np.random.default_rng(SEED + h)
    for _ in range(h):
        Z_h  = rng_h.standard_normal((N_MC, 4))
        R_step = (mu_vec + (L_chol @ Z_h.T).T) @ weights
        R_h  += R_step
    v_h, e_h = historical_var_es(R_h, ALPHA)
    var_mc_h.append(v_h)
    es_mc_h.append(e_h)

ax.plot(horizons, np.array(var_sqrt)*100, color=YELLOW, lw=2.5,
        label="VaR (sqrt-of-time rule)")
ax.plot(horizons, np.array(es_sqrt)*100,  color=ORANGE, lw=2.5,
        linestyle="--", label="ES (sqrt-of-time)")
ax.plot(horizons, np.array(var_mc_h)*100, "o", color=BLUE, ms=5,
        label="VaR MC (compounded)")
ax.plot(horizons, np.array(es_mc_h)*100,  "s", color=GREEN, ms=5,
        label="ES MC (compounded)")
ax.set_xlabel("Horizon (days)"); ax.set_ylabel("Risk Measure (daily %)")
ax.set_title("VaR Scaling: sqrt(T) Rule vs MC\n"
             "Valid only for i.i.d. Normal returns",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

plt.tight_layout()
p3 = os.path.join(OUT_DIR, "m19_03_portfolio_var.png")
plt.savefig(p3, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p3}  ({time.perf_counter()-t0:.1f}s)")

print()
print("=" * 65)
print("  MODULE 19 COMPLETE -- 3 figures saved")
print("  Key Concepts:")
print("  [1] VaR = -quantile_{1-alpha}(L)  (not coherent)")
print("  [2] ES = mean of tail losses beyond VaR  (coherent)")
print("  [3] HS: nonparametric, captures fat tails")
print("  [4] Parametric: Normal underestimates tail risk")
print("  [5] Student-t: better fat-tail approximation")
print("  [6] Diversification: portfolio VaR < sum of VaRs")
print(f"  [7] Div. benefit = {div_benefit*100:.4f}%  "
      f"(USD {div_benefit*PORTFOLIO:,.0f})")
print(f"      Backtesting: {n_except} exceptions / {n_days} days "
      f"({n_except/n_days:.2%} vs {1-ALPHA:.2%} expected)")
print("=" * 65)
