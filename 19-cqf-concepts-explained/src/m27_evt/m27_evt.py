#!/usr/bin/env python3
"""
M27 — Extreme Value Theory: GEV and GPD
========================================
Module 27 | CQF Concepts Explained
Group 5   | Risk Management

Theory
------
Classical statistics focuses on the mean. EVT focuses on the tails —
the statistical behavior of rare, extreme events.

Fisher-Tippett-Gnedenko Theorem (Block Maxima)
-----------------------------------------------
Let M_n = max(X_1, ..., X_n) be the maximum of n i.i.d. r.v.s.
Under regularity conditions, the normalized maximum converges in
distribution to the Generalized Extreme Value (GEV) distribution:

    P((M_n - b_n)/a_n <= x) -> G_{xi}(x)

    G_{xi}(x) = exp(-(1 + xi*(x-mu)/sigma)^{-1/xi})  if xi != 0
    G_0(x)    = exp(-exp(-(x-mu)/sigma))               if xi = 0 (Gumbel)

Three families determined by the tail index xi (shape parameter):
    xi > 0 : Fréchet — heavy tail (power law), e.g. equity returns
    xi = 0 : Gumbel  — light tail (exponential), e.g. normal-like
    xi < 0 : Weibull — bounded tail (finite right endpoint)

Pickands-Balkema-de Haan Theorem (POT Method)
----------------------------------------------
For large enough threshold u, exceedances (X - u | X > u) follow
the Generalized Pareto Distribution (GPD):

    F_{xi,beta}(y) = 1 - (1 + xi*y/beta)^{-1/xi}  if xi != 0
    F_{0,beta}(y)  = 1 - exp(-y/beta)               if xi = 0

where y = x - u >= 0, beta > 0 is the scale parameter.

The GPD tail index xi is the same as the GEV tail index.

Threshold Selection
--------------------
Mean Excess Function (MEF):
    e(u) = E[X - u | X > u]
For GPD: e(u) = (beta + xi*u) / (1 - xi)  — linear in u.
A plot of empirical MEF vs u should become linear above the optimal u.

Extreme Risk Measures
---------------------
VaR at level alpha (very high, alpha -> 1) via GPD:
    VaR_alpha = u + (beta/xi) * [((1-alpha)/N_u * N)^{-xi} - 1]

where N_u = number of exceedances above u, N = total sample size.

Expected Shortfall:
    ES_alpha = VaR_alpha / (1 - xi) + (beta - xi*u) / (1 - xi)

Return Level x_m (level exceeded once every m observations):
    x_m = u + (beta/xi) * [(m * N_u/N)^xi - 1]

References
----------
- Embrechts, P., Kluppelberg, C., Mikosch, T. (1997). Modelling
  Extremal Events. Springer. (The definitive EVT reference)
- McNeil, A.J., Frey, R., Embrechts, P. (2015). Quantitative Risk
  Management. Princeton University Press.
- Coles, S. (2001). An Introduction to Statistical Modeling of
  Extreme Values. Springer.
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import genextreme, genpareto, norm
from scipy.optimize import minimize

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
# SECTION 1 — SYNTHETIC LOSS DATA (fat-tailed returns)
# ============================================================

def simulate_fat_tail_losses(n, seed=42):
    """
    Simulate daily losses from a Student-t(4) process (fat tails, xi~0.25).
    Losses = positive values (left tail of returns).
    """
    rng = np.random.default_rng(seed)
    # Student-t(4): tail index xi = 1/nu = 0.25
    from scipy.stats import t as student_t
    returns = student_t.rvs(df=4, loc=0, scale=0.01, size=n, random_state=rng)
    losses  = -returns   # losses are positive for negative returns
    return losses

N_OBS = 5000
losses = simulate_fat_tail_losses(N_OBS)
print(f"[M27] Simulated {N_OBS} daily losses (Student-t(4), scale=1%)")
print(f"      Mean loss: {losses.mean()*100:.4f}%  |  "
      f"Std: {losses.std()*100:.4f}%  |  "
      f"Max: {losses.max()*100:.4f}%")

# ============================================================
# SECTION 2 — BLOCK MAXIMA (GEV)
# ============================================================

def block_maxima(data, block_size=63):
    """Extract quarterly (63-day) block maxima."""
    n_blocks = len(data) // block_size
    maxima   = np.array([data[i*block_size:(i+1)*block_size].max()
                         for i in range(n_blocks)])
    return maxima

maxima = block_maxima(losses, block_size=63)

# Fit GEV via scipy (shape=xi, loc=mu, scale=sigma)
xi_gev, mu_gev, sigma_gev = genextreme.fit(maxima)
print(f"\n[M27] GEV fit to quarterly block maxima ({len(maxima)} blocks):")
print(f"      xi={xi_gev:.4f}  mu={mu_gev:.6f}  sigma={sigma_gev:.6f}")
gev_family = ("Fréchet (heavy tail)" if xi_gev > 0.01 else
              "Gumbel (light tail)"  if abs(xi_gev) < 0.01 else
              "Weibull (bounded)")
print(f"      Family: {gev_family}")

# ============================================================
# SECTION 3 — PEAKS OVER THRESHOLD (GPD)
# ============================================================

def mean_excess_function(data, u_grid):
    """
    Empirical MEF: e(u) = mean(data[data>u] - u).
    Theoretical GPD MEF is linear: (beta + xi*u) / (1 - xi).
    """
    mef = []
    for u in u_grid:
        excess = data[data > u] - u
        mef.append(excess.mean() if len(excess) > 10 else np.nan)
    return np.array(mef)

def fit_gpd(excesses):
    """
    Fit GPD to excesses via MLE.
    scipy genpareto: shape=xi, loc=0, scale=beta.
    """
    xi_gpd, _, beta_gpd = genpareto.fit(excesses, floc=0)
    return xi_gpd, beta_gpd

def extreme_var(alpha, u, xi, beta, N_u, N):
    """EVT-based VaR at extreme confidence level alpha."""
    if xi == 0:
        return u - beta * np.log((1-alpha)*N/N_u)
    return u + (beta/xi) * (((1-alpha)*N/N_u)**(-xi) - 1)

def extreme_es(var_alpha, u, xi, beta):
    """EVT-based Expected Shortfall."""
    return (var_alpha + beta - xi*u) / (1 - xi)

# Threshold selection: choose u at ~90th percentile
u_thresh = np.percentile(losses, 90)
exceedances = losses[losses > u_thresh] - u_thresh
N_u = len(exceedances); N = len(losses)
xi_gpd, beta_gpd = fit_gpd(exceedances)

print(f"\n[M27] GPD fit — threshold u = {u_thresh*100:.4f}% "
      f"({N_u} exceedances, {N_u/N*100:.1f}% of sample):")
print(f"      xi={xi_gpd:.4f}  beta={beta_gpd:.6f}")

# Extreme quantile estimates
for alpha in [0.99, 0.995, 0.999, 0.9999]:
    v = extreme_var(alpha, u_thresh, xi_gpd, beta_gpd, N_u, N)
    e = extreme_es(v, u_thresh, xi_gpd, beta_gpd)
    v_norm = norm.ppf(alpha) * losses.std()
    print(f"      alpha={alpha:.4f}: EVT VaR={v*100:.4f}%  "
          f"ES={e*100:.4f}%  |  Normal VaR={v_norm*100:.4f}%")

# ============================================================
# FIGURE 1 — GEV families and block maxima fit
# ============================================================
t0 = time.perf_counter()
print("\n[M27] Figure 1: GEV families and block maxima ...")

x_gev = np.linspace(-3, 6, 500)
gev_shapes = [
    ("Fréchet (xi=0.5)",  0.5,  GREEN),
    ("Fréchet (xi=0.2)",  0.2,  BLUE),
    ("Gumbel  (xi=0.0)",  0.0,  WHITE),
    ("Weibull (xi=-0.2)", -0.2, ORANGE),
    ("Weibull (xi=-0.5)", -0.5, RED),
]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor(DARK)
fig.suptitle("M27 — Extreme Value Theory: GEV Families\n"
             "Fisher-Tippett-Gnedenko Theorem",
             color=WHITE, fontsize=12, fontweight="bold")

# (a) GEV PDFs
ax = axes[0]
for label, xi_, col in gev_shapes:
    pdf = genextreme.pdf(x_gev, xi_, loc=0, scale=1)
    ax.plot(x_gev, pdf, color=col, lw=2, label=label)
ax.set_xlabel("x"); ax.set_ylabel("PDF")
ax.set_title("GEV Families: Fréchet | Gumbel | Weibull\n"
             "xi > 0: heavy tail  |  xi = 0: Gumbel  |  xi < 0: bounded",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (b) GEV survival functions (tail focus)
ax = axes[1]
x_tail = np.linspace(2, 8, 300)
for label, xi_, col in gev_shapes:
    sf = genextreme.sf(x_tail, xi_, loc=0, scale=1)
    ax.semilogy(x_tail, sf, color=col, lw=2, label=label)
ax.set_xlabel("x (tail region)"); ax.set_ylabel("P(X > x)  [log scale]")
ax.set_title("Tail Survival Functions (log scale)\n"
             "Fréchet tail decays slowest (power law)",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (c) Block maxima fit
ax = axes[2]
x_fit = np.linspace(maxima.min()*0.8, maxima.max()*1.2, 200)
pdf_fit = genextreme.pdf(x_fit, xi_gev, loc=mu_gev, scale=sigma_gev)
ax.hist(maxima, bins=15, density=True, color=BLUE, alpha=0.6,
        label=f"Quarterly maxima ({len(maxima)} blocks)")
ax.plot(x_fit, pdf_fit, color=ORANGE, lw=2.5,
        label=f"GEV fit: xi={xi_gev:.3f}")
ax.set_xlabel("Block maximum loss"); ax.set_ylabel("Density")
ax.set_title(f"GEV Fit to Block Maxima\n{gev_family}",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

plt.tight_layout()
p1 = os.path.join(OUT_DIR, "m27_01_gev.png")
plt.savefig(p1, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p1}  ({time.perf_counter()-t0:.1f}s)")

# ============================================================
# FIGURE 2 — POT method: MEF, threshold, GPD fit
# ============================================================
t0 = time.perf_counter()
print("[M27] Figure 2: POT method — MEF and GPD fit ...")

u_grid = np.percentile(losses, np.linspace(50, 97, 80))
mef    = mean_excess_function(losses, u_grid)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor(DARK)
fig.suptitle("M27 — Peaks Over Threshold (POT)\n"
             "Mean Excess Function | GPD Fit | QQ-Plot",
             color=WHITE, fontsize=12, fontweight="bold")

# (a) Mean Excess Function
ax = axes[0]
valid = ~np.isnan(mef)
ax.plot(u_grid[valid]*100, mef[valid]*100, color=BLUE, lw=2)
ax.axvline(u_thresh*100, color=ORANGE, lw=2, linestyle="--",
           label=f"Selected u = {u_thresh*100:.3f}%")
# Theoretical GPD MEF: (beta + xi*u) / (1-xi)
mef_theory = (beta_gpd + xi_gpd*(u_grid[valid]-u_thresh)) / (1-xi_gpd)
ax.plot(u_grid[valid]*100, mef_theory*100, color=GREEN, lw=2,
        linestyle=":", label="GPD MEF (linear above u)")
ax.set_xlabel("Threshold u (% loss)"); ax.set_ylabel("Mean excess e(u) (%)")
ax.set_title("Mean Excess Function\nLinear above u => GPD approximation valid",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (b) GPD fit to exceedances
ax = axes[1]
x_exc = np.linspace(0, exceedances.max()*1.1, 300)
pdf_gpd = genpareto.pdf(x_exc, xi_gpd, loc=0, scale=beta_gpd)
ax.hist(exceedances, bins=40, density=True, color=GREEN, alpha=0.55,
        label=f"Exceedances (N_u={N_u})")
ax.plot(x_exc, pdf_gpd, color=ORANGE, lw=2.5,
        label=f"GPD: xi={xi_gpd:.4f}, beta={beta_gpd:.5f}")
ax.set_xlabel("Excess y = X - u"); ax.set_ylabel("Density")
ax.set_title(f"GPD Fit to Excesses over u={u_thresh*100:.3f}%\n"
             f"Pickands-Balkema-de Haan Theorem",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (c) GPD probability plot (QQ plot)
ax = axes[2]
n_exc  = len(exceedances)
exc_sorted = np.sort(exceedances)
probs  = (np.arange(1, n_exc+1)) / (n_exc + 1)
q_gpd  = genpareto.ppf(probs, xi_gpd, loc=0, scale=beta_gpd)
ax.scatter(q_gpd*100, exc_sorted*100, s=8, alpha=0.6, color=PURPLE,
           label="Empirical vs GPD quantiles")
max_q = max(q_gpd.max(), exc_sorted.max()) * 100
ax.plot([0, max_q], [0, max_q], color=WHITE, lw=2, linestyle="--",
        label="45-degree line (perfect fit)")
ax.set_xlabel("GPD theoretical quantile (%)"); ax.set_ylabel("Empirical quantile (%)")
ax.set_title("GPD Probability Plot\nPoints on diagonal = good fit",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

plt.tight_layout()
p2 = os.path.join(OUT_DIR, "m27_02_pot_gpd.png")
plt.savefig(p2, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p2}  ({time.perf_counter()-t0:.1f}s)")

# ============================================================
# FIGURE 3 — Extreme VaR, ES and return levels
# ============================================================
t0 = time.perf_counter()
print("[M27] Figure 3: Extreme VaR, ES, and return levels ...")

alphas_ext = np.linspace(0.90, 0.9999, 200)
var_evt    = [extreme_var(a, u_thresh, xi_gpd, beta_gpd, N_u, N)
              for a in alphas_ext]
es_evt     = [extreme_es(v, u_thresh, xi_gpd, beta_gpd) for v in var_evt]
var_norm   = [norm.ppf(a) * losses.std() for a in alphas_ext]

# Return levels (observations per exceedance)
m_arr = np.logspace(1, 4, 100)   # 10 to 10000 observations
rl    = [extreme_var(1-1/m, u_thresh, xi_gpd, beta_gpd, N_u, N)
         for m in m_arr]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor(DARK)
fig.suptitle("M27 — Extreme Risk Measures\n"
             "EVT VaR vs Normal | Return Levels | Tail Extrapolation",
             color=WHITE, fontsize=12, fontweight="bold")

# (a) EVT vs Normal VaR
ax = axes[0]
ax.plot(alphas_ext*100, np.array(var_evt)*100,  color=RED,    lw=2.5,
        label="EVT VaR (GPD tail)")
ax.plot(alphas_ext*100, np.array(es_evt)*100,   color=ORANGE, lw=2.5,
        linestyle="--", label="EVT ES")
ax.plot(alphas_ext*100, np.array(var_norm)*100, color=BLUE,   lw=2,
        label="Normal VaR")
ax.fill_between(alphas_ext*100,
                np.array(var_norm)*100, np.array(var_evt)*100,
                color=RED, alpha=0.15, label="EVT excess over Normal")
ax.set_xlabel("Confidence level (%)"); ax.set_ylabel("VaR / ES (%)")
ax.set_title("EVT vs Normal VaR\n"
             "Fat tail => EVT >> Normal at extreme quantiles",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (b) Return level plot
ax = axes[1]
ax.semilogx(m_arr, np.array(rl)*100, color=GREEN, lw=2.5,
            label="EVT return level")
ax.fill_between(m_arr, np.array(rl)*100, color=GREEN, alpha=0.12)
for m_, lbl_ in [(252,"1Y"), (252*5,"5Y"), (252*10,"10Y"), (252*25,"25Y")]:
    v_ = extreme_var(1-1/m_, u_thresh, xi_gpd, beta_gpd, N_u, N)
    ax.scatter([m_], [v_*100], color=YELLOW, s=60, zorder=5)
    ax.annotate(f"{lbl_}: {v_*100:.2f}%", (m_, v_*100),
                textcoords="offset points", xytext=(5, 3),
                fontsize=7, color=YELLOW)
ax.set_xlabel("Return period m (observations, log scale)")
ax.set_ylabel("Loss level (%)")
ax.set_title("Return Level Plot\nx_m: loss exceeded once per m obs",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (c) Tail comparison: empirical vs EVT vs Normal
ax = axes[2]
# Empirical tail
losses_sorted = np.sort(losses)[::-1]
n_tail = min(300, len(losses_sorted))
emp_probs = np.arange(1, n_tail+1) / (len(losses)+1)
ax.semilogy(losses_sorted[:n_tail]*100, emp_probs[:n_tail],
            "o", ms=3, alpha=0.5, color=BLUE, label="Empirical tail")
x_tail2 = np.linspace(u_thresh, losses_sorted[0]*1.5, 200)
# GPD tail: P(X > x) = (N_u/N) * P_gpd(X-u > x-u)
sf_evt  = (N_u/N) * genpareto.sf(x_tail2-u_thresh, xi_gpd, loc=0, scale=beta_gpd)
sf_norm = norm.sf(x_tail2, loc=losses.mean(), scale=losses.std())
ax.semilogy(x_tail2*100, sf_evt,  color=RED,  lw=2.5, label="EVT tail (GPD)")
ax.semilogy(x_tail2*100, sf_norm, color=GREEN, lw=2,
            linestyle="--", label="Normal tail")
ax.set_xlabel("Loss level (%)"); ax.set_ylabel("P(X > x)  [log scale]")
ax.set_title("Tail Probability Comparison\n"
             "EVT captures empirical fat tail; Normal underestimates",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

plt.tight_layout()
p3 = os.path.join(OUT_DIR, "m27_03_extreme_risk.png")
plt.savefig(p3, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p3}  ({time.perf_counter()-t0:.1f}s)")

# ── Summary ──────────────────────────────────────────────────────────────────
v99   = extreme_var(0.99,   u_thresh, xi_gpd, beta_gpd, N_u, N)
v999  = extreme_var(0.999,  u_thresh, xi_gpd, beta_gpd, N_u, N)
v9999 = extreme_var(0.9999, u_thresh, xi_gpd, beta_gpd, N_u, N)
print()
print("=" * 65)
print("  MODULE 27 COMPLETE -- 3 figures saved")
print("  Key Concepts:")
print("  [1] GEV: limit dist of block maxima (Fisher-Tippett)")
print("  [2] xi>0 Frechet (heavy), xi=0 Gumbel, xi<0 Weibull")
print("  [3] GPD: limit dist of threshold exceedances (POT)")
print("  [4] MEF e(u) linear in u => GPD approximation valid")
print("  [5] EVT VaR extrapolates beyond historical sample range")
print("  [6] Return level x_m: exceeded once per m observations")
print(f"  GEV fit: xi={xi_gev:.4f}  [{gev_family}]")
print(f"  GPD fit: xi={xi_gpd:.4f}  beta={beta_gpd:.6f}")
print(f"  EVT VaR 99.00%: {v99*100:.4f}%  |  Normal: "
      f"{norm.ppf(0.99)*losses.std()*100:.4f}%")
print(f"  EVT VaR 99.90%: {v999*100:.4f}%  |  Normal: "
      f"{norm.ppf(0.999)*losses.std()*100:.4f}%")
print(f"  EVT VaR 99.99%: {v9999*100:.4f}%  |  Normal: "
      f"{norm.ppf(0.9999)*losses.std()*100:.4f}%")
print("=" * 65)
