"""
M54 -- Extreme Value Theory & Tail Risk
========================================
CQF Concepts Explained | Project 19 | Quantitative Finance Portfolio

Theory
------
Extreme Value Theory (EVT) provides a rigorous framework for modelling
the tails of distributions -- the region where standard Gaussian or
historical simulation methods fail most critically.

1. Generalised Extreme Value (GEV) Distribution -- Block Maxima
---------------------------------------------------------------
Fisher-Tippett-Gnedenko theorem: for iid X_1,...,X_n, normalised block
maxima converge in distribution to the GEV:

    G(x; mu, sigma, xi) = exp{ -[1 + xi*(x-mu)/sigma]^{-1/xi} }

  xi > 0  (Frechet):  heavy tail  -- financial losses, insurance
  xi = 0  (Gumbel):   light tail  -- normal, lognormal
  xi < 0  (Weibull):  bounded tail -- beta, uniform

2. Peaks-Over-Threshold (POT) -- Generalised Pareto Distribution
----------------------------------------------------------------
For exceedances Y = X - u | X > u, Pickands-Balkema-de Haan theorem:
as threshold u -> x_F (right endpoint),

    H(y; sigma_u, xi) = 1 - (1 + xi*y/sigma_u)^{-1/xi}   xi != 0
                      = 1 - exp(-y/sigma_u)                xi = 0

POT is more data-efficient than block maxima for financial series.

3. Hill Estimator (tail index alpha = 1/xi)
-------------------------------------------
For order statistics X_{(1)} >= ... >= X_{(n)}:

    alpha_Hill(k) = k / sum_{i=1}^{k} log(X_{(i)} / X_{(k+1)})

Stable region of Hill plot identifies the tail index.

4. EVT-Based Risk Measures
--------------------------
VaR_p  = u + (sigma_u/xi) * [ (n/N_u * (1-p))^{-xi} - 1 ]
CVaR_p = VaR_p/(1-xi) + (sigma_u - xi*u)/(1-xi)

where N_u = number of exceedances above threshold u,
      n   = total observations, p = confidence level.

References
----------
Embrechts, Kluppelberg & Mikosch (1997) "Modelling Extremal Events"
McNeil, Frey & Embrechts (2005) "Quantitative Risk Management", PUP
Pickands (1975) "Statistical Inference using Extreme Order Statistics"
Hill (1975) "A Simple General Approach to Inference About the Tail"
"""

import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

# =============================================================================
# STYLE
# =============================================================================
DARK   = "#0d1117"
PANEL  = "#161b22"
TEXT   = "#c9d1d9"
GREEN  = "#3fb950"
RED    = "#f85149"
ACCENT = "#58a6ff"
GOLD   = "#d29922"
PURPLE = "#bc8cff"
ORANGE = "#f0883e"

plt.rcParams.update({
    "figure.facecolor":  DARK,
    "axes.facecolor":    PANEL,
    "axes.edgecolor":    TEXT,
    "axes.labelcolor":   TEXT,
    "xtick.color":       TEXT,
    "ytick.color":       TEXT,
    "text.color":        TEXT,
    "grid.color":        "#21262d",
    "grid.linestyle":    "--",
    "grid.linewidth":    0.5,
    "font.family":       "monospace",
    "font.size":         8,
    "legend.facecolor":  PANEL,
    "legend.edgecolor":  TEXT,
})

FIGS = os.path.join(os.path.dirname(__file__), "..", "..", "figures", "m54_evt_tail_risk")
os.makedirs(FIGS, exist_ok=True)

SEED = 42
np.random.seed(SEED)

print()
print("=" * 65)
print("  MODULE 54: EXTREME VALUE THEORY & TAIL RISK")
print("  GEV | GPD-POT | Hill Estimator | EVT-VaR | EVT-CVaR")
print("=" * 65)

# =============================================================================
# 1. SYNTHETIC FINANCIAL LOSS SERIES (fat-tailed)
# =============================================================================
# Simulate daily log-losses from a Student-t(nu=4) scaled process,
# reflecting the empirical excess kurtosis of equity return series.

N     = 3000
NU    = 4          # degrees of freedom -- heavy tail
SIGMA = 0.015      # daily volatility scale
MU    = 0.0004     # daily drift (losses = -returns)

t_draws   = np.random.standard_t(NU, size=N)
returns   = MU + SIGMA * t_draws
losses    = -returns          # work with losses (positive = bad)

# Empirical moments
emp_mean = losses.mean()
emp_std  = losses.std()
emp_kurt = float(stats.kurtosis(losses, fisher=True))  # excess kurtosis

print(f"  [01] Loss series: N={N}  nu(t)={NU}  sigma={SIGMA}")
print(f"       Mean={emp_mean:.5f}  Std={emp_std:.4f}  "
      f"Excess kurtosis={emp_kurt:.2f}")

# =============================================================================
# 2. BLOCK MAXIMA -- GEV FITTING
# =============================================================================
BLOCK_SIZE = 21    # monthly blocks

n_blocks   = N // BLOCK_SIZE
block_max  = np.array([
    losses[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE].max()
    for i in range(n_blocks)
])

# MLE for GEV via scipy
def neg_log_lik_gev(params, data):
    mu, sigma, xi = params
    if sigma <= 0:
        return 1e10
    z = 1 + xi * (data - mu) / sigma
    if np.any(z <= 0):
        return 1e10
    if abs(xi) < 1e-6:   # Gumbel limit
        t = (data - mu) / sigma
        return len(data) * np.log(sigma) + np.sum(t + np.exp(-t))
    return (len(data) * np.log(sigma)
            + (1 + 1/xi) * np.sum(np.log(z))
            + np.sum(z**(-1/xi)))

mu0 = block_max.mean()
s0  = block_max.std() * np.sqrt(6) / np.pi
x0  = [mu0, s0, 0.1]

res_gev = minimize(neg_log_lik_gev, x0, args=(block_max,),
                   method="Nelder-Mead",
                   options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 10000})
mu_gev, sigma_gev, xi_gev = res_gev.x

type_map = {True: "Frechet (heavy)",
            False: ("Gumbel (light)" if abs(xi_gev) < 0.05 else "Weibull (bounded)")}
gev_type = "Frechet (heavy)" if xi_gev > 0.05 else (
           "Gumbel (light)"  if abs(xi_gev) < 0.05 else "Weibull (bounded)")

print(f"  [02] GEV (Block Maxima, block={BLOCK_SIZE}d): "
      f"n_blocks={n_blocks}")
print(f"       mu={mu_gev:.5f}  sigma={sigma_gev:.5f}  xi={xi_gev:.4f}")
print(f"       Type: {gev_type}")

# =============================================================================
# 3. PEAKS-OVER-THRESHOLD -- GPD FITTING
# =============================================================================
# Mean Excess Plot: E[X-u | X>u] should be linear in u for GPD.
# Threshold selection: 90th percentile of losses.

THRESHOLD_Q = 0.90
u = np.quantile(losses, THRESHOLD_Q)
exceedances = losses[losses > u] - u   # Y = X - u | X > u
N_u = len(exceedances)

# MLE for GPD: H(y; sigma, xi) = 1 - (1 + xi*y/sigma)^{-1/xi}
def neg_log_lik_gpd(params, data):
    sigma, xi = params
    if sigma <= 0:
        return 1e10
    if abs(xi) < 1e-6:   # exponential limit
        return len(data) * np.log(sigma) + np.sum(data) / sigma
    z = 1 + xi * data / sigma
    if np.any(z <= 0):
        return 1e10
    return (len(data) * np.log(sigma)
            + (1 + 1/xi) * np.sum(np.log(z)))

res_gpd = minimize(neg_log_lik_gpd, [exceedances.mean(), 0.1],
                   args=(exceedances,),
                   method="Nelder-Mead",
                   options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 10000})
sigma_gpd, xi_gpd = res_gpd.x

print(f"  [03] GPD-POT: threshold u={u:.5f} (q={THRESHOLD_Q})  "
      f"N_u={N_u}")
print(f"       sigma={sigma_gpd:.5f}  xi={xi_gpd:.4f}")

# =============================================================================
# 4. HILL ESTIMATOR
# =============================================================================
losses_sorted = np.sort(losses)[::-1]   # descending order
k_range = np.arange(5, N // 4)

def hill_estimator(x_sorted: np.ndarray, k: int) -> float:
    """Hill estimator of tail index alpha for top-k order statistics."""
    log_ratios = np.log(x_sorted[:k]) - np.log(x_sorted[k])
    return float(k / np.sum(log_ratios))

# Only compute Hill for positive values (shift if needed)
shift = max(0, -losses_sorted[-1]) + 1e-6
ls_pos = losses_sorted + shift

hill_alpha = np.array([hill_estimator(ls_pos, k) for k in k_range])
hill_xi    = 1.0 / hill_alpha   # tail index xi = 1/alpha

# Stable region: median of Hill estimates in k=[50,200]
stable_mask = (k_range >= 50) & (k_range <= 200)
xi_hill = float(np.median(hill_xi[stable_mask]))

print(f"  [04] Hill Estimator: xi (stable region k=50-200) = {xi_hill:.4f}")
print(f"       Implied tail alpha = 1/xi = {1/xi_hill:.3f}")

# =============================================================================
# 5. EVT-BASED VaR AND CVaR
# =============================================================================
CONF_LEVELS = [0.95, 0.99, 0.999]

def evt_var_cvar(p, u, sigma, xi, n, n_u):
    """EVT VaR and CVaR at confidence level p via POT/GPD."""
    if abs(xi) < 1e-6:
        # Exponential case
        var  = u + sigma * np.log(n / n_u * (1 - p))
        cvar = var + sigma
    else:
        ratio = (n / n_u) * (1 - p)
        if ratio <= 0:
            return np.nan, np.nan
        var  = u + (sigma / xi) * (ratio**(-xi) - 1)
        cvar = var / (1 - xi) + (sigma_gpd - xi_gpd * u) / (1 - xi)
    return float(var), float(cvar)

def hist_var_cvar(losses, p):
    var  = float(np.quantile(losses, p))
    cvar = float(losses[losses >= var].mean())
    return var, cvar

def norm_var_cvar(losses, p):
    mu_, s_ = losses.mean(), losses.std()
    var  = float(mu_ + s_ * stats.norm.ppf(p))
    cvar = float(mu_ + s_ * stats.norm.pdf(stats.norm.ppf(p)) / (1 - p))
    return var, cvar

print(f"  [05] Risk Measures Comparison:")
print(f"       {'Level':>6}  {'Method':<12} {'VaR':>9} {'CVaR':>9}")
for p in CONF_LEVELS:
    v_evt, c_evt = evt_var_cvar(p, u, sigma_gpd, xi_gpd, N, N_u)
    v_his, c_his = hist_var_cvar(losses, p)
    v_nor, c_nor = norm_var_cvar(losses, p)
    print(f"       {p:>6.3f}  {'EVT-GPD':<12} {v_evt:>9.5f} {c_evt:>9.5f}")
    print(f"       {p:>6.3f}  {'Historical':<12} {v_his:>9.5f} {c_his:>9.5f}")
    print(f"       {p:>6.3f}  {'Normal':<12} {v_nor:>9.5f} {c_nor:>9.5f}")
    print()

# =============================================================================
# 6. MEAN EXCESS FUNCTION (for threshold selection diagnostics)
# =============================================================================
thresholds_me = np.linspace(np.quantile(losses, 0.70),
                             np.quantile(losses, 0.98), 40)
mean_excess = np.array([
    losses[losses > t].mean() - t if np.sum(losses > t) > 10 else np.nan
    for t in thresholds_me
])

# =============================================================================
# 7. FIGURE 1 -- Loss Distribution & GEV Block Maxima
# =============================================================================
fig = plt.figure(figsize=(15, 9), facecolor=DARK)
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
fig.suptitle("M54 -- Extreme Value Theory: Loss Distribution & GEV",
             color=TEXT, fontsize=11)

# 7A: Loss histogram vs Normal vs t-distribution
ax = fig.add_subplot(gs[0, 0])
x_lin = np.linspace(losses.min(), losses.max(), 300)
ax.hist(losses, bins=80, density=True, color=ACCENT, alpha=0.5,
        label="Empirical", edgecolor=DARK, linewidth=0.3)
ax.plot(x_lin,
        stats.norm.pdf(x_lin, losses.mean(), losses.std()),
        color=GREEN, lw=1.5, label="Normal fit")
ax.plot(x_lin,
        stats.t.pdf(x_lin, NU, losses.mean(), losses.std() * np.sqrt((NU-2)/NU)),
        color=GOLD, lw=1.5, label=f"t({NU}) fit")
ax.set_title("Daily Loss Distribution\nvs Parametric Fits")
ax.set_xlabel("Loss")
ax.set_ylabel("Density")
ax.legend(fontsize=7)
ax.set_facecolor(PANEL)
ax.grid(True)

# 7B: QQ plot -- empirical vs Normal
ax = fig.add_subplot(gs[0, 1])
(osm, osr), (slope, intercept, _) = stats.probplot(losses, dist="norm")
ax.scatter(osm, osr, s=4, color=ACCENT, alpha=0.5, label="Empirical quantiles")
qq_line = slope * np.array([osm[0], osm[-1]]) + intercept
ax.plot([osm[0], osm[-1]], qq_line, color=GREEN, lw=1.5, label="Normal reference")
ax.set_title("QQ Plot: Losses vs Normal\n(tail deviation = fat tails)")
ax.set_xlabel("Theoretical Normal Quantile")
ax.set_ylabel("Empirical Quantile")
ax.legend(fontsize=7)
ax.set_facecolor(PANEL)
ax.grid(True)

# 7C: Block maxima histogram + GEV fit
ax = fig.add_subplot(gs[0, 2])
x_bm = np.linspace(block_max.min(), block_max.max() * 1.2, 200)

def gev_pdf(x, mu, sigma, xi):
    if abs(xi) < 1e-6:
        t = (x - mu) / sigma
        return np.exp(-t - np.exp(-t)) / sigma
    z = 1 + xi * (x - mu) / sigma
    z = np.where(z > 0, z, np.nan)
    return (1/sigma) * z**(-(1+1/xi)) * np.exp(-z**(-1/xi))

pdf_gev = gev_pdf(x_bm, mu_gev, sigma_gev, xi_gev)
ax.hist(block_max, bins=20, density=True, color=GOLD, alpha=0.5,
        label="Block maxima", edgecolor=DARK, linewidth=0.3)
ax.plot(x_bm, pdf_gev, color=RED, lw=2, label=f"GEV fit (xi={xi_gev:.3f})")
ax.set_title(f"Block Maxima (block={BLOCK_SIZE}d)\nGEV: {gev_type}")
ax.set_xlabel("Max Loss")
ax.set_ylabel("Density")
ax.legend(fontsize=7)
ax.set_facecolor(PANEL)
ax.grid(True)

# 7D: GPD fit to exceedances
ax = fig.add_subplot(gs[1, 0])
x_exc = np.linspace(0, exceedances.max() * 1.2, 200)

def gpd_pdf(y, sigma, xi):
    if abs(xi) < 1e-6:
        return np.exp(-y / sigma) / sigma
    z = 1 + xi * y / sigma
    z = np.where(z > 0, z, np.nan)
    return (1/sigma) * z**(-(1+1/xi))

pdf_gpd = gpd_pdf(x_exc, sigma_gpd, xi_gpd)
ax.hist(exceedances, bins=30, density=True, color=PURPLE, alpha=0.5,
        label=f"Exceedances (u={u:.4f})", edgecolor=DARK, linewidth=0.3)
ax.plot(x_exc, pdf_gpd, color=RED, lw=2,
        label=f"GPD fit (xi={xi_gpd:.3f})")
ax.set_title(f"POT Exceedances\nGPD: sigma={sigma_gpd:.4f}  xi={xi_gpd:.4f}")
ax.set_xlabel("Exceedance Y = X - u")
ax.set_ylabel("Density")
ax.legend(fontsize=7)
ax.set_facecolor(PANEL)
ax.grid(True)

# 7E: Mean excess function
ax = fig.add_subplot(gs[1, 1])
valid = ~np.isnan(mean_excess)
ax.plot(thresholds_me[valid], mean_excess[valid],
        color=ORANGE, lw=1.5, marker="o", markersize=3)
ax.axvline(u, color=RED, lw=1.2, ls="--", label=f"Selected u={u:.4f}")
ax.set_title("Mean Excess Function\n(linear region -> GPD)")
ax.set_xlabel("Threshold u")
ax.set_ylabel("E[X - u | X > u]")
ax.legend(fontsize=7)
ax.set_facecolor(PANEL)
ax.grid(True)

# 7F: Hill plot
ax = fig.add_subplot(gs[1, 2])
ax.plot(k_range, hill_xi, color=ACCENT, lw=0.8, alpha=0.6)
ax.axhline(xi_gpd, color=RED, lw=1.2, ls="--",
           label=f"GPD xi={xi_gpd:.4f}")
ax.axhline(xi_hill, color=GREEN, lw=1.2, ls="-.",
           label=f"Hill xi={xi_hill:.4f}")
ax.axvspan(50, 200, color=GOLD, alpha=0.12, label="Stable region")
ax.set_xlim(0, 400)
ax.set_ylim(max(0, hill_xi.min() - 0.1), hill_xi.max() + 0.1)
ax.set_title("Hill Plot: Tail Index xi(k)")
ax.set_xlabel("k (number of upper order statistics)")
ax.set_ylabel("Hill xi = 1/alpha")
ax.legend(fontsize=7)
ax.set_facecolor(PANEL)
ax.grid(True)

fig.savefig(os.path.join(FIGS, "m54_fig1_evt_distributions.png"),
            dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close(fig)
print("  [06] Fig 1 saved: loss distribution, GEV, GPD, Hill plot")

# =============================================================================
# 8. FIGURE 2 -- VaR / CVaR Comparison Across Methods
# =============================================================================
conf_grid = np.linspace(0.90, 0.999, 60)

var_evt  = np.array([evt_var_cvar(p, u, sigma_gpd, xi_gpd, N, N_u)[0]
                      for p in conf_grid])
cvar_evt = np.array([evt_var_cvar(p, u, sigma_gpd, xi_gpd, N, N_u)[1]
                      for p in conf_grid])
var_his  = np.array([hist_var_cvar(losses, p)[0] for p in conf_grid])
cvar_his = np.array([hist_var_cvar(losses, p)[1] for p in conf_grid])
var_nor  = np.array([norm_var_cvar(losses, p)[0] for p in conf_grid])
cvar_nor = np.array([norm_var_cvar(losses, p)[1] for p in conf_grid])

fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=DARK)
fig.suptitle("M54 -- VaR & CVaR: EVT vs Historical vs Normal",
             color=TEXT, fontsize=11, y=1.01)

# 8A: VaR comparison
ax = axes[0]
ax.plot(conf_grid * 100, var_evt * 100, color=RED,    lw=2,   label="EVT-GPD")
ax.plot(conf_grid * 100, var_his * 100, color=GOLD,   lw=1.5, label="Historical", ls="--")
ax.plot(conf_grid * 100, var_nor * 100, color=GREEN,  lw=1.5, label="Normal",     ls="-.")
ax.set_title("VaR vs Confidence Level")
ax.set_xlabel("Confidence Level (%)")
ax.set_ylabel("VaR (% of portfolio)")
ax.legend(fontsize=8)
ax.set_facecolor(PANEL)
ax.grid(True)

# 8B: CVaR comparison
ax = axes[1]
ax.plot(conf_grid * 100, cvar_evt * 100, color=RED,   lw=2,   label="EVT-GPD")
ax.plot(conf_grid * 100, cvar_his * 100, color=GOLD,  lw=1.5, label="Historical", ls="--")
ax.plot(conf_grid * 100, cvar_nor * 100, color=GREEN, lw=1.5, label="Normal",     ls="-.")
ax.set_title("CVaR (Expected Shortfall)\nvs Confidence Level")
ax.set_xlabel("Confidence Level (%)")
ax.set_ylabel("CVaR (% of portfolio)")
ax.legend(fontsize=8)
ax.set_facecolor(PANEL)
ax.grid(True)

# 8C: VaR ratio EVT/Normal (illustrates normal underestimation)
ax = axes[2]
ratio_var  = var_evt  / (var_nor  + 1e-9)
ratio_cvar = cvar_evt / (cvar_nor + 1e-9)
ax.plot(conf_grid * 100, ratio_var,  color=RED,    lw=2,   label="VaR ratio EVT/Normal")
ax.plot(conf_grid * 100, ratio_cvar, color=PURPLE, lw=2,   label="CVaR ratio EVT/Normal")
ax.axhline(1.0, color=TEXT, lw=0.8, ls="--", label="Ratio = 1 (no underestimation)")
ax.set_title("EVT / Normal Risk Ratio\n(>1 = Normal underestimates tail risk)")
ax.set_xlabel("Confidence Level (%)")
ax.set_ylabel("Risk Ratio")
ax.legend(fontsize=7)
ax.set_facecolor(PANEL)
ax.grid(True)

for ax in axes:
    ax.set_facecolor(PANEL)

fig.tight_layout()
fig.savefig(os.path.join(FIGS, "m54_fig2_var_cvar_comparison.png"),
            dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close(fig)
print("  [07] Fig 2 saved: VaR & CVaR comparison")

# =============================================================================
# 9. FIGURE 3 -- Return Level Plot & Tail Probability
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=DARK)
fig.suptitle("M54 -- Return Level Plot & Tail Probability",
             color=TEXT, fontsize=11, y=1.01)

# 9A: Return level plot (GEV)
ax = axes[0]
T_range  = np.logspace(0.3, 3, 100)   # return periods in blocks
p_T      = 1 - 1.0 / T_range
# GEV quantile function: Q(p) = mu - sigma/xi * [1 - (-log p)^{-xi}]
def gev_quantile(p, mu, sigma, xi):
    if abs(xi) < 1e-6:
        return mu - sigma * np.log(-np.log(p))
    return mu + sigma / xi * ((-np.log(p))**(-xi) - 1)

rl_gev = np.array([gev_quantile(p, mu_gev, sigma_gev, xi_gev) for p in p_T])

# Empirical return levels from sorted block maxima
T_emp = (n_blocks + 1) / np.arange(1, n_blocks + 1)
bm_sorted = np.sort(block_max)[::-1]

ax.semilogx(T_range * BLOCK_SIZE, rl_gev * 100, color=RED, lw=2,
            label="GEV Return Level")
ax.scatter(T_emp * BLOCK_SIZE, bm_sorted * 100, s=15,
           color=GOLD, alpha=0.7, zorder=5, label="Empirical")
ax.axhline(u * 100, color=GREEN, lw=1, ls="--",
           label=f"POT threshold {u*100:.2f}%")
ax.set_title(f"Return Level Plot\n(block={BLOCK_SIZE}d, GEV xi={xi_gev:.3f})")
ax.set_xlabel("Return Period (days)")
ax.set_ylabel("Return Level (% loss)")
ax.legend(fontsize=7)
ax.set_facecolor(PANEL)
ax.grid(True)

# 9B: Tail probability comparison P(X > x)
ax = axes[1]
x_tail = np.linspace(u, losses.max() * 1.3, 200)

# EVT tail: P(X>x) = N_u/N * (1 + xi*(x-u)/sigma)^{-1/xi}
def evt_tail_prob(x, u, sigma, xi, n, n_u):
    y = x - u
    if abs(xi) < 1e-6:
        return (n_u / n) * np.exp(-y / sigma)
    z = 1 + xi * y / sigma
    z = np.where(z > 0, z, np.nan)
    return (n_u / n) * z**(-1/xi)

p_evt  = evt_tail_prob(x_tail, u, sigma_gpd, xi_gpd, N, N_u)
p_nor  = 1 - stats.norm.cdf(x_tail, losses.mean(), losses.std())
p_emp  = np.array([np.mean(losses > x) for x in x_tail])

ax.semilogy(x_tail * 100, p_evt,  color=RED,   lw=2,   label="EVT-GPD")
ax.semilogy(x_tail * 100, p_nor,  color=GREEN, lw=1.5, ls="-.", label="Normal")
ax.semilogy(x_tail * 100, p_emp,  color=GOLD,  lw=1.5, ls="--", label="Empirical")
ax.set_title("Tail Probability P(X > x)\n(log scale)")
ax.set_xlabel("Loss Threshold (%)")
ax.set_ylabel("P(X > x)")
ax.legend(fontsize=7)
ax.set_facecolor(PANEL)
ax.grid(True)

# 9C: Tail loss time series with EVT-VaR overlay
ax = axes[2]
t_ts = np.arange(N)
ax.plot(t_ts, losses * 100, color=TEXT, lw=0.4, alpha=0.5, label="Daily Loss")

v99_evt, _ = evt_var_cvar(0.99, u, sigma_gpd, xi_gpd, N, N_u)
v99_nor, _ = norm_var_cvar(losses, 0.99)
v99_his, _ = hist_var_cvar(losses, 0.99)

ax.axhline(v99_evt * 100, color=RED,   lw=1.5, ls="--",
           label=f"99% VaR EVT={v99_evt*100:.2f}%")
ax.axhline(v99_nor * 100, color=GREEN, lw=1.5, ls="-.",
           label=f"99% VaR Normal={v99_nor*100:.2f}%")
ax.axhline(v99_his * 100, color=GOLD,  lw=1.5, ls=":",
           label=f"99% VaR Hist={v99_his*100:.2f}%")

# Highlight breaches of EVT-VaR
breach = losses > v99_evt
ax.scatter(t_ts[breach], losses[breach] * 100, s=10, color=RED,
           zorder=5, alpha=0.8)
n_breach = int(breach.sum())
ax.set_title(f"Daily Losses & 99% VaR Lines\n"
             f"EVT breaches: {n_breach} ({n_breach/N*100:.1f}%)")
ax.set_xlabel("Day")
ax.set_ylabel("Loss (%)")
ax.legend(fontsize=6)
ax.set_facecolor(PANEL)
ax.grid(True)

for ax in axes:
    ax.set_facecolor(PANEL)

fig.tight_layout()
fig.savefig(os.path.join(FIGS, "m54_fig3_return_level_tail.png"),
            dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close(fig)
print("  [08] Fig 3 saved: return level, tail probability, VaR overlay")

# =============================================================================
# SUMMARY
# =============================================================================
v99_e, c99_e   = evt_var_cvar(0.99,  u, sigma_gpd, xi_gpd, N, N_u)
v999_e, c999_e = evt_var_cvar(0.999, u, sigma_gpd, xi_gpd, N, N_u)
v99_n, c99_n   = norm_var_cvar(losses, 0.99)
v999_n, c999_n = norm_var_cvar(losses, 0.999)

print()
print("  MODULE 54 COMPLETE -- 3 figures saved")
print("  Key Concepts:")
print("  [1] GEV: Fisher-Tippett-Gnedenko -- block maxima convergence")
print(f"  [2] GEV fit: xi={xi_gev:.4f} => {gev_type}")
print(f"  [3] GPD-POT: threshold u={u:.5f}  N_u={N_u}  xi={xi_gpd:.4f}")
print(f"  [4] Hill estimator: xi={xi_hill:.4f} (stable region k=50-200)")
print(f"  [5] 99% VaR  -- EVT={v99_e*100:.3f}%  Normal={v99_n*100:.3f}%  "
      f"ratio={v99_e/v99_n:.2f}x")
print(f"  [6] 99% CVaR -- EVT={c99_e*100:.3f}%  Normal={c99_n*100:.3f}%  "
      f"ratio={c99_e/c99_n:.2f}x")
print(f"  [7] 99.9% VaR -- EVT={v999_e*100:.3f}%  Normal={v999_n*100:.3f}%  "
      f"ratio={v999_e/v999_n:.2f}x")
print(f"  [8] Normal underestimates extreme tail risk by {v999_e/v999_n:.1f}x at 99.9%")
print("  NEXT: M55 -- Copulas & Dependence Modelling")
print()
