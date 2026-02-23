#!/usr/bin/env python3
"""
M29 — Portfolio Optimization: Markowitz and Beyond
====================================================
Module 29 | CQF Concepts Explained
Group 6   | Portfolio Construction

Theory
------
Mean-Variance Optimization (Markowitz, 1952)
---------------------------------------------
Given N assets with expected returns mu (Nx1) and covariance Sigma (NxN):

    min_{w}  (1/2) * w' * Sigma * w
    s.t.     w' * mu  = mu_target
             w' * 1   = 1
             (w >= 0  if long-only)

Lagrangian solution (unconstrained):
    w* = lambda * Sigma^{-1} * mu + gamma * Sigma^{-1} * 1

The efficient frontier is a parabola in (sigma, mu) space.
Every efficient portfolio is a linear combination of two
"fund separation" portfolios:
    (1) Global Minimum Variance: w_GMV = Sigma^{-1}*1 / (1'*Sigma^{-1}*1)
    (2) Maximum Sharpe Ratio (tangency portfolio): w_T = Sigma^{-1}*(mu-rf)

Sharpe Ratio: SR = (mu_p - rf) / sigma_p

Capital Market Line (CML)
--------------------------
Combining the tangency portfolio w_T with the risk-free asset:
    E[R_p] = rf + SR_T * sigma_p

All mean-variance investors hold w_T (mutual fund theorem).

Black-Litterman (1992)
-----------------------
Addresses Markowitz instability (extreme weights, sensitivity to mu).
Two inputs:
    Pi  = tau * Sigma * w_mkt   (implied equilibrium returns)
    Q   = investor views (K x 1 vector)
    P   = pick matrix (K x N), linking views to assets
    Omega = view uncertainty covariance (K x K, diagonal)

Posterior mean:
    mu_BL = [(tau*Sigma)^{-1} + P'*Omega^{-1}*P]^{-1}
            * [(tau*Sigma)^{-1}*Pi + P'*Omega^{-1}*Q]

Posterior covariance:
    Sigma_BL = Sigma + [(tau*Sigma)^{-1} + P'*Omega^{-1}*P]^{-1}

Risk Parity
-----------
Each asset contributes equally to total portfolio variance:
    RC_i = w_i * (Sigma*w)_i = Var_p / N  for all i

Total variance: Var_p = w' * Sigma * w
Risk contribution: RC_i = w_i * partial_i(Var_p) = w_i * (Sigma*w)_i
Risk parity condition: RC_1 = RC_2 = ... = RC_N = Var_p / N

Solved numerically (no closed form for N > 2).

References
----------
- Markowitz, H. (1952). Portfolio selection. Journal of Finance, 7(1), 77-91.
- Black, F., Litterman, R. (1992). Global portfolio optimization.
  Financial Analysts Journal, 48(5), 28-43.
- Maillard, S., Roncalli, T., Teiletche, J. (2010). The properties of
  equally weighted risk contribution portfolios. Journal of Portfolio
  Management, 36(4), 60-70.
- Sharpe, W. (1964). Capital asset prices. Journal of Finance, 19(3).
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ── Styling ──────────────────────────────────────────────────────────────────
DARK   = "#0a0a0a";  PANEL  = "#111111"; GRID   = "#1e1e1e"
WHITE  = "#e8e8e8";  BLUE   = "#4a9eff"; GREEN  = "#00d4aa"
ORANGE = "#ff8c42";  RED    = "#ff4757"; PURPLE = "#a855f7"
YELLOW = "#ffd700";  CYAN   = "#00bcd4"

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
# SECTION 1 — ASSET UNIVERSE
# ============================================================

NAMES  = ["US Equity", "EM Equity", "US Bonds", "EM Bonds",
          "Commodities", "Real Estate"]
N      = len(NAMES)
RF     = 0.04   # risk-free rate

MU     = np.array([0.10, 0.13, 0.04, 0.07, 0.08, 0.09])
VOL    = np.array([0.18, 0.25, 0.05, 0.12, 0.20, 0.16])
CORR   = np.array([
    [1.00,  0.75,  0.10,  0.35,  0.20,  0.60],
    [0.75,  1.00,  0.00,  0.45,  0.30,  0.50],
    [0.10,  0.00,  1.00,  0.40, -0.10,  0.15],
    [0.35,  0.45,  0.40,  1.00,  0.10,  0.30],
    [0.20,  0.30, -0.10,  0.10,  1.00,  0.15],
    [0.60,  0.50,  0.15,  0.30,  0.15,  1.00],
])
SIGMA  = np.diag(VOL) @ CORR @ np.diag(VOL)
W_MKT  = np.array([0.35, 0.15, 0.20, 0.10, 0.10, 0.10])  # market cap weights

# ============================================================
# SECTION 2 — MEAN-VARIANCE ENGINE
# ============================================================

def portfolio_stats(w, mu, sigma):
    ret = w @ mu
    vol = np.sqrt(w @ sigma @ w)
    sr  = (ret - RF) / vol
    return ret, vol, sr

def gmv_portfolio(sigma):
    """Global Minimum Variance: analytical solution."""
    inv_S = np.linalg.inv(sigma)
    ones  = np.ones(len(sigma))
    w     = inv_S @ ones / (ones @ inv_S @ ones)
    return w

def tangency_portfolio(mu, sigma, rf):
    """Maximum Sharpe Ratio portfolio: Sigma^{-1}*(mu-rf), normalized."""
    inv_S  = np.linalg.inv(sigma)
    excess = mu - rf
    w      = inv_S @ excess
    return w / w.sum()

def efficient_frontier(mu, sigma, rf, n_points=200, long_only=True):
    """Trace efficient frontier by solving min-var for each target return."""
    mu_min  = mu.min() * 0.8
    mu_max  = mu.max() * 1.1
    targets = np.linspace(mu_min, mu_max, n_points)
    results = []
    n       = len(mu)
    for mu_t in targets:
        constraints = [
            {"type": "eq", "fun": lambda w: w @ mu - mu_t},
            {"type": "eq", "fun": lambda w: w.sum() - 1},
        ]
        bounds = [(0, 1)] * n if long_only else [(None, None)] * n
        res = minimize(lambda w: w @ sigma @ w,
                       x0=np.ones(n)/n, method="SLSQP",
                       constraints=constraints, bounds=bounds,
                       options={"ftol":1e-12, "maxiter":500})
        if res.success:
            w  = res.x
            r_ = w @ mu
            v_ = np.sqrt(w @ sigma @ w)
            results.append((r_, v_, w))
    return results

# ============================================================
# SECTION 3 — BLACK-LITTERMAN
# ============================================================

def black_litterman(mu_eq, sigma, P, Q, omega, tau=0.05):
    """
    Black-Litterman posterior mean and covariance.
    mu_eq : equilibrium returns (N,)
    P     : pick matrix (K, N)
    Q     : view returns (K,)
    omega : view uncertainty (K, K)
    """
    ts    = tau * sigma
    ts_inv = np.linalg.inv(ts)
    omega_inv = np.linalg.inv(omega)
    M_inv = ts_inv + P.T @ omega_inv @ P
    M     = np.linalg.inv(M_inv)
    mu_bl = M @ (ts_inv @ mu_eq + P.T @ omega_inv @ Q)
    sigma_bl = sigma + M
    return mu_bl, sigma_bl

# ============================================================
# SECTION 4 — RISK PARITY
# ============================================================

def risk_parity(sigma, tol=1e-10):
    """Equal risk contribution portfolio via sequential quadratic programming."""
    n = len(sigma)
    def objective(w):
        w  = np.maximum(w, 1e-8)
        Sw = sigma @ w
        vp = w @ Sw
        rc = w * Sw           # risk contributions
        rc_target = vp / n
        return np.sum((rc - rc_target)**2)

    def total_var_grad(w):
        return 2 * sigma @ w

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds      = [(1e-4, 1)] * n
    best, best_val = None, np.inf
    for w0 in [np.ones(n)/n,
               1/(np.diag(sigma)**0.5) / (1/(np.diag(sigma)**0.5)).sum()]:
        res = minimize(objective, w0, method="SLSQP",
                       constraints=constraints, bounds=bounds,
                       options={"ftol":tol, "maxiter":2000})
        if res.fun < best_val:
            best_val = res.fun; best = res
    return best.x

# ============================================================
# COMPUTE ALL PORTFOLIOS
# ============================================================

w_gmv  = gmv_portfolio(SIGMA)
w_tan  = tangency_portfolio(MU, SIGMA, RF)
w_rp   = risk_parity(SIGMA)
w_ew   = np.ones(N) / N

# Equilibrium returns (reverse-optimized from market weights)
tau_bl  = 0.05
Pi      = tau_bl * SIGMA @ W_MKT * (1/tau_bl)   # delta * Sigma * w_mkt, delta~1
Pi     += RF    # add risk-free baseline

# Views: EM Equity outperforms US Equity by 3%, Commodities return 10%
P     = np.array([[0, 1, 0, 0, 0, 0],      # EM Equity view
                  [0, 0, 0, 0, 1, 0]])      # Commodities view
Q     = np.array([0.13, 0.10])
omega = np.diag([0.0004, 0.0009])           # view confidence (variance)
mu_bl, sigma_bl = black_litterman(Pi, SIGMA, P, Q, omega)
w_bl  = tangency_portfolio(mu_bl, sigma_bl, RF)
w_bl  = np.maximum(w_bl, 0); w_bl /= w_bl.sum()

# Efficient frontier (long-only)
ef_lo = efficient_frontier(MU, SIGMA, RF, n_points=150, long_only=True)
ef_lo_ret = np.array([x[0] for x in ef_lo])
ef_lo_vol = np.array([x[1] for x in ef_lo])

# Monte Carlo portfolios (random weights)
rng_mc  = np.random.default_rng(7)
W_mc    = rng_mc.dirichlet(np.ones(N), size=3000)
mc_ret  = W_mc @ MU
mc_vol  = np.array([np.sqrt(w @ SIGMA @ w) for w in W_mc])
mc_sr   = (mc_ret - RF) / mc_vol

print("[M29] Portfolio Statistics:")
header = f"{'Portfolio':18s}  {'Return':>7s}  {'Vol':>7s}  {'Sharpe':>7s}"
print("      " + header)
print("      " + "-"*48)
for lbl, w in [("Equal Weight", w_ew), ("GMV", w_gmv),
               ("Tangency", w_tan), ("Risk Parity", w_rp),
               ("Black-Litterman", w_bl)]:
    r_, v_, sr_ = portfolio_stats(w, MU, SIGMA)
    print(f"      {lbl:18s}  {r_*100:6.2f}%  {v_*100:6.2f}%  {sr_:7.4f}")

# Risk contributions
print("\n[M29] Risk Contributions — Risk Parity Portfolio:")
Sw    = SIGMA @ w_rp
vp    = w_rp @ Sw
rc    = w_rp * Sw / vp * 100
for nm, rc_ in zip(NAMES, rc):
    print(f"      {nm:15s}: {rc_:.2f}%")

# ============================================================
# FIGURE 1 — Efficient Frontier and special portfolios
# ============================================================
t0 = time.perf_counter()
print("\n[M29] Figure 1: Efficient Frontier ...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor(DARK)
fig.suptitle("M29 — Markowitz Portfolio Optimization\n"
             "Efficient Frontier | Capital Market Line | Weight Analysis",
             color=WHITE, fontsize=12, fontweight="bold")

# (a) Efficient frontier with special portfolios
ax = axes[0]
sc = ax.scatter(mc_vol*100, mc_ret*100, c=mc_sr, cmap="plasma",
                s=4, alpha=0.4, vmin=0, vmax=mc_sr.max())
plt.colorbar(sc, ax=ax, label="Sharpe Ratio")
ax.plot(ef_lo_vol*100, ef_lo_ret*100, color=WHITE, lw=2.5,
        label="Efficient Frontier (long-only)")
specials = [("GMV",    w_gmv, GREEN),  ("Tangency", w_tan, YELLOW),
            ("RP",     w_rp,  ORANGE), ("EW",       w_ew,  CYAN),
            ("BL",     w_bl,  RED)]
for lbl, w, col in specials:
    r_, v_, _ = portfolio_stats(w, MU, SIGMA)
    ax.scatter(v_*100, r_*100, color=col, s=120, zorder=6,
               marker="*" if lbl=="Tangency" else "o", label=lbl)
# CML
v_tan = portfolio_stats(w_tan, MU, SIGMA)[1]
v_line = np.linspace(0, ef_lo_vol.max()*100*1.1, 100)
r_cml  = RF*100 + (portfolio_stats(w_tan, MU, SIGMA)[2]) * v_line
ax.plot(v_line, r_cml, color=YELLOW, lw=1.5, linestyle="--",
        alpha=0.8, label="CML")
ax.set_xlabel("Volatility (%)"); ax.set_ylabel("Expected Return (%)")
ax.set_title("Efficient Frontier (Long-Only)\n"
             "Color = Sharpe Ratio | Star = Tangency",
             color=WHITE, fontsize=9)
ax.legend(fontsize=6.5); ax.grid(True); watermark(ax)

# (b) Weight comparison
ax = axes[1]
x  = np.arange(N)
w_data = [w_ew, w_gmv, w_tan, w_rp, w_bl]
labels = ["EW", "GMV", "Tangency", "Risk Parity", "BL"]
colors = [CYAN, GREEN, YELLOW, ORANGE, RED]
bar_w  = 0.15
for i, (w, lbl, col) in enumerate(zip(w_data, labels, colors)):
    ax.bar(x + i*bar_w - 2*bar_w, w*100, width=bar_w,
           color=col, alpha=0.85, label=lbl)
ax.set_xticks(x)
ax.set_xticklabels([n.replace(" ", "\n") for n in NAMES], fontsize=7)
ax.set_ylabel("Weight (%)")
ax.set_title("Portfolio Weights Comparison\n"
             "5 Strategies | 6 Assets",
             color=WHITE, fontsize=9)
ax.legend(fontsize=6.5); ax.grid(True, axis="y"); watermark(ax)

# (c) Sharpe ratio vs volatility budget
ax = axes[2]
sigma_budgets = np.linspace(0.04, 0.25, 200)
sr_cml_arr = [(portfolio_stats(w_tan, MU, SIGMA)[2])] * len(sigma_budgets)
ret_ef   = RF + portfolio_stats(w_tan, MU, SIGMA)[2] * sigma_budgets
for lbl, w, col in specials:
    r_, v_, sr_ = portfolio_stats(w, MU, SIGMA)
    ax.scatter(v_*100, sr_, color=col, s=100, zorder=5, label=f"{lbl}: SR={sr_:.3f}")
ax.axhline(portfolio_stats(w_tan, MU, SIGMA)[2], color=YELLOW,
           lw=1.5, linestyle="--", label="Max SR (CML)")
ax.set_xlabel("Portfolio Volatility (%)"); ax.set_ylabel("Sharpe Ratio")
ax.set_title("Sharpe Ratio by Strategy\nTangency maximizes SR on the CML",
             color=WHITE, fontsize=9)
ax.legend(fontsize=6.5); ax.grid(True); watermark(ax)

plt.tight_layout()
p1 = os.path.join(OUT_DIR, "m29_01_efficient_frontier.png")
plt.savefig(p1, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p1}  ({time.perf_counter()-t0:.1f}s)")

# ============================================================
# FIGURE 2 — Risk Parity and contributions
# ============================================================
t0 = time.perf_counter()
print("[M29] Figure 2: Risk parity and contributions ...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor(DARK)
fig.suptitle("M29 — Risk Parity: Equal Risk Contribution\n"
             "RC_i = w_i*(Sigma*w)_i = Var_p/N",
             color=WHITE, fontsize=12, fontweight="bold")

colors6 = [BLUE, GREEN, ORANGE, RED, PURPLE, YELLOW]

def risk_contribs(w, sigma):
    Sw = sigma @ w; vp = w @ Sw
    return w * Sw / vp * 100

# (a) Risk contributions for each strategy
ax = axes[0]
x  = np.arange(N)
for i, (w, lbl, col) in enumerate(zip(w_data, labels, colors)):
    rc = risk_contribs(w, SIGMA)
    ax.bar(x + i*bar_w - 2*bar_w, rc, width=bar_w,
           color=col, alpha=0.85, label=lbl)
ax.axhline(100/N, color=WHITE, lw=1.5, linestyle="--",
           label=f"Equal RC = {100/N:.1f}%")
ax.set_xticks(x)
ax.set_xticklabels([n.replace(" ", "\n") for n in NAMES], fontsize=7)
ax.set_ylabel("Risk Contribution (%)")
ax.set_title("Risk Contributions by Strategy\n"
             "RP forces equal bar heights",
             color=WHITE, fontsize=9)
ax.legend(fontsize=6.5); ax.grid(True, axis="y"); watermark(ax)

# (b) Risk parity vs equal weight pie comparison
ax = axes[1]
rc_ew = risk_contribs(w_ew, SIGMA)
rc_rp = risk_contribs(w_rp, SIGMA)
x_idx = np.arange(N)
ax.barh(x_idx + 0.2, w_ew*100,  height=0.35, color=CYAN,   alpha=0.8,
        label="EW weights")
ax.barh(x_idx - 0.2, w_rp*100,  height=0.35, color=ORANGE, alpha=0.8,
        label="RP weights")
ax2 = ax.twiny()
ax2.barh(x_idx + 0.2, rc_ew, height=0.35, color=CYAN,   alpha=0.3,
         linestyle="--")
ax2.barh(x_idx - 0.2, rc_rp, height=0.35, color=ORANGE, alpha=0.3)
ax2.set_xlabel("Risk contribution (%)", color=ORANGE)
ax.set_yticks(x_idx)
ax.set_yticklabels(NAMES, fontsize=8)
ax.set_xlabel("Portfolio weight (%)")
ax.set_title("EW vs Risk Parity\nWeights (solid) | Risk Contribs (faded)",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True, axis="x"); watermark(ax)

# (c) Convergence of risk parity optimization
ax = axes[2]
# Show how RC deviation shrinks over iterations (proxy: varying tolerance)
n_iter = np.arange(1, 51)
# Simulate convergence by fitting with increasing iterations
dev_arr = []
for n_it in n_iter:
    res_it = minimize(
        lambda w: np.sum((w*np.maximum(w,1e-8)*np.diag(SIGMA)/
                          (np.maximum(w,1e-8)@SIGMA@np.maximum(w,1e-8)) - 1/N)**2),
        np.ones(N)/N,
        method="SLSQP",
        constraints=[{"type":"eq","fun":lambda w: w.sum()-1}],
        bounds=[(1e-4,1)]*N,
        options={"maxiter": int(n_it*10), "ftol":1e-14}
    )
    rc_it  = risk_contribs(np.maximum(res_it.x, 1e-8), SIGMA)
    dev_arr.append(np.std(rc_it))
ax.semilogy(n_iter*10, dev_arr, color=GREEN, lw=2.5)
ax.fill_between(n_iter*10, dev_arr, color=GREEN, alpha=0.15)
ax.set_xlabel("Optimizer iterations"); ax.set_ylabel("Std(RC_i) [log]")
ax.set_title("Risk Parity Convergence\n"
             "std(RC_i) -> 0 as RC_i -> equal",
             color=WHITE, fontsize=9)
ax.grid(True); watermark(ax)

plt.tight_layout()
p2 = os.path.join(OUT_DIR, "m29_02_risk_parity.png")
plt.savefig(p2, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p2}  ({time.perf_counter()-t0:.1f}s)")

# ============================================================
# FIGURE 3 — Black-Litterman
# ============================================================
t0 = time.perf_counter()
print("[M29] Figure 3: Black-Litterman analysis ...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor(DARK)
fig.suptitle("M29 — Black-Litterman: Bayesian Return Estimation\n"
             "Equilibrium Pi + Views Q -> Posterior mu_BL",
             color=WHITE, fontsize=12, fontweight="bold")

# (a) Returns comparison: equilibrium vs BL vs historical
ax = axes[0]
x  = np.arange(N)
w_ = 0.25
ax.bar(x - w_, Pi*100,  width=w_, color=BLUE,   alpha=0.85, label="Equilibrium Pi")
ax.bar(x,      MU*100,  width=w_, color=GREEN,  alpha=0.85, label="Historical MU")
ax.bar(x + w_, mu_bl*100, width=w_, color=ORANGE, alpha=0.85, label="BL Posterior")
ax.set_xticks(x)
ax.set_xticklabels([n.replace(" ", "\n") for n in NAMES], fontsize=7)
ax.set_ylabel("Expected Return (%)")
ax.set_title("Return Estimates: Equilibrium | Historical | BL\n"
             "Views: EM Equity 13%, Commodities 10%",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True, axis="y"); watermark(ax)

# (b) Weight shift from equilibrium to BL
ax = axes[1]
w_eq_tan = tangency_portfolio(Pi, SIGMA, RF)
w_eq_tan = np.maximum(w_eq_tan, 0); w_eq_tan /= w_eq_tan.sum()
delta_w  = (w_bl - w_eq_tan) * 100
colors_d = [GREEN if d > 0 else RED for d in delta_w]
ax.bar(NAMES, delta_w, color=colors_d, alpha=0.85)
ax.axhline(0, color=WHITE, lw=1, linestyle=":")
for i, d in enumerate(delta_w):
    ax.text(i, d + (0.5 if d >= 0 else -1.0),
            f"{d:+.1f}%", ha="center", fontsize=8, color=WHITE)
ax.set_ylabel("Weight change (%)")
ax.set_xticklabels([n.replace(" ", "\n") for n in NAMES],
                   rotation=0, fontsize=7)
ax.set_title("BL Weight Change vs Equilibrium Tangency\n"
             "Views tilt weights toward EM Equity & Commodities",
             color=WHITE, fontsize=9)
ax.grid(True, axis="y"); watermark(ax)

# (c) Sensitivity to view confidence (omega scaling)
ax = axes[2]
tau_scales = np.logspace(-2, 1, 30)
sr_bl_arr  = []
for scale in tau_scales:
    omega_s = omega * scale
    mu_s, sig_s = black_litterman(Pi, SIGMA, P, Q, omega_s)
    w_s  = tangency_portfolio(mu_s, sig_s, RF)
    w_s  = np.maximum(w_s, 0); w_s /= w_s.sum()
    r_s, v_s, sr_s = portfolio_stats(w_s, MU, SIGMA)
    sr_bl_arr.append(sr_s)

sr_eq = portfolio_stats(w_eq_tan, MU, SIGMA)[2]
sr_hist = portfolio_stats(w_tan, MU, SIGMA)[2]

ax.semilogx(tau_scales, sr_bl_arr, color=ORANGE, lw=2.5,
            label="BL Sharpe (ex-post)")
ax.axhline(sr_eq,   color=BLUE,  lw=1.5, linestyle="--",
           label=f"Equilibrium SR = {sr_eq:.3f}")
ax.axhline(sr_hist, color=GREEN, lw=1.5, linestyle=":",
           label=f"Historical SR = {sr_hist:.3f}")
ax.set_xlabel("Omega scale factor (view uncertainty)")
ax.set_ylabel("Ex-post Sharpe Ratio")
ax.set_title("BL Sensitivity to View Confidence\n"
             "Low omega = confident views | High omega = equilibrium",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

plt.tight_layout()
p3 = os.path.join(OUT_DIR, "m29_03_black_litterman.png")
plt.savefig(p3, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p3}  ({time.perf_counter()-t0:.1f}s)")

# ── Summary ──────────────────────────────────────────────────────────────────
r_tan, v_tan, sr_tan = portfolio_stats(w_tan, MU, SIGMA)
r_rp,  v_rp,  sr_rp  = portfolio_stats(w_rp,  MU, SIGMA)
r_bl,  v_bl,  sr_bl  = portfolio_stats(w_bl,  MU, SIGMA)
print()
print("=" * 65)
print("  MODULE 29 COMPLETE -- 3 figures saved")
print("  Key Concepts:")
print("  [1] MV: min w'Sw  s.t. w'mu=mu_t, w'1=1  (parabola in mu-sigma)")
print("  [2] GMV: Sigma^{-1}*1 / (1'*Sigma^{-1}*1)  (analytic)")
print("  [3] Tangency: max Sharpe  w = Sigma^{-1}*(mu-rf) (normalized)")
print("  [4] CML: E[R] = rf + SR_T * sigma  (all efficient investors)")
print("  [5] RP: RC_i = w_i*(Sw)_i = Var/N  (equal risk)")
print("  [6] BL: posterior mu = weighted avg(Pi, views Q)")
print(f"  Tangency:      SR={sr_tan:.4f}  ret={r_tan*100:.2f}%  "
      f"vol={v_tan*100:.2f}%")
print(f"  Risk Parity:   SR={sr_rp:.4f}  ret={r_rp*100:.2f}%  "
      f"vol={v_rp*100:.2f}%")
print(f"  Black-Litterman: SR={sr_bl:.4f}  ret={r_bl*100:.2f}%  "
      f"vol={v_bl*100:.2f}%")
print("=" * 65)
