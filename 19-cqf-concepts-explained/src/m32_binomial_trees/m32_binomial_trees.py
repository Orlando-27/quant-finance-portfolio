#!/usr/bin/env python3
"""
M32 — Binomial Trees: CRR and American Options
===============================================
Module 32 | CQF Concepts Explained
Group 7   | Derivatives Pricing

Theory
------
Cox-Ross-Rubinstein (CRR) Binomial Model (1979)
-------------------------------------------------
Discretize time [0,T] into n steps of length dt = T/n.
At each node the asset can move up by factor u or down by d:

    u = exp(sigma * sqrt(dt))
    d = 1/u  = exp(-sigma * sqrt(dt))   [recombining tree]

Risk-neutral probability (no-arbitrage):
    p = (exp((r-q)*dt) - d) / (u - d)
    1-p = (u - exp((r-q)*dt)) / (u - d)

Asset price at node (i, j):  S_{i,j} = S_0 * u^j * d^{i-j}
    where i = time step, j = number of up-moves (0 <= j <= i)

Backward induction (European):
    V_{i,j} = exp(-r*dt) * [p*V_{i+1,j+1} + (1-p)*V_{i+1,j}]
Terminal condition: V_{n,j} = max(S_{n,j} - K, 0)  (call)

American Option Modification
-----------------------------
At each interior node, compare continuation value with immediate
exercise value:
    V_{i,j}^{Am} = max(payoff(S_{i,j}), exp(-r*dt)*[p*V^{Am}_{i+1,j+1}
                                                     + (1-p)*V^{Am}_{i+1,j}])

Early exercise is optimal when the intrinsic value exceeds the
discounted expected future value. This occurs:
    - American put: deep ITM when interest savings > optionality value
    - American call (no dividend): NEVER optimal (proven analytically)
    - American call (with dividend): may be optimal just before ex-date

Convergence to Black-Scholes
------------------------------
As n -> inf:
    CRR call price -> BS call price
    Convergence rate: O(1/n) with oscillation (even/odd n behavior)
    Smooth convergence achieved by averaging n and n+1 step results,
    or using Richardson extrapolation.

Early Exercise Boundary
------------------------
The critical stock price S*(t) below which immediate exercise is
optimal for an American put. Satisfies the free-boundary condition:
    V(S*(t), t) = K - S*(t)  (value = intrinsic)
    dV/dS|_{S=S*(t)} = -1    (smooth pasting condition)

Binomial Greeks
---------------
Delta: [V(up) - V(down)] / [S*u - S*d]  (centered finite difference)
Gamma: [Delta(up) - Delta(down)] / [0.5*(S*u^2 - S*d^2)]

References
----------
- Cox, J., Ross, S., Rubinstein, M. (1979). Option pricing: a simplified
  approach. Journal of Financial Economics, 7(3), 229-263.
- Broadie, M., Detemple, J. (1996). American option valuation.
  Review of Financial Studies, 9(4), 1211-1250.
- Hull, J.C. (2022). Options, Futures, and Other Derivatives.
  Chapters 13, 21. Pearson.
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

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
# SECTION 1 — CRR BINOMIAL ENGINE
# ============================================================

def crr_params(sigma, r, q, dt):
    """CRR up/down factors and risk-neutral probability."""
    u  = np.exp(sigma * np.sqrt(dt))
    d  = 1.0 / u
    p  = (np.exp((r - q) * dt) - d) / (u - d)
    df = np.exp(-r * dt)          # one-step discount factor
    return u, d, p, df

def binomial_price(S, K, r, T, sigma, n, q=0.0,
                   option="call", style="european"):
    """
    CRR binomial tree pricer.
    Returns option price and optionally the exercise boundary.
    style: 'european' or 'american'
    """
    dt       = T / n
    u, d, p, df = crr_params(sigma, r, q, dt)

    # Terminal asset prices and payoffs
    j_arr    = np.arange(n + 1)
    S_T      = S * (u ** j_arr) * (d ** (n - j_arr))
    if option == "call":
        V = np.maximum(S_T - K, 0.0)
    else:
        V = np.maximum(K - S_T, 0.0)

    # Backward induction
    early_boundary = []   # (time_step, S*) for American puts
    for i in range(n - 1, -1, -1):
        j_arr_i = np.arange(i + 1)
        S_i     = S * (u ** j_arr_i) * (d ** (i - j_arr_i))
        cont    = df * (p * V[1:i+2] + (1-p) * V[0:i+1])
        if style == "american":
            if option == "call":
                intrinsic = np.maximum(S_i - K, 0.0)
            else:
                intrinsic = np.maximum(K - S_i, 0.0)
            V = np.maximum(cont, intrinsic)
            # Early exercise boundary: lowest S where exercise is optimal
            ex_mask = V == intrinsic
            if ex_mask.any() and option == "put":
                S_boundary = S_i[ex_mask].max()
                early_boundary.append((i * dt, S_boundary))
        else:
            V = cont

    return V[0], early_boundary

def bs_price(S, K, r, T, sigma, q=0.0, option="call"):
    """Black-Scholes reference price."""
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*sqrt_T)
    d2 = d1 - sigma*sqrt_T
    if option == "call":
        return (S*np.exp(-q*T)*norm.cdf(d1)
                - K*np.exp(-r*T)*norm.cdf(d2))
    else:
        return (K*np.exp(-r*T)*norm.cdf(-d2)
                - S*np.exp(-q*T)*norm.cdf(-d1))

def binomial_greeks(S, K, r, T, sigma, n, q=0.0, option="call",
                    style="european"):
    """Delta and Gamma via two-step centered differences."""
    dt      = T / n
    u, d, _, _ = crr_params(sigma, r, q, dt)
    Su = S * u;  Sd = S * d
    Suu = S * u**2; Sdd = S * d**2; Sud = S

    Vu  = binomial_price(Su,  K, r, T-dt,   sigma, n-1, q, option, style)[0]
    Vd  = binomial_price(Sd,  K, r, T-dt,   sigma, n-1, q, option, style)[0]
    Vuu = binomial_price(Suu, K, r, T-2*dt, sigma, n-2, q, option, style)[0]
    Vdd = binomial_price(Sdd, K, r, T-2*dt, sigma, n-2, q, option, style)[0]
    Vud = binomial_price(Sud, K, r, T-2*dt, sigma, n-2, q, option, style)[0]

    delta = (Vu - Vd) / (Su - Sd)
    gamma = ((Vuu - Vud) / (Suu - Sud) - (Vud - Vdd) / (Sud - Sdd)) / (
             0.5 * (Suu - Sdd))
    return delta, gamma

# ============================================================
# SECTION 2 — BASE CASE
# ============================================================

S0 = 100.0; K0 = 100.0; r0 = 0.05; T0 = 1.0; sigma0 = 0.20; q0 = 0.0

bs_eu_call = bs_price(S0, K0, r0, T0, sigma0, q0, "call")
bs_eu_put  = bs_price(S0, K0, r0, T0, sigma0, q0, "put")

N_STEPS = 200
crr_eu_call, _ = binomial_price(S0, K0, r0, T0, sigma0, N_STEPS, q0,
                                 "call", "european")
crr_eu_put,  _ = binomial_price(S0, K0, r0, T0, sigma0, N_STEPS, q0,
                                 "put",  "european")
crr_am_call, _ = binomial_price(S0, K0, r0, T0, sigma0, N_STEPS, q0,
                                 "call", "american")
crr_am_put, bd = binomial_price(S0, K0, r0, T0, sigma0, N_STEPS, q0,
                                 "put",  "american")

print(f"[M32] CRR Binomial Tree (n={N_STEPS})")
print(f"      European call: CRR={crr_eu_call:.4f}  BS={bs_eu_call:.4f}  "
      f"err={abs(crr_eu_call-bs_eu_call):.2e}")
print(f"      European put:  CRR={crr_eu_put:.4f}   BS={bs_eu_put:.4f}  "
      f"err={abs(crr_eu_put-bs_eu_put):.2e}")
print(f"      American call: {crr_am_call:.4f}  "
      f"(= European: {crr_am_call==crr_eu_call})")
print(f"      American put:  {crr_am_put:.4f}  "
      f"early exercise premium: {crr_am_put-crr_eu_put:.4f}")

delta_eu, gamma_eu = binomial_greeks(S0, K0, r0, T0, sigma0, 50, q0,
                                      "put", "european")
delta_am, gamma_am = binomial_greeks(S0, K0, r0, T0, sigma0, 50, q0,
                                      "put", "american")
print(f"      European put Greeks: Delta={delta_eu:.4f}  Gamma={gamma_eu:.6f}")
print(f"      American put Greeks: Delta={delta_am:.4f}  Gamma={gamma_am:.6f}")

# ============================================================
# FIGURE 1 — Tree visualization and convergence
# ============================================================
t0 = time.perf_counter()
print("\n[M32] Figure 1: Tree visualization and BS convergence ...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor(DARK)
fig.suptitle("M32 — CRR Binomial Tree\n"
             "Tree Structure | BS Convergence | Error Analysis",
             color=WHITE, fontsize=12, fontweight="bold")

# (a) Small tree: visualize nodes (n=5)
ax = axes[0]
n_vis = 5
dt_v  = T0 / n_vis
u_v, d_v, p_v, df_v = crr_params(sigma0, r0, q0, dt_v)
for i in range(n_vis + 1):
    for j in range(i + 1):
        S_node = S0 * u_v**j * d_v**(i-j)
        t_node = i * dt_v
        ax.scatter(t_node, S_node, color=BLUE, s=60, zorder=5)
        ax.text(t_node + 0.02, S_node, f"{S_node:.1f}",
                fontsize=6, color=WHITE, va="center")
        if i < n_vis:
            Su_n = S0 * u_v**(j+1) * d_v**((i+1)-(j+1))
            Sd_n = S0 * u_v**j     * d_v**((i+1)-j)
            ax.plot([t_node, (i+1)*dt_v], [S_node, Su_n], color=GREEN,
                    lw=1.5, alpha=0.7)
            ax.plot([t_node, (i+1)*dt_v], [S_node, Sd_n], color=RED,
                    lw=1.5, alpha=0.7)
ax.axhline(K0, color=YELLOW, lw=1, linestyle="--", label=f"Strike K={K0}")
ax.set_xlabel("Time"); ax.set_ylabel("Asset price S")
ax.set_title(f"CRR Recombining Tree (n={n_vis})\n"
             f"u={u_v:.4f}  d={d_v:.4f}  p={p_v:.4f}",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (b) Convergence to BS (European call)
ax = axes[1]
n_arr  = np.arange(2, 201)
prices = [binomial_price(S0, K0, r0, T0, sigma0, n, q0,
                          "call", "european")[0] for n in n_arr]
prices = np.array(prices)
ax.plot(n_arr, prices, color=BLUE, lw=1.5, alpha=0.8, label="CRR price")
ax.axhline(bs_eu_call, color=ORANGE, lw=2, linestyle="--",
           label=f"BS limit = {bs_eu_call:.4f}")
ax.fill_between(n_arr, prices, bs_eu_call,
                color=BLUE, alpha=0.15, label="Oscillation band")
ax.set_xlabel("Number of steps n"); ax.set_ylabel("Call price")
ax.set_title("CRR Convergence to Black-Scholes\n"
             "Even/odd oscillation -> BS as n->inf",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (c) Absolute error vs n (log-log)
ax = axes[2]
err_arr = np.abs(prices - bs_eu_call)
valid   = err_arr > 1e-8
ax.loglog(n_arr[valid], err_arr[valid], color=GREEN, lw=1.5,
          alpha=0.8, label="|CRR - BS|")
# Fit O(1/n) reference line
from numpy.polynomial import polynomial as P_fit
log_n   = np.log(n_arr[valid])
log_err = np.log(err_arr[valid])
slope, intercept = np.polyfit(log_n, log_err, 1)
ax.loglog(n_arr[valid], np.exp(intercept)*n_arr[valid]**slope,
          color=ORANGE, lw=2, linestyle="--",
          label=f"O(n^{{{slope:.2f}}}) reference")
ax.set_xlabel("n (log scale)"); ax.set_ylabel("|Error| (log scale)")
ax.set_title(f"Convergence Rate: O(1/n)\n"
             f"Empirical slope = {slope:.3f}",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

plt.tight_layout()
p1 = os.path.join(OUT_DIR, "m32_01_tree_convergence.png")
plt.savefig(p1, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p1}  ({time.perf_counter()-t0:.1f}s)")

# ============================================================
# FIGURE 2 — American vs European and early exercise boundary
# ============================================================
t0 = time.perf_counter()
print("[M32] Figure 2: American vs European and early exercise boundary ...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor(DARK)
fig.suptitle("M32 — American Options: Early Exercise Premium\n"
             "V_Am = max(intrinsic, continuation)  |  S*(t): free boundary",
             color=WHITE, fontsize=12, fontweight="bold")

# (a) American vs European put value vs spot
ax = axes[0]
S_arr = np.linspace(60, 140, 80)
n_fast = 100
eu_puts = [binomial_price(s, K0, r0, T0, sigma0, n_fast, q0,
                           "put", "european")[0] for s in S_arr]
am_puts = [binomial_price(s, K0, r0, T0, sigma0, n_fast, q0,
                           "put", "american")[0] for s in S_arr]
intrinsic = np.maximum(K0 - S_arr, 0)

ax.plot(S_arr, am_puts,  color=RED,   lw=2.5, label="American put")
ax.plot(S_arr, eu_puts,  color=BLUE,  lw=2.5, label="European put")
ax.plot(S_arr, intrinsic,color=WHITE, lw=1.5, linestyle=":",
        alpha=0.7, label="Intrinsic value")
ax.fill_between(S_arr, am_puts, eu_puts,
                color=RED, alpha=0.20, label="Early exercise premium")
ax.axvline(K0, color=YELLOW, lw=1, linestyle="--", alpha=0.5,
           label=f"K={K0}")
ax.set_xlabel("Spot S"); ax.set_ylabel("Put price")
ax.set_title("American vs European Put\n"
             "Early exercise premium = Am - Eu > 0",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (b) Early exercise boundary S*(t) for American put
if bd:
    bd_arr  = np.array(bd)
    t_bd    = bd_arr[:, 0]
    S_bd    = bd_arr[:, 1]
    # Sort by time
    sort_idx = np.argsort(t_bd)
    t_bd    = t_bd[sort_idx]
    S_bd    = S_bd[sort_idx]
    ax = axes[1]
    ax.plot(t_bd, S_bd, color=RED, lw=2.5, label="Early exercise boundary S*(t)")
    ax.fill_between(t_bd, 0, S_bd, color=RED, alpha=0.15,
                    label="Exercise region (S < S*)")
    ax.fill_between(t_bd, S_bd, K0*1.5, color=BLUE, alpha=0.10,
                    label="Hold region (S > S*)")
    ax.axhline(K0, color=YELLOW, lw=1.5, linestyle="--",
               label=f"Strike K={K0}")
    ax.set_xlabel("Time t"); ax.set_ylabel("Critical stock price S*(t)")
    ax.set_title("Free Boundary: Early Exercise Region\n"
                 "Exercise immediately when S < S*(t)",
                 color=WHITE, fontsize=9)
    ax.legend(fontsize=7); ax.grid(True); watermark(ax)
else:
    axes[1].text(0.5, 0.5, "No early exercise\nboundary detected",
                 ha="center", va="center", color=WHITE,
                 transform=axes[1].transAxes)

# (c) Early exercise premium vs volatility and interest rate
ax = axes[2]
sigma_arr = np.linspace(0.05, 0.50, 30)
r_arr     = [0.01, 0.05, 0.10]
r_cols    = [BLUE, GREEN, ORANGE]
for r_, col in zip(r_arr, r_cols):
    prems = []
    for s_ in sigma_arr:
        eu = binomial_price(S0, K0, r_, T0, s_, 80, 0, "put", "european")[0]
        am = binomial_price(S0, K0, r_, T0, s_, 80, 0, "put", "american")[0]
        prems.append(am - eu)
    ax.plot(sigma_arr*100, prems, color=col, lw=2,
            label=f"r={r_*100:.0f}%")
ax.set_xlabel("Volatility sigma (%)"); ax.set_ylabel("Early exercise premium")
ax.set_title("Early Exercise Premium\nvs Volatility and Interest Rate\n"
             "Higher r => larger premium (interest on K)",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

plt.tight_layout()
p2 = os.path.join(OUT_DIR, "m32_02_american_early_exercise.png")
plt.savefig(p2, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p2}  ({time.perf_counter()-t0:.1f}s)")

# ============================================================
# FIGURE 3 — Dividend effects and Greeks from tree
# ============================================================
t0 = time.perf_counter()
print("[M32] Figure 3: Dividend effects and binomial Greeks ...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor(DARK)
fig.suptitle("M32 — Dividend Effects and Binomial Greeks\n"
             "American call exercised early only with dividends",
             color=WHITE, fontsize=12, fontweight="bold")

# (a) American call: effect of dividend yield on early exercise
ax = axes[0]
q_arr = np.linspace(0.0, 0.10, 40)
n_div = 100
eu_calls_q = [binomial_price(S0, K0, r0, T0, sigma0, n_div, q,
                              "call", "european")[0] for q in q_arr]
am_calls_q = [binomial_price(S0, K0, r0, T0, sigma0, n_div, q,
                              "call", "american")[0] for q in q_arr]
prem_q = np.array(am_calls_q) - np.array(eu_calls_q)

ax.plot(q_arr*100, am_calls_q, color=GREEN, lw=2.5, label="American call")
ax.plot(q_arr*100, eu_calls_q, color=BLUE,  lw=2.5, label="European call")
ax.fill_between(q_arr*100, am_calls_q, eu_calls_q,
                color=GREEN, alpha=0.20, label="Early exercise premium")
ax.set_xlabel("Dividend yield q (%)"); ax.set_ylabel("Call price")
ax.set_title("American Call: Early Exercise with Dividends\n"
             "q=0: Am=Eu (no early exercise) | q>0: Am>Eu",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (b) Delta comparison: European vs American put vs BS
ax = axes[1]
S_delta = np.linspace(70, 130, 50)
n_gr = 60
delta_eu_arr = [binomial_greeks(s, K0, r0, T0, sigma0, n_gr, q0,
                                 "put", "european")[0] for s in S_delta]
delta_am_arr = [binomial_greeks(s, K0, r0, T0, sigma0, n_gr, q0,
                                 "put", "american")[0] for s in S_delta]
# BS delta for European put
from scipy.stats import norm as norm_sp
def bs_delta_put(S, K, r, T, sigma, q=0):
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S/K) + (r-q+0.5*sigma**2)*T) / (sigma*sqrt_T)
    return -np.exp(-q*T) * norm_sp.cdf(-d1)

bs_delta_arr = [bs_delta_put(s, K0, r0, T0, sigma0, q0) for s in S_delta]

ax.plot(S_delta, delta_eu_arr, color=BLUE,   lw=2.5, label="European put delta (CRR)")
ax.plot(S_delta, delta_am_arr, color=RED,    lw=2.5, label="American put delta (CRR)")
ax.plot(S_delta, bs_delta_arr, color=ORANGE, lw=1.5, linestyle="--",
        label="BS European delta")
ax.axhline(-1, color=WHITE, lw=1, linestyle=":", alpha=0.5, label="Delta=-1 (deep ITM)")
ax.axvline(K0, color=YELLOW, lw=1, linestyle="--", alpha=0.5)
ax.set_xlabel("Spot S"); ax.set_ylabel("Delta")
ax.set_title("Put Delta: European vs American\n"
             "Am delta reaches -1 faster (early exercise)",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (c) Gamma profile
ax = axes[2]
gamma_eu_arr = [binomial_greeks(s, K0, r0, T0, sigma0, n_gr, q0,
                                 "put", "european")[1] for s in S_delta]
gamma_am_arr = [binomial_greeks(s, K0, r0, T0, sigma0, n_gr, q0,
                                 "put", "american")[1] for s in S_delta]
ax.plot(S_delta, gamma_eu_arr, color=BLUE, lw=2.5, label="European put gamma")
ax.plot(S_delta, gamma_am_arr, color=RED,  lw=2.5, label="American put gamma")
ax.axvline(K0, color=YELLOW, lw=1, linestyle="--", alpha=0.5,
           label=f"ATM K={K0}")
ax.axhline(0, color=WHITE, lw=0.8, linestyle=":", alpha=0.5)
ax.set_xlabel("Spot S"); ax.set_ylabel("Gamma")
ax.set_title("Put Gamma: European vs American\n"
             "Am gamma can be negative in exercise region",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

plt.tight_layout()
p3 = os.path.join(OUT_DIR, "m32_03_dividends_greeks.png")
plt.savefig(p3, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p3}  ({time.perf_counter()-t0:.1f}s)")

# ── Summary ──────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("  MODULE 32 COMPLETE -- 3 figures saved")
print("  Key Concepts:")
print("  [1] CRR: u=exp(s*sqrt(dt)), d=1/u, p=(e^{r*dt}-d)/(u-d)")
print("  [2] Backward induction: V=df*[p*Vu+(1-p)*Vd] (European)")
print("  [3] American: V=max(intrinsic, continuation) at each node")
print("  [4] Am call (no dividend) = Eu call  (never exercise early)")
print("  [5] Am put > Eu put: early exercise premium = interest on K")
print("  [6] Free boundary S*(t): smooth pasting dV/dS|_{S*} = -1")
print(f"  European call: CRR={crr_eu_call:.4f}  BS={bs_eu_call:.4f}  "
      f"err={abs(crr_eu_call-bs_eu_call):.2e}")
print(f"  American put:  {crr_am_put:.4f}  "
      f"premium={crr_am_put-crr_eu_put:.4f} over European")
print(f"  Am call (q=0): {crr_am_call:.4f} = European: "
      f"{abs(crr_am_call-crr_eu_call)<1e-6}")
print("=" * 65)
