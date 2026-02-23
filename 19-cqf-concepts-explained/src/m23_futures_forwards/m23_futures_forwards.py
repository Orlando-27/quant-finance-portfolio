#!/usr/bin/env python3
"""
M23 — Futures & Forwards: Pricing and Basis Risk
=================================================
Module 23 | CQF Concepts Explained
Group 4   | Fixed Income & Interest Rates

Theory
------
A forward contract fixes today (t=0) the price F at which two parties
will exchange an asset at T. No money changes hands at inception.

Cost-of-Carry Pricing (no arbitrage)
-------------------------------------
For a non-dividend-paying asset:
    F(0,T) = S_0 * exp(r * T)

Intuition: borrow S_0 at rate r, buy the asset, deliver at T, repay loan.
Any deviation creates a riskless arbitrage.

With continuous dividend yield q (equities) or foreign rate r_f (FX):
    F(0,T) = S_0 * exp((r - q) * T)

With discrete dividends D_i at times t_i:
    F(0,T) = (S_0 - PV(dividends)) * exp(r * T)

For commodities with storage cost u and convenience yield y:
    F(0,T) = S_0 * exp((r + u - y) * T)

Backwardation: y > r + u  =>  F < S  (futures curve slopes down)
Contango:      y < r + u  =>  F > S  (futures curve slopes up)

Futures vs Forwards
-------------------
Futures are marked-to-market daily. When asset price is positively
correlated with interest rates, futures price > forward price (futures
gains are reinvested at higher rates). The adjustment is small for short
maturities.

Basis and Basis Risk
--------------------
Basis B_t = S_t - F_t (converges to 0 at expiry by no-arbitrage).
Hedgers face basis risk: uncertainty in B at the time the hedge is lifted.
The hedge is imperfect when hedge horizon != futures expiry.

Optimal Hedge Ratio (minimum variance)
----------------------------------------
A portfolio: one unit long spot, h units short futures.
    Var(dS - h*dF) minimised at:
    h* = Cov(dS, dF) / Var(dF) = rho * (sigma_S / sigma_F)

where rho is the correlation between spot and futures price changes.
Hedge effectiveness = rho^2 (R^2 of the hedged portfolio).

References
----------
- Hull, J.C. (2022). Options, Futures, and Other Derivatives. 11th ed.
  Pearson. Chapters 2-5.
- Keynes, J.M. (1930). A Treatise on Money. (Normal backwardation theory)
- Working, H. (1949). The theory of price of storage. American Economic
  Review, 39(6), 1254-1262.
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Styling ──────────────────────────────────────────────────────────────────
DARK   = "#0a0a0a"
PANEL  = "#111111"
GRID   = "#1e1e1e"
WHITE  = "#e8e8e8"
BLUE   = "#4a9eff"
GREEN  = "#00d4aa"
ORANGE = "#ff8c42"
RED    = "#ff4757"
PURPLE = "#a855f7"
YELLOW = "#ffd700"

WATERMARK = "Jose O. Bobadilla | CQF"
OUT_DIR   = os.path.expanduser(
    "~/quant-finance-portfolio/19-cqf-concepts-explained/outputs"
)
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor":  DARK,  "axes.facecolor":  PANEL,
    "axes.edgecolor":    GRID,  "axes.labelcolor": WHITE,
    "axes.titlecolor":   WHITE, "xtick.color":     WHITE,
    "ytick.color":       WHITE, "text.color":      WHITE,
    "grid.color":        GRID,  "grid.linewidth":  0.6,
    "legend.facecolor":  PANEL, "legend.edgecolor":GRID,
    "font.family":       "monospace",
    "axes.spines.top":   False, "axes.spines.right": False,
})

def watermark(ax):
    ax.text(0.99, 0.02, WATERMARK, transform=ax.transAxes,
            fontsize=7, color=WHITE, alpha=0.35, ha="right", va="bottom",
            fontstyle="italic")

# ============================================================
# SECTION 1 — PRICING FUNCTIONS
# ============================================================

def forward_price(S0, r, T, q=0.0):
    """
    Forward/futures price under cost-of-carry.
    S0: spot price
    r : risk-free rate (continuous)
    T : time to maturity (years)
    q : continuous dividend/foreign rate yield
    """
    return S0 * np.exp((r - q) * T)

def forward_price_commodity(S0, r, T, u, y):
    """
    Commodity forward with storage cost u and convenience yield y.
    F = S0 * exp((r + u - y) * T)
    """
    return S0 * np.exp((r + u - y) * T)

def forward_value(S0, F0, r, T, t):
    """
    Value of a long forward at time t (entered at t=0 with delivery price F0).
    V_t = S_t * exp(-q*(T-t)) - F0 * exp(-r*(T-t))
    For q=0: V_t = S_t - F0 * exp(-r*(T-t))
    """
    tau = T - t
    return S0 - F0 * np.exp(-r * tau)

def implied_convenience_yield(S0, F, r, u, T):
    """Back out convenience yield from observed futures price."""
    return r + u - np.log(F / S0) / T

def optimal_hedge_ratio(rho, sigma_S, sigma_F):
    """Minimum-variance hedge ratio h* = rho * sigma_S / sigma_F."""
    return rho * sigma_S / sigma_F

# ============================================================
# SECTION 2 — BASIS SIMULATION
# ============================================================

def simulate_basis(S0, r, q, T, n_steps, n_paths, sigma_S, seed=42):
    """
    Simulate spot price paths (GBM) and corresponding futures prices.
    Futures: F(t,T) = S_t * exp((r-q)*(T-t))
    Basis: B_t = S_t - F(t,T)  -> converges to 0 at T.
    Returns times, spot paths, futures paths, basis paths.
    """
    rng    = np.random.default_rng(seed)
    dt     = T / n_steps
    times  = np.linspace(0, T, n_steps + 1)
    S      = np.zeros((n_paths, n_steps + 1))
    S[:, 0] = S0
    Z = rng.standard_normal((n_paths, n_steps))
    sqrt_dt = np.sqrt(dt)
    for i in range(n_steps):
        S[:, i+1] = S[:, i] * np.exp(
            (r - q - 0.5*sigma_S**2)*dt + sigma_S*sqrt_dt*Z[:, i]
        )
    # Futures price at each t
    tau  = T - times                              # (n_steps+1,)
    F    = S * np.exp((r - q) * tau[np.newaxis, :])
    basis = S - F
    return times, S, F, basis

# ============================================================
# FIGURE 1 — Cost-of-carry across asset classes
# ============================================================
t0 = time.perf_counter()
print("[M23] Figure 1: Cost-of-carry pricing across asset classes ...")

S0 = 100.0; r = 0.05; T_max = 2.0
T_arr = np.linspace(0.01, T_max, 200)

# Asset class parameters
scenarios = [
    ("Equity (q=0%)",         forward_price(S0, r, T_arr, q=0.00),   BLUE),
    ("Equity (q=2%)",         forward_price(S0, r, T_arr, q=0.02),   GREEN),
    ("FX (r_f=3%)",           forward_price(S0, r, T_arr, q=0.03),   ORANGE),
    ("Commodity (contango)",  forward_price_commodity(S0,r,T_arr,u=0.03,y=0.01), PURPLE),
    ("Commodity (backw.)",    forward_price_commodity(S0,r,T_arr,u=0.01,y=0.08), RED),
]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor(DARK)
fig.suptitle("M23 — Futures & Forwards: Pricing\nCost-of-Carry Model",
             color=WHITE, fontsize=12, fontweight="bold")

ax = axes[0]
for label, F_arr, col in scenarios:
    ax.plot(T_arr, F_arr, color=col, lw=2, label=label)
ax.axhline(S0, color=WHITE, lw=1, linestyle=":", alpha=0.7, label=f"Spot S0={S0}")
ax.set_xlabel("Maturity T (years)"); ax.set_ylabel("Forward price")
ax.set_title("F(0,T) = S0 * exp((r-q)*T)\nAcross asset classes", color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# Carry decomposition: r, q, and net carry
ax = axes[1]
q_arr = np.linspace(0, 0.10, 200)
net_carry = r - q_arr
F_matrix  = S0 * np.exp(net_carry[:, np.newaxis] * T_arr[np.newaxis, :])
c = ax.contourf(T_arr, q_arr*100, F_matrix, levels=30, cmap="plasma")
plt.colorbar(c, ax=ax, label="Forward price")
ax.set_xlabel("Maturity T (years)"); ax.set_ylabel("Dividend yield q (%)")
ax.set_title(f"F(0,T) surface  (S0={S0}, r={r*100:.0f}%)\nContour: forward price level",
             color=WHITE, fontsize=9)
watermark(ax)

# Forward value over time (mark-to-market)
ax = axes[2]
F0 = forward_price(S0, r, T_max)  # delivery price locked at inception
print(f"      Delivery price F0 = {F0:.4f}")
t_arr = np.linspace(0, T_max * 0.99, 200)
for S_t in [80, 90, 100, 110, 120]:
    V = forward_value(S_t, F0, r, T_max, t_arr)
    col = GREEN if S_t > S0 else (RED if S_t < S0 else WHITE)
    ax.plot(t_arr, V, color=col, lw=2, label=f"S_t={S_t}")
ax.axhline(0, color=WHITE, lw=1, linestyle=":", alpha=0.7)
ax.set_xlabel("Time t"); ax.set_ylabel("Forward value V_t")
ax.set_title(f"Value of Long Forward Over Time\nF0={F0:.2f}, r={r*100:.0f}%, T={T_max}Y",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

plt.tight_layout()
p1 = os.path.join(OUT_DIR, "m23_01_pricing.png")
plt.savefig(p1, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p1}  ({time.perf_counter()-t0:.1f}s)")

# ============================================================
# FIGURE 2 — Basis convergence and basis risk
# ============================================================
t0 = time.perf_counter()
print("[M23] Figure 2: Basis convergence and basis risk ...")

S0_sim = 100.0; r_sim = 0.05; q_sim = 0.02
T_sim  = 1.0;   sigma_S = 0.20
N_PATHS = 300;  N_STEPS = 252

times, S_paths, F_paths, basis_paths = simulate_basis(
    S0_sim, r_sim, q_sim, T_sim, N_STEPS, N_PATHS, sigma_S
)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.patch.set_facecolor(DARK)
fig.suptitle("M23 — Basis Dynamics and Basis Risk\n"
             "B_t = S_t - F(t,T)  ->  0 as t -> T",
             color=WHITE, fontsize=12, fontweight="bold")

# Spot and futures paths
ax = axes[0, 0]
for j in range(min(80, N_PATHS)):
    ax.plot(times, S_paths[j], lw=0.4, alpha=0.25, color=BLUE)
    ax.plot(times, F_paths[j], lw=0.4, alpha=0.15, color=GREEN)
ax.plot(times, S_paths.mean(axis=0), color=BLUE,  lw=2, label="Mean spot S_t")
ax.plot(times, F_paths.mean(axis=0), color=GREEN, lw=2, label="Mean futures F(t,T)")
ax.set_xlabel("Time (years)"); ax.set_ylabel("Price")
ax.set_title("Spot vs Futures Paths (GBM)\n80 sample paths shown",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# Basis paths — convergence to zero
ax = axes[0, 1]
for j in range(min(80, N_PATHS)):
    ax.plot(times, basis_paths[j], lw=0.4, alpha=0.3, color=ORANGE)
ax.plot(times, basis_paths.mean(axis=0), color=WHITE,  lw=2, label="Mean basis")
ax.fill_between(times,
                np.percentile(basis_paths, 5,  axis=0),
                np.percentile(basis_paths, 95, axis=0),
                color=ORANGE, alpha=0.15, label="5th-95th percentile")
ax.axhline(0, color=RED, lw=1.5, linestyle="--", label="Convergence target (0)")
ax.set_xlabel("Time (years)"); ax.set_ylabel("Basis B_t = S_t - F(t,T)")
ax.set_title("Basis Convergence to Zero at Expiry\nBasis risk = uncertainty in B at hedge lifting",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# Basis risk: distribution at different horizons
ax = axes[1, 0]
horizons = [0.25, 0.50, 0.75, 1.00]
colors_h  = [BLUE, GREEN, ORANGE, RED]
for h, col in zip(horizons, colors_h):
    idx = int(h / T_sim * N_STEPS)
    b   = basis_paths[:, idx]
    ax.hist(b, bins=30, alpha=0.55, density=True, color=col,
            label=f"t={h:.2f}Y  std={b.std():.3f}")
ax.set_xlabel("Basis value"); ax.set_ylabel("Density")
ax.set_title("Basis Distribution at Different Horizons\nVariance shrinks as t -> T",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# Basis std dev over time — decreases monotonically
ax = axes[1, 1]
basis_std = basis_paths.std(axis=0)
ax.plot(times, basis_std, color=PURPLE, lw=2.5, label="std(Basis_t)")
# Theoretical: Var(B_t) = Var(S_t - F(t,T)) = Var(S_t)*exp(-2*(r-q)*(T-t)) deviation
# Approx: std(B_t) ~ sigma_S * S0 * sqrt(t) * |1 - exp(...)| (complex; show MC)
ax.fill_between(times, 0, basis_std, color=PURPLE, alpha=0.12)
ax.set_xlabel("Time (years)"); ax.set_ylabel("Basis std deviation")
ax.set_title("Basis Risk Decreasing Towards Expiry\nstd(B_t) -> 0 as t -> T",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

plt.tight_layout()
p2 = os.path.join(OUT_DIR, "m23_02_basis_risk.png")
plt.savefig(p2, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p2}  ({time.perf_counter()-t0:.1f}s)")

# ============================================================
# FIGURE 3 — Optimal hedge ratio and hedging effectiveness
# ============================================================
t0 = time.perf_counter()
print("[M23] Figure 3: Optimal hedge ratio and hedging effectiveness ...")

# Simulate correlated spot and futures returns
rng     = np.random.default_rng(0)
n_obs   = 1000
rho_true = 0.85
sigma_S_ = 0.20 / np.sqrt(252)
sigma_F_ = 0.18 / np.sqrt(252)

# Correlated normals via Cholesky
L = np.array([[1, 0],
              [rho_true, np.sqrt(1 - rho_true**2)]])
Z = rng.standard_normal((2, n_obs))
W = L @ Z
dS = sigma_S_ * W[0]
dF = sigma_F_ * W[1]

# Empirical hedge ratio
h_star_emp = np.cov(dS, dF)[0, 1] / np.var(dF)
rho_emp    = np.corrcoef(dS, dF)[0, 1]
h_star_th  = optimal_hedge_ratio(rho_true, sigma_S_, sigma_F_)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor(DARK)
fig.suptitle("M23 — Optimal Hedge Ratio\nh* = rho * (sigma_S / sigma_F)  |  Effectiveness = rho^2",
             color=WHITE, fontsize=12, fontweight="bold")

# (a) Scatter dS vs dF with regression line
ax = axes[0]
ax.scatter(dF*100, dS*100, alpha=0.15, s=8, color=BLUE, label="Observations")
x_line = np.array([dF.min(), dF.max()])
ax.plot(x_line*100, h_star_emp*x_line*100, color=ORANGE, lw=2,
        label=f"OLS slope = h* = {h_star_emp:.4f}")
ax.set_xlabel("dF (%)"); ax.set_ylabel("dS (%)")
ax.set_title(f"Spot vs Futures Returns\nrho={rho_emp:.4f}  h*={h_star_emp:.4f}",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (b) Variance reduction across hedge ratios
ax = axes[1]
h_arr     = np.linspace(-0.2, 1.8, 300)
var_hedged = np.var(dS) + h_arr**2 * np.var(dF) - 2*h_arr*np.cov(dS, dF)[0, 1]
var_unhd   = np.var(dS)
ax.plot(h_arr, var_hedged / var_unhd * 100, color=GREEN, lw=2.5)
ax.axvline(h_star_emp, color=ORANGE, lw=1.5, linestyle="--",
           label=f"h* = {h_star_emp:.4f}  (minimum)")
ax.axhline((1 - rho_emp**2)*100, color=RED, lw=1.5, linestyle=":",
           label=f"Min = {(1-rho_emp**2)*100:.1f}%  (1 - rho^2)")
ax.set_xlabel("Hedge ratio h"); ax.set_ylabel("Hedged variance / Unhedged variance (%)")
ax.set_title("Variance Reduction vs Hedge Ratio\nMinimum at h = rho*(sigma_S/sigma_F)",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (c) Hedge effectiveness vs rho (across different asset pairs)
ax = axes[2]
rho_arr = np.linspace(0, 1, 200)
eff_arr = rho_arr**2 * 100
ax.plot(rho_arr, eff_arr, color=PURPLE, lw=2.5)
ax.fill_between(rho_arr, eff_arr, alpha=0.12, color=PURPLE)
# Mark representative asset pairs
pairs = [
    ("S&P vs E-mini", 0.999, GREEN),
    ("WTI vs Brent",  0.97,  ORANGE),
    ("Jet vs WTI",    0.85,  YELLOW),
    ("Corn vs Wheat", 0.70,  RED),
]
for label, rho_p, col in pairs:
    eff = rho_p**2 * 100
    ax.scatter([rho_p], [eff], color=col, s=80, zorder=5)
    ax.annotate(f"{label}\n({eff:.1f}%)", (rho_p, eff),
                textcoords="offset points", xytext=(-40, 8),
                fontsize=6.5, color=col)
ax.set_xlabel("Correlation rho"); ax.set_ylabel("Hedge effectiveness (%)")
ax.set_title("Hedge Effectiveness = rho^2\nAcross representative asset pairs",
             color=WHITE, fontsize=9)
ax.grid(True); watermark(ax)

plt.tight_layout()
p3 = os.path.join(OUT_DIR, "m23_03_hedge_ratio.png")
plt.savefig(p3, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p3}  ({time.perf_counter()-t0:.1f}s)")

# ── Summary ──────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("  MODULE 23 COMPLETE -- 3 figures saved")
print("  Key Concepts:")
print("  [1] F(0,T) = S0*exp((r-q)*T)  — cost-of-carry pricing")
print("  [2] Commodity: F = S0*exp((r+u-y)*T)  u=storage, y=conv.yield")
print("  [3] Backwardation: y > r+u  =>  F < S  (convenience dominates)")
print("  [4] Basis B_t = S_t - F(t,T)  ->  0 at expiry (no-arb)")
print("  [5] Hedge ratio h* = rho*(sigma_S/sigma_F)  (min variance)")
print("  [6] Hedge effectiveness = rho^2  (R-squared of hedge)")
print(f"  Empirical h* = {h_star_emp:.4f}  |  Theoretical = {h_star_th:.4f}")
print(f"  Hedge effectiveness = {rho_emp**2*100:.2f}%  (rho={rho_emp:.4f})")
print("=" * 65)
