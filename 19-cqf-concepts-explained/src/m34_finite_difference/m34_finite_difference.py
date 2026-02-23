#!/usr/bin/env python3
"""
M34 — Finite Difference Methods: Explicit, Implicit, Crank-Nicolson
=====================================================================
Module 34 | CQF Concepts Explained
Group 7   | Derivatives Pricing

Theory
------
The Black-Scholes PDE for a European option V(S,t):
    dV/dt + (1/2)*sigma^2*S^2 * d2V/dS2 + r*S*dV/dS - r*V = 0

with terminal condition V(S,T) = payoff(S) and boundary conditions:
    V(0,t)    = 0           (call); K*exp(-r*(T-t)) (put)
    V(S_max,t) ~ S - K*exp(-r*(T-t)) (call); 0 (put, large S)

Log-space transformation: x = ln(S), tau = T - t
    dV/dtau = (1/2)*sigma^2*d2V/dx2 + (r - sigma^2/2)*dV/dx - r*V
(constant coefficient PDE => simpler finite difference stencils)

Discretization
--------------
Grid: x_i = x_min + i*dx  (i=0,...,M)
      tau_j = j*dtau       (j=0,...,N)

Explicit (Forward Euler in tau):
    V_i^{j+1} = V_i^j + dtau * L(V^j)
    where L = (sig^2/2)*D_{xx} + (r-sig^2/2)*D_x - r
    Stability: dtau <= dx^2 / sigma^2  (CFL condition)

Implicit (Backward Euler in tau):
    V_i^{j+1} - dtau*L(V^{j+1}) = V_i^j
    => Tridiagonal system A*V^{j+1} = V^j  (solved via Thomas algorithm)
    Unconditionally stable.

Crank-Nicolson (average of explicit and implicit):
    V^{j+1} - (dtau/2)*L(V^{j+1}) = V^j + (dtau/2)*L(V^j)
    => A_CN * V^{j+1} = B_CN * V^j
    Second-order in both dtau and dx. Unconditionally stable.
    May exhibit spurious oscillations near discontinuities (Rannacher fix).

Thomas Algorithm (tridiagonal solver)
---------------------------------------
For system: a_i*V_{i-1} + b_i*V_i + c_i*V_{i+1} = d_i
Forward sweep: c'_i = c_i/(b_i - a_i*c'_{i-1})
               d'_i = (d_i - a_i*d'_{i-1})/(b_i - a_i*c'_{i-1})
Backward sweep: V_i = d'_i - c'_i*V_{i+1}
O(N) complexity vs O(N^3) for general matrix solve.

Truncation Error
-----------------
Explicit:    O(dtau + dx^2)  — first-order in time
Implicit:    O(dtau + dx^2)  — first-order in time
CN:          O(dtau^2 + dx^2)  — second-order in time

References
----------
- Wilmott, P., Dewynne, J., Howison, S. (1993). Option Pricing:
  Mathematical Models and Computation. Oxford Financial Press.
- Thomas, L.H. (1949). Elliptic Problems in Linear Difference Equations
  over a Network. Watson Sci. Comput. Lab. Report.
- Rannacher, R. (1984). Finite element solution of diffusion problems
  with irregular data. Numerische Mathematik, 43(2), 309-327.
- Duffy, D.J. (2006). Finite Difference Methods in Financial Engineering.
  Wiley Finance.
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.linalg import solve_banded

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
# SECTION 1 — BLACK-SCHOLES REFERENCE
# ============================================================

def bs_price(S, K, r, T, sigma, q=0.0, option="call"):
    if T <= 0:
        return max(S-K, 0) if option=="call" else max(K-S, 0)
    d1 = (np.log(S/K) + (r-q+0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if option == "call":
        return S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1)

# ============================================================
# SECTION 2 — FINITE DIFFERENCE SOLVERS (log-space)
# ============================================================

def build_grid(S0, K, r, sigma, T, M, N, S_max_mult=4.0):
    """Build log-space grid parameters."""
    x_min = np.log(K / S_max_mult)
    x_max = np.log(K * S_max_mult)
    dx    = (x_max - x_min) / M
    dtau  = T / N
    x_arr = np.linspace(x_min, x_max, M+1)
    S_arr = np.exp(x_arr)
    return x_arr, S_arr, dx, dtau

def terminal_condition(S_arr, K, option="call"):
    if option == "call":
        return np.maximum(S_arr - K, 0.0)
    return np.maximum(K - S_arr, 0.0)

def boundary_conditions(S_arr, K, r, tau, option="call"):
    """Boundary values at S_min and S_max."""
    S_min, S_max = S_arr[0], S_arr[-1]
    if option == "call":
        bc_low  = 0.0
        bc_high = S_max - K * np.exp(-r * tau)
    else:
        bc_low  = K * np.exp(-r * tau) - S_min
        bc_high = 0.0
    return bc_low, bc_high

def fdm_coefficients(sigma, r, dx):
    """
    FDM coefficients for the PDE in log-space.
    alpha, beta, gamma for central difference discretization.
    """
    a = 0.5 * sigma**2 / dx**2
    b = (r - 0.5*sigma**2) / (2*dx)
    alpha = a - b          # sub-diagonal coefficient
    gamma = a + b          # super-diagonal coefficient
    beta  = -(2*a + r)     # diagonal coefficient
    return alpha, beta, gamma

def thomas_solve(a_sub, b_diag, c_sup, d_rhs):
    """
    Thomas algorithm for tridiagonal system.
    a_sub: sub-diagonal (length M-1, aligned with rows 1..M-1)
    b_diag: diagonal (length M)
    c_sup: super-diagonal (length M-1, aligned with rows 0..M-2)
    d_rhs: right-hand side (length M)
    Returns solution vector x of length M.
    """
    n   = len(b_diag)
    c_p = np.zeros(n-1)
    d_p = np.zeros(n)
    c_p[0] = c_sup[0] / b_diag[0]
    d_p[0] = d_rhs[0] / b_diag[0]
    for i in range(1, n):
        denom  = b_diag[i] - a_sub[i-1]*c_p[i-1]
        d_p[i] = (d_rhs[i] - a_sub[i-1]*d_p[i-1]) / denom
        if i < n-1:
            c_p[i] = c_sup[i] / denom
    x = np.zeros(n)
    x[-1] = d_p[-1]
    for i in range(n-2, -1, -1):
        x[i] = d_p[i] - c_p[i]*x[i+1]
    return x

def fdm_explicit(S0, K, r, T, sigma, M=200, N=2000, option="call"):
    """Explicit (Forward Euler) FDM in log-space."""
    x_arr, S_arr, dx, dtau = build_grid(S0, K, r, sigma, T, M, N)
    V = terminal_condition(S_arr, K, option)
    alpha, beta, gamma = fdm_coefficients(sigma, r, dx)
    # Stability check
    lam = dtau / dx**2
    if lam * sigma**2 > 1:
        pass  # may be unstable, continue anyway for demonstration

    for j in range(N):
        tau = (j+1) * dtau
        bc_low, bc_high = boundary_conditions(S_arr, K, r, tau, option)
        V_new = V.copy()
        V_new[1:-1] = (V[1:-1]
                       + dtau*(alpha*V[:-2] + beta*V[1:-1] + gamma*V[2:]))
        V_new[0]  = bc_low
        V_new[-1] = bc_high
        V = V_new
    return S_arr, V

def fdm_implicit(S0, K, r, T, sigma, M=200, N=500, option="call"):
    """Implicit (Backward Euler) FDM in log-space. Unconditionally stable."""
    x_arr, S_arr, dx, dtau = build_grid(S0, K, r, sigma, T, M, N)
    V = terminal_condition(S_arr, K, option)
    alpha, beta, gamma = fdm_coefficients(sigma, r, dx)
    m_int = M - 1   # interior points
    a_sub  = -dtau * alpha * np.ones(m_int - 1)
    b_diag = (1 - dtau * beta) * np.ones(m_int)
    c_sup  = -dtau * gamma * np.ones(m_int - 1)

    for j in range(N):
        tau = (j+1) * dtau
        bc_low, bc_high = boundary_conditions(S_arr, K, r, tau, option)
        rhs = V[1:-1].copy()
        rhs[0]  += dtau * alpha * bc_low
        rhs[-1] += dtau * gamma * bc_high
        V[1:-1] = thomas_solve(a_sub, b_diag, c_sup, rhs)
        V[0]    = bc_low
        V[-1]   = bc_high
    return S_arr, V

def fdm_crank_nicolson(S0, K, r, T, sigma, M=200, N=500, option="call"):
    """Crank-Nicolson FDM. Second-order in time and space."""
    x_arr, S_arr, dx, dtau = build_grid(S0, K, r, sigma, T, M, N)
    V = terminal_condition(S_arr, K, option)
    alpha, beta, gamma = fdm_coefficients(sigma, r, dx)
    m_int = M - 1
    theta = 0.5   # CN = 0.5 explicit + 0.5 implicit
    # LHS matrix coefficients (implicit part)
    a_sub  = -theta*dtau*alpha * np.ones(m_int - 1)
    b_diag = (1 - theta*dtau*beta) * np.ones(m_int)
    c_sup  = -theta*dtau*gamma * np.ones(m_int - 1)
    # RHS matrix coefficients (explicit part)
    ra = (1-theta)*dtau*alpha
    rb = (1-theta)*dtau*beta
    rg = (1-theta)*dtau*gamma

    for j in range(N):
        tau_new = (j+1)*dtau
        tau_old = j*dtau
        bc_low_new, bc_high_new = boundary_conditions(S_arr, K, r, tau_new, option)
        bc_low_old, bc_high_old = boundary_conditions(S_arr, K, r, tau_old, option)
        V_int = V[1:-1]
        rhs   = (V_int
                 + ra*V[:-2] + rb*V_int + rg*V[2:])
        rhs[0]  += theta*dtau*alpha*bc_low_new + ra*bc_low_old
        rhs[-1] += theta*dtau*gamma*bc_high_new + rg*bc_high_old
        V[1:-1] = thomas_solve(a_sub, b_diag, c_sup, rhs)
        V[0]    = bc_low_new
        V[-1]   = bc_high_new
    return S_arr, V

# ============================================================
# SECTION 3 — DIAGNOSTICS
# ============================================================

S0 = 100.0; K = 100.0; r = 0.05; T = 1.0; sigma = 0.20
OPT = "call"
BS_REF = bs_price(S0, K, r, T, sigma, option=OPT)

print(f"[M34] Finite Difference Methods — BS reference: {BS_REF:.6f}")

S_ex, V_ex = fdm_explicit(S0, K, r, T, sigma, M=200, N=3000, option=OPT)
S_im, V_im = fdm_implicit(S0, K, r, T, sigma, M=200, N=500,  option=OPT)
S_cn, V_cn = fdm_crank_nicolson(S0, K, r, T, sigma, M=200, N=500, option=OPT)

def interpolate_price(S_arr, V_arr, S_target):
    idx = np.searchsorted(S_arr, S_target)
    if idx == 0: return V_arr[0]
    if idx >= len(S_arr): return V_arr[-1]
    w = (S_target - S_arr[idx-1]) / (S_arr[idx] - S_arr[idx-1])
    return V_arr[idx-1]*(1-w) + V_arr[idx]*w

p_ex = interpolate_price(S_ex, V_ex, S0)
p_im = interpolate_price(S_im, V_im, S0)
p_cn = interpolate_price(S_cn, V_cn, S0)

print(f"      Explicit (M=200,N=3000): {p_ex:.6f}  err={abs(p_ex-BS_REF):.2e}")
print(f"      Implicit (M=200,N=500):  {p_im:.6f}  err={abs(p_im-BS_REF):.2e}")
print(f"      Crank-Nicolson (M=200,N=500): {p_cn:.6f}  err={abs(p_cn-BS_REF):.2e}")

# ============================================================
# FIGURE 1 — Solution profiles and comparison with BS
# ============================================================
t0 = time.perf_counter()
print("\n[M34] Figure 1: FDM solution profiles vs BS ...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor(DARK)
fig.suptitle("M34 — Finite Difference Methods for BS PDE\n"
             "Explicit | Implicit | Crank-Nicolson vs Analytical",
             color=WHITE, fontsize=12, fontweight="bold")

S_plot = np.linspace(60, 140, 300)
BS_arr = [bs_price(s, K, r, T, sigma, option=OPT) for s in S_plot]

# (a) Price profiles
ax = axes[0]
ax.plot(S_plot, BS_arr, color=WHITE, lw=3, linestyle="--",
        label="BS Analytical", zorder=5)
mask = (S_ex > 60) & (S_ex < 140)
ax.plot(S_ex[mask], V_ex[mask], color=BLUE,   lw=2, alpha=0.8,
        label=f"Explicit (N=3000)")
ax.plot(S_im[mask], V_im[mask], color=GREEN,  lw=2, alpha=0.8,
        label=f"Implicit (N=500)")
ax.plot(S_cn[mask], V_cn[mask], color=ORANGE, lw=2, alpha=0.8,
        label=f"Crank-Nicolson (N=500)")
ax.axvline(K, color=YELLOW, lw=1, linestyle=":", alpha=0.5, label=f"K={K}")
ax.set_xlabel("Spot S"); ax.set_ylabel("Call price")
ax.set_title("FDM Price Profiles vs Black-Scholes\n"
             "All three methods converge to analytical",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (b) Pricing error vs BS
ax = axes[1]
for S_fd, V_fd, lbl, col in [
    (S_ex, V_ex, "Explicit",        BLUE),
    (S_im, V_im, "Implicit",        GREEN),
    (S_cn, V_cn, "Crank-Nicolson",  ORANGE),
]:
    mask2 = (S_fd > 65) & (S_fd < 135)
    bs_ref_ = np.array([bs_price(s, K, r, T, sigma, option=OPT)
                         for s in S_fd[mask2]])
    err_ = V_fd[mask2] - bs_ref_
    ax.plot(S_fd[mask2], err_, color=col, lw=2, label=lbl)
ax.axhline(0, color=WHITE, lw=1, linestyle=":", alpha=0.6)
ax.axvline(K, color=YELLOW, lw=1, linestyle=":", alpha=0.5)
ax.set_xlabel("Spot S"); ax.set_ylabel("FDM price - BS price")
ax.set_title("Pricing Error vs Black-Scholes\n"
             "CN has smallest and smoothest error",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (c) Grid spacing effect on accuracy (CN, fixed N)
ax = axes[2]
M_arr = [25, 50, 100, 200, 300, 400]
err_im_M = []; err_cn_M = []
for M_ in M_arr:
    S_, V_ = fdm_implicit(S0, K, r, T, sigma, M=M_, N=200, option=OPT)
    err_im_M.append(abs(interpolate_price(S_, V_, S0) - BS_REF))
    S_, V_ = fdm_crank_nicolson(S0, K, r, T, sigma, M=M_, N=200, option=OPT)
    err_cn_M.append(abs(interpolate_price(S_, V_, S0) - BS_REF))

ax.loglog(M_arr, err_im_M, color=GREEN,  lw=2.5, marker="o", ms=6,
          label="Implicit")
ax.loglog(M_arr, err_cn_M, color=ORANGE, lw=2.5, marker="s", ms=6,
          label="Crank-Nicolson")
M_ref = np.array([M_arr[0], M_arr[-1]], dtype=float)
ax.loglog(M_ref, err_im_M[0]*(M_arr[0]/M_ref)**2, color=WHITE,
          lw=1.5, linestyle=":", alpha=0.6, label="O(1/M^2) ref")
ax.set_xlabel("Number of space steps M (log)");
ax.set_ylabel("|Error| at S=S0 (log)")
ax.set_title("Spatial Convergence (N=200 fixed)\n"
             "Error ~ O(dx^2) for both schemes",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

plt.tight_layout()
p1 = os.path.join(OUT_DIR, "m34_01_fdm_profiles.png")
plt.savefig(p1, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p1}  ({time.perf_counter()-t0:.1f}s)")

# ============================================================
# FIGURE 2 — Convergence in time steps and stability
# ============================================================
t0 = time.perf_counter()
print("[M34] Figure 2: Temporal convergence and stability analysis ...")

N_arr = [50, 100, 200, 400, 800, 1600]
err_im_N = []; err_cn_N = []; err_ex_N = []
for N_ in N_arr:
    # Implicit
    S_, V_ = fdm_implicit(S0, K, r, T, sigma, M=200, N=N_, option=OPT)
    err_im_N.append(abs(interpolate_price(S_, V_, S0) - BS_REF))
    # CN
    S_, V_ = fdm_crank_nicolson(S0, K, r, T, sigma, M=200, N=N_, option=OPT)
    err_cn_N.append(abs(interpolate_price(S_, V_, S0) - BS_REF))
    # Explicit (stable only for large N)
    N_ex_stab = max(N_, 3000)
    S_, V_ = fdm_explicit(S0, K, r, T, sigma, M=100, N=N_ex_stab, option=OPT)
    err_ex_N.append(abs(interpolate_price(S_, V_, S0) - BS_REF))

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor(DARK)
fig.suptitle("M34 — Temporal Convergence and Stability\n"
             "Implicit O(dt) | CN O(dt^2) | Explicit stability constraint",
             color=WHITE, fontsize=12, fontweight="bold")

# (a) Temporal convergence (M fixed, vary N)
ax = axes[0]
ax.loglog(N_arr, err_im_N, color=GREEN,  lw=2.5, marker="o", ms=6,
          label="Implicit O(dt)")
ax.loglog(N_arr, err_cn_N, color=ORANGE, lw=2.5, marker="s", ms=6,
          label="Crank-Nicolson O(dt^2)")
N_ref2 = np.array([N_arr[0], N_arr[-1]], dtype=float)
ax.loglog(N_ref2, err_im_N[0]*(N_arr[0]/N_ref2)**1, color=GREEN,
          lw=1, linestyle=":", alpha=0.5, label="O(1/N) ref")
ax.loglog(N_ref2, err_cn_N[0]*(N_arr[0]/N_ref2)**2, color=ORANGE,
          lw=1, linestyle=":", alpha=0.5, label="O(1/N^2) ref")
ax.set_xlabel("Number of time steps N (log)");
ax.set_ylabel("|Error| (log)")
ax.set_title("Temporal Convergence (M=200 fixed)\n"
             "CN converges quadratically in dt",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (b) Explicit stability: solution for different N (lambda = dtau/dx^2)
ax = axes[1]
S_plot2 = np.linspace(70, 130, 200)
BS_arr2 = [bs_price(s, K, r, T, sigma, option=OPT) for s in S_plot2]
ax.plot(S_plot2, BS_arr2, color=WHITE, lw=2.5, linestyle="--",
        label="BS Analytical")
for N_ex_, col, stab in [
    (500,  RED,    "Unstable (N=500)"),
    (1500, ORANGE, "Marginal (N=1500)"),
    (5000, GREEN,  "Stable (N=5000)"),
]:
    try:
        S_, V_ = fdm_explicit(S0, K, r, T, sigma, M=100, N=N_ex_, option=OPT)
        mask_s = (S_ > 70) & (S_ < 130)
        ax.plot(S_[mask_s], V_[mask_s], color=col, lw=2, label=stab, alpha=0.8)
    except Exception:
        pass
ax.set_ylim(-5, 50); ax.set_xlabel("Spot S"); ax.set_ylabel("Call price")
ax.set_title("Explicit Scheme: Stability Requirement\n"
             "dtau <= dx^2/sigma^2  (CFL condition)",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (c) Greeks from CN: Delta and Gamma via FDM
ax = axes[2]
S_greeks = np.linspace(70, 130, 60)
delta_cn = []; gamma_cn = []; delta_bs = []; gamma_bs = []
for s in S_greeks:
    h = s * 0.01
    V_up = interpolate_price(*fdm_crank_nicolson(s+h, K, r, T, sigma, M=100, N=200), s+h)
    V_dn = interpolate_price(*fdm_crank_nicolson(s-h, K, r, T, sigma, M=100, N=200), s-h)
    V_0  = interpolate_price(*fdm_crank_nicolson(s,   K, r, T, sigma, M=100, N=200), s)
    delta_cn.append((V_up - V_dn) / (2*h))
    gamma_cn.append((V_up - 2*V_0 + V_dn) / h**2)
    d1 = (np.log(s/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    delta_bs.append(norm.cdf(d1))
    gamma_bs.append(norm.pdf(d1)/(s*sigma*np.sqrt(T)))

ax2 = ax.twinx()
ax.plot(S_greeks, delta_cn, color=BLUE,   lw=2.5, label="Delta (CN FDM)")
ax.plot(S_greeks, delta_bs, color=BLUE,   lw=1.5, linestyle="--",
        alpha=0.6, label="Delta (BS)")
ax2.plot(S_greeks, gamma_cn, color=ORANGE, lw=2.5, label="Gamma (CN FDM)")
ax2.plot(S_greeks, gamma_bs, color=ORANGE, lw=1.5, linestyle="--",
         alpha=0.6, label="Gamma (BS)")
ax.set_xlabel("Spot S"); ax.set_ylabel("Delta", color=BLUE)
ax2.set_ylabel("Gamma", color=ORANGE)
ax.set_title("Greeks from CN Grid\nCentred differences: Delta=(V+−V−)/(2h)",
             color=WHITE, fontsize=9)
lines1, l1 = ax.get_legend_handles_labels()
lines2, l2 = ax2.get_legend_handles_labels()
ax.legend(lines1+lines2, l1+l2, fontsize=6.5)
ax.grid(True); watermark(ax)

plt.tight_layout()
p2 = os.path.join(OUT_DIR, "m34_02_convergence_stability.png")
plt.savefig(p2, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p2}  ({time.perf_counter()-t0:.1f}s)")

# ============================================================
# FIGURE 3 — American put via FDM (CN + free boundary)
# ============================================================
t0 = time.perf_counter()
print("[M34] Figure 3: American put via FDM and early exercise ...")

def fdm_cn_american_put(S0, K, r, T, sigma, M=200, N=500):
    """
    Crank-Nicolson FDM for American put with PSOR (penalty method):
    At each time step, enforce V >= max(K-S, 0) via direct comparison.
    """
    x_arr, S_arr, dx, dtau = build_grid(S0, K, r, sigma, T, M, N)
    V = np.maximum(K - S_arr, 0.0)    # terminal = put payoff
    alpha, beta, gamma = fdm_coefficients(sigma, r, dx)
    m_int = M - 1
    theta = 0.5
    a_sub  = -theta*dtau*alpha * np.ones(m_int - 1)
    b_diag = (1 - theta*dtau*beta) * np.ones(m_int)
    c_sup  = -theta*dtau*gamma * np.ones(m_int - 1)
    ra = (1-theta)*dtau*alpha
    rb = (1-theta)*dtau*beta
    rg = (1-theta)*dtau*gamma

    for j in range(N):
        tau_new = (j+1)*dtau; tau_old = j*dtau
        bc_low_new = K*np.exp(-r*tau_new) - S_arr[0]
        bc_high_new = 0.0
        bc_low_old  = K*np.exp(-r*tau_old) - S_arr[0]
        V_int = V[1:-1]
        rhs   = V_int + ra*V[:-2] + rb*V_int + rg*V[2:]
        rhs[0]  += theta*dtau*alpha*bc_low_new + ra*bc_low_old
        rhs[-1] += theta*dtau*gamma*bc_high_new
        V_new_int = thomas_solve(a_sub, b_diag, c_sup, rhs)
        # Enforce American constraint: V >= intrinsic
        intrinsic_int = np.maximum(K - S_arr[1:-1], 0.0)
        V[1:-1] = np.maximum(V_new_int, intrinsic_int)
        V[0]  = bc_low_new; V[-1] = bc_high_new
    return S_arr, V

S_am, V_am = fdm_cn_american_put(S0, K, r, T, sigma, M=200, N=500)
S_eu, V_eu = fdm_crank_nicolson(S0, K, r, T, sigma, M=200, N=500, option="put")

p_am_fdm = interpolate_price(S_am, V_am, S0)
p_eu_fdm = interpolate_price(S_eu, V_eu, S0)
bs_eu_put = bs_price(S0, K, r, T, sigma, option="put")

print(f"      American put (FDM CN): {p_am_fdm:.4f}")
print(f"      European put (FDM CN): {p_eu_fdm:.4f}  BS={bs_eu_put:.4f}")
print(f"      Early exercise premium: {p_am_fdm-p_eu_fdm:.4f}")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor(DARK)
fig.suptitle("M34 — American Put via FDM (Crank-Nicolson + Constraint)\n"
             "V >= max(K-S,0) enforced at each time step",
             color=WHITE, fontsize=12, fontweight="bold")

mask3 = (S_am > 60) & (S_am < 140)
S_plot3 = np.linspace(60, 140, 200)
intrinsic3 = np.maximum(K - S_plot3, 0)
bs_eu_arr3 = [bs_price(s, K, r, T, sigma, option="put") for s in S_plot3]

# (a) American vs European put price
ax = axes[0]
ax.plot(S_am[mask3], V_am[mask3], color=RED,   lw=2.5,
        label="American put (FDM CN)")
ax.plot(S_eu[mask3], V_eu[mask3], color=BLUE,  lw=2.5,
        label="European put (FDM CN)")
ax.plot(S_plot3, bs_eu_arr3, color=WHITE, lw=1.5, linestyle="--",
        alpha=0.7, label="European put (BS)")
ax.plot(S_plot3, intrinsic3, color=YELLOW, lw=1.5, linestyle=":",
        alpha=0.7, label="Intrinsic K-S")
ax.fill_between(S_am[mask3], V_am[mask3], V_eu[mask3],
                color=RED, alpha=0.18, label="Early exercise premium")
ax.set_xlabel("Spot S"); ax.set_ylabel("Put price")
ax.set_title("American vs European Put (FDM)\n"
             "Constraint: V_Am >= max(K-S,0)",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (b) Early exercise premium vs moneyness
ax = axes[1]
S_mono = np.linspace(60, 110, 40)
prems = []
eu_p  = []
am_p  = []
for s_ in S_mono:
    Sa, Va = fdm_cn_american_put(s_, K, r, T, sigma, M=100, N=200)
    Se, Ve = fdm_crank_nicolson(s_, K, r, T, sigma, M=100, N=200, option="put")
    pa = interpolate_price(Sa, Va, s_)
    pe = interpolate_price(Se, Ve, s_)
    prems.append(pa - pe)
    am_p.append(pa); eu_p.append(pe)

ax.plot(S_mono, prems, color=RED, lw=2.5, label="Early exercise premium")
ax.fill_between(S_mono, prems, color=RED, alpha=0.18)
ax.axvline(K, color=YELLOW, lw=1, linestyle="--", alpha=0.6, label=f"ATM K={K}")
ax.set_xlabel("Spot S"); ax.set_ylabel("American - European put")
ax.set_title("Early Exercise Premium vs Moneyness\n"
             "Premium largest for deep ITM puts",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (c) Scheme comparison table (text-based visual)
ax = axes[2]
ax.axis("off")
table_data = [
    ["Scheme",       "Stability",   "Order(dt)", "Order(dx)", "System"],
    ["Explicit",     "Conditional", "O(dt)",     "O(dx^2)",   "Explicit"],
    ["Implicit",     "Uncondit.",   "O(dt)",     "O(dx^2)",   "Tridiag"],
    ["Crank-Nicolson","Uncondit.",  "O(dt^2)",   "O(dx^2)",   "Tridiag"],
    ["","","","",""],
    ["Key metrics (M=200, N=500, S=100):","","","",""],
    [f"Explicit err: {abs(p_ex-BS_REF):.2e}","","","",""],
    [f"Implicit err: {abs(p_im-BS_REF):.2e}","","","",""],
    [f"CN err:       {abs(p_cn-BS_REF):.2e}","","","",""],
]
colors_tbl = [
    [BLUE]*5, [ORANGE]*5, [GREEN]*5, [PURPLE]*5,
    [DARK]*5, [DARK]*5, [BLUE]*5, [GREEN]*5, [PURPLE]*5
]
for row_i, (row, row_col) in enumerate(zip(table_data, colors_tbl)):
    for col_i, (cell, col) in enumerate(zip(row, row_col)):
        ax.text(col_i*0.22, 1.0 - row_i*0.11, cell,
                transform=ax.transAxes, fontsize=7.5,
                color=WHITE if row_i==0 else WHITE,
                fontweight="bold" if row_i==0 else "normal")

ax.set_title("FDM Scheme Summary",color=WHITE, fontsize=9)
watermark(ax)

plt.tight_layout()
p3 = os.path.join(OUT_DIR, "m34_03_american_put_fdm.png")
plt.savefig(p3, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p3}  ({time.perf_counter()-t0:.1f}s)")

# ── Summary ──────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("  MODULE 34 COMPLETE -- 3 figures saved")
print("  Key Concepts:")
print("  [1] BS PDE: dV/dt + 0.5*s^2*S^2*Vss + r*S*Vs - r*V = 0")
print("  [2] Explicit: O(dt+dx^2), conditional stability dtau<dx^2/s^2")
print("  [3] Implicit: O(dt+dx^2), unconditional, tridiagonal system")
print("  [4] CN: O(dt^2+dx^2), unconditional, highest accuracy")
print("  [5] Thomas: O(N) tridiagonal solver (vs O(N^3) dense)")
print("  [6] American: enforce V>=intrinsic at each CN time step")
print(f"  Explicit err:       {abs(p_ex-BS_REF):.2e}")
print(f"  Implicit err:       {abs(p_im-BS_REF):.2e}")
print(f"  Crank-Nicolson err: {abs(p_cn-BS_REF):.2e}")
print(f"  American put (FDM): {p_am_fdm:.4f}  "
      f"premium={p_am_fdm-p_eu_fdm:.4f} over European")
print("=" * 65)
