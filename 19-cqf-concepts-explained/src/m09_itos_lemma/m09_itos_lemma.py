#!/usr/bin/env python3
"""
M09 — Ito's Lemma: Numerical dS vs d(lnS)
==========================================
Module 2 of 9 | CQF Concepts Explained

Theory
------
Given GBM:
    dS_t = mu * S_t * dt + sigma * S_t * dW_t

Apply Ito's Lemma to f(S) = ln(S):

    df = f'(S) dS + 1/2 f''(S) (dS)^2

where (dS)^2 = sigma^2 * S^2 * dt  (using Ito's table: dW^2 = dt)

    f'(S)  = 1/S
    f''(S) = -1/S^2

Therefore:
    d(lnS) = (1/S)(mu*S*dt + sigma*S*dW) + 1/2*(-1/S^2)*sigma^2*S^2*dt
           = mu*dt + sigma*dW - (sigma^2/2)*dt
           = (mu - sigma^2/2)*dt + sigma*dW

The KEY insight: the -sigma^2/2 correction (the Ito term) arises from
the second-order term in the Taylor expansion that survives because
(dW)^2 = dt is O(dt), not O(dt^2) as in classical calculus.

Without Ito's lemma (classical calculus):
    d(lnS) = dS/S = mu*dt + sigma*dW     <-- WRONG: misses -sigma^2/2

This module verifies numerically:
    1. Direct simulation of dS/S  (no Ito correction)
    2. Simulation of d(lnS) = ln(S_{t+dt}) - ln(S_t)
    3. Convergence of the -sigma^2/2 correction as dt -> 0
    4. Impact on long-run wealth (Jensen's inequality interpretation)

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
from scipy.stats import norm

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
# Parameters
# ---------------------------------------------------------------------------
S0      = 100.0
mu      = 0.10
sigma   = 0.25
T       = 5.0       # longer horizon to make Jensen gap visible
N_PATHS = 5000
N_STEPS = 1260      # 5Y * 252
SEED    = 42

rng = np.random.default_rng(SEED)

# Shared Brownian increments for fair comparison
dt    = T / N_STEPS
sqdt  = np.sqrt(dt)
Z_all = rng.standard_normal((N_PATHS, N_STEPS))
dW_all = sqdt * Z_all

# ===========================================================================
# FIGURE 1 — Step-by-step illustration: dS vs dS/S vs d(lnS)
# ===========================================================================
print("[M09] Figure 1: dS vs d(lnS) step comparison ...")
t0 = time.perf_counter()

# Use a single path for illustration
n_ill  = 252         # 1-year illustration
Z_ill  = Z_all[0, :n_ill]
dW_ill = dW_all[0, :n_ill]
dt_ill = T / N_STEPS  # same dt
t_ill  = np.arange(n_ill + 1) * dt_ill

# Build exact path
log_S = np.zeros(n_ill + 1)
for i in range(n_ill):
    log_S[i+1] = log_S[i] + (mu - 0.5*sigma**2)*dt_ill + sigma*dW_ill[i]
S_path = S0 * np.exp(log_S)

# Compute increments three ways
dS_actual  = np.diff(S_path)                           # true dS
dS_over_S  = dS_actual / S_path[:-1]                  # dS/S (classical approx)
d_lnS      = np.diff(np.log(S_path))                  # true d(lnS) via log

# Theoretical means per step
mean_dS_over_S = mu * dt_ill                           # classical: E[dS/S] = mu*dt
mean_d_lnS     = (mu - 0.5*sigma**2) * dt_ill         # Ito: E[d(lnS)] = (mu-sig^2/2)*dt
ito_correction  = 0.5 * sigma**2 * dt_ill              # the gap

fig, axes = plt.subplots(2, 2, figsize=(14, 8), facecolor=DARK)
fig.suptitle(
    "M09 — Ito's Lemma: dS/S vs d(lnS)\n"
    f"GBM: mu={mu:.0%}, sigma={sigma:.0%}, S0={S0:.0f}",
    color=WHITE, fontsize=11
)

# (0,0) Price path
ax = axes[0, 0]
ax.plot(t_ill, S_path, color=BLUE, lw=1.5, label="$S_t$ (exact GBM)")
ax.set_xlabel("Time (years)"); ax.set_ylabel("$S_t$")
ax.set_title("Sample GBM Path (1 Year, 252 Steps)", color=WHITE, fontsize=9)
ax.legend(fontsize=8); ax.grid(True); watermark(ax)

# (0,1) dS/S vs d(lnS) for first 60 steps
ax = axes[0, 1]
steps_show = np.arange(1, 61)
ax.bar(steps_show - 0.2, dS_over_S[:60] * 100, width=0.35,
       color=ORANGE, alpha=0.75, label="dS/S  (classical)")
ax.bar(steps_show + 0.2, d_lnS[:60]   * 100, width=0.35,
       color=GREEN,  alpha=0.75, label="d(lnS) (Ito)")
ax.axhline(mean_dS_over_S * 100, color=ORANGE, lw=1.5, linestyle="--",
           label=f"E[dS/S] = mu*dt = {mean_dS_over_S*100:.4f}%")
ax.axhline(mean_d_lnS * 100, color=GREEN, lw=1.5, linestyle="--",
           label=f"E[d(lnS)] = (mu-sig^2/2)*dt = {mean_d_lnS*100:.4f}%")
ax.set_xlabel("Step"); ax.set_ylabel("Increment (%)")
ax.set_title("Step Increments: dS/S vs d(lnS)\n(first 60 steps)", color=WHITE, fontsize=9)
ax.legend(fontsize=6); ax.grid(True, axis="y"); watermark(ax)

# (1,0) Cumulative sum: sum(dS/S) vs ln(S_T/S_0)
ax = axes[1, 0]
cum_dS_over_S = np.cumsum(dS_over_S)
cum_d_lnS     = np.cumsum(d_lnS)
ax.plot(t_ill[1:], cum_dS_over_S, color=ORANGE, lw=2,
        label=r"$\sum dS/S$ (classical sum)")
ax.plot(t_ill[1:], cum_d_lnS,     color=GREEN,  lw=2,
        label=r"$\ln(S_t/S_0)$ (true log-return)")
ax.fill_between(t_ill[1:], cum_d_lnS, cum_dS_over_S,
                color=RED, alpha=0.20,
                label=f"Ito gap = sigma^2/2 * t")
# Theoretical Ito gap line
t_arr = t_ill[1:]
ax.plot(t_arr, 0.5*sigma**2 * t_arr, color=RED, lw=1.5, linestyle=":",
        label=f"Theoretical gap = sigma^2/2 * t")
ax.set_xlabel("Time (years)"); ax.set_ylabel("Cumulative value")
ax.set_title("Cumulative Sum: Classical vs Ito", color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (1,1) Distribution of single-step error: dS/S - d(lnS)
ax = axes[1, 1]
# Use all N_PATHS paths, first step only
Z_step = Z_all[:, 0]
dW_step = dW_all[:, 0]
dS_over_S_mc = mu*dt + sigma*dW_step                        # = dS/S approx
d_lnS_mc     = (mu - 0.5*sigma**2)*dt + sigma*dW_step      # = d(lnS) exact
diff = dS_over_S_mc - d_lnS_mc                              # should = sigma^2/2 * dt exactly

ax.hist(diff * 1e6, bins=50, density=True, color=PURPLE, alpha=0.7,
        edgecolor="none", label="dS/S - d(lnS)  x10^6")
ax.axvline(0.5*sigma**2*dt * 1e6, color=YELLOW, lw=2, linestyle="--",
           label=f"Theoretical = sigma^2/2*dt = {0.5*sigma**2*dt*1e6:.4f} x10^-6")
ax.axvline(diff.mean() * 1e6, color=RED, lw=2, linestyle=":",
           label=f"Simulated mean = {diff.mean()*1e6:.4f} x10^-6")
ax.set_xlabel("dS/S - d(lnS)  (x 10^-6)"); ax.set_ylabel("Density")
ax.set_title("Per-Step Ito Correction Distribution\n"
             "dS/S - d(lnS) = sigma^2/2 * dt  (deterministic!)",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

plt.tight_layout()
p1 = os.path.join(OUT_DIR, "m09_01_dS_vs_dlnS.png")
plt.savefig(p1, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p1}  ({time.perf_counter()-t0:.1f}s)")


# ===========================================================================
# FIGURE 2 — Convergence of Ito correction as dt -> 0
# ===========================================================================
print("[M09] Figure 2: Convergence of Ito correction ...")
t0 = time.perf_counter()

n_steps_range = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
N_CONV = 2000
T_conv = 1.0

measured_corrections = []
theoretical_per_step = []

for ns in n_steps_range:
    dt_c   = T_conv / ns
    sqdt_c = np.sqrt(dt_c)
    rng_c  = np.random.default_rng(SEED + ns)
    Z_c    = rng_c.standard_normal((N_CONV, ns))
    dW_c   = sqdt_c * Z_c

    # d(lnS) per step (Ito): (mu - sig^2/2)*dt + sigma*dW
    d_lnS_c = (mu - 0.5*sigma**2)*dt_c + sigma*dW_c

    # dS/S per step (classical / Euler):  mu*dt + sigma*dW
    dS_S_c  = mu*dt_c + sigma*dW_c

    # Average per-step difference = sigma^2/2 * dt
    diff_c = (dS_S_c - d_lnS_c).mean()
    measured_corrections.append(diff_c)
    theoretical_per_step.append(0.5 * sigma**2 * dt_c)

measured_corrections  = np.array(measured_corrections)
theoretical_per_step  = np.array(theoretical_per_step)
dt_vals = T_conv / np.array(n_steps_range)

# MAE of terminal log-return: sum(dS/S) vs exact ln(S_T/S_0)
mae_terminal = []
for ns in n_steps_range:
    dt_c   = T_conv / ns
    sqdt_c = np.sqrt(dt_c)
    rng_c  = np.random.default_rng(SEED + ns + 1)
    Z_c    = rng_c.standard_normal((N_CONV, ns))
    dW_c   = sqdt_c * Z_c

    sum_dS_S   = (mu*dt_c + sigma*dW_c).sum(axis=1)          # classical sum
    true_lnS   = ((mu - 0.5*sigma**2)*dt_c + sigma*dW_c).sum(axis=1)  # exact
    mae_terminal.append(np.abs(sum_dS_S - true_lnS).mean())

mae_terminal = np.array(mae_terminal)

fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=DARK)
fig.suptitle("M09 — Convergence of the Ito Correction  sigma^2/2 * dt",
             color=WHITE, fontsize=11)

ax = axes[0]
ax.loglog(dt_vals, theoretical_per_step, color=YELLOW, lw=2.5,
          label=r"Theoretical: $\sigma^2/2 \cdot \Delta t$")
ax.loglog(dt_vals, measured_corrections, "o", color=GREEN, ms=7,
          label="Simulated mean correction")
ax.loglog(dt_vals, dt_vals, "--", color=GREY, lw=1,
          label=r"$O(\Delta t)$ reference")
ax.set_xlabel(r"Step size $\Delta t$")
ax.set_ylabel("Per-step Ito correction")
ax.set_title("Per-Step Ito Correction vs Step Size\n"
             "dS/S - d(lnS) = sigma^2/2 * dt  (linear in dt)",
             color=WHITE, fontsize=9)
ax.legend(fontsize=8); ax.grid(True); watermark(ax)

ax = axes[1]
ax.loglog(dt_vals, mae_terminal, "s-", color=ORANGE, lw=2, ms=7,
          label="MAE: sum(dS/S) vs ln(S_T/S_0)")
theo_gap = 0.5 * sigma**2 * T_conv * np.ones_like(dt_vals)
ax.axhline(0.5*sigma**2*T_conv, color=RED, lw=2, linestyle="--",
           label=f"Cumulative gap = sigma^2/2*T = {0.5*sigma**2*T_conv:.4f}")
ax.set_xlabel(r"Step size $\Delta t$")
ax.set_ylabel("MAE of terminal log-return")
ax.set_title("Terminal Error: Classical Sum vs Exact lnS\n"
             "Gap converges to sigma^2/2 * T as dt -> 0",
             color=WHITE, fontsize=9)
ax.legend(fontsize=8); ax.grid(True); watermark(ax)

plt.tight_layout()
p2 = os.path.join(OUT_DIR, "m09_02_ito_convergence.png")
plt.savefig(p2, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p2}  ({time.perf_counter()-t0:.1f}s)")


# ===========================================================================
# FIGURE 3 — Long-run wealth impact: Jensen's inequality
# ===========================================================================
print("[M09] Figure 3: Long-run wealth and Jensen's inequality ...")
t0 = time.perf_counter()

# Simulate N_PATHS full paths over T=5Y with N_STEPS steps
S_terminal_exact = np.zeros(N_PATHS)
S_terminal_wrong = np.zeros(N_PATHS)   # using exp(sum dS/S) -- wrong

for k in range(N_PATHS):
    dW = dW_all[k]
    # Exact: ln(S_T/S0) = sum[(mu-sig^2/2)*dt + sigma*dW]
    log_ret_exact = np.sum((mu - 0.5*sigma**2)*dt + sigma*dW)
    # Wrong (no Ito correction): exp(sum[mu*dt + sigma*dW])
    log_ret_wrong = np.sum(mu*dt + sigma*dW)
    S_terminal_exact[k] = S0 * np.exp(log_ret_exact)
    S_terminal_wrong[k] = S0 * np.exp(log_ret_wrong)

# Theoretical values
E_ST_theo   = S0 * np.exp(mu * T)                # E[S_T] = S0 * e^(mu*T)
med_ST_theo = S0 * np.exp((mu - 0.5*sigma**2)*T) # median = S0 * e^((mu-sig^2/2)*T)
E_ST_wrong  = S0 * np.exp((mu + 0.5*sigma**2)*T) # wrong: double counts sigma^2/2

fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=DARK)
fig.suptitle(
    "M09 — Long-Run Wealth Impact of the Ito Correction\n"
    f"Jensen's Inequality: E[e^X] > e^E[X]  |  T={T}Y, sigma={sigma:.0%}",
    color=WHITE, fontsize=10
)

# (0) Distribution comparison exact vs wrong
ax = axes[0]
x_min = min(S_terminal_exact.min(), S_terminal_wrong.min())
x_max = max(S_terminal_exact.max(), S_terminal_wrong.max()) * 0.9
bins = np.linspace(x_min, x_max, 80)
ax.hist(S_terminal_exact, bins=bins, density=True, color=GREEN,  alpha=0.55,
        edgecolor="none", label="Exact GBM")
ax.hist(S_terminal_wrong, bins=bins, density=True, color=ORANGE, alpha=0.55,
        edgecolor="none", label="Classical (no Ito correction)")
ax.axvline(S_terminal_exact.mean(), color=GREEN,  lw=2, linestyle="--",
           label=f"Mean exact = {S_terminal_exact.mean():.2f}")
ax.axvline(S_terminal_wrong.mean(), color=ORANGE, lw=2, linestyle="--",
           label=f"Mean wrong = {S_terminal_wrong.mean():.2f}")
ax.axvline(E_ST_theo, color=YELLOW, lw=2, linestyle=":",
           label=f"Theo. mean = {E_ST_theo:.2f}")
ax.set_xlabel("$S_T$"); ax.set_ylabel("Density")
ax.set_title(f"Terminal Wealth Distribution  (T={T}Y)", color=WHITE, fontsize=9)
ax.legend(fontsize=6); ax.grid(True); watermark(ax)

# (1) Median vs Mean over time (using exact paths)
ax = axes[1]
# Reconstruct paths mean and median at each time point (use subset)
n_sub  = 500
t_sub_steps = 63   # quarterly checkpoints
t_checkpoints = np.arange(0, N_STEPS+1, N_STEPS // t_sub_steps)
t_check_years = t_checkpoints * dt

S_sub = np.zeros((n_sub, len(t_checkpoints)))
S_sub[:, 0] = S0
log_S_sub = np.zeros(n_sub)
for j, tc in enumerate(t_checkpoints[1:], 1):
    prev_tc = t_checkpoints[j-1]
    chunk = dW_all[:n_sub, prev_tc:tc]
    log_S_sub += (mu - 0.5*sigma**2) * (tc - prev_tc) * dt + sigma * chunk.sum(axis=1)
    S_sub[:, j] = S0 * np.exp(log_S_sub)

sim_mean_t   = S_sub.mean(axis=0)
sim_median_t = np.median(S_sub, axis=0)
theo_mean_t  = S0 * np.exp(mu * t_check_years)
theo_med_t   = S0 * np.exp((mu - 0.5*sigma**2) * t_check_years)

ax.plot(t_check_years, theo_mean_t,   color=YELLOW, lw=2.5,
        label=r"$E[S_t] = S_0 e^{\mu t}$")
ax.plot(t_check_years, sim_mean_t,    color=WHITE,  lw=1.5, linestyle="--",
        label="Simulated mean")
ax.plot(t_check_years, theo_med_t,    color=GREEN,  lw=2.5,
        label=r"Median $= S_0 e^{(\mu-\sigma^2/2)t}$")
ax.plot(t_check_years, sim_median_t,  color=BLUE,   lw=1.5, linestyle="--",
        label="Simulated median")
ax.fill_between(t_check_years, theo_med_t, theo_mean_t,
                color=RED, alpha=0.12,
                label=r"Jensen gap = $S_0 e^{\mu t}(1 - e^{-\sigma^2 t/2})$")
ax.set_xlabel("Time (years)"); ax.set_ylabel("$S_t$")
ax.set_title("Mean vs Median Over Time\nJensen Gap Widens with Horizon",
             color=WHITE, fontsize=9)
ax.legend(fontsize=6); ax.grid(True); watermark(ax)

# (2) Summary table
ax = axes[2]
ax.axis("off")
ito_corr_pct = 0.5 * sigma**2
rows = [
    ["Quantity",                "Value"],
    ["mu  (drift)",             f"{mu:.2%}"],
    ["sigma  (volatility)",     f"{sigma:.2%}"],
    ["Ito correction: sig^2/2", f"{ito_corr_pct:.4f}  ({ito_corr_pct:.2%})"],
    ["Geom. drift: mu-sig^2/2", f"{mu - ito_corr_pct:.4f}  ({mu-ito_corr_pct:.2%})"],
    ["T  (horizon)",            f"{T:.0f} years"],
    ["E[S_T] = S0*exp(mu*T)",   f"{E_ST_theo:.4f}"],
    ["Med[S_T]",                f"{med_ST_theo:.4f}"],
    ["Sim. mean S_T",           f"{S_terminal_exact.mean():.4f}"],
    ["Sim. median S_T",         f"{np.median(S_terminal_exact):.4f}"],
    ["Jensen gap at T",         f"{(E_ST_theo - med_ST_theo):.4f}"],
    ["Paths",                   f"{N_PATHS:,}"],
]
colors_row = [[PANEL]*2] + [[DARK if i%2==0 else PANEL]*2 for i in range(len(rows)-1)]
tbl = ax.table(cellText=rows, cellLoc="center", loc="center",
               cellColours=colors_row)
tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1.0, 1.55)
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor(GREY)
    cell.set_text_props(color=YELLOW if r == 0 else WHITE,
                        weight="bold" if r == 0 else "normal")
ax.set_title("Ito Correction Summary", color=WHITE, fontsize=9, pad=8)

plt.tight_layout()
p3 = os.path.join(OUT_DIR, "m09_03_jensens_inequality.png")
plt.savefig(p3, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p3}  ({time.perf_counter()-t0:.1f}s)")

print()
print("=" * 65)
print("  MODULE 09 COMPLETE -- 3 figures saved")
print("  Key Concepts:")
print("  [1] Ito's Lemma: d(lnS) = (mu - sigma^2/2)*dt + sigma*dW")
print("  [2] Classical error: dS/S = mu*dt + sigma*dW (missing -sig^2/2)")
print(f"  [3] Ito correction = sigma^2/2 = {0.5*sigma**2:.4f}  ({0.5*sigma**2:.2%})")
print(f"  [4] Over T={T}Y: cumulative gap = {0.5*sigma**2*T:.4f}")
print("  [5] Jensen's inequality: E[e^X] > e^E[X]")
print(f"  [6] E[S_T] = {E_ST_theo:.4f}  |  Median S_T = {med_ST_theo:.4f}")
print("=" * 65)
