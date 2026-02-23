#!/usr/bin/env python3
"""
M21 — Bond Pricing, Duration & Convexity
=========================================
Module 6 of 9 | CQF Concepts Explained

Theory
------
A coupon bond with face value F, coupon rate c, maturity T,
and yield-to-maturity y has price:

    P = sum_{i=1}^{N} C / (1+y/m)^{m*t_i} + F / (1+y/m)^{m*T}

where C = c*F/m is the periodic coupon, m = payment frequency.

Continuous compounding equivalent:
    P = sum_i C * exp(-y*t_i) + F * exp(-y*T)

Modified Duration (price sensitivity to yield):
    D_mod = -1/P * dP/dy
          = sum_i t_i * PV(CF_i) / P      (Macaulay Duration / (1+y/m))

Convexity (second-order sensitivity):
    Cx = 1/P * d2P/dy2
       = sum_i t_i^2 * PV(CF_i) / P

Price change approximation (Taylor expansion):
    dP/P = -D_mod * dy + 0.5 * Cx * dy^2

Dollar Duration (DV01):
    DV01 = -dP/dy * 0.0001 = D_mod * P * 0.0001

Key relationships:
    - Zero-coupon bond: D_mac = T  (duration = maturity)
    - Higher coupon => lower duration (more cash flows near term)
    - Higher yield => lower duration
    - Convexity always positive for option-free bonds
    - Price-yield relationship is convex (not linear)

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
from scipy.optimize import brentq

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
# Bond pricing engine
# ---------------------------------------------------------------------------
def bond_cashflows(F, c, T, m=2):
    """Returns (times, cashflows) for a coupon bond."""
    n      = int(T * m)
    times  = np.arange(1, n+1) / m
    cflows = np.full(n, F * c / m)
    cflows[-1] += F
    return times, cflows

def bond_price(F, c, T, y, m=2):
    """Dirty price of coupon bond (continuous compounding)."""
    times, cflows = bond_cashflows(F, c, T, m)
    return np.sum(cflows * np.exp(-y * times))

def bond_ytm(F, c, T, P_mkt, m=2):
    """Yield-to-maturity from market price via Brent."""
    f = lambda y: bond_price(F, c, T, y, m) - P_mkt
    return brentq(f, 1e-6, 0.99)

def macaulay_duration(F, c, T, y, m=2):
    """Macaulay duration (years)."""
    times, cflows = bond_cashflows(F, c, T, m)
    pv = cflows * np.exp(-y * times)
    P  = pv.sum()
    return np.sum(times * pv) / P

def modified_duration(F, c, T, y, m=2):
    return macaulay_duration(F, c, T, y, m)   # continuous: D_mod = D_mac

def convexity(F, c, T, y, m=2):
    """Convexity (years^2)."""
    times, cflows = bond_cashflows(F, c, T, m)
    pv = cflows * np.exp(-y * times)
    P  = pv.sum()
    return np.sum(times**2 * pv) / P

def dv01(F, c, T, y, m=2):
    """Dollar value of 1 basis point."""
    P  = bond_price(F, c, T, y, m)
    Dm = modified_duration(F, c, T, y, m)
    return Dm * P * 0.0001

def price_approx(P0, D_mod, Cx, dy):
    """Taylor approximation: linear + convexity correction."""
    return P0 * (1 - D_mod*dy + 0.5*Cx*dy**2)

# ---------------------------------------------------------------------------
# Parameters — benchmark bond
# ---------------------------------------------------------------------------
F  = 1000.0   # face value
c  = 0.05     # coupon rate (5%)
T  = 10.0     # maturity (10 years)
y0 = 0.05     # yield (5% — at par)
m  = 2        # semi-annual

P0   = bond_price(F, c, T, y0, m)
D_mac= macaulay_duration(F, c, T, y0, m)
D_mod= modified_duration(F, c, T, y0, m)
Cx   = convexity(F, c, T, y0, m)
DV01_= dv01(F, c, T, y0, m)

print(f"[M21] Bond: F={F}, c={c:.0%}, T={T}Y, y={y0:.0%}, m={m}")
print(f"      Price = {P0:.4f}  |  D_mac = {D_mac:.4f}Y  "
      f"|  D_mod = {D_mod:.4f}  |  Cx = {Cx:.4f}  |  DV01 = {DV01_:.4f}")


# ===========================================================================
# FIGURE 1 — Price-yield curve + duration/convexity illustration
# ===========================================================================
print("[M21] Figure 1: Price-yield curve ...")
t0 = time.perf_counter()

y_range = np.linspace(0.005, 0.15, 300)
P_exact = np.array([bond_price(F, c, T, y, m) for y in y_range])

# Taylor approximations around y0
dy_range    = y_range - y0
P_linear    = price_approx(P0, D_mod, 0,  dy_range)
P_convex    = price_approx(P0, D_mod, Cx, dy_range)

fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=DARK)
fig.suptitle(
    f"M21 — Bond Pricing: Price-Yield Relationship\n"
    f"F={F:.0f}, c={c:.0%}, T={T:.0f}Y, y0={y0:.0%}  |  "
    f"D_mac={D_mac:.3f}Y  D_mod={D_mod:.3f}  Cx={Cx:.3f}",
    color=WHITE, fontsize=10
)

# (0) Price vs yield
ax = axes[0]
ax.plot(y_range*100, P_exact,   color=BLUE,   lw=2.5, label="Exact price")
ax.plot(y_range*100, P_linear,  color=ORANGE, lw=2,   linestyle="--",
        label="Linear approx (duration only)")
ax.plot(y_range*100, P_convex,  color=GREEN,  lw=2,   linestyle="--",
        label="Quadratic approx (+ convexity)")
ax.scatter([y0*100], [P0], color=YELLOW, s=100, zorder=5,
           label=f"Current: y={y0:.0%}, P={P0:.2f}")
ax.axvline(y0*100, color=WHITE, lw=1, linestyle=":", alpha=0.5)
ax.set_xlabel("Yield (%)"); ax.set_ylabel("Price")
ax.set_title("Price-Yield Curve\nConvexity: curve lies above tangent line",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (1) Approximation error vs yield change
ax = axes[1]
err_lin = np.abs(P_linear - P_exact)
err_cx  = np.abs(P_convex - P_exact)
ax.semilogy(dy_range*100, err_lin + 1e-6, color=ORANGE, lw=2,
            label="Linear error")
ax.semilogy(dy_range*100, err_cx  + 1e-6, color=GREEN,  lw=2,
            label="Convexity error")
ax.axvline(0, color=WHITE, lw=1, linestyle=":", alpha=0.5)
ax.set_xlabel("Yield change dy (bps/100)"); ax.set_ylabel("|Price error|  (log)")
ax.set_title("Approximation Error vs Yield Shock\n"
             "Convexity correction dominates for large shocks",
             color=WHITE, fontsize=9)
ax.legend(fontsize=8); ax.grid(True); watermark(ax)

# (2) Cash flow timeline and PV weights
ax = axes[2]
times, cflows = bond_cashflows(F, c, T, m)
pv_cf = cflows * np.exp(-y0 * times)
weights = pv_cf / pv_cf.sum()
ax.bar(times, pv_cf, width=0.3, color=BLUE, alpha=0.7, label="PV(cashflow)")
ax.bar(times[-1], pv_cf[-1], width=0.3, color=YELLOW, alpha=0.9,
       label=f"Principal PV = {pv_cf[-1]:.2f}")
ax.axvline(D_mac, color=RED, lw=2.5, linestyle="--",
           label=f"D_mac = {D_mac:.3f}Y (balance point)")
ax.set_xlabel("Time (years)"); ax.set_ylabel("PV of Cash Flow")
ax.set_title("Cash Flow PV Weights\n"
             "Duration = weighted average maturity",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

plt.tight_layout()
p1 = os.path.join(OUT_DIR, "m21_01_price_yield.png")
plt.savefig(p1, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p1}  ({time.perf_counter()-t0:.1f}s)")


# ===========================================================================
# FIGURE 2 — Duration & Convexity across bond characteristics
# ===========================================================================
print("[M21] Figure 2: Duration and convexity analysis ...")
t0 = time.perf_counter()

y_ax   = np.linspace(0.01, 0.15, 200)
T_vals = [2, 5, 10, 20, 30]
c_vals = [0.02, 0.05, 0.08, 0.12]
colors_T = [BLUE, GREEN, YELLOW, ORANGE, RED]
colors_c = [PURPLE, BLUE, GREEN, ORANGE]

fig, axes = plt.subplots(2, 3, figsize=(16, 9), facecolor=DARK)
fig.suptitle("M21 — Duration and Convexity: Sensitivity Analysis",
             color=WHITE, fontsize=11)

# (0,0) Duration vs yield for different maturities
ax = axes[0, 0]
for T_v, col in zip(T_vals, colors_T):
    D_arr = [macaulay_duration(F, c, T_v, y, m) for y in y_ax]
    ax.plot(y_ax*100, D_arr, color=col, lw=2, label=f"T={T_v}Y")
ax.set_xlabel("Yield (%)"); ax.set_ylabel("Macaulay Duration (years)")
ax.set_title("Duration vs Yield\n(higher yield => shorter duration)",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (0,1) Duration vs coupon for different maturities
ax = axes[0, 1]
c_range = np.linspace(0.01, 0.15, 200)
for T_v, col in zip(T_vals, colors_T):
    D_arr = [macaulay_duration(F, c_v, T_v, y0, m) for c_v in c_range]
    ax.plot(c_range*100, D_arr, color=col, lw=2, label=f"T={T_v}Y")
ax.set_xlabel("Coupon Rate (%)"); ax.set_ylabel("Macaulay Duration (years)")
ax.set_title("Duration vs Coupon Rate\n(higher coupon => shorter duration)",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (0,2) Convexity vs yield
ax = axes[0, 2]
for T_v, col in zip(T_vals, colors_T):
    Cx_arr = [convexity(F, c, T_v, y, m) for y in y_ax]
    ax.plot(y_ax*100, Cx_arr, color=col, lw=2, label=f"T={T_v}Y")
ax.set_xlabel("Yield (%)"); ax.set_ylabel("Convexity (years^2)")
ax.set_title("Convexity vs Yield\n(always positive for option-free bonds)",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (1,0) DV01 vs maturity and coupon
ax = axes[1, 0]
T_range = np.linspace(0.5, 30, 100)
for c_v, col in zip(c_vals, colors_c):
    dv_arr = [dv01(F, c_v, T_v, y0, m) for T_v in T_range]
    ax.plot(T_range, dv_arr, color=col, lw=2, label=f"c={c_v:.0%}")
ax.set_xlabel("Maturity T (years)"); ax.set_ylabel("DV01 (USD per bp)")
ax.set_title("DV01 vs Maturity\n(longer maturity => higher rate sensitivity)",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (1,1) Price change: actual vs linear vs convexity-adjusted
ax = axes[1, 1]
dy_shocks = np.linspace(-0.03, 0.03, 300)
dP_exact  = np.array([bond_price(F,c,T,y0+dy,m) - P0 for dy in dy_shocks])
dP_lin    = -D_mod * P0 * dy_shocks
dP_cx     = -D_mod * P0 * dy_shocks + 0.5 * Cx * P0 * dy_shocks**2
ax.plot(dy_shocks*100, dP_exact, color=BLUE,   lw=2.5, label="Exact dP")
ax.plot(dy_shocks*100, dP_lin,   color=ORANGE, lw=2,   linestyle="--",
        label="Linear: -D_mod*P*dy")
ax.plot(dy_shocks*100, dP_cx,    color=GREEN,  lw=2,   linestyle="--",
        label="Quadratic: + 0.5*Cx*P*dy^2")
ax.axhline(0, color=GREY, lw=0.8)
ax.axvline(0, color=GREY, lw=0.8)
ax.set_xlabel("Yield change dy (%)"); ax.set_ylabel("Price change dP")
ax.set_title("Price Change Decomposition\n"
             "dP = -D_mod*P*dy + 0.5*Cx*P*dy^2 + ...",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (1,2) Summary table
ax = axes[1, 2]
ax.axis("off")
bonds = [
    ("ZCB 10Y",   F, 0.00, 10, y0),
    ("5% 10Y",    F, 0.05, 10, y0),
    ("10% 10Y",   F, 0.10, 10, y0),
    ("5%  5Y",    F, 0.05,  5, y0),
    ("5% 30Y",    F, 0.05, 30, y0),
]
rows = [["Bond", "Price", "D_mac", "D_mod", "Convex", "DV01"]]
for name, f_, c_, t_, y_ in bonds:
    if c_ == 0:
        p_ = f_ * np.exp(-y_*t_)
        dm = t_; dmod = t_; cx_ = t_**2
        dv_ = dmod * p_ * 0.0001
    else:
        p_   = bond_price(f_, c_, t_, y_, m)
        dm   = macaulay_duration(f_, c_, t_, y_, m)
        dmod = modified_duration(f_, c_, t_, y_, m)
        cx_  = convexity(f_, c_, t_, y_, m)
        dv_  = dv01(f_, c_, t_, y_, m)
    rows.append([name, f"{p_:.2f}", f"{dm:.3f}", f"{dmod:.3f}",
                 f"{cx_:.3f}", f"{dv_:.4f}"])

cols_row = [[PANEL]*6]+[[DARK if i%2==0 else PANEL]*6 for i in range(len(rows)-1)]
tbl = ax.table(cellText=rows, cellLoc="center", loc="center",
               cellColours=cols_row)
tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1.0, 1.55)
for (ri, ci_), cell in tbl.get_celld().items():
    cell.set_edgecolor(GREY)
    cell.set_text_props(color=YELLOW if ri==0 else WHITE,
                        weight="bold" if ri==0 else "normal")
ax.set_title("Bond Risk Metrics Summary  (y=5%)", color=WHITE, fontsize=9, pad=8)

plt.tight_layout()
p2 = os.path.join(OUT_DIR, "m21_02_duration_convexity.png")
plt.savefig(p2, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p2}  ({time.perf_counter()-t0:.1f}s)")


# ===========================================================================
# FIGURE 3 — Immunization: duration matching
# ===========================================================================
print("[M21] Figure 3: Immunization ...")
t0 = time.perf_counter()

# Liability: PV = 1,000,000 due in 7 years
PV_liab   = 1_000_000.0
T_liab    = 7.0
D_liab    = T_liab             # zero-coupon liability: duration = maturity

# Two bonds available for immunization
# Bond A: 5Y, 4% coupon
# Bond B: 10Y, 6% coupon
T_A, c_A = 5.0,  0.04
T_B, c_B = 10.0, 0.06
y_imm    = 0.05

P_A   = bond_price(F, c_A, T_A, y_imm, m)
P_B   = bond_price(F, c_B, T_B, y_imm, m)
D_A   = macaulay_duration(F, c_A, T_A, y_imm, m)
D_B   = macaulay_duration(F, c_B, T_B, y_imm, m)
Cx_A  = convexity(F, c_A, T_A, y_imm, m)
Cx_B  = convexity(F, c_B, T_B, y_imm, m)

# Solve: w_A*D_A + w_B*D_B = D_liab,  w_A + w_B = 1
# => w_A = (D_liab - D_B) / (D_A - D_B)
w_A = (D_liab - D_B) / (D_A - D_B)
w_B = 1 - w_A

# Number of bonds to buy (total investment = PV_liab)
N_A = w_A * PV_liab / P_A
N_B = w_B * PV_liab / P_B

D_port = w_A*D_A + w_B*D_B
Cx_port= w_A*Cx_A + w_B*Cx_B

print(f"      Immunization: w_A={w_A:.4f}, w_B={w_B:.4f}  "
      f"|  D_port={D_port:.4f}Y  D_liab={D_liab:.1f}Y")

# Surplus (Assets - PV_liab) vs yield shock
dy_surp  = np.linspace(-0.03, 0.03, 300)
PV_A     = np.array([N_A * bond_price(F, c_A, T_A, y_imm+dy, m) for dy in dy_surp])
PV_B     = np.array([N_B * bond_price(F, c_B, T_B, y_imm+dy, m) for dy in dy_surp])
PV_assets= PV_A + PV_B
PV_liab_ = np.array([PV_liab * np.exp(-D_liab*dy) for dy in dy_surp])  # ZCB-like liab
surplus   = PV_assets - PV_liab_

# Duration-matched vs unhedged (all in Bond A)
N_A_only  = PV_liab / P_A
PV_A_only = np.array([N_A_only * bond_price(F, c_A, T_A, y_imm+dy, m) for dy in dy_surp])
surplus_unhedged = PV_A_only - PV_liab_

fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=DARK)
fig.suptitle(
    f"M21 — Duration Immunization\n"
    f"Liability PV=USD {PV_liab:,.0f}  D_liab={D_liab}Y  |  "
    f"Bond A: {T_A}Y {c_A:.0%}  Bond B: {T_B}Y {c_B:.0%}  y={y_imm:.0%}",
    color=WHITE, fontsize=10
)

# (0) Duration matching diagram
ax = axes[0]
items  = ["Bond A\n(5Y 4%)", "Bond B\n(10Y 6%)", "Portfolio", "Liability"]
durs   = [D_A, D_B, D_port, D_liab]
colors_d= [BLUE, GREEN, YELLOW, RED]
bars = ax.bar(items, durs, color=colors_d, alpha=0.75, edgecolor=GREY)
for bar, d in zip(bars, durs):
    ax.text(bar.get_x()+bar.get_width()/2, d+0.05, f"{d:.3f}Y",
            ha="center", va="bottom", fontsize=9, color=WHITE)
ax.axhline(D_liab, color=RED, lw=2, linestyle="--",
           label=f"Liability duration = {D_liab}Y")
ax.set_ylabel("Duration (years)")
ax.set_title(f"Duration Matching\n"
             f"w_A={w_A:.3f}, w_B={w_B:.3f}  |  D_port={D_port:.4f}Y",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True, axis="y"); watermark(ax)

# (1) Surplus vs yield shock
ax = axes[1]
ax.plot(dy_surp*100, surplus,          color=GREEN,  lw=2.5,
        label="Immunized surplus")
ax.plot(dy_surp*100, surplus_unhedged, color=ORANGE, lw=2, linestyle="--",
        label="Unhedged surplus (Bond A only)")
ax.axhline(0, color=WHITE, lw=1, linestyle=":", alpha=0.7)
ax.fill_between(dy_surp*100, surplus, 0,
                where=surplus>=0, color=GREEN, alpha=0.12, label="Positive surplus")
ax.fill_between(dy_surp*100, surplus, 0,
                where=surplus<0,  color=RED,   alpha=0.12, label="Negative surplus")
ax.set_xlabel("Yield shock dy (%)"); ax.set_ylabel("Surplus (USD)")
ax.set_title("Asset - Liability Surplus vs Yield Shock\n"
             "Immunized portfolio: convex surplus around 0",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

# (2) Assets and Liability separately
ax = axes[2]
ax.plot(dy_surp*100, PV_assets/1e3, color=BLUE,   lw=2.5, label="Assets (immunized)")
ax.plot(dy_surp*100, PV_liab_/1e3,  color=RED,    lw=2.5, linestyle="--",
        label="Liability PV")
ax.plot(dy_surp*100, PV_A_only/1e3, color=ORANGE, lw=2, linestyle=":",
        label="Assets (Bond A only)")
ax.set_xlabel("Yield shock dy (%)"); ax.set_ylabel("Value (USD thousands)")
ax.set_title("Asset vs Liability PV vs Yield\n"
             "Higher convexity of assets => positive surplus",
             color=WHITE, fontsize=9)
ax.legend(fontsize=7); ax.grid(True); watermark(ax)

plt.tight_layout()
p3 = os.path.join(OUT_DIR, "m21_03_immunization.png")
plt.savefig(p3, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"    [OK] {p3}  ({time.perf_counter()-t0:.1f}s)")

print()
print("=" * 65)
print("  MODULE 21 COMPLETE -- 3 figures saved")
print("  Key Concepts:")
print("  [1] P = sum CF_i * exp(-y*t_i)  (continuous compounding)")
print("  [2] D_mac = weighted average maturity of cash flows")
print("  [3] D_mod = dP/P / dy  (price sensitivity per unit yield)")
print("  [4] Convexity: dP/P = -D_mod*dy + 0.5*Cx*dy^2")
print("  [5] DV01 = D_mod * P * 0.0001  (price change per 1bp)")
print("  [6] Immunization: match D_port = D_liab to hedge rate risk")
print(f"  Bond: P={P0:.4f}  D_mac={D_mac:.4f}Y  "
      f"Cx={Cx:.4f}  DV01={DV01_:.4f}")
print(f"  Immunization: w_A={w_A:.4f}, w_B={w_B:.4f}  "
      f"D_port={D_port:.6f}Y")
print("=" * 65)
