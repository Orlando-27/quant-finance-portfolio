#!/usr/bin/env python3
"""
=============================================================================
MODULE 1: DISCOUNTED CASH FLOW (DCF) VALUATION ENGINE
=============================================================================
Author      : Jose Orlando Bobadilla Fuentes
Credentials : CQF | MSc Artificial Intelligence
Role        : Senior Quantitative Portfolio Manager & Lead Data Scientist
Institution : Colombian Pension Fund -- Vicepresidencia de Inversiones

Description
-----------
A comprehensive, institutional-grade DCF valuation engine that implements
the complete workflow from revenue projection to equity value per share:

    1. Revenue Projection (growth assumptions)
    2. Income Statement Build-Down (margins, D&A, interest)
    3. Unlevered Free Cash Flow Computation
    4. WACC Calculation (CAPM, Hamada beta adjustment)
    5. Terminal Value (Gordon Growth + Exit Multiple cross-check)
    6. Enterprise Value Bridge (EV -> Equity Value -> Per Share)
    7. Sensitivity Analysis (WACC vs. Growth, WACC vs. Exit Multiple)
    8. Football Field Valuation Summary

The model uses a fictional company ("TechCorp Inc.") with realistic
technology-sector assumptions for demonstration purposes.

Theoretical Foundations
-----------------------
The DCF methodology values an asset as the present value of its expected
future free cash flows.  For a going concern:

    EV = sum_{t=1}^{N} UFCF_t / (1 + WACC)^t  +  TV_N / (1 + WACC)^N

where:
    UFCF_t  = Unlevered Free Cash Flow in period t
    WACC    = Weighted Average Cost of Capital
    TV_N    = Terminal Value at end of explicit forecast period
    N       = Number of explicit forecast years

Enterprise Value (EV) is then bridged to equity value:
    Equity Value = EV - Net Debt - Minority Interest + Associates

References
----------
    - Damodaran, A. (2012). "Investment Valuation", 3rd ed., Wiley.
    - Koller, T., Goedhart, M., Wessels, D. (2020). "Valuation:
      Measuring and Managing the Value of Companies", 7th ed., McKinsey.
    - Berk, J. & DeMarzo, P. (2019). "Corporate Finance", 5th ed., Pearson.

Output
------
    - Console: Full model tables (Income Statement, FCF, Valuation)
    - Figures: 6 publication-quality charts saved to outputs/figures/dcf/
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch

# ---------------------------------------------------------------------------
# Path setup for Cloud Shell execution
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.common.style import (
    COLORS, PALETTE, save_figure, print_table, print_section,
    fmt_millions, fmt_pct, fmt_currency, fmt_multiple
)
from src.common.finance_utils import (
    cost_of_equity_capm, wacc, unlever_beta, relever_beta,
    unlevered_fcf, terminal_value_gordon, terminal_value_exit_multiple,
    discount_cashflows, mid_year_discount_factors, compute_irr,
    project_working_capital
)


# =============================================================================
# 1. COMPANY ASSUMPTIONS
# =============================================================================
print_section("MODULE 1: DCF VALUATION ENGINE -- TechCorp Inc.")

# --- Company profile ---
COMPANY = "TechCorp Inc."
SHARES_OUTSTANDING = 500e6       # 500 million diluted shares
CURRENT_SHARE_PRICE = 45.00      # Current market price for comparison

# --- Historical base year (Year 0) ---
BASE_REVENUE     = 10_000e6      # $10.0B revenue
BASE_COGS_PCT    = 0.40          # 40% cost of goods sold
BASE_OPEX_PCT    = 0.25          # 25% operating expenses (ex-D&A)
BASE_DA_PCT      = 0.05          # 5% D&A as % of revenue
BASE_CAPEX_PCT   = 0.07          # 7% capex as % of revenue

# --- Projection assumptions (5-year explicit forecast) ---
N_YEARS = 5
YEARS = np.arange(1, N_YEARS + 1)
YEAR_LABELS = [f"Year {y}" for y in YEARS]

# Revenue growth rates (decelerating growth pattern)
REVENUE_GROWTH = np.array([0.12, 0.10, 0.08, 0.07, 0.06])

# Margin assumptions (gradual improvement reflecting operating leverage)
COGS_PCT   = np.array([0.39, 0.38, 0.37, 0.36, 0.35])
OPEX_PCT   = np.array([0.24, 0.23, 0.22, 0.22, 0.21])
DA_PCT     = np.full(N_YEARS, 0.05)
CAPEX_PCT  = np.array([0.07, 0.065, 0.06, 0.06, 0.055])
TAX_RATE   = 0.25

# Working capital assumptions (days)
DSO = 45.0      # Days Sales Outstanding
DIO = 30.0      # Days Inventory Outstanding
DPO = 40.0      # Days Payable Outstanding

# --- WACC assumptions ---
RISK_FREE_RATE    = 0.04          # 10Y US Treasury yield
EQUITY_RISK_PREM  = 0.055         # Equity risk premium
LEVERED_BETA      = 1.15          # Observed equity beta
PRE_TAX_COST_DEBT = 0.05          # Pre-tax cost of debt
EQUITY_WEIGHT     = 0.75          # E/(E+D) target capital structure
DEBT_WEIGHT       = 0.25          # D/(E+D)

# --- Terminal value assumptions ---
TERMINAL_GROWTH   = 0.025         # 2.5% perpetuity growth
EXIT_EV_EBITDA    = 12.0          # Exit multiple for cross-check

# --- Balance sheet items for EV bridge ---
TOTAL_DEBT        = 3_000e6       # $3.0B total debt
CASH              = 1_500e6       # $1.5B cash & equivalents
MINORITY_INTEREST = 200e6         # $200M minority interest
NET_DEBT = TOTAL_DEBT - CASH + MINORITY_INTEREST

print(f"  Company         : {COMPANY}")
print(f"  Base Revenue    : {fmt_currency(BASE_REVENUE)}")
print(f"  Shares Out      : {SHARES_OUTSTANDING/1e6:.0f}M diluted")
print(f"  Forecast Horizon: {N_YEARS} years")


# =============================================================================
# 2. WACC CALCULATION
# =============================================================================
print_section("WACC CALCULATION")

ke = cost_of_equity_capm(RISK_FREE_RATE, LEVERED_BETA, EQUITY_RISK_PREM)
wacc_rate = wacc(ke, PRE_TAX_COST_DEBT, TAX_RATE, EQUITY_WEIGHT, DEBT_WEIGHT)
kd_after_tax = PRE_TAX_COST_DEBT * (1.0 - TAX_RATE)

# Unlevered beta for reference
de_ratio = DEBT_WEIGHT / EQUITY_WEIGHT
unlevered = unlever_beta(LEVERED_BETA, TAX_RATE, de_ratio)

wacc_rows = [
    ["Risk-Free Rate (Rf)",       f"{RISK_FREE_RATE:.2%}"],
    ["Equity Risk Premium (ERP)", f"{EQUITY_RISK_PREM:.2%}"],
    ["Levered Beta",              f"{LEVERED_BETA:.2f}"],
    ["Unlevered Beta",            f"{unlevered:.3f}"],
    ["Cost of Equity (Ke)",       f"{ke:.2%}"],
    ["Pre-Tax Cost of Debt (Kd)", f"{PRE_TAX_COST_DEBT:.2%}"],
    ["Tax Rate",                  f"{TAX_RATE:.2%}"],
    ["After-Tax Kd",              f"{kd_after_tax:.2%}"],
    ["Equity Weight (E/(E+D))",   f"{EQUITY_WEIGHT:.2%}"],
    ["Debt Weight (D/(E+D))",     f"{DEBT_WEIGHT:.2%}"],
    ["WACC",                      f"{wacc_rate:.2%}"],
]
print_table("WACC Components", ["Parameter", "Value"], wacc_rows)


# =============================================================================
# 3. FINANCIAL PROJECTIONS
# =============================================================================
print_section("FINANCIAL PROJECTIONS (5-YEAR)")

# Revenue projection
revenue = np.zeros(N_YEARS)
revenue[0] = BASE_REVENUE * (1.0 + REVENUE_GROWTH[0])
for i in range(1, N_YEARS):
    revenue[i] = revenue[i - 1] * (1.0 + REVENUE_GROWTH[i])

# Income statement build-down
cogs      = revenue * COGS_PCT
gross_profit = revenue - cogs
gross_margin = gross_profit / revenue

opex      = revenue * OPEX_PCT
da        = revenue * DA_PCT
ebitda    = gross_profit - opex
ebit      = ebitda - da
ebit_margin = ebit / revenue

# Tax (on EBIT for UFCF purposes)
nopat     = ebit * (1.0 - TAX_RATE)

# Capital expenditures
capex     = revenue * CAPEX_PCT

# Working capital
ar, inv, ap, nwc, delta_nwc = project_working_capital(revenue, cogs, DSO, DIO, DPO)

# Unlevered Free Cash Flow
ufcf = unlevered_fcf(ebit, TAX_RATE, da, capex, delta_nwc)

# Print Income Statement
is_headers = ["Item"] + YEAR_LABELS
is_rows = [
    ["Revenue"]        + [fmt_currency(v) for v in revenue],
    ["  Growth %"]     + [f"{g:.1%}" for g in REVENUE_GROWTH],
    ["(-) COGS"]       + [fmt_currency(v) for v in cogs],
    ["Gross Profit"]   + [fmt_currency(v) for v in gross_profit],
    ["  Gross Margin"]  + [f"{m:.1%}" for m in gross_margin],
    ["(-) OpEx"]       + [fmt_currency(v) for v in opex],
    ["EBITDA"]         + [fmt_currency(v) for v in ebitda],
    ["(-) D&A"]        + [fmt_currency(v) for v in da],
    ["EBIT"]           + [fmt_currency(v) for v in ebit],
    ["  EBIT Margin"]  + [f"{m:.1%}" for m in ebit_margin],
]
print_table("PRO FORMA INCOME STATEMENT", is_headers, is_rows)

# Print FCF Waterfall
fcf_headers = ["Item"] + YEAR_LABELS
fcf_rows = [
    ["EBIT"]             + [fmt_currency(v) for v in ebit],
    ["(-) Taxes on EBIT"] + [fmt_currency(v) for v in ebit * TAX_RATE],
    ["NOPAT"]            + [fmt_currency(v) for v in nopat],
    ["(+) D&A"]          + [fmt_currency(v) for v in da],
    ["(-) CapEx"]        + [fmt_currency(v) for v in capex],
    ["(-) Delta NWC"]    + [fmt_currency(v) for v in delta_nwc],
    ["= UFCF"]           + [fmt_currency(v) for v in ufcf],
]
print_table("UNLEVERED FREE CASH FLOW", fcf_headers, fcf_rows)


# =============================================================================
# 4. TERMINAL VALUE
# =============================================================================
print_section("TERMINAL VALUE")

# Method 1: Gordon Growth Model
tv_gordon = terminal_value_gordon(ufcf[-1], wacc_rate, TERMINAL_GROWTH)

# Method 2: Exit Multiple (EV/EBITDA)
tv_exit = terminal_value_exit_multiple(ebitda[-1], EXIT_EV_EBITDA)

# Implied perpetuity growth from exit multiple
# TV = UFCF * (1+g) / (WACC - g)  =>  g = (TV * WACC - UFCF) / (TV + UFCF)
implied_growth = (tv_exit * wacc_rate - ufcf[-1]) / (tv_exit + ufcf[-1])

# Implied exit multiple from Gordon Growth
implied_multiple = tv_gordon / ebitda[-1]

tv_rows = [
    ["Gordon Growth Model", ""],
    ["  Terminal FCF",         fmt_currency(ufcf[-1])],
    ["  Growth Rate (g)",      f"{TERMINAL_GROWTH:.2%}"],
    ["  WACC",                 f"{wacc_rate:.2%}"],
    ["  Terminal Value",       fmt_currency(tv_gordon)],
    ["  Implied EV/EBITDA",    f"{implied_multiple:.1f}x"],
    ["", ""],
    ["Exit Multiple Method", ""],
    ["  Terminal EBITDA",      fmt_currency(ebitda[-1])],
    ["  Exit Multiple",        f"{EXIT_EV_EBITDA:.1f}x"],
    ["  Terminal Value",       fmt_currency(tv_exit)],
    ["  Implied Growth Rate",  f"{implied_growth:.2%}"],
]
print_table("TERMINAL VALUE CROSS-CHECK", ["Component", "Value"], tv_rows)


# =============================================================================
# 5. ENTERPRISE VALUE & EQUITY BRIDGE
# =============================================================================
print_section("DCF VALUATION -- ENTERPRISE VALUE TO EQUITY")

# Mid-year convention discount factors
mid_year_df = mid_year_discount_factors(N_YEARS, wacc_rate)

# PV of projected FCFs
pv_fcf = ufcf * mid_year_df
total_pv_fcf = pv_fcf.sum()

# PV of terminal value (discounted from end of Year N)
tv_discount_factor = 1.0 / (1.0 + wacc_rate) ** N_YEARS
pv_tv_gordon = tv_gordon * tv_discount_factor
pv_tv_exit   = tv_exit * tv_discount_factor

# Enterprise Value (using Gordon Growth as primary)
ev_gordon = total_pv_fcf + pv_tv_gordon
ev_exit   = total_pv_fcf + pv_tv_exit

# Equity Value
equity_gordon = ev_gordon - NET_DEBT
equity_exit   = ev_exit - NET_DEBT

# Per Share
price_gordon = equity_gordon / SHARES_OUTSTANDING
price_exit   = equity_exit / SHARES_OUTSTANDING

# TV as % of EV
tv_pct_gordon = pv_tv_gordon / ev_gordon * 100
tv_pct_exit   = pv_tv_exit / ev_exit * 100

bridge_rows = [
    ["PV of Projected FCFs",      fmt_currency(total_pv_fcf)],
    ["", ""],
    ["--- Gordon Growth Method ---", ""],
    ["PV of Terminal Value",       fmt_currency(pv_tv_gordon)],
    ["Enterprise Value",           fmt_currency(ev_gordon)],
    ["(-) Net Debt + Minority",    fmt_currency(NET_DEBT)],
    ["Equity Value",               fmt_currency(equity_gordon)],
    ["Shares Outstanding",         f"{SHARES_OUTSTANDING/1e6:.0f}M"],
    ["Implied Share Price",        f"${price_gordon:.2f}"],
    ["TV as % of EV",              f"{tv_pct_gordon:.1f}%"],
    ["", ""],
    ["--- Exit Multiple Method ---", ""],
    ["PV of Terminal Value",       fmt_currency(pv_tv_exit)],
    ["Enterprise Value",           fmt_currency(ev_exit)],
    ["(-) Net Debt + Minority",    fmt_currency(NET_DEBT)],
    ["Equity Value",               fmt_currency(equity_exit)],
    ["Implied Share Price",        f"${price_exit:.2f}"],
    ["TV as % of EV",              f"{tv_pct_exit:.1f}%"],
    ["", ""],
    ["Current Market Price",       f"${CURRENT_SHARE_PRICE:.2f}"],
    ["Upside (Gordon)",            f"{(price_gordon/CURRENT_SHARE_PRICE - 1)*100:.1f}%"],
    ["Upside (Exit Multiple)",     f"{(price_exit/CURRENT_SHARE_PRICE - 1)*100:.1f}%"],
]
print_table("EV-TO-EQUITY BRIDGE", ["Component", "Value"], bridge_rows)


# =============================================================================
# 6. SENSITIVITY ANALYSIS
# =============================================================================
print_section("SENSITIVITY ANALYSIS")

# Grid: WACC vs Terminal Growth Rate
wacc_range   = np.arange(0.07, 0.12 + 0.005, 0.005)
growth_range = np.arange(0.015, 0.035 + 0.0025, 0.0025)

sensitivity_wg = np.zeros((len(wacc_range), len(growth_range)))
for i, w in enumerate(wacc_range):
    df_mid = mid_year_discount_factors(N_YEARS, w)
    pv_cf = (ufcf * df_mid).sum()
    for j, g in enumerate(growth_range):
        if w <= g:
            sensitivity_wg[i, j] = np.nan
            continue
        tv = terminal_value_gordon(ufcf[-1], w, g)
        pv_t = tv / (1.0 + w) ** N_YEARS
        eq_val = pv_cf + pv_t - NET_DEBT
        sensitivity_wg[i, j] = eq_val / SHARES_OUTSTANDING

# Grid: WACC vs Exit Multiple
multiple_range = np.arange(8.0, 16.0 + 0.5, 1.0)
sensitivity_wm = np.zeros((len(wacc_range), len(multiple_range)))
for i, w in enumerate(wacc_range):
    df_mid = mid_year_discount_factors(N_YEARS, w)
    pv_cf = (ufcf * df_mid).sum()
    for j, m in enumerate(multiple_range):
        tv = terminal_value_exit_multiple(ebitda[-1], m)
        pv_t = tv / (1.0 + w) ** N_YEARS
        eq_val = pv_cf + pv_t - NET_DEBT
        sensitivity_wm[i, j] = eq_val / SHARES_OUTSTANDING

# Print sensitivity tables
print("\n  WACC vs. Terminal Growth Rate (Implied Share Price):")
hdr = ["WACC \\ g"] + [f"{g:.2%}" for g in growth_range]
rows_wg = []
for i, w in enumerate(wacc_range):
    row = [f"{w:.2%}"]
    for j in range(len(growth_range)):
        val = sensitivity_wg[i, j]
        if np.isnan(val):
            row.append("N/A")
        else:
            row.append(f"${val:.2f}")
    rows_wg.append(row)
print_table("SENSITIVITY: WACC vs. Growth", hdr, rows_wg)

print("\n  WACC vs. Exit EV/EBITDA Multiple (Implied Share Price):")
hdr2 = ["WACC \\ Mult"] + [f"{m:.1f}x" for m in multiple_range]
rows_wm = []
for i, w in enumerate(wacc_range):
    row = [f"{w:.2%}"]
    for j in range(len(multiple_range)):
        row.append(f"${sensitivity_wm[i, j]:.2f}")
    rows_wm.append(row)
print_table("SENSITIVITY: WACC vs. Exit Multiple", hdr2, rows_wm)


# =============================================================================
# 7. VISUALIZATIONS
# =============================================================================
print_section("GENERATING VISUALIZATIONS")

# -------------------------------------------------------------------------
# FIGURE 1: Revenue & EBITDA Projection
# -------------------------------------------------------------------------
fig1, ax1 = plt.subplots(figsize=(10, 6))
x = np.arange(N_YEARS)
width = 0.35
bars1 = ax1.bar(x - width/2, revenue / 1e9, width, color=COLORS["primary"],
                alpha=0.85, label="Revenue", edgecolor="none")
bars2 = ax1.bar(x + width/2, ebitda / 1e9, width, color=COLORS["secondary"],
                alpha=0.85, label="EBITDA", edgecolor="none")

# Add value labels
for bar in bars1:
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, h + 0.1,
             f"${h:.1f}B", ha="center", va="bottom",
             fontsize=8, color=COLORS["white"])
for bar in bars2:
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, h + 0.1,
             f"${h:.1f}B", ha="center", va="bottom",
             fontsize=8, color=COLORS["white"])

# EBITDA margin line on secondary axis
ax1b = ax1.twinx()
ebitda_margin = ebitda / revenue * 100
ax1b.plot(x, ebitda_margin, color=COLORS["accent"], marker="o",
          linewidth=2, markersize=6, label="EBITDA Margin %")
ax1b.set_ylabel("EBITDA Margin (%)", color=COLORS["accent"])
ax1b.tick_params(axis="y", labelcolor=COLORS["accent"])
ax1b.set_ylim(25, 45)
ax1b.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_pct))
ax1b.spines["right"].set_color(COLORS["accent"])

ax1.set_xlabel("Forecast Period")
ax1.set_ylabel("USD (Billions)")
ax1.set_title(f"{COMPANY} -- Revenue & EBITDA Projections", fontsize=14,
              fontweight="bold", pad=15)
ax1.set_xticks(x)
ax1.set_xticklabels(YEAR_LABELS)

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1b.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

fig1.tight_layout()
save_figure(fig1, "dcf_01_revenue_ebitda", subdir="dcf")

# -------------------------------------------------------------------------
# FIGURE 2: FCF Waterfall Chart
# -------------------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(11, 6))

# Use Year 3 (midpoint) as representative waterfall
yr_idx = 2
waterfall_labels = ["EBIT", "(-) Taxes", "NOPAT", "(+) D&A",
                    "(-) CapEx", "(-) dNWC", "UFCF"]
waterfall_values = [
    ebit[yr_idx],
    -ebit[yr_idx] * TAX_RATE,
    nopat[yr_idx],
    da[yr_idx],
    -capex[yr_idx],
    -delta_nwc[yr_idx],
    ufcf[yr_idx],
]

# Calculate waterfall positions
cumulative = np.zeros(len(waterfall_values))
cumulative[0] = waterfall_values[0]
for i in range(1, len(waterfall_values) - 1):
    cumulative[i] = cumulative[i-1] + waterfall_values[i]
cumulative[-1] = waterfall_values[-1]  # Final bar starts at 0

# Determine bar bottoms
bottoms = np.zeros(len(waterfall_values))
bottoms[0] = 0
for i in range(1, len(waterfall_values) - 1):
    if waterfall_values[i] >= 0:
        bottoms[i] = cumulative[i] - waterfall_values[i]
    else:
        bottoms[i] = cumulative[i]
bottoms[-1] = 0  # UFCF bar starts from 0

bar_colors = []
for i, v in enumerate(waterfall_values):
    if i == 0:
        bar_colors.append(COLORS["primary"])
    elif i == len(waterfall_values) - 1:
        bar_colors.append(COLORS["teal"])
    elif v >= 0:
        bar_colors.append(COLORS["secondary"])
    else:
        bar_colors.append(COLORS["danger"])

ax2.bar(waterfall_labels, [abs(v) for v in waterfall_values],
        bottom=bottoms, color=bar_colors, edgecolor="none", alpha=0.85,
        width=0.6)

# Value labels
for i, (lbl, val) in enumerate(zip(waterfall_labels, waterfall_values)):
    y_pos = bottoms[i] + abs(val) / 2
    ax2.text(i, y_pos, fmt_currency(abs(val)),
             ha="center", va="center", fontsize=8, color=COLORS["white"],
             fontweight="bold")

# Connector lines between bars
for i in range(len(waterfall_values) - 2):
    next_top = cumulative[i]
    ax2.plot([i + 0.3, i + 0.7], [next_top / 1e6 * 1e6] * 2,
             color="#555555", linewidth=0.8, linestyle="--")

ax2.set_title(f"{COMPANY} -- FCF Waterfall (Year {yr_idx + 1})",
              fontsize=14, fontweight="bold", pad=15)
ax2.set_ylabel("USD")
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_millions))
ax2.set_ylim(0, ebit[yr_idx] * 1.15)

fig2.tight_layout()
save_figure(fig2, "dcf_02_fcf_waterfall", subdir="dcf")

# -------------------------------------------------------------------------
# FIGURE 3: EV Bridge (Stacked Bar)
# -------------------------------------------------------------------------
fig3, ax3 = plt.subplots(figsize=(10, 6))

# Components for both methods
methods = ["Gordon Growth", "Exit Multiple"]
pv_fcf_vals = [total_pv_fcf / 1e9, total_pv_fcf / 1e9]
pv_tv_vals  = [pv_tv_gordon / 1e9, pv_tv_exit / 1e9]
net_debt_vals = [NET_DEBT / 1e9, NET_DEBT / 1e9]

x3 = np.arange(len(methods))
w3 = 0.4

# Stacked: PV FCF + PV TV = EV, then subtract net debt
b1 = ax3.bar(x3, pv_fcf_vals, w3, color=COLORS["primary"], alpha=0.85,
             label="PV of Projected FCFs")
b2 = ax3.bar(x3, pv_tv_vals, w3, bottom=pv_fcf_vals, color=COLORS["secondary"],
             alpha=0.85, label="PV of Terminal Value")

# Net debt as separate negative-offset bar
for i in range(len(methods)):
    ev_total = pv_fcf_vals[i] + pv_tv_vals[i]
    ax3.bar(x3[i] + 0.5, net_debt_vals[i], w3 * 0.8, color=COLORS["danger"],
            alpha=0.7, label="Net Debt + Minority" if i == 0 else "")
    equity = ev_total - net_debt_vals[i]
    ax3.bar(x3[i] + 1.0, equity, w3 * 0.8, color=COLORS["accent"],
            alpha=0.85, label="Equity Value" if i == 0 else "")

# Annotations
for i in range(len(methods)):
    ev_total = pv_fcf_vals[i] + pv_tv_vals[i]
    equity = ev_total - net_debt_vals[i]
    price = equity * 1e9 / SHARES_OUTSTANDING
    ax3.text(x3[i], ev_total + 0.3, f"EV: ${ev_total:.1f}B",
             ha="center", fontsize=9, color=COLORS["white"], fontweight="bold")
    ax3.text(x3[i] + 1.0, equity + 0.3, f"${price:.2f}/sh",
             ha="center", fontsize=9, color=COLORS["accent"], fontweight="bold")

ax3.set_title(f"{COMPANY} -- Enterprise Value Bridge",
              fontsize=14, fontweight="bold", pad=15)
ax3.set_ylabel("USD (Billions)")
ax3.set_xticks([0.5, 1.5])
ax3.set_xticklabels(["Gordon Growth\nMethod", "Exit Multiple\nMethod"])
ax3.legend(loc="upper right")

fig3.tight_layout()
save_figure(fig3, "dcf_03_ev_bridge", subdir="dcf")

# -------------------------------------------------------------------------
# FIGURE 4: Sensitivity Heatmap -- WACC vs Growth
# -------------------------------------------------------------------------
fig4, ax4 = plt.subplots(figsize=(10, 7))

# Mask invalid cells
masked = np.ma.masked_invalid(sensitivity_wg)

im = ax4.imshow(masked, cmap="RdYlGn", aspect="auto",
                vmin=np.nanpercentile(sensitivity_wg, 5),
                vmax=np.nanpercentile(sensitivity_wg, 95))

# Annotate cells
for i in range(len(wacc_range)):
    for j in range(len(growth_range)):
        val = sensitivity_wg[i, j]
        if np.isnan(val):
            txt = "N/A"
            clr = "#666666"
        else:
            txt = f"${val:.1f}"
            # Highlight current market price vicinity
            clr = "black" if val > 40 else COLORS["white"]
        ax4.text(j, i, txt, ha="center", va="center", fontsize=7.5, color=clr)

ax4.set_xticks(np.arange(len(growth_range)))
ax4.set_xticklabels([f"{g:.2%}" for g in growth_range], rotation=45, ha="right")
ax4.set_yticks(np.arange(len(wacc_range)))
ax4.set_yticklabels([f"{w:.2%}" for w in wacc_range])
ax4.set_xlabel("Terminal Growth Rate (g)")
ax4.set_ylabel("WACC")
ax4.set_title(f"{COMPANY} -- Sensitivity: Implied Share Price (WACC vs Growth)",
              fontsize=13, fontweight="bold", pad=15)

cbar = fig4.colorbar(im, ax=ax4, shrink=0.8)
cbar.set_label("Share Price ($)", color=COLORS["white"])
cbar.ax.yaxis.set_tick_params(color=COLORS["white"])
plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=COLORS["white"])

fig4.tight_layout()
save_figure(fig4, "dcf_04_sensitivity_wacc_growth", subdir="dcf")

# -------------------------------------------------------------------------
# FIGURE 5: Sensitivity Heatmap -- WACC vs Exit Multiple
# -------------------------------------------------------------------------
fig5, ax5 = plt.subplots(figsize=(10, 7))

im5 = ax5.imshow(sensitivity_wm, cmap="RdYlGn", aspect="auto",
                 vmin=np.nanpercentile(sensitivity_wm, 5),
                 vmax=np.nanpercentile(sensitivity_wm, 95))

for i in range(len(wacc_range)):
    for j in range(len(multiple_range)):
        val = sensitivity_wm[i, j]
        clr = "black" if val > 50 else COLORS["white"]
        ax5.text(j, i, f"${val:.1f}", ha="center", va="center",
                 fontsize=7.5, color=clr)

ax5.set_xticks(np.arange(len(multiple_range)))
ax5.set_xticklabels([f"{m:.0f}x" for m in multiple_range])
ax5.set_yticks(np.arange(len(wacc_range)))
ax5.set_yticklabels([f"{w:.2%}" for w in wacc_range])
ax5.set_xlabel("Exit EV/EBITDA Multiple")
ax5.set_ylabel("WACC")
ax5.set_title(f"{COMPANY} -- Sensitivity: Implied Share Price (WACC vs Multiple)",
              fontsize=13, fontweight="bold", pad=15)

cbar5 = fig5.colorbar(im5, ax=ax5, shrink=0.8)
cbar5.set_label("Share Price ($)", color=COLORS["white"])
cbar5.ax.yaxis.set_tick_params(color=COLORS["white"])
plt.setp(plt.getp(cbar5.ax.axes, "yticklabels"), color=COLORS["white"])

fig5.tight_layout()
save_figure(fig5, "dcf_05_sensitivity_wacc_multiple", subdir="dcf")

# -------------------------------------------------------------------------
# FIGURE 6: Football Field Valuation Summary
# -------------------------------------------------------------------------
fig6, ax6 = plt.subplots(figsize=(12, 5))

# Valuation ranges from different methodologies
methods_ff = [
    "DCF (Gordon Growth)",
    "DCF (Exit Multiple)",
    "52-Week Range",
    "Analyst Consensus",
]

# Ranges: [low, base, high]
ranges_ff = [
    [sensitivity_wg[~np.isnan(sensitivity_wg)].min(),
     price_gordon,
     sensitivity_wg[~np.isnan(sensitivity_wg)].max()],
    [sensitivity_wm.min(), price_exit, sensitivity_wm.max()],
    [38.00, 45.00, 52.00],           # Fictional 52-week range
    [42.00, 48.00, 55.00],           # Fictional analyst consensus
]

colors_ff = [COLORS["primary"], COLORS["secondary"],
             COLORS["purple"], COLORS["accent"]]

y_pos = np.arange(len(methods_ff))

for i, (method, rng, clr) in enumerate(zip(methods_ff, ranges_ff, colors_ff)):
    low, mid, high = rng
    # Range bar
    ax6.barh(i, high - low, left=low, height=0.5, color=clr, alpha=0.4,
             edgecolor=clr)
    # Midpoint marker
    ax6.plot(mid, i, "D", color=clr, markersize=10, zorder=5)
    # Labels
    ax6.text(low - 1, i, f"${low:.0f}", ha="right", va="center",
             fontsize=8, color=COLORS["white"])
    ax6.text(high + 1, i, f"${high:.0f}", ha="left", va="center",
             fontsize=8, color=COLORS["white"])
    ax6.text(mid, i + 0.3, f"${mid:.1f}", ha="center", va="bottom",
             fontsize=8, color=clr, fontweight="bold")

# Current price line
ax6.axvline(x=CURRENT_SHARE_PRICE, color=COLORS["danger"], linewidth=1.5,
            linestyle="--", alpha=0.8, label=f"Current: ${CURRENT_SHARE_PRICE:.2f}")

ax6.set_yticks(y_pos)
ax6.set_yticklabels(methods_ff)
ax6.set_xlabel("Implied Share Price ($)")
ax6.set_title(f"{COMPANY} -- Football Field Valuation Summary",
              fontsize=14, fontweight="bold", pad=15)
ax6.legend(loc="upper right")
ax6.invert_yaxis()

fig6.tight_layout()
save_figure(fig6, "dcf_06_football_field", subdir="dcf")


# =============================================================================
# 8. SUMMARY
# =============================================================================
print_section("DCF VALUATION SUMMARY")
print(f"  Company              : {COMPANY}")
print(f"  WACC                 : {wacc_rate:.2%}")
print(f"  PV of Projected FCFs : {fmt_currency(total_pv_fcf)}")
print(f"  ")
print(f"  Gordon Growth Method:")
print(f"    Terminal Value     : {fmt_currency(tv_gordon)}")
print(f"    Enterprise Value   : {fmt_currency(ev_gordon)}")
print(f"    Equity Value       : {fmt_currency(equity_gordon)}")
print(f"    Price per Share    : ${price_gordon:.2f}")
print(f"  ")
print(f"  Exit Multiple Method:")
print(f"    Terminal Value     : {fmt_currency(tv_exit)}")
print(f"    Enterprise Value   : {fmt_currency(ev_exit)}")
print(f"    Equity Value       : {fmt_currency(equity_exit)}")
print(f"    Price per Share    : ${price_exit:.2f}")
print(f"  ")
print(f"  Current Market Price : ${CURRENT_SHARE_PRICE:.2f}")
print(f"  Upside (Gordon)      : {(price_gordon/CURRENT_SHARE_PRICE - 1)*100:.1f}%")
print(f"  Upside (Exit Mult)   : {(price_exit/CURRENT_SHARE_PRICE - 1)*100:.1f}%")
print(f"  ")
print(f"  Figures saved to: outputs/figures/dcf/")
print(f"  Module 1 of 6 complete.")
print("=" * 70)
