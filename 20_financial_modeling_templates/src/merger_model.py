#!/usr/bin/env python3
"""
=============================================================================
MODULE 5: MERGER MODEL -- ACCRETION / DILUTION ANALYSIS
=============================================================================
Author      : Jose Orlando Bobadilla Fuentes
Credentials : CQF | MSc Artificial Intelligence
Role        : Senior Quantitative Portfolio Manager & Lead Data Scientist
Institution : Colombian Pension Fund -- Vicepresidencia de Inversiones

Description
-----------
A comprehensive merger model analyzing the financial impact of a proposed
acquisition of "TargetCo" by "AcquirerCo".  The model covers:

    1. Transaction Structure (cash, stock, mixed consideration)
    2. Sources & Uses of Funds
    3. Purchase Price Allocation (goodwill computation)
    4. Pro Forma Combined Income Statement
    5. Accretion / Dilution Analysis at various offer premiums
    6. Synergy Analysis (revenue & cost synergies, phased-in)
    7. Contribution Analysis (who contributes what)
    8. Breakeven Synergy Analysis (minimum synergies for accretion)

Theoretical Foundations
-----------------------
A merger is accretive if the combined EPS exceeds the acquirer's
standalone EPS, and dilutive otherwise:

    Accretion / Dilution = (Pro Forma EPS - Standalone EPS) / Standalone EPS

Key drivers:
    (a) Target's P/E vs Acquirer's P/E (relative valuation)
    (b) Financing mix (debt vs equity -- debt is generally more accretive
        if Target P/E > after-tax cost of debt)
    (c) Synergies (cost savings + revenue enhancement)
    (d) Purchase accounting adjustments (D&A step-up)

    Breakeven Synergy = Dilution * Combined Shares / (1 - Tax Rate)

References
----------
    - DePamphilis, D. (2019). "Mergers, Acquisitions, and Other
      Restructuring Activities", 10th ed., Academic Press.
    - Rosenbaum, J. & Pearl, J. (2020). "Investment Banking", Ch. 7-8.
    - Bruner, R. (2004). "Applied Mergers and Acquisitions", Wiley.

Output
------
    - Console: Transaction summary, pro forma IS, accretion/dilution
    - Figures: 6 charts saved to outputs/figures/merger/
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.common.style import (
    COLORS, PALETTE, save_figure, print_table, print_section,
    fmt_millions, fmt_pct, fmt_currency
)


# =============================================================================
# 1. COMPANY PROFILES
# =============================================================================
print_section("MODULE 5: MERGER MODEL -- AcquirerCo + TargetCo")

ACQUIRER = {
    "name"        : "AcquirerCo",
    "share_price" : 80.00,
    "shares_out"  : 500e6,          # 500M diluted shares
    "revenue"     : 20_000e6,
    "ebitda"      : 5_000e6,
    "da"          : 800e6,
    "interest"    : 400e6,
    "tax_rate"    : 0.25,
    "net_income"  : 2_850e6,        # Pre-calculated for consistency
    "total_debt"  : 8_000e6,
    "cash"        : 3_000e6,
}
ACQUIRER["market_cap"] = ACQUIRER["share_price"] * ACQUIRER["shares_out"]
ACQUIRER["ev"] = ACQUIRER["market_cap"] + ACQUIRER["total_debt"] - ACQUIRER["cash"]
ACQUIRER["eps"] = ACQUIRER["net_income"] / ACQUIRER["shares_out"]

TARGET = {
    "name"        : "TargetCo",
    "share_price" : 30.00,
    "shares_out"  : 200e6,
    "revenue"     : 5_000e6,
    "ebitda"      : 1_250e6,
    "da"          : 200e6,
    "interest"    : 100e6,
    "tax_rate"    : 0.25,
    "net_income"  : 712.5e6,
    "total_debt"  : 2_000e6,
    "cash"        : 500e6,
    "book_value_assets" : 4_000e6,
    "book_value_liab"   : 2_500e6,
}
TARGET["market_cap"] = TARGET["share_price"] * TARGET["shares_out"]
TARGET["ev"] = TARGET["market_cap"] + TARGET["total_debt"] - TARGET["cash"]
TARGET["eps"] = TARGET["net_income"] / TARGET["shares_out"]
TARGET["pe"]  = TARGET["share_price"] / TARGET["eps"]

# --- Transaction terms ---
OFFER_PREMIUM    = 0.30           # 30% premium to current price
OFFER_PRICE      = TARGET["share_price"] * (1.0 + OFFER_PREMIUM)
EQUITY_PURCHASE  = OFFER_PRICE * TARGET["shares_out"]

# Consideration mix
CASH_PCT  = 0.50                  # 50% cash
STOCK_PCT = 0.50                  # 50% stock

CASH_CONSIDERATION  = EQUITY_PURCHASE * CASH_PCT
STOCK_CONSIDERATION = EQUITY_PURCHASE * STOCK_PCT

# New shares issued (stock portion)
EXCHANGE_RATIO = OFFER_PRICE * STOCK_PCT / ACQUIRER["share_price"]
NEW_SHARES = TARGET["shares_out"] * EXCHANGE_RATIO

# Financing of cash portion
# Mix: existing cash + new term loan
CASH_FROM_BALANCE   = min(1_000e6, ACQUIRER["cash"])
NEW_DEBT_FOR_DEAL   = CASH_CONSIDERATION - CASH_FROM_BALANCE
NEW_DEBT_RATE       = 0.055       # 5.5% on acquisition financing
TRANSACTION_FEES    = EQUITY_PURCHASE * 0.02  # 2% advisory fees

# --- Synergy assumptions ---
COST_SYNERGIES      = 300e6       # $300M annual cost synergies (run-rate)
REVENUE_SYNERGIES   = 150e6       # $150M annual revenue synergies (run-rate)
SYNERGY_TAX_RATE    = 0.25
SYNERGY_PHASE_IN    = np.array([0.25, 0.60, 1.00])  # 3-year phase-in

# --- Purchase accounting ---
ASSET_STEP_UP       = 500e6       # Fair value step-up on intangible assets
STEP_UP_AMORT_YEARS = 10          # Amortize step-up over 10 years
STEP_UP_ANNUAL_DA   = ASSET_STEP_UP / STEP_UP_AMORT_YEARS

print(f"  Acquirer          : {ACQUIRER['name']}")
print(f"  Target            : {TARGET['name']}")
print(f"  Offer Premium     : {OFFER_PREMIUM:.0%}")
print(f"  Offer Price       : ${OFFER_PRICE:.2f}/share")
print(f"  Consideration     : {CASH_PCT:.0%} Cash / {STOCK_PCT:.0%} Stock")
print(f"  Cost Synergies    : {fmt_currency(COST_SYNERGIES)} (run-rate)")
print(f"  Revenue Synergies : {fmt_currency(REVENUE_SYNERGIES)} (run-rate)")


# =============================================================================
# 2. SOURCES & USES
# =============================================================================
print_section("SOURCES & USES OF FUNDS")

total_uses = EQUITY_PURCHASE + TARGET["total_debt"] + TRANSACTION_FEES

sources_rows = [
    ["Cash from Acquirer Balance Sheet", fmt_currency(CASH_FROM_BALANCE)],
    ["New Acquisition Debt",             fmt_currency(NEW_DEBT_FOR_DEAL)],
    ["Stock Consideration",              fmt_currency(STOCK_CONSIDERATION)],
    ["TOTAL SOURCES",                    fmt_currency(CASH_FROM_BALANCE +
                                         NEW_DEBT_FOR_DEAL + STOCK_CONSIDERATION)],
]
uses_rows = [
    ["Equity Purchase Price",    fmt_currency(EQUITY_PURCHASE)],
    ["Refinance Target Debt",    fmt_currency(TARGET["total_debt"])],
    ["Transaction Fees",         fmt_currency(TRANSACTION_FEES)],
    ["TOTAL USES",               fmt_currency(total_uses)],
]

# Note: For simplicity, target debt assumed at acquirer level post-close
# Sources balanced by assuming remaining uses covered at close
print_table("SOURCES", ["Source", "Amount"], sources_rows)
print_table("USES", ["Use", "Amount"], uses_rows)


# =============================================================================
# 3. PURCHASE PRICE ALLOCATION
# =============================================================================
print_section("PURCHASE PRICE ALLOCATION")

book_equity = TARGET["book_value_assets"] - TARGET["book_value_liab"]
total_consideration = EQUITY_PURCHASE  # equity value paid
goodwill = total_consideration - book_equity - ASSET_STEP_UP

ppa_rows = [
    ["Equity Purchase Price",        fmt_currency(total_consideration)],
    ["(-) Book Value of Equity",     fmt_currency(book_equity)],
    ["(-) Asset Step-Up (FV adj)",   fmt_currency(ASSET_STEP_UP)],
    ["= Goodwill",                   fmt_currency(goodwill)],
    ["", ""],
    ["Step-Up Amortization Period",  f"{STEP_UP_AMORT_YEARS} years"],
    ["Annual Step-Up D&A",           fmt_currency(STEP_UP_ANNUAL_DA)],
]
print_table("PURCHASE PRICE ALLOCATION", ["Component", "Value"], ppa_rows)


# =============================================================================
# 4. PRO FORMA COMBINED INCOME STATEMENT (3-YEAR)
# =============================================================================
print_section("PRO FORMA COMBINED INCOME STATEMENT")

N_PROJ = 3
proj_labels = [f"Year {y}" for y in range(1, N_PROJ + 1)]

# Standalone acquirer (flat for simplicity)
acq_revenue = np.full(N_PROJ, ACQUIRER["revenue"])
acq_ebitda  = np.full(N_PROJ, ACQUIRER["ebitda"])
acq_da      = np.full(N_PROJ, ACQUIRER["da"])
acq_ebit    = acq_ebitda - acq_da
acq_interest = np.full(N_PROJ, ACQUIRER["interest"])

# Standalone target (flat)
tgt_revenue = np.full(N_PROJ, TARGET["revenue"])
tgt_ebitda  = np.full(N_PROJ, TARGET["ebitda"])
tgt_da      = np.full(N_PROJ, TARGET["da"])
tgt_ebit    = tgt_ebitda - tgt_da
tgt_interest = np.full(N_PROJ, TARGET["interest"])

# Synergies (phased-in)
syn_cost_savings = COST_SYNERGIES * SYNERGY_PHASE_IN
syn_revenue      = REVENUE_SYNERGIES * SYNERGY_PHASE_IN

# New interest from acquisition debt
new_interest = np.full(N_PROJ, NEW_DEBT_FOR_DEAL * NEW_DEBT_RATE)

# Lost interest on cash used
lost_interest_income = np.full(N_PROJ, CASH_FROM_BALANCE * 0.03)

# Step-up amortization
step_up_da = np.full(N_PROJ, STEP_UP_ANNUAL_DA)

# Pro Forma combined
pf_revenue  = acq_revenue + tgt_revenue + syn_revenue
pf_ebitda   = acq_ebitda + tgt_ebitda + syn_cost_savings + syn_revenue
pf_da       = acq_da + tgt_da + step_up_da
pf_ebit     = pf_ebitda - pf_da
pf_interest = acq_interest + tgt_interest + new_interest + lost_interest_income
pf_ebt      = pf_ebit - pf_interest
pf_taxes    = np.maximum(0, pf_ebt * ACQUIRER["tax_rate"])
pf_ni       = pf_ebt - pf_taxes

# Pro forma shares
pf_shares = ACQUIRER["shares_out"] + NEW_SHARES
pf_eps = pf_ni / pf_shares

# Standalone acquirer EPS
acq_ebt      = acq_ebit - acq_interest
acq_taxes    = np.maximum(0, acq_ebt * ACQUIRER["tax_rate"])
acq_ni       = acq_ebt - acq_taxes
acq_eps      = acq_ni / ACQUIRER["shares_out"]

# Accretion / Dilution
accretion_pct = (pf_eps - acq_eps) / acq_eps * 100

pf_headers = ["Item"] + proj_labels
pf_rows = [
    ["--- Acquirer Standalone ---", ""] + [""] * (N_PROJ - 1),
    ["  Revenue"]          + [fmt_currency(v) for v in acq_revenue],
    ["  EBITDA"]           + [fmt_currency(v) for v in acq_ebitda],
    ["  Net Income"]       + [fmt_currency(v) for v in acq_ni],
    ["  EPS"]              + [f"${v:.2f}" for v in acq_eps],
    ["", ""] + [""] * (N_PROJ - 1),
    ["--- Target Standalone ---", ""] + [""] * (N_PROJ - 1),
    ["  Revenue"]          + [fmt_currency(v) for v in tgt_revenue],
    ["  EBITDA"]           + [fmt_currency(v) for v in tgt_ebitda],
    ["", ""] + [""] * (N_PROJ - 1),
    ["--- Adjustments ---", ""] + [""] * (N_PROJ - 1),
    ["  Cost Synergies"]   + [fmt_currency(v) for v in syn_cost_savings],
    ["  Revenue Synergies"]+ [fmt_currency(v) for v in syn_revenue],
    ["  New Debt Interest"] + [fmt_currency(v) for v in new_interest],
    ["  Step-Up D&A"]      + [fmt_currency(v) for v in step_up_da],
    ["", ""] + [""] * (N_PROJ - 1),
    ["--- Pro Forma Combined ---", ""] + [""] * (N_PROJ - 1),
    ["  Revenue"]          + [fmt_currency(v) for v in pf_revenue],
    ["  EBITDA"]           + [fmt_currency(v) for v in pf_ebitda],
    ["  EBIT"]             + [fmt_currency(v) for v in pf_ebit],
    ["  Interest"]         + [fmt_currency(v) for v in pf_interest],
    ["  Net Income"]       + [fmt_currency(v) for v in pf_ni],
    ["  Shares Outstanding"] + [f"{pf_shares/1e6:.0f}M"] * N_PROJ,
    ["  Pro Forma EPS"]    + [f"${v:.2f}" for v in pf_eps],
    ["", ""] + [""] * (N_PROJ - 1),
    ["  Accretion/(Dilution)"] + [f"{v:+.1f}%" for v in accretion_pct],
]
print_table("PRO FORMA INCOME STATEMENT", pf_headers, pf_rows)


# =============================================================================
# 5. ACCRETION/DILUTION AT VARIOUS PREMIUMS
# =============================================================================
print_section("ACCRETION/DILUTION SENSITIVITY -- OFFER PREMIUM")

premium_range = np.arange(0.10, 0.55, 0.05)
ad_by_premium = {"all_cash": [], "all_stock": [], "mixed_50_50": []}

for prem in premium_range:
    offer_px = TARGET["share_price"] * (1.0 + prem)
    eq_purchase = offer_px * TARGET["shares_out"]

    for scenario, cash_pct in [("all_cash", 1.0), ("all_stock", 0.0),
                                ("mixed_50_50", 0.5)]:
        cash_cons = eq_purchase * cash_pct
        stock_cons = eq_purchase * (1.0 - cash_pct)

        # New shares
        if stock_cons > 0:
            exch_ratio = offer_px * (1.0 - cash_pct) / ACQUIRER["share_price"]
            new_sh = TARGET["shares_out"] * exch_ratio
        else:
            new_sh = 0

        # Financing
        cash_used = min(CASH_FROM_BALANCE, cash_cons)
        new_debt = cash_cons - cash_used

        # Year 1 pro forma (with full synergies at 25% phase-in)
        syn_yr1 = (COST_SYNERGIES + REVENUE_SYNERGIES) * SYNERGY_PHASE_IN[0]
        combined_ebitda = ACQUIRER["ebitda"] + TARGET["ebitda"] + syn_yr1
        combined_da = ACQUIRER["da"] + TARGET["da"] + STEP_UP_ANNUAL_DA
        combined_ebit = combined_ebitda - combined_da
        combined_int = (ACQUIRER["interest"] + TARGET["interest"] +
                       new_debt * NEW_DEBT_RATE + cash_used * 0.03)
        combined_ebt = combined_ebit - combined_int
        combined_tax = max(0, combined_ebt * 0.25)
        combined_ni = combined_ebt - combined_tax
        combined_shares = ACQUIRER["shares_out"] + new_sh
        combined_eps = combined_ni / combined_shares

        ad_pct = (combined_eps - ACQUIRER["eps"]) / ACQUIRER["eps"] * 100
        ad_by_premium[scenario].append(ad_pct)

# Print sensitivity
ad_headers = ["Premium"] + [f"{p:.0%}" for p in premium_range]
ad_rows = [
    ["All Cash"]    + [f"{v:+.1f}%" for v in ad_by_premium["all_cash"]],
    ["All Stock"]   + [f"{v:+.1f}%" for v in ad_by_premium["all_stock"]],
    ["50/50 Mixed"] + [f"{v:+.1f}%" for v in ad_by_premium["mixed_50_50"]],
]
print_table("ACCRETION/(DILUTION) BY PREMIUM & CONSIDERATION",
            ad_headers, ad_rows)


# =============================================================================
# 6. CONTRIBUTION ANALYSIS
# =============================================================================
print_section("CONTRIBUTION ANALYSIS")

total_rev    = ACQUIRER["revenue"] + TARGET["revenue"]
total_ebitda = ACQUIRER["ebitda"] + TARGET["ebitda"]
total_ni     = ACQUIRER["net_income"] + TARGET["net_income"]

contrib_rows = [
    ["Revenue",
     f"{ACQUIRER['revenue']/total_rev*100:.1f}%",
     f"{TARGET['revenue']/total_rev*100:.1f}%"],
    ["EBITDA",
     f"{ACQUIRER['ebitda']/total_ebitda*100:.1f}%",
     f"{TARGET['ebitda']/total_ebitda*100:.1f}%"],
    ["Net Income",
     f"{ACQUIRER['net_income']/total_ni*100:.1f}%",
     f"{TARGET['net_income']/total_ni*100:.1f}%"],
    ["Equity Value (implied)",
     f"{ACQUIRER['market_cap']/(ACQUIRER['market_cap']+EQUITY_PURCHASE)*100:.1f}%",
     f"{EQUITY_PURCHASE/(ACQUIRER['market_cap']+EQUITY_PURCHASE)*100:.1f}%"],
]
print_table("CONTRIBUTION ANALYSIS",
            ["Metric", ACQUIRER["name"], TARGET["name"]], contrib_rows)


# =============================================================================
# 7. BREAKEVEN SYNERGY ANALYSIS
# =============================================================================
print_section("BREAKEVEN SYNERGY ANALYSIS")

# Minimum synergies needed for Year 1 EPS accretion (at current deal terms)
# Without any synergies, Year 1 EPS:
no_syn_ebitda = ACQUIRER["ebitda"] + TARGET["ebitda"]
no_syn_da     = ACQUIRER["da"] + TARGET["da"] + STEP_UP_ANNUAL_DA
no_syn_ebit   = no_syn_ebitda - no_syn_da
no_syn_int    = (ACQUIRER["interest"] + TARGET["interest"] +
                 NEW_DEBT_FOR_DEAL * NEW_DEBT_RATE + CASH_FROM_BALANCE * 0.03)
no_syn_ebt    = no_syn_ebit - no_syn_int
no_syn_tax    = max(0, no_syn_ebt * 0.25)
no_syn_ni     = no_syn_ebt - no_syn_tax
no_syn_eps    = no_syn_ni / pf_shares

eps_gap = ACQUIRER["eps"] - no_syn_eps  # per-share gap to break even

if eps_gap > 0:
    # Dilutive without synergies -- need synergies
    breakeven_ni_needed = eps_gap * pf_shares
    breakeven_pretax_syn = breakeven_ni_needed / (1.0 - ACQUIRER["tax_rate"])
    be_note = "Deal is DILUTIVE without synergies"
else:
    breakeven_pretax_syn = 0
    be_note = "Deal is ACCRETIVE even without synergies"

be_rows = [
    ["Standalone Acquirer EPS",     f"${ACQUIRER['eps']:.2f}"],
    ["PF EPS (No Synergies)",       f"${no_syn_eps:.2f}"],
    ["EPS Gap",                     f"${eps_gap:.2f}"],
    ["Status",                      be_note],
    ["Breakeven Pre-Tax Synergies", fmt_currency(breakeven_pretax_syn)],
    ["As % of Target EBITDA",       f"{breakeven_pretax_syn/TARGET['ebitda']*100:.1f}%"
                                    if breakeven_pretax_syn > 0 else "N/A"],
]
print_table("BREAKEVEN SYNERGY", ["Metric", "Value"], be_rows)


# =============================================================================
# 8. VISUALIZATIONS
# =============================================================================
print_section("GENERATING VISUALIZATIONS")

# -------------------------------------------------------------------------
# FIGURE 1: Accretion/Dilution by Year with Synergy Phase-In
# -------------------------------------------------------------------------
fig1, ax1 = plt.subplots(figsize=(10, 6))

x1 = np.arange(N_PROJ)
bar_width = 0.5

colors_ad = [COLORS["secondary"] if v > 0 else COLORS["danger"]
             for v in accretion_pct]
bars1 = ax1.bar(x1, accretion_pct, width=bar_width, color=colors_ad,
                alpha=0.85, edgecolor="none")

for bar, val in zip(bars1, accretion_pct):
    y_off = 0.3 if val >= 0 else -0.8
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + y_off,
             f"{val:+.1f}%", ha="center", fontsize=10, color=COLORS["white"],
             fontweight="bold")

ax1.axhline(y=0, color=COLORS["white"], linewidth=0.8, alpha=0.5)
ax1.set_xlabel("Year")
ax1.set_ylabel("Accretion / (Dilution) %")
ax1.set_title("Pro Forma EPS Accretion / (Dilution) with Synergy Phase-In",
              fontsize=14, fontweight="bold", pad=15)
ax1.set_xticks(x1)
ax1.set_xticklabels(proj_labels)

# Synergy annotation
ax1_twin = ax1.twinx()
syn_total = (syn_cost_savings + syn_revenue) / 1e6
ax1_twin.plot(x1, syn_total, color=COLORS["accent"], marker="o",
              linewidth=2, markersize=7, label="Total Synergies ($M)")
ax1_twin.set_ylabel("Synergies ($M)", color=COLORS["accent"])
ax1_twin.tick_params(axis="y", labelcolor=COLORS["accent"])

fig1.tight_layout()
save_figure(fig1, "merger_01_accretion_dilution", subdir="merger")

# -------------------------------------------------------------------------
# FIGURE 2: Accretion/Dilution by Premium & Consideration
# -------------------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(11, 6))

ax2.plot(premium_range * 100, ad_by_premium["all_cash"],
         color=COLORS["primary"], marker="o", linewidth=2.5,
         markersize=7, label="All Cash")
ax2.plot(premium_range * 100, ad_by_premium["all_stock"],
         color=COLORS["secondary"], marker="s", linewidth=2.5,
         markersize=7, label="All Stock")
ax2.plot(premium_range * 100, ad_by_premium["mixed_50_50"],
         color=COLORS["accent"], marker="^", linewidth=2.5,
         markersize=7, label="50/50 Mix")

ax2.axhline(y=0, color=COLORS["danger"], linewidth=1.5, linestyle="--",
            alpha=0.7, label="Breakeven")
ax2.axvline(x=OFFER_PREMIUM * 100, color=COLORS["white"], linewidth=1,
            linestyle=":", alpha=0.5, label=f"Current Premium: {OFFER_PREMIUM:.0%}")

ax2.set_xlabel("Offer Premium (%)")
ax2.set_ylabel("Year 1 Accretion / (Dilution) %")
ax2.set_title("Accretion/(Dilution) Sensitivity to Premium & Consideration",
              fontsize=13, fontweight="bold", pad=15)
ax2.legend(loc="lower left")
ax2.fill_between(premium_range * 100, 0, max(max(ad_by_premium["all_cash"]), 10),
                 alpha=0.05, color=COLORS["secondary"])
ax2.fill_between(premium_range * 100, min(min(ad_by_premium["all_stock"]), -10), 0,
                 alpha=0.05, color=COLORS["danger"])

fig2.tight_layout()
save_figure(fig2, "merger_02_premium_sensitivity", subdir="merger")

# -------------------------------------------------------------------------
# FIGURE 3: Contribution Analysis (Stacked Horizontal)
# -------------------------------------------------------------------------
fig3, ax3 = plt.subplots(figsize=(10, 5))

contrib_metrics = ["Revenue", "EBITDA", "Net Income", "Equity\n(Implied)"]
acq_contribs = [
    ACQUIRER["revenue"] / total_rev * 100,
    ACQUIRER["ebitda"] / total_ebitda * 100,
    ACQUIRER["net_income"] / total_ni * 100,
    ACQUIRER["market_cap"] / (ACQUIRER["market_cap"] + EQUITY_PURCHASE) * 100,
]
tgt_contribs = [100 - v for v in acq_contribs]

y3 = np.arange(len(contrib_metrics))
h3 = 0.5

ax3.barh(y3, acq_contribs, height=h3, color=COLORS["primary"], alpha=0.85,
         label=ACQUIRER["name"])
ax3.barh(y3, tgt_contribs, left=acq_contribs, height=h3,
         color=COLORS["accent"], alpha=0.85, label=TARGET["name"])

for i, (a, t) in enumerate(zip(acq_contribs, tgt_contribs)):
    ax3.text(a/2, i, f"{a:.1f}%", ha="center", va="center",
             fontsize=10, color=COLORS["white"], fontweight="bold")
    ax3.text(a + t/2, i, f"{t:.1f}%", ha="center", va="center",
             fontsize=10, color=COLORS["white"], fontweight="bold")

ax3.set_yticks(y3)
ax3.set_yticklabels(contrib_metrics)
ax3.set_xlabel("Contribution (%)")
ax3.set_title("Contribution Analysis -- Who Brings What",
              fontsize=14, fontweight="bold", pad=15)
ax3.legend(loc="lower right")
ax3.invert_yaxis()

fig3.tight_layout()
save_figure(fig3, "merger_03_contribution", subdir="merger")

# -------------------------------------------------------------------------
# FIGURE 4: Synergy Phase-In & Impact
# -------------------------------------------------------------------------
fig4, ax4 = plt.subplots(figsize=(10, 6))

x4 = np.arange(N_PROJ)
syn_cost_plot = syn_cost_savings / 1e6
syn_rev_plot  = syn_revenue / 1e6

ax4.bar(x4 - 0.2, syn_cost_plot, 0.35, color=COLORS["secondary"],
        alpha=0.85, label="Cost Synergies")
ax4.bar(x4 + 0.2, syn_rev_plot, 0.35, color=COLORS["accent"],
        alpha=0.85, label="Revenue Synergies")

# Total line
total_syn = (syn_cost_savings + syn_revenue) / 1e6
ax4.plot(x4, total_syn, color=COLORS["primary"], marker="D",
         linewidth=2.5, markersize=8, label="Total Synergies", zorder=5)

for i, v in enumerate(total_syn):
    ax4.text(i, v + 8, f"${v:.0f}M", ha="center", fontsize=9,
             color=COLORS["white"], fontweight="bold")

# Phase-in % annotation
for i, pct in enumerate(SYNERGY_PHASE_IN):
    ax4.text(i, -15, f"{pct:.0%} realized", ha="center", fontsize=8,
             color=COLORS["white"], alpha=0.7)

ax4.set_xlabel("Year")
ax4.set_ylabel("Synergies ($M)")
ax4.set_title("Synergy Realization Schedule",
              fontsize=14, fontweight="bold", pad=15)
ax4.set_xticks(x4)
ax4.set_xticklabels(proj_labels)
ax4.legend(loc="upper left")

fig4.tight_layout()
save_figure(fig4, "merger_04_synergies", subdir="merger")

# -------------------------------------------------------------------------
# FIGURE 5: EPS Bridge -- Standalone to Pro Forma
# -------------------------------------------------------------------------
fig5, ax5 = plt.subplots(figsize=(11, 6))

# Year 1 EPS bridge components
standalone_eps = ACQUIRER["eps"]
target_contrib = TARGET["net_income"] / pf_shares
syn_impact = ((COST_SYNERGIES + REVENUE_SYNERGIES) * SYNERGY_PHASE_IN[0] *
              (1 - ACQUIRER["tax_rate"])) / pf_shares
financing_cost = (-(NEW_DEBT_FOR_DEAL * NEW_DEBT_RATE +
                    CASH_FROM_BALANCE * 0.03) * (1 - ACQUIRER["tax_rate"])) / pf_shares
step_up_impact = -(STEP_UP_ANNUAL_DA * (1 - ACQUIRER["tax_rate"])) / pf_shares
dilution_impact = -(ACQUIRER["net_income"] / ACQUIRER["shares_out"] -
                    ACQUIRER["net_income"] / pf_shares)

bridge_labels = ["Standalone\nEPS", "Target\nEarnings", "Synergies\n(Y1)",
                 "Financing\nCost", "Step-Up\nD&A", "Share\nDilution",
                 "Pro Forma\nEPS"]
bridge_vals = [standalone_eps, target_contrib, syn_impact,
               financing_cost, step_up_impact, dilution_impact, pf_eps[0]]

cum5 = np.zeros(len(bridge_vals))
cum5[0] = bridge_vals[0]
for k in range(1, len(bridge_vals) - 1):
    cum5[k] = cum5[k-1] + bridge_vals[k]
cum5[-1] = bridge_vals[-1]

bot5 = np.zeros(len(bridge_vals))
bot5[0] = 0
for k in range(1, len(bridge_vals) - 1):
    if bridge_vals[k] >= 0:
        bot5[k] = cum5[k] - bridge_vals[k]
    else:
        bot5[k] = cum5[k]
bot5[-1] = 0

c5 = []
for k, v in enumerate(bridge_vals):
    if k == 0:
        c5.append(COLORS["primary"])
    elif k == len(bridge_vals) - 1:
        c5.append(COLORS["teal"])
    elif v >= 0:
        c5.append(COLORS["secondary"])
    else:
        c5.append(COLORS["danger"])

ax5.bar(bridge_labels, [abs(v) for v in bridge_vals], bottom=bot5,
        color=c5, alpha=0.85, edgecolor="none", width=0.55)

for k, val in enumerate(bridge_vals):
    y_pos = bot5[k] + abs(val) / 2
    sign = "+" if val > 0 and 0 < k < len(bridge_vals)-1 else ""
    ax5.text(k, y_pos, f"{sign}${val:.2f}", ha="center", va="center",
             fontsize=9, color=COLORS["white"], fontweight="bold")

ax5.set_title("EPS Bridge: Standalone to Pro Forma (Year 1)",
              fontsize=14, fontweight="bold", pad=15)
ax5.set_ylabel("Earnings Per Share ($)")

fig5.tight_layout()
save_figure(fig5, "merger_05_eps_bridge", subdir="merger")

# -------------------------------------------------------------------------
# FIGURE 6: Breakeven Synergy Sensitivity
# -------------------------------------------------------------------------
fig6, ax6 = plt.subplots(figsize=(10, 6))

# How breakeven synergies change with premium
be_premiums = np.arange(0.10, 0.55, 0.05)
be_synergies = {"all_cash": [], "all_stock": [], "mixed": []}

for prem in be_premiums:
    offer_px = TARGET["share_price"] * (1.0 + prem)
    eq_pur = offer_px * TARGET["shares_out"]

    for scenario, cash_pct in [("all_cash", 1.0), ("all_stock", 0.0),
                                ("mixed", 0.5)]:
        cash_cons = eq_pur * cash_pct
        stock_cons = eq_pur * (1.0 - cash_pct)
        if stock_cons > 0:
            exch_r = offer_px * (1.0 - cash_pct) / ACQUIRER["share_price"]
            n_sh = TARGET["shares_out"] * exch_r
        else:
            n_sh = 0

        cash_u = min(CASH_FROM_BALANCE, cash_cons)
        new_d = cash_cons - cash_u

        # No-synergy EPS
        ns_ebitda = ACQUIRER["ebitda"] + TARGET["ebitda"]
        ns_da = ACQUIRER["da"] + TARGET["da"] + STEP_UP_ANNUAL_DA
        ns_ebit = ns_ebitda - ns_da
        ns_int = (ACQUIRER["interest"] + TARGET["interest"] +
                  new_d * NEW_DEBT_RATE + cash_u * 0.03)
        ns_ebt = ns_ebit - ns_int
        ns_tax = max(0, ns_ebt * 0.25)
        ns_ni = ns_ebt - ns_tax
        comb_sh = ACQUIRER["shares_out"] + n_sh
        ns_eps = ns_ni / comb_sh

        gap = ACQUIRER["eps"] - ns_eps
        if gap > 0:
            be_ni = gap * comb_sh
            be_pretax = be_ni / (1 - 0.25)
        else:
            be_pretax = 0

        be_synergies[scenario].append(be_pretax / 1e6)

ax6.plot(be_premiums * 100, be_synergies["all_cash"],
         color=COLORS["primary"], marker="o", linewidth=2.5,
         markersize=7, label="All Cash")
ax6.plot(be_premiums * 100, be_synergies["all_stock"],
         color=COLORS["secondary"], marker="s", linewidth=2.5,
         markersize=7, label="All Stock")
ax6.plot(be_premiums * 100, be_synergies["mixed"],
         color=COLORS["accent"], marker="^", linewidth=2.5,
         markersize=7, label="50/50 Mix")

ax6.axhline(y=(COST_SYNERGIES + REVENUE_SYNERGIES) / 1e6,
            color=COLORS["danger"], linewidth=1.5, linestyle="--", alpha=0.7,
            label=f"Run-Rate Synergies: ${(COST_SYNERGIES+REVENUE_SYNERGIES)/1e6:.0f}M")

ax6.set_xlabel("Offer Premium (%)")
ax6.set_ylabel("Breakeven Pre-Tax Synergies ($M)")
ax6.set_title("Breakeven Synergy Analysis by Premium & Consideration",
              fontsize=13, fontweight="bold", pad=15)
ax6.legend(loc="upper left")

fig6.tight_layout()
save_figure(fig6, "merger_06_breakeven_synergy", subdir="merger")


# =============================================================================
# 9. SUMMARY
# =============================================================================
print_section("MERGER MODEL SUMMARY")
print(f"  Acquirer             : {ACQUIRER['name']}")
print(f"  Target               : {TARGET['name']}")
print(f"  Offer Premium        : {OFFER_PREMIUM:.0%} (${OFFER_PRICE:.2f}/share)")
print(f"  Consideration        : {CASH_PCT:.0%} Cash / {STOCK_PCT:.0%} Stock")
print(f"  Equity Purchase      : {fmt_currency(EQUITY_PURCHASE)}")
print(f"  Goodwill Created     : {fmt_currency(goodwill)}")
print(f"  New Shares Issued    : {NEW_SHARES/1e6:.1f}M")
print(f"  Standalone EPS       : ${ACQUIRER['eps']:.2f}")
print(f"  Pro Forma Y1 EPS     : ${pf_eps[0]:.2f} ({accretion_pct[0]:+.1f}%)")
print(f"  Pro Forma Y3 EPS     : ${pf_eps[2]:.2f} ({accretion_pct[2]:+.1f}%)")
print(f"  Breakeven Synergies  : {fmt_currency(breakeven_pretax_syn)} pre-tax")
print(f"  Figures saved to     : outputs/figures/merger/")
print(f"  Module 5 of 6 complete.")
print("=" * 70)
