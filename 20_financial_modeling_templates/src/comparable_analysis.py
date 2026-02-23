#!/usr/bin/env python3
"""
=============================================================================
MODULE 4: COMPARABLE COMPANY ANALYSIS
=============================================================================
Author      : Jose Orlando Bobadilla Fuentes
Credentials : CQF | MSc Artificial Intelligence
Role        : Senior Quantitative Portfolio Manager & Lead Data Scientist
Institution : Colombian Pension Fund -- Vicepresidencia de Inversiones

Description
-----------
A comprehensive comparable company analysis engine implementing both
Trading Comps and Transaction (Precedent) Comps methodologies to derive
an implied valuation range for a target company ("TechTarget Inc.").

The module covers:

    1. Trading Comps Universe Selection & Screening
       - EV/Revenue, EV/EBITDA, EV/EBIT, P/E multiples
       - Growth-adjusted metrics (PEG ratio, EV/EBITDA/Growth)
       - Profitability & return metrics (margins, ROIC, ROE)
       - Statistical analysis (mean, median, quartiles, trimmed mean)

    2. Precedent Transaction Comps
       - Acquisition multiples from recent M&A transactions
       - Control premium analysis
       - Time-decay weighting (more recent = higher weight)

    3. Implied Valuation Range
       - Football field from multiple methodologies
       - Weighted composite valuation
       - Confidence intervals via bootstrapping

    4. Regression-Based Valuation
       - EV/EBITDA vs EBITDA Margin regression
       - Implied multiple from target's fundamentals
       - R-squared and statistical significance

Theoretical Foundations
-----------------------
Relative valuation assumes that similar companies should trade at
similar multiples.  The key challenges are:

    (a) Defining "comparable" -- industry, size, growth, profitability
    (b) Selecting the right multiple -- enterprise vs equity, forward vs trailing
    (c) Adjusting for differences -- normalizing one-time items, leverage

    Implied EV = Target Metric * Peer Multiple
    Implied Equity = Implied EV - Net Debt

The approach complements intrinsic valuation (DCF) by providing a
market-based cross-check anchored in observable prices.

References
----------
    - Damodaran, A. (2012). "Investment Valuation", 3rd ed., Wiley, Ch. 17-20.
    - Rosenbaum, J. & Pearl, J. (2020). "Investment Banking", 3rd ed., Wiley.
    - Liu, J., Nissim, D. & Thomas, J. (2002). "Equity Valuation Using
      Multiples", Journal of Accounting Research, 40(1), 135-172.

Output
------
    - Console: Comp tables, statistical summaries, implied valuations
    - Figures: 6 charts saved to outputs/figures/comps/
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.common.style import (
    COLORS, PALETTE, save_figure, print_table, print_section,
    fmt_millions, fmt_pct, fmt_currency, fmt_multiple
)


# =============================================================================
# 1. TARGET COMPANY PROFILE
# =============================================================================
print_section("MODULE 4: COMPARABLE COMPANY ANALYSIS -- TechTarget Inc.")

TARGET = {
    "name"          : "TechTarget Inc.",
    "revenue"       : 3_500e6,      # $3.5B LTM revenue
    "ebitda"        : 875e6,        # $875M LTM EBITDA (25.0% margin)
    "ebit"          : 700e6,        # $700M LTM EBIT
    "net_income"    : 490e6,        # $490M LTM net income
    "revenue_fwd"   : 3_850e6,      # NTM consensus revenue
    "ebitda_fwd"    : 1_001e6,      # NTM consensus EBITDA
    "eps_fwd"       : 3.50,         # NTM consensus EPS
    "rev_growth"    : 0.10,         # LTM revenue growth
    "ebitda_margin" : 0.250,        # LTM EBITDA margin
    "net_debt"      : 1_200e6,      # Net debt
    "shares_out"    : 200e6,        # Diluted shares
    "share_price"   : 42.00,        # Current share price
}
TARGET["market_cap"]  = TARGET["shares_out"] * TARGET["share_price"]
TARGET["ev"]          = TARGET["market_cap"] + TARGET["net_debt"]

print(f"  Company       : {TARGET['name']}")
print(f"  Revenue (LTM) : {fmt_currency(TARGET['revenue'])}")
print(f"  EBITDA (LTM)  : {fmt_currency(TARGET['ebitda'])}")
print(f"  Market Cap    : {fmt_currency(TARGET['market_cap'])}")
print(f"  EV            : {fmt_currency(TARGET['ev'])}")


# =============================================================================
# 2. TRADING COMPS UNIVERSE
# =============================================================================
print_section("TRADING COMPS UNIVERSE")

# Fictional peer universe -- 10 comparable technology companies
# Each dict: name, mcap, ev, rev, ebitda, ebit, ni, rev_growth, ebitda_margin
PEERS = [
    {"name": "AlphaTech",    "mcap": 12_000e6, "ev": 13_500e6,
     "rev": 5_200e6,  "ebitda": 1_456e6, "ebit": 1_196e6, "ni": 832e6,
     "rev_fwd": 5_720e6, "ebitda_fwd": 1_659e6, "eps_fwd": 4.10,
     "rev_growth": 0.12, "ebitda_margin": 0.280, "roic": 0.18, "roe": 0.22,
     "shares": 250e6},
    {"name": "BetaSoft",     "mcap": 8_000e6,  "ev": 8_800e6,
     "rev": 3_800e6,  "ebitda": 950e6,   "ebit": 760e6,   "ni": 532e6,
     "rev_fwd": 4_104e6, "ebitda_fwd": 1_067e6, "eps_fwd": 3.20,
     "rev_growth": 0.08, "ebitda_margin": 0.250, "roic": 0.15, "roe": 0.19,
     "shares": 200e6},
    {"name": "GammaSys",     "mcap": 15_000e6, "ev": 16_200e6,
     "rev": 6_000e6,  "ebitda": 1_680e6, "ebit": 1_380e6, "ni": 966e6,
     "rev_fwd": 6_600e6, "ebitda_fwd": 1_914e6, "eps_fwd": 4.80,
     "rev_growth": 0.10, "ebitda_margin": 0.280, "roic": 0.20, "roe": 0.25,
     "shares": 280e6},
    {"name": "DeltaCloud",   "mcap": 6_500e6,  "ev": 7_200e6,
     "rev": 2_800e6,  "ebitda": 644e6,   "ebit": 504e6,   "ni": 352e6,
     "rev_fwd": 3_136e6, "ebitda_fwd": 753e6,   "eps_fwd": 2.60,
     "rev_growth": 0.12, "ebitda_margin": 0.230, "roic": 0.14, "roe": 0.17,
     "shares": 180e6},
    {"name": "EpsilonData",  "mcap": 10_500e6, "ev": 11_800e6,
     "rev": 4_500e6,  "ebitda": 1_215e6, "ebit": 990e6,   "ni": 693e6,
     "rev_fwd": 4_950e6, "ebitda_fwd": 1_386e6, "eps_fwd": 3.85,
     "rev_growth": 0.10, "ebitda_margin": 0.270, "roic": 0.17, "roe": 0.21,
     "shares": 230e6},
    {"name": "ZetaNet",      "mcap": 4_200e6,  "ev": 4_900e6,
     "rev": 2_100e6,  "ebitda": 462e6,   "ebit": 357e6,   "ni": 250e6,
     "rev_fwd": 2_268e6, "ebitda_fwd": 522e6,   "eps_fwd": 2.10,
     "rev_growth": 0.08, "ebitda_margin": 0.220, "roic": 0.13, "roe": 0.16,
     "shares": 150e6},
    {"name": "EtaCyber",     "mcap": 9_000e6,  "ev": 10_000e6,
     "rev": 3_600e6,  "ebitda": 972e6,   "ebit": 792e6,   "ni": 554e6,
     "rev_fwd": 4_032e6, "ebitda_fwd": 1_129e6, "eps_fwd": 3.40,
     "rev_growth": 0.12, "ebitda_margin": 0.270, "roic": 0.16, "roe": 0.20,
     "shares": 220e6},
    {"name": "ThetaAI",      "mcap": 18_000e6, "ev": 19_500e6,
     "rev": 7_000e6,  "ebitda": 2_100e6, "ebit": 1_750e6, "ni": 1_225e6,
     "rev_fwd": 8_050e6, "ebitda_fwd": 2_496e6, "eps_fwd": 5.60,
     "rev_growth": 0.15, "ebitda_margin": 0.300, "roic": 0.22, "roe": 0.28,
     "shares": 300e6},
    {"name": "IotaLogic",    "mcap": 5_000e6,  "ev": 5_600e6,
     "rev": 2_400e6,  "ebitda": 552e6,   "ebit": 432e6,   "ni": 302e6,
     "rev_fwd": 2_616e6, "ebitda_fwd": 627e6,   "eps_fwd": 2.30,
     "rev_growth": 0.09, "ebitda_margin": 0.230, "roic": 0.14, "roe": 0.18,
     "shares": 170e6},
    {"name": "KappaWare",    "mcap": 7_500e6,  "ev": 8_400e6,
     "rev": 3_200e6,  "ebitda": 832e6,   "ebit": 672e6,   "ni": 470e6,
     "rev_fwd": 3_520e6, "ebitda_fwd": 950e6,   "eps_fwd": 2.95,
     "rev_growth": 0.10, "ebitda_margin": 0.260, "roic": 0.16, "roe": 0.20,
     "shares": 210e6},
]

# Compute multiples for each peer
for p in PEERS:
    p["ev_rev"]        = p["ev"] / p["rev"]
    p["ev_ebitda"]     = p["ev"] / p["ebitda"]
    p["ev_ebit"]       = p["ev"] / p["ebit"]
    p["pe"]            = p["mcap"] / p["ni"]
    p["ev_rev_fwd"]    = p["ev"] / p["rev_fwd"]
    p["ev_ebitda_fwd"] = p["ev"] / p["ebitda_fwd"]
    p["pe_fwd"]        = p["mcap"] / (p["eps_fwd"] * p["shares"])
    p["peg"]           = p["pe"] / (p["rev_growth"] * 100) if p["rev_growth"] > 0 else np.nan

# Print trading comps table
tc_headers = ["Company", "EV/Rev", "EV/EBITDA", "EV/EBIT", "P/E",
              "Fwd EV/EBITDA", "Fwd P/E", "Growth", "Margin"]
tc_rows = []
for p in PEERS:
    tc_rows.append([
        p["name"],
        f"{p['ev_rev']:.2f}x",
        f"{p['ev_ebitda']:.1f}x",
        f"{p['ev_ebit']:.1f}x",
        f"{p['pe']:.1f}x",
        f"{p['ev_ebitda_fwd']:.1f}x",
        f"{p['pe_fwd']:.1f}x",
        f"{p['rev_growth']:.0%}",
        f"{p['ebitda_margin']:.0%}",
    ])
print_table("TRADING COMPS -- KEY MULTIPLES", tc_headers, tc_rows)


# =============================================================================
# 3. STATISTICAL SUMMARY
# =============================================================================
print_section("STATISTICAL SUMMARY OF PEER MULTIPLES")

METRICS = {
    "EV/Revenue (LTM)"      : [p["ev_rev"] for p in PEERS],
    "EV/EBITDA (LTM)"       : [p["ev_ebitda"] for p in PEERS],
    "EV/EBIT (LTM)"         : [p["ev_ebit"] for p in PEERS],
    "P/E (LTM)"             : [p["pe"] for p in PEERS],
    "EV/EBITDA (NTM)"       : [p["ev_ebitda_fwd"] for p in PEERS],
    "P/E (NTM)"             : [p["pe_fwd"] for p in PEERS],
}

stat_headers = ["Multiple", "Mean", "Median", "25th Pct", "75th Pct",
                "Trimmed Mean", "Std Dev"]
stat_rows = []
peer_stats = {}

for metric_name, values in METRICS.items():
    arr = np.array(values)
    mean_val    = np.mean(arr)
    median_val  = np.median(arr)
    p25         = np.percentile(arr, 25)
    p75         = np.percentile(arr, 75)
    trim_mean   = stats.trim_mean(arr, 0.1)  # 10% trimmed
    std_val     = np.std(arr, ddof=1)

    peer_stats[metric_name] = {
        "mean": mean_val, "median": median_val,
        "p25": p25, "p75": p75, "trim_mean": trim_mean, "std": std_val
    }

    stat_rows.append([
        metric_name,
        f"{mean_val:.2f}x",
        f"{median_val:.2f}x",
        f"{p25:.2f}x",
        f"{p75:.2f}x",
        f"{trim_mean:.2f}x",
        f"{std_val:.2f}x",
    ])

print_table("PEER MULTIPLE STATISTICS", stat_headers, stat_rows)


# =============================================================================
# 4. PRECEDENT TRANSACTION COMPS
# =============================================================================
print_section("PRECEDENT TRANSACTION COMPS")

# Fictional M&A transactions in the technology sector
TRANSACTIONS = [
    {"target": "OmegaSoft",   "acquirer": "MegaCorp",    "date": "2025-Q4",
     "ev": 8_500e6,  "rev": 3_000e6, "ebitda": 780e6,
     "premium": 0.28, "months_ago": 3},
    {"target": "SigmaTech",   "acquirer": "GlobalPE",    "date": "2025-Q3",
     "ev": 5_200e6,  "rev": 2_000e6, "ebitda": 520e6,
     "premium": 0.32, "months_ago": 6},
    {"target": "LambdaIO",    "acquirer": "StratBuyer",  "date": "2025-Q2",
     "ev": 12_000e6, "rev": 4_800e6, "ebitda": 1_344e6,
     "premium": 0.25, "months_ago": 9},
    {"target": "PsiAnalytics", "acquirer": "DataHolding", "date": "2025-Q1",
     "ev": 3_800e6,  "rev": 1_500e6, "ebitda": 375e6,
     "premium": 0.35, "months_ago": 12},
    {"target": "PhiNetworks",  "acquirer": "TeleGroup",   "date": "2024-Q4",
     "ev": 7_000e6,  "rev": 2_800e6, "ebitda": 700e6,
     "premium": 0.22, "months_ago": 15},
    {"target": "ChiPlatform",  "acquirer": "CloudPE",     "date": "2024-Q3",
     "ev": 9_500e6,  "rev": 3_500e6, "ebitda": 950e6,
     "premium": 0.30, "months_ago": 18},
    {"target": "UpsilonAI",    "acquirer": "VentureHold", "date": "2024-Q2",
     "ev": 6_200e6,  "rev": 2_200e6, "ebitda": 572e6,
     "premium": 0.27, "months_ago": 21},
]

# Compute transaction multiples
for t in TRANSACTIONS:
    t["ev_rev"]    = t["ev"] / t["rev"]
    t["ev_ebitda"] = t["ev"] / t["ebitda"]
    # Time-decay weight: more recent transactions get higher weight
    t["weight"] = np.exp(-0.03 * t["months_ago"])

# Normalize weights
total_w = sum(t["weight"] for t in TRANSACTIONS)
for t in TRANSACTIONS:
    t["weight_norm"] = t["weight"] / total_w

# Print transaction comps
txn_headers = ["Target", "Acquirer", "Date", "EV ($M)", "EV/Rev",
               "EV/EBITDA", "Premium", "Weight"]
txn_rows = []
for t in TRANSACTIONS:
    txn_rows.append([
        t["target"], t["acquirer"], t["date"],
        f"${t['ev']/1e6:,.0f}M",
        f"{t['ev_rev']:.2f}x",
        f"{t['ev_ebitda']:.1f}x",
        f"{t['premium']:.0%}",
        f"{t['weight_norm']:.1%}",
    ])
print_table("PRECEDENT TRANSACTIONS", txn_headers, txn_rows)

# Transaction statistics
txn_ev_ebitda = np.array([t["ev_ebitda"] for t in TRANSACTIONS])
txn_premiums  = np.array([t["premium"] for t in TRANSACTIONS])
txn_weights   = np.array([t["weight_norm"] for t in TRANSACTIONS])

txn_weighted_ebitda = np.average(txn_ev_ebitda, weights=txn_weights)
txn_weighted_prem   = np.average(txn_premiums, weights=txn_weights)

txn_stat_rows = [
    ["EV/EBITDA -- Simple Mean",    f"{np.mean(txn_ev_ebitda):.1f}x"],
    ["EV/EBITDA -- Simple Median",  f"{np.median(txn_ev_ebitda):.1f}x"],
    ["EV/EBITDA -- Weighted Mean",  f"{txn_weighted_ebitda:.1f}x"],
    ["Control Premium -- Mean",     f"{np.mean(txn_premiums):.1%}"],
    ["Control Premium -- Weighted", f"{txn_weighted_prem:.1%}"],
]
print_table("TRANSACTION STATISTICS", ["Metric", "Value"], txn_stat_rows)


# =============================================================================
# 5. IMPLIED VALUATION RANGE
# =============================================================================
print_section("IMPLIED VALUATION FOR TARGET")

# Method 1: Trading Comps (median multiples)
tc_ev_ebitda_med = peer_stats["EV/EBITDA (LTM)"]["median"]
tc_ev_ebitda_p25 = peer_stats["EV/EBITDA (LTM)"]["p25"]
tc_ev_ebitda_p75 = peer_stats["EV/EBITDA (LTM)"]["p75"]

tc_ev_rev_med = peer_stats["EV/Revenue (LTM)"]["median"]
tc_pe_med     = peer_stats["P/E (LTM)"]["median"]

# Forward multiples
tc_fwd_ebitda_med = peer_stats["EV/EBITDA (NTM)"]["median"]
tc_fwd_pe_med     = peer_stats["P/E (NTM)"]["median"]

# Implied EV from trading comps
implied_ev_tc_ebitda     = TARGET["ebitda"] * tc_ev_ebitda_med
implied_ev_tc_ebitda_low = TARGET["ebitda"] * tc_ev_ebitda_p25
implied_ev_tc_ebitda_hi  = TARGET["ebitda"] * tc_ev_ebitda_p75
implied_ev_tc_rev        = TARGET["revenue"] * tc_ev_rev_med

# Implied equity value per share
implied_eq_tc_ebitda     = (implied_ev_tc_ebitda - TARGET["net_debt"]) / TARGET["shares_out"]
implied_eq_tc_ebitda_low = (implied_ev_tc_ebitda_low - TARGET["net_debt"]) / TARGET["shares_out"]
implied_eq_tc_ebitda_hi  = (implied_ev_tc_ebitda_hi - TARGET["net_debt"]) / TARGET["shares_out"]
implied_eq_tc_rev        = (implied_ev_tc_rev - TARGET["net_debt"]) / TARGET["shares_out"]

# Forward-based
implied_ev_fwd_ebitda = TARGET["ebitda_fwd"] * tc_fwd_ebitda_med
implied_eq_fwd_ebitda = (implied_ev_fwd_ebitda - TARGET["net_debt"]) / TARGET["shares_out"]
implied_eq_fwd_pe     = TARGET["eps_fwd"] * tc_fwd_pe_med

# Method 2: Transaction comps (weighted mean)
implied_ev_txn = TARGET["ebitda"] * txn_weighted_ebitda
implied_eq_txn = (implied_ev_txn - TARGET["net_debt"]) / TARGET["shares_out"]

# Method 3: Transaction comps with premium
implied_eq_premium = TARGET["share_price"] * (1.0 + txn_weighted_prem)

val_rows = [
    ["--- Trading Comps ---", "", "", ""],
    ["EV/EBITDA (LTM Median)",
     f"{tc_ev_ebitda_med:.1f}x", fmt_currency(implied_ev_tc_ebitda),
     f"${implied_eq_tc_ebitda:.2f}"],
    ["EV/EBITDA (25th Pct)",
     f"{tc_ev_ebitda_p25:.1f}x", fmt_currency(implied_ev_tc_ebitda_low),
     f"${implied_eq_tc_ebitda_low:.2f}"],
    ["EV/EBITDA (75th Pct)",
     f"{tc_ev_ebitda_p75:.1f}x", fmt_currency(implied_ev_tc_ebitda_hi),
     f"${implied_eq_tc_ebitda_hi:.2f}"],
    ["EV/Revenue (LTM Median)",
     f"{tc_ev_rev_med:.2f}x", fmt_currency(implied_ev_tc_rev),
     f"${implied_eq_tc_rev:.2f}"],
    ["EV/EBITDA (NTM Median)",
     f"{tc_fwd_ebitda_med:.1f}x", fmt_currency(implied_ev_fwd_ebitda),
     f"${implied_eq_fwd_ebitda:.2f}"],
    ["P/E (NTM Median)",
     f"{tc_fwd_pe_med:.1f}x", "--",
     f"${implied_eq_fwd_pe:.2f}"],
    ["", "", "", ""],
    ["--- Transaction Comps ---", "", "", ""],
    ["EV/EBITDA (Wtd Mean)",
     f"{txn_weighted_ebitda:.1f}x", fmt_currency(implied_ev_txn),
     f"${implied_eq_txn:.2f}"],
    ["Implied w/ Control Premium",
     f"{txn_weighted_prem:.0%} prem", "--",
     f"${implied_eq_premium:.2f}"],
    ["", "", "", ""],
    ["Current Share Price", "", "", f"${TARGET['share_price']:.2f}"],
]
print_table("IMPLIED VALUATION",
            ["Method", "Multiple", "Implied EV", "Implied Price"],
            val_rows)


# =============================================================================
# 6. REGRESSION-BASED VALUATION
# =============================================================================
print_section("REGRESSION: EV/EBITDA vs EBITDA MARGIN")

margins = np.array([p["ebitda_margin"] * 100 for p in PEERS])
ev_ebitda_arr = np.array([p["ev_ebitda"] for p in PEERS])

slope, intercept, r_value, p_value, std_err = stats.linregress(margins, ev_ebitda_arr)
r_squared = r_value ** 2

# Implied multiple for target
target_margin_pct = TARGET["ebitda_margin"] * 100
implied_multiple_reg = slope * target_margin_pct + intercept
implied_ev_reg = TARGET["ebitda"] * implied_multiple_reg
implied_eq_reg = (implied_ev_reg - TARGET["net_debt"]) / TARGET["shares_out"]

reg_rows = [
    ["Regression Equation",  f"EV/EBITDA = {slope:.3f} * Margin% + {intercept:.2f}"],
    ["R-squared",            f"{r_squared:.3f}"],
    ["p-value (slope)",      f"{p_value:.4f}"],
    ["Std Error (slope)",    f"{std_err:.4f}"],
    ["Target EBITDA Margin", f"{target_margin_pct:.1f}%"],
    ["Implied EV/EBITDA",    f"{implied_multiple_reg:.1f}x"],
    ["Implied EV",           fmt_currency(implied_ev_reg)],
    ["Implied Price/Share",  f"${implied_eq_reg:.2f}"],
]
print_table("REGRESSION-BASED VALUATION", ["Metric", "Value"], reg_rows)


# =============================================================================
# 7. BOOTSTRAP CONFIDENCE INTERVAL
# =============================================================================
print_section("BOOTSTRAP CONFIDENCE INTERVAL (EV/EBITDA)")

N_BOOTSTRAP = 10_000
np.random.seed(42)
bootstrap_medians = np.zeros(N_BOOTSTRAP)
for b in range(N_BOOTSTRAP):
    sample = np.random.choice(ev_ebitda_arr, size=len(ev_ebitda_arr), replace=True)
    bootstrap_medians[b] = np.median(sample)

ci_low = np.percentile(bootstrap_medians, 2.5)
ci_high = np.percentile(bootstrap_medians, 97.5)
ci_med = np.median(bootstrap_medians)

implied_ci_low  = (TARGET["ebitda"] * ci_low - TARGET["net_debt"]) / TARGET["shares_out"]
implied_ci_high = (TARGET["ebitda"] * ci_high - TARGET["net_debt"]) / TARGET["shares_out"]

boot_rows = [
    ["Bootstrap Iterations",        f"{N_BOOTSTRAP:,}"],
    ["Median EV/EBITDA",            f"{ci_med:.2f}x"],
    ["95% CI Lower (2.5th pct)",    f"{ci_low:.2f}x"],
    ["95% CI Upper (97.5th pct)",   f"{ci_high:.2f}x"],
    ["Implied Price (Low)",         f"${implied_ci_low:.2f}"],
    ["Implied Price (High)",        f"${implied_ci_high:.2f}"],
]
print_table("BOOTSTRAP 95% CONFIDENCE INTERVAL", ["Metric", "Value"], boot_rows)


# =============================================================================
# 8. VISUALIZATIONS
# =============================================================================
print_section("GENERATING VISUALIZATIONS")

# -------------------------------------------------------------------------
# FIGURE 1: Peer Multiple Comparison (Dot Plot)
# -------------------------------------------------------------------------
fig1, ax1 = plt.subplots(figsize=(12, 6))

names = [p["name"] for p in PEERS]
y_pos = np.arange(len(names))

ev_ebitda_vals = [p["ev_ebitda"] for p in PEERS]

bars1 = ax1.barh(y_pos, ev_ebitda_vals, height=0.5, color=COLORS["primary"],
                 alpha=0.85, edgecolor="none")

# Target reference line
target_ev_ebitda = TARGET["ev"] / TARGET["ebitda"]
ax1.axvline(x=target_ev_ebitda, color=COLORS["danger"], linewidth=2,
            linestyle="--", alpha=0.8,
            label=f"Target: {target_ev_ebitda:.1f}x")

# Median reference
ax1.axvline(x=tc_ev_ebitda_med, color=COLORS["accent"], linewidth=1.5,
            linestyle=":", alpha=0.8,
            label=f"Peer Median: {tc_ev_ebitda_med:.1f}x")

for i, (bar, val) in enumerate(zip(bars1, ev_ebitda_vals)):
    ax1.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
             f"{val:.1f}x", va="center", fontsize=9, color=COLORS["white"])

ax1.set_yticks(y_pos)
ax1.set_yticklabels(names)
ax1.set_xlabel("EV / EBITDA (LTM)")
ax1.set_title(f"Trading Comps -- EV/EBITDA Comparison",
              fontsize=14, fontweight="bold", pad=15)
ax1.legend(loc="lower right")
ax1.invert_yaxis()

fig1.tight_layout()
save_figure(fig1, "comps_01_peer_ev_ebitda", subdir="comps")

# -------------------------------------------------------------------------
# FIGURE 2: Multiple Box Plots
# -------------------------------------------------------------------------
fig2, axes2 = plt.subplots(1, 3, figsize=(14, 5))

multiples_data = [
    ([p["ev_rev"] for p in PEERS], "EV/Revenue", COLORS["primary"]),
    ([p["ev_ebitda"] for p in PEERS], "EV/EBITDA", COLORS["secondary"]),
    ([p["pe"] for p in PEERS], "P/E", COLORS["accent"]),
]

target_multiples = [
    TARGET["ev"] / TARGET["revenue"],
    TARGET["ev"] / TARGET["ebitda"],
    TARGET["market_cap"] / TARGET["net_income"],
]

for ax, (data, label, color), t_mult in zip(axes2, multiples_data, target_multiples):
    bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                    boxprops=dict(facecolor=color, alpha=0.4, edgecolor=color),
                    whiskerprops=dict(color=color),
                    capprops=dict(color=color),
                    medianprops=dict(color=COLORS["white"], linewidth=2),
                    flierprops=dict(markeredgecolor=color))
    ax.axhline(y=t_mult, color=COLORS["danger"], linewidth=1.5,
               linestyle="--", label=f"Target: {t_mult:.1f}x")
    ax.set_title(label, fontsize=11, fontweight="bold")
    ax.set_ylabel("Multiple (x)")
    ax.set_xticklabels(["Peers"])
    ax.legend(fontsize=8)

fig2.suptitle("Peer Multiple Distributions vs Target",
              fontsize=14, fontweight="bold", y=1.02)
fig2.tight_layout()
save_figure(fig2, "comps_02_box_plots", subdir="comps")

# -------------------------------------------------------------------------
# FIGURE 3: Regression -- EV/EBITDA vs Margin
# -------------------------------------------------------------------------
fig3, ax3 = plt.subplots(figsize=(10, 7))

# Scatter peers
ax3.scatter(margins, ev_ebitda_arr, s=100, color=COLORS["primary"],
            alpha=0.8, zorder=5, edgecolors=COLORS["white"], linewidths=0.5)
for p, m, ev_e in zip(PEERS, margins, ev_ebitda_arr):
    ax3.annotate(p["name"], (m, ev_e), textcoords="offset points",
                 xytext=(8, 5), fontsize=7, color=COLORS["white"], alpha=0.8)

# Regression line
x_line = np.linspace(margins.min() - 1, margins.max() + 1, 100)
y_line = slope * x_line + intercept
ax3.plot(x_line, y_line, color=COLORS["accent"], linewidth=2, linestyle="--",
         label=f"Regression (R2={r_squared:.2f})")

# 95% confidence band
y_pred = slope * margins + intercept
residuals = ev_ebitda_arr - y_pred
se_resid = np.sqrt(np.sum(residuals**2) / (len(margins) - 2))
x_mean = np.mean(margins)
x_ss = np.sum((margins - x_mean)**2)
for x_val in x_line:
    se_line = se_resid * np.sqrt(1/len(margins) + (x_val - x_mean)**2 / x_ss)
y_band_lo = slope * x_line + intercept - 1.96 * se_resid
y_band_hi = slope * x_line + intercept + 1.96 * se_resid
ax3.fill_between(x_line, y_band_lo, y_band_hi, alpha=0.15, color=COLORS["accent"])

# Target point
ax3.scatter([target_margin_pct], [implied_multiple_reg], s=200,
            color=COLORS["danger"], marker="*", zorder=10,
            label=f"Target Implied: {implied_multiple_reg:.1f}x")

ax3.set_xlabel("EBITDA Margin (%)")
ax3.set_ylabel("EV / EBITDA (x)")
ax3.set_title("Regression: EV/EBITDA vs EBITDA Margin",
              fontsize=14, fontweight="bold", pad=15)
ax3.legend(loc="upper left")

fig3.tight_layout()
save_figure(fig3, "comps_03_regression", subdir="comps")

# -------------------------------------------------------------------------
# FIGURE 4: Transaction Comps Timeline
# -------------------------------------------------------------------------
fig4, ax4 = plt.subplots(figsize=(11, 6))

months = [t["months_ago"] for t in TRANSACTIONS]
txn_mults = [t["ev_ebitda"] for t in TRANSACTIONS]
txn_sizes = [t["ev"] / 1e6 for t in TRANSACTIONS]  # bubble size
txn_names = [t["target"] for t in TRANSACTIONS]

# Bubble size proportional to deal EV
size_scale = np.array(txn_sizes) / max(txn_sizes) * 400 + 100

scatter4 = ax4.scatter(months, txn_mults, s=size_scale,
                       c=[t["premium"] for t in TRANSACTIONS],
                       cmap="YlOrRd", alpha=0.8, edgecolors=COLORS["white"],
                       linewidths=0.5, zorder=5)

for t, m, ev_e in zip(TRANSACTIONS, months, txn_mults):
    ax4.annotate(t["target"], (m, ev_e), textcoords="offset points",
                 xytext=(10, 5), fontsize=7, color=COLORS["white"], alpha=0.9)

# Weighted mean reference
ax4.axhline(y=txn_weighted_ebitda, color=COLORS["accent"], linewidth=1.5,
            linestyle="--", label=f"Weighted Mean: {txn_weighted_ebitda:.1f}x")

ax4.set_xlabel("Months Ago")
ax4.set_ylabel("EV / EBITDA (x)")
ax4.set_title("Precedent Transactions -- EV/EBITDA by Recency",
              fontsize=14, fontweight="bold", pad=15)
ax4.legend(loc="upper right")
ax4.invert_xaxis()

cbar4 = fig4.colorbar(scatter4, ax=ax4, shrink=0.8)
cbar4.set_label("Control Premium (%)", color=COLORS["white"])
cbar4.ax.yaxis.set_tick_params(color=COLORS["white"])
plt.setp(plt.getp(cbar4.ax.axes, "yticklabels"), color=COLORS["white"])

fig4.tight_layout()
save_figure(fig4, "comps_04_transactions", subdir="comps")

# -------------------------------------------------------------------------
# FIGURE 5: Bootstrap Distribution
# -------------------------------------------------------------------------
fig5, ax5 = plt.subplots(figsize=(10, 6))

ax5.hist(bootstrap_medians, bins=60, color=COLORS["primary"], alpha=0.7,
         edgecolor="none", density=True)
ax5.axvline(x=ci_low, color=COLORS["danger"], linewidth=2, linestyle="--",
            label=f"2.5th Pct: {ci_low:.2f}x")
ax5.axvline(x=ci_high, color=COLORS["danger"], linewidth=2, linestyle="--",
            label=f"97.5th Pct: {ci_high:.2f}x")
ax5.axvline(x=ci_med, color=COLORS["accent"], linewidth=2,
            label=f"Median: {ci_med:.2f}x")

ax5.set_xlabel("Median EV/EBITDA (x)")
ax5.set_ylabel("Density")
ax5.set_title("Bootstrap Distribution of Median EV/EBITDA (10,000 iterations)",
              fontsize=13, fontweight="bold", pad=15)
ax5.legend(loc="upper right")

fig5.tight_layout()
save_figure(fig5, "comps_05_bootstrap", subdir="comps")

# -------------------------------------------------------------------------
# FIGURE 6: Football Field -- All Methodologies
# -------------------------------------------------------------------------
fig6, ax6 = plt.subplots(figsize=(13, 6))

ff_methods = [
    "Trading Comps\n(EV/EBITDA LTM)",
    "Trading Comps\n(EV/Revenue LTM)",
    "Trading Comps\n(NTM EV/EBITDA)",
    "Trading Comps\n(NTM P/E)",
    "Regression\n(Margin-Based)",
    "Transaction Comps\n(Wtd EV/EBITDA)",
    "Transaction Comps\n(Control Premium)",
    "Bootstrap\n(95% CI)",
]

ff_ranges = [
    [implied_eq_tc_ebitda_low, implied_eq_tc_ebitda, implied_eq_tc_ebitda_hi],
    [implied_eq_tc_rev * 0.90, implied_eq_tc_rev, implied_eq_tc_rev * 1.10],
    [implied_eq_fwd_ebitda * 0.90, implied_eq_fwd_ebitda, implied_eq_fwd_ebitda * 1.10],
    [implied_eq_fwd_pe * 0.90, implied_eq_fwd_pe, implied_eq_fwd_pe * 1.10],
    [implied_eq_reg * 0.90, implied_eq_reg, implied_eq_reg * 1.10],
    [implied_eq_txn * 0.85, implied_eq_txn, implied_eq_txn * 1.15],
    [implied_eq_premium * 0.90, implied_eq_premium, implied_eq_premium * 1.10],
    [implied_ci_low, (implied_ci_low + implied_ci_high)/2, implied_ci_high],
]

ff_colors = [COLORS["primary"], COLORS["secondary"], COLORS["accent"],
             COLORS["purple"], COLORS["teal"], COLORS["pink"],
             COLORS["danger"], "#aed581"]

y6 = np.arange(len(ff_methods))

for i, (method, rng, clr) in enumerate(zip(ff_methods, ff_ranges, ff_colors)):
    low, mid, high = rng
    ax6.barh(i, high - low, left=low, height=0.5, color=clr, alpha=0.4,
             edgecolor=clr)
    ax6.plot(mid, i, "D", color=clr, markersize=9, zorder=5)
    ax6.text(low - 0.5, i, f"${low:.0f}", ha="right", va="center",
             fontsize=7, color=COLORS["white"])
    ax6.text(high + 0.5, i, f"${high:.0f}", ha="left", va="center",
             fontsize=7, color=COLORS["white"])

ax6.axvline(x=TARGET["share_price"], color=COLORS["danger"], linewidth=2,
            linestyle="--", alpha=0.8,
            label=f"Current: ${TARGET['share_price']:.2f}")

ax6.set_yticks(y6)
ax6.set_yticklabels(ff_methods, fontsize=8)
ax6.set_xlabel("Implied Share Price ($)")
ax6.set_title(f"{TARGET['name']} -- Football Field (Comparable Analysis)",
              fontsize=14, fontweight="bold", pad=15)
ax6.legend(loc="upper right")
ax6.invert_yaxis()

fig6.tight_layout()
save_figure(fig6, "comps_06_football_field", subdir="comps")


# =============================================================================
# 9. SUMMARY
# =============================================================================
print_section("COMPARABLE ANALYSIS SUMMARY")
print(f"  Target                  : {TARGET['name']}")
print(f"  Current Price           : ${TARGET['share_price']:.2f}")
print(f"  Trading Comps Range     : ${implied_eq_tc_ebitda_low:.2f} - ${implied_eq_tc_ebitda_hi:.2f}")
print(f"  Transaction Comps       : ${implied_eq_txn:.2f}")
print(f"  Regression Implied      : ${implied_eq_reg:.2f}")
print(f"  Bootstrap 95% CI        : ${implied_ci_low:.2f} - ${implied_ci_high:.2f}")
print(f"  Figures saved to        : outputs/figures/comps/")
print(f"  Module 4 of 6 complete.")
print("=" * 70)
