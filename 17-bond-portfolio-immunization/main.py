"""
main.py — Bond Portfolio Immunization: Full Pipeline
======================================================
Execution flow:
  1. Construct bond universe (7 bonds, 2Y–30Y)
  2. Define pension liability stream
  3. Redington immunization (duration + convexity matching)
  4. Cash flow matching LP (dedication)
  5. Build Bullet, Barbell, Ladder alternatives
  6. Parallel & non-parallel yield shock scenarios
  7. Key Rate Duration analysis (full portfolio)
  8. Generate all 8 publication-quality dark-theme charts

Author  : Jose Orlando Bobadilla Fuentes | CQF | MSc AI
Project : 16 — Bond Portfolio Immunization
"""

import numpy as np
import json
from pathlib import Path

from src.bond        import Bond
from src.immunization import ImmunizationEngine, Liability
from src.key_rate_duration import KeyRateDuration
from src.portfolio    import PortfolioScenarioEngine
from src.visualization import (
    yield_price_curve,
    duration_convexity_map,
    immunization_bar,
    parallel_shift_pnl,
    key_rate_duration_bar,
    cash_flow_matching_chart,
    portfolio_structures_chart,
    scenario_heatmap,
)

OUTDIR = Path("outputs/figures")
OUTDIR.mkdir(parents=True, exist_ok=True)

REPORT_DIR = Path("outputs/reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# 1. Bond Universe
# ==============================================================================
print("\n" + "="*70)
print("  PROJECT 16 — BOND PORTFOLIO IMMUNIZATION")
print("  Author: Jose Orlando Bobadilla Fuentes | CQF | MSc AI")
print("="*70)

print("\n[1/8] Building bond universe...")

# Sovereign-style bond universe (analogous to US Treasuries)
bonds = [
    Bond(face_value=1000, coupon_rate=0.030, maturity=2.0,  frequency=2, issuer="T-2Y",  isin="US-2Y"),
    Bond(face_value=1000, coupon_rate=0.035, maturity=3.0,  frequency=2, issuer="T-3Y",  isin="US-3Y"),
    Bond(face_value=1000, coupon_rate=0.040, maturity=5.0,  frequency=2, issuer="T-5Y",  isin="US-5Y"),
    Bond(face_value=1000, coupon_rate=0.045, maturity=7.0,  frequency=2, issuer="T-7Y",  isin="US-7Y"),
    Bond(face_value=1000, coupon_rate=0.050, maturity=10.0, frequency=2, issuer="T-10Y", isin="US-10Y"),
    Bond(face_value=1000, coupon_rate=0.055, maturity=20.0, frequency=2, issuer="T-20Y", isin="US-20Y"),
    Bond(face_value=1000, coupon_rate=0.058, maturity=30.0, frequency=2, issuer="T-30Y", isin="US-30Y"),
]

# Market yields (flat-ish curve with modest upward slope)
yields = [0.031, 0.038, 0.042, 0.046, 0.050, 0.054, 0.057]

print(f"  Universe: {len(bonds)} bonds | Maturities: 2Y → 30Y")

# Print analytics table
print(f"\n  {'Issuer':<10} {'Mat':>5} {'Cpn%':>6} {'YTM%':>6} {'Price':>9} "
      f"{'Dmod':>7} {'Conv':>8} {'DV01':>8}")
print("  " + "-"*65)
for b, y in zip(bonds, yields):
    s = b.summary(y)
    print(f"  {s['issuer']:<10} {s['maturity_yrs']:>5.1f} {s['coupon_rate_pct']:>6.2f} "
          f"{s['ytm_pct']:>6.2f} {s['price']:>9.2f} "
          f"{s['modified_dur']:>7.4f} {s['convexity']:>8.4f} {s['dv01']:>8.4f}")


# ==============================================================================
# 2. Liability Stream (Pension-style)
# ==============================================================================
print("\n[2/8] Defining liability stream (pension obligations)...")

liabilities = [
    Liability(time=2.0,  amount=120_000),
    Liability(time=3.0,  amount=150_000),
    Liability(time=5.0,  amount=200_000),
    Liability(time=7.0,  amount=180_000),
    Liability(time=10.0, amount=250_000),
]

total_liab_nominal = sum(L.amount for L in liabilities)
print(f"  Liabilities: {len(liabilities)} payments | Total nominal: ${total_liab_nominal:,.0f}")

engine = ImmunizationEngine(bonds, liabilities, yields)


# ==============================================================================
# 3. Redington Immunization
# ==============================================================================
print("\n[3/8] Redington immunization (duration + convexity matching)...")

immu = engine.redington_immunization(flat_ytm=0.046)   # mid-curve rate

print(f"  Portfolio PV      : ${immu['portfolio_pv']:>12,.2f}")
print(f"  Liability PV      : ${immu['liability_pv']:>12,.2f}")
print(f"  PV Match          : {'YES' if immu['pv_match'] else 'NO'}")
print(f"  Portfolio D_mod   : {immu['portfolio_duration']:>8.4f} yrs")
print(f"  Liability D_mod   : {immu['liability_duration']:>8.4f} yrs")
print(f"  Duration Match    : {'YES' if immu['duration_match'] else 'NO'}")
print(f"  Convexity Surplus : {immu['convexity_surplus']:>8.4f}")
print(f"  IMMUNIZED         : {'✓ YES' if immu['immunized'] else '✗ NO'}")


# ==============================================================================
# 4. Cash Flow Matching
# ==============================================================================
print("\n[4/8] Cash flow matching (dedication LP)...")

cfm = engine.cash_flow_matching()

print(f"  LP Status         : {'SUCCESS' if cfm['optimizer_success'] else 'FAILED'}")
print(f"  Total Cost        : ${cfm['total_cost']:>12,.2f}")
print(f"  Fully Matched     : {'YES' if cfm['fully_matched'] else 'NO'}")
print(f"  Coverage Ratios   : {['%.3f' % r for r in cfm['coverage_ratios']]}")


# ==============================================================================
# 5. Portfolio Structures
# ==============================================================================
print("\n[5/8] Constructing Bullet, Barbell, Ladder portfolios...")

budget = immu["liability_pv"]

w_bullet  = ImmunizationEngine.build_bullet(bonds, yields, target_maturity=5.0, budget=budget)
w_barbell = ImmunizationEngine.build_barbell(bonds, yields, budget=budget)
w_ladder  = ImmunizationEngine.build_ladder(bonds, yields, budget=budget)
w_matched = immu["weights"]

def _port_stats(w):
    pvs   = np.array([b.price(y) for b, y in zip(bonds, yields)])
    pv    = float(np.dot(w, pvs))
    d_mod = float(np.dot(w, pvs * np.array([b.modified_duration(y) for b, y in zip(bonds, yields)])) / pv)
    conv  = float(np.dot(w, pvs * np.array([b.convexity(y) for b, y in zip(bonds, yields)])) / pv)
    dv01  = float(np.dot(w, np.array([b.dv01(y) for b, y in zip(bonds, yields)])))
    return {"pv": pv, "duration": d_mod, "convexity": conv, "dv01": dv01}

structures = {
    "Bullet"   : _port_stats(w_bullet),
    "Barbell"  : _port_stats(w_barbell),
    "Ladder"   : _port_stats(w_ladder),
    "Immunized": _port_stats(w_matched),
}

for name, stats in structures.items():
    print(f"  {name:<12}: PV=${stats['pv']:>10,.0f} | D={stats['duration']:.3f} | C={stats['convexity']:.3f} | DV01=${stats['dv01']:.2f}")


# ==============================================================================
# 6. Scenario Analysis
# ==============================================================================
print("\n[6/8] Running parallel & non-parallel yield shock scenarios...")

scenario_engine = PortfolioScenarioEngine(bonds, yields, w_matched, liabilities)

shocks_bps = [-300, -200, -100, -50, 50, 100, 200, 300]
parallel   = scenario_engine.parallel_shift_analysis(shocks_bps)

print(f"\n  {'Shock':>7} {'Port PV':>12} {'Liab PV':>12} {'Surplus':>12} {'Surplus Δ':>12}")
print("  " + "-"*60)
for r in parallel:
    print(f"  {r['shock_bps']:>+7} bps  ${r['portfolio_pv']:>10,.0f}  ${r['liability_pv']:>10,.0f}  "
          f"${r['surplus']:>10,.0f}  ${r['surplus_change']:>+10,.0f}")

curve_scenarios = [
    {"name": "Base",       "short_shock":  0, "mid_shock":  0, "long_shock":  0},
    {"name": "Steepener",  "short_shock": -50,"mid_shock":  0, "long_shock": 100},
    {"name": "Flattener",  "short_shock": 100,"mid_shock": 25, "long_shock": -50},
    {"name": "Butterfly",  "short_shock": -50,"mid_shock": 75, "long_shock": -50},
    {"name": "+100bps",    "short_shock": 100,"mid_shock":100, "long_shock": 100},
    {"name": "-100bps",    "short_shock":-100,"mid_shock":-100,"long_shock":-100},
]

curve_results = scenario_engine.curve_scenario_analysis(curve_scenarios)


# ==============================================================================
# 7. Key Rate Duration
# ==============================================================================
print("\n[7/8] Key Rate Duration analysis...")

krd_engine  = KeyRateDuration(
    key_tenors=[0.0833, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
    shock_bps=1.0
)
key_labels  = ["1M","3M","6M","1Y","2Y","5Y","10Y","30Y"]

krd_portfolio = krd_engine.compute_portfolio_krd(bonds, yields, w_matched)

print(f"  ΣD (KRD) = {krd_portfolio['sum_krd']:.4f} yrs")
for label, k, kv in zip(key_labels, krd_portfolio["krd_vector"], krd_portfolio["krdv01"]):
    print(f"    {label:<5}: KRD={k:>8.4f} yrs   KR-DV01=${kv:>8.4f}")


# ==============================================================================
# 8. Visualizations
# ==============================================================================
print("\n[8/8] Generating charts (dark theme)...")

ytm_range = np.linspace(0.005, 0.12, 200)

p1 = yield_price_curve(bonds, ytm_range, yields, OUTDIR)
print(f"  [OK] {p1}")

p2 = duration_convexity_map(bonds, yields, OUTDIR)
print(f"  [OK] {p2}")

p3 = immunization_bar(immu, OUTDIR)
print(f"  [OK] {p3}")

p4 = parallel_shift_pnl(parallel, OUTDIR)
print(f"  [OK] {p4}")

p5 = key_rate_duration_bar(krd_portfolio, key_labels, OUTDIR)
print(f"  [OK] {p5}")

p6 = cash_flow_matching_chart(bonds, w_matched, liabilities, OUTDIR)
print(f"  [OK] {p6}")

p7 = portfolio_structures_chart(structures, OUTDIR)
print(f"  [OK] {p7}")

# Heatmap: rows = portfolio structures, cols = curve scenarios
struct_names   = list(structures.keys())
scenario_names = [sc["name"] for sc in curve_scenarios]
heatmap_arr    = np.zeros((len(struct_names), len(scenario_names)))

for i, (sname, stats) in enumerate(structures.items()):
    # recompute scenario surplus for each structure
    _eng = PortfolioScenarioEngine(
        bonds, yields,
        [w_bullet, w_barbell, w_ladder, w_matched][i],
        liabilities
    )
    _res = _eng.curve_scenario_analysis(curve_scenarios)
    for j, r in enumerate(_res):
        heatmap_arr[i, j] = r["surplus_change"]

p8 = scenario_heatmap(heatmap_arr, struct_names, scenario_names,
                       "Surplus Change (USD) by Structure & Scenario", OUTDIR)
print(f"  [OK] {p8}")


# ==============================================================================
# Summary report
# ==============================================================================
report = {
    "immunization" : {k: v for k, v in immu.items() if k != "weights"},
    "cfm"          : {"total_cost": cfm["total_cost"], "fully_matched": cfm["fully_matched"]},
    "structures"   : structures,
    "krd_sum"      : krd_portfolio["sum_krd"],
}

report_path = REPORT_DIR / "summary.json"
with open(report_path, "w") as f:
    json.dump(report, f, indent=2, default=str)

print(f"\n  [OK] Report saved: {report_path}")

print("\n" + "="*70)
print("  PROJECT 16 COMPLETE")
print(f"  Charts  : {OUTDIR}")
print(f"  Reports : {REPORT_DIR}")
print("  Bond Portfolio Immunization — Jose O. Bobadilla | CQF")
print("="*70 + "\n")
