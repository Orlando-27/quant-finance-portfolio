"""
Black-Scholes Options Pricing Engine - Main Analysis
======================================================

Demonstrates:
    1. European option pricing with put-call parity verification
    2. Full Greeks computation and sensitivity analysis
    3. American option pricing via binomial tree
    4. Implied volatility extraction
    5. Publication-quality visualization generation

Author: Jose Orlando Bobadilla Fuentes | CQF
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from models.black_scholes import BlackScholesEngine, OptionParameters, OptionType
from models.implied_volatility import ImpliedVolatilitySolver
from models.binomial_tree import BinomialTreePricer
from visualization.surface_plots import (
    plot_greeks_sensitivity, plot_option_payoff_pnl, plot_volatility_surface_3d)


def header(text):
    print(f"\n{'='*70}\n  {text}\n{'='*70}")


def main():
    header("BLACK-SCHOLES OPTIONS PRICING ENGINE")
    print("  Author: Jose Orlando Bobadilla Fuentes | CQF")

    # --- 1. European Pricing ---
    header("1. EUROPEAN OPTION PRICING")
    params = OptionParameters(S=100, K=100, T=1.0, r=0.05, sigma=0.20, q=0.02)
    engine = BlackScholesEngine()

    call_price = engine.price(params, OptionType.CALL)
    put_price = engine.price(params, OptionType.PUT)

    print(f"\n  S={params.S}, K={params.K}, T={params.T}y, "
          f"r={params.r:.2%}, sigma={params.sigma:.2%}, q={params.q:.2%}")
    print(f"\n  European Call = {call_price:.6f}")
    print(f"  European Put  = {put_price:.6f}")

    parity = engine.put_call_parity_check(params)
    print(f"\n  Put-Call Parity:")
    print(f"    C - P (actual)     = {parity['actual_C_minus_P']:.10f}")
    print(f"    Theoretical        = {parity['theoretical_C_minus_P']:.10f}")
    print(f"    Error              = {parity['parity_error']:.2e}")
    print(f"    Holds              = {parity['parity_holds']}")

    # --- 2. Greeks ---
    header("2. GREEKS ANALYSIS")
    for ot in [OptionType.CALL, OptionType.PUT]:
        g = engine.compute_all_greeks(params, ot)
        print(f"\n  {ot.value.upper()} Greeks:")
        for name, val in g.items():
            print(f"    {name:8s} = {val:+.6f}")

    # --- 3. American Options ---
    header("3. AMERICAN OPTIONS - CRR BINOMIAL TREE")
    pricer = BinomialTreePricer(n_steps=500)
    am = pricer.price(S=100, K=100, T=1.0, r=0.05, sigma=0.20, q=0.02,
                      option_type=OptionType.PUT, american=True)
    eu_tree = pricer.price(S=100, K=100, T=1.0, r=0.05, sigma=0.20, q=0.02,
                           option_type=OptionType.PUT, american=False)
    am_rich = pricer.price_with_richardson(
        S=100, K=100, T=1.0, r=0.05, sigma=0.20, q=0.02)

    print(f"\n  European Put (BS)       = {put_price:.6f}")
    print(f"  European Put (tree)     = {eu_tree.price:.6f}")
    print(f"  American Put (tree)     = {am.price:.6f}")
    print(f"  American Put (Rich.)    = {am_rich:.6f}")
    print(f"  Early exercise premium  = {am.price - put_price:.6f}")

    # --- 4. Implied Volatility ---
    header("4. IMPLIED VOLATILITY SOLVER")
    solver = ImpliedVolatilitySolver()
    iv = solver.solve(market_price=call_price, S=100, K=100, T=1.0,
                      r=0.05, q=0.02, option_type=OptionType.CALL)
    print(f"\n  Input price  = {call_price:.6f} (sigma=20%)")
    print(f"  Recovered IV = {iv:.6%}" if iv else "  Solver failed")
    print(f"  Error        = {abs(iv - 0.20):.2e}" if iv else "  N/A")

    # --- 5. Visualizations ---
    header("5. GENERATING VISUALIZATIONS")
    out = "outputs/figures"

    print("  [1/3] Greeks sensitivity...")
    plot_greeks_sensitivity(S_range=(50, 150), K=100, T=1.0, r=0.05,
                            sigma=0.20, output_dir=out)

    print("  [2/3] Payoff/P&L diagrams...")
    plot_option_payoff_pnl(S_range=(50, 150), K=100,
                           premium_call=call_price, premium_put=put_price,
                           output_dir=out)

    print("  [3/3] Synthetic IV surface...")
    strikes = np.linspace(80, 120, 20)
    exps = np.linspace(0.1, 2.0, 15)
    K_g, T_g = np.meshgrid(strikes, exps)
    iv_surf = 0.20 + 0.15 * ((K_g / 100) - 1) ** 2 - 0.02 * np.sqrt(T_g)
    iv_surf = np.maximum(iv_surf, 0.05)
    plot_volatility_surface_3d(iv_surf, strikes, exps, output_dir=out)

    header("ANALYSIS COMPLETE")
    print(f"\n  Outputs: {out}/")
    print("  Run: streamlit run app.py  (for interactive dashboard)")


if __name__ == "__main__":
    main()
