"""
Monte Carlo Exotic Derivatives Engine - Main Analysis
=======================================================

Demonstrates: Asian (3 methods), Barrier (4 types), Lookback (3 types),
convergence analysis, variance reduction comparison.

Author: Jose Orlando Bobadilla Fuentes | CQF
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from models.path_generator import SimulationConfig, GBMPathGenerator
from models.asian_options import AsianOptionPricer
from models.barrier_options import BarrierOptionPricer, BarrierType
from models.lookback_options import LookbackOptionPricer
from visualization.convergence_plots import plot_convergence_analysis, plot_sample_paths


def header(t):
    print(f"\n{'='*70}\n  {t}\n{'='*70}")

def pr(r, indent="    "):
    print(f"{indent}Price:   {r.price:.6f}")
    print(f"{indent}SE:      {r.std_error:.6f}")
    print(f"{indent}95% CI:  [{r.ci_lower:.6f}, {r.ci_upper:.6f}]")
    if hasattr(r, 'method'):
        print(f"{indent}Method:  {r.method}")
    if hasattr(r, 'knock_frequency'):
        print(f"{indent}Knock%:  {r.knock_frequency:.2%}")


def main():
    header("MONTE CARLO EXOTIC DERIVATIVES ENGINE")

    cfg = SimulationConfig(S0=100, r=0.05, sigma=0.20, T=1.0,
                           n_steps=252, n_paths=200000, seed=42)
    K = 100
    print(f"\n  S0={cfg.S0}, K={K}, T={cfg.T}y, r={cfg.r:.1%}, "
          f"sigma={cfg.sigma:.1%}, paths={cfg.n_paths:,}")

    # --- 1. ASIAN ---
    header("1. ASIAN OPTIONS")
    asian = AsianOptionPricer(cfg, K=K)

    print("\n  [a] Standard MC:")
    r1 = asian.price_arithmetic_call()
    pr(r1)
    print("\n  [b] Control Variate:")
    r2 = asian.price_with_control_variate()
    pr(r2)
    print("\n  [c] Antithetic:")
    r3 = asian.price_with_antithetic()
    pr(r3)

    if r1.std_error > 0:
        print(f"\n  Variance Reduction:")
        print(f"    Standard SE:   {r1.std_error:.6f}")
        print(f"    CV SE:         {r2.std_error:.6f}  "
              f"({(1-r2.std_error/r1.std_error)*100:.1f}% reduction)")
        print(f"    Antithetic SE: {r3.std_error:.6f}  "
              f"({(1-r3.std_error/r1.std_error)*100:.1f}% reduction)")

    # --- 2. BARRIER ---
    header("2. BARRIER OPTIONS")
    for bt, H in [(BarrierType.DOWN_AND_OUT, 90), (BarrierType.DOWN_AND_IN, 90),
                  (BarrierType.UP_AND_OUT, 120), (BarrierType.UP_AND_IN, 120)]:
        p = BarrierOptionPricer(cfg, K=K, H=H, barrier_type=bt)
        print(f"\n  {bt.value} Call (H={H}):")
        pr(p.price_call())

    # In-Out parity
    print("\n  In-Out Parity (Down, H=90):")
    do = BarrierOptionPricer(cfg, K=K, H=90, barrier_type=BarrierType.DOWN_AND_OUT)
    di = BarrierOptionPricer(cfg, K=K, H=90, barrier_type=BarrierType.DOWN_AND_IN)
    do_r, di_r = do.price_call(), di.price_call()
    gen = GBMPathGenerator(cfg)
    vanilla = np.mean(np.exp(-cfg.r*cfg.T) * np.maximum(gen.terminal_values()-K, 0))
    print(f"    DO + DI  = {do_r.price + di_r.price:.4f}")
    print(f"    Vanilla  = {vanilla:.4f}")
    print(f"    Error    = {abs(do_r.price + di_r.price - vanilla):.4f}")

    # --- 3. LOOKBACK ---
    header("3. LOOKBACK OPTIONS")
    lb = LookbackOptionPricer(cfg, K=K)
    for name, method in [("Floating Call", lb.floating_strike_call),
                         ("Floating Put", lb.floating_strike_put),
                         ("Fixed Call", lb.fixed_strike_call)]:
        print(f"\n  {name}:")
        pr(method())

    # --- 4. VISUALIZATIONS ---
    header("4. VISUALIZATIONS")
    out = "outputs/figures"

    small = SimulationConfig(S0=100, r=0.05, sigma=0.20, T=1.0,
                             n_steps=252, n_paths=200, seed=42)
    plot_sample_paths(GBMPathGenerator(small).generate(), n_show=100,
                      barrier=90, output_dir=out)

    print("\n  Running convergence analysis...")
    ns = [1000, 5000, 10000, 25000, 50000, 100000, 200000]
    ps, ss, pc, sc = [], [], [], []
    for n in ns:
        c = SimulationConfig(S0=100, r=0.05, sigma=0.20, T=1.0, n_paths=n, seed=42)
        ap = AsianOptionPricer(c, K=100)
        r_s = ap.price_arithmetic_call()
        r_c = ap.price_with_control_variate()
        ps.append(r_s.price); ss.append(r_s.std_error)
        pc.append(r_c.price); sc.append(r_c.std_error)

    plot_convergence_analysis(
        np.array(ns), np.array(ps), np.array(ss),
        np.array(pc), np.array(sc), output_dir=out,
        title="Asian Call: MC Convergence with Control Variate")

    header("ANALYSIS COMPLETE")
    print(f"\n  Outputs: {out}/")


if __name__ == "__main__":
    main()
