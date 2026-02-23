"""
Stochastic Processes Simulator - Main Analysis
Author: Jose Orlando Bobadilla Fuentes | CQF
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from models.processes import (
    SimConfig, WienerProcess, GeometricBrownianMotion,
    OrnsteinUhlenbeck, CIRProcess, HestonModel, MertonJumpDiffusion)
from visualization.process_plots import plot_process_panel


def header(t):
    print(f"\n{'='*70}\n  {t}\n{'='*70}")

def main():
    header("STOCHASTIC PROCESSES SIMULATOR")
    cfg = SimConfig(T=2.0, n_steps=500, n_paths=1000, seed=42)

    processes = [
        ("Wiener Process", WienerProcess()),
        ("GBM (S0=100, mu=8%, sigma=20%)", GeometricBrownianMotion(100, 0.08, 0.20)),
        ("OU (kappa=3, theta=0.05)", OrnsteinUhlenbeck(0.08, 3.0, 0.05, 0.02)),
        ("CIR (kappa=2, theta=0.05)", CIRProcess(0.03, 2.0, 0.05, 0.1)),
    ]

    panel_data = []
    for name, proc in processes:
        header(name)
        result = proc.simulate(cfg)
        t, paths = result[0], result[1]

        print(f"  Paths: {paths.shape[0]}, Steps: {paths.shape[1]-1}")
        print(f"  E[X(T)] sample:      {np.mean(paths[:, -1]):.6f}")
        print(f"  E[X(T)] theoretical: {proc.theoretical_mean(np.array([cfg.T]))[0]:.6f}")
        tv = proc.theoretical_var(np.array([cfg.T]))[0]
        if not np.isnan(tv):
            print(f"  Var[X(T)] sample:    {np.var(paths[:, -1]):.6f}")
            print(f"  Var[X(T)] theoretical:{tv:.6f}")

        panel_data.append({
            "name": name, "t": t, "paths": paths,
            "mean_fn": proc.theoretical_mean, "var_fn": proc.theoretical_var})

    # Heston
    header("Heston Stochastic Volatility")
    heston = HestonModel(S0=100, v0=0.04, mu=0.05, kappa=2.0,
                         theta=0.04, xi=0.3, rho=-0.7)
    t, S, v = heston.simulate(cfg)
    print(f"  E[S(T)]: {np.mean(S[:, -1]):.2f}")
    print(f"  E[v(T)]: {np.mean(v[:, -1]):.4f}")

    # Merton
    header("Merton Jump-Diffusion")
    merton = MertonJumpDiffusion(S0=100, mu=0.08, sigma=0.15,
                                  lam=1.0, mu_J=-0.05, sigma_J=0.10)
    t_m, S_m = merton.simulate(cfg)
    print(f"  E[S(T)]: {np.mean(S_m[:, -1]):.2f}")
    print(f"  Std[S(T)]: {np.std(S_m[:, -1]):.2f}")

    panel_data.append({"name": "Heston (price)", "t": t, "paths": S,
                        "mean_fn": heston.theoretical_mean, "var_fn": None})
    panel_data.append({"name": "Merton Jump-Diffusion", "t": t_m, "paths": S_m,
                        "mean_fn": merton.theoretical_mean, "var_fn": None})

    header("GENERATING VISUALIZATIONS")
    plot_process_panel(panel_data)

    header("ANALYSIS COMPLETE")

if __name__ == "__main__":
    main()
