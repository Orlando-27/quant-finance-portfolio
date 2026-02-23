"""
Portfolio Optimization Analysis - Main Entry Point
=====================================================

Pipeline: Market data -> Markowitz -> Black-Litterman -> Mean-CVaR ->
          Risk Parity -> Backtest comparison -> Visualizations

Author: Jose Orlando Bobadilla Fuentes | CQF
"""

import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from models.markowitz import MarkowitzOptimizer
from models.black_litterman import BlackLittermanModel
from models.mean_cvar import MeanCVaROptimizer
from models.risk_parity import RiskParityOptimizer
from utils.performance_metrics import compute_all_metrics
from visualization.efficient_frontier import (
    plot_frontier_comparison, plot_weight_comparison)


def header(t):
    print(f"\n{'='*70}\n  {t}\n{'='*70}")


def main():
    header("PORTFOLIO OPTIMIZATION ANALYSIS")
    print("  Methods: Markowitz | Black-Litterman | Mean-CVaR | Risk Parity")

    # --- 1. Market Data (synthetic for demonstration) ---
    header("1. MARKET DATA")
    np.random.seed(42)
    names = ["US Equity", "Int'l Equity", "EM Equity", "US Bonds", "Gold", "REITs"]
    n = len(names)
    mu = np.array([0.10, 0.09, 0.12, 0.035, 0.05, 0.08])

    corr = np.array([
        [1.00, 0.75, 0.65,-0.10, 0.05, 0.60],
        [0.75, 1.00, 0.70,-0.05, 0.10, 0.50],
        [0.65, 0.70, 1.00, 0.00, 0.15, 0.45],
        [-0.10,-0.05,0.00, 1.00, 0.25, 0.10],
        [0.05, 0.10, 0.15, 0.25, 1.00, 0.05],
        [0.60, 0.50, 0.45, 0.10, 0.05, 1.00]])
    vols = np.array([0.18, 0.20, 0.25, 0.05, 0.15, 0.22])
    Sigma = np.outer(vols, vols) * corr
    w_mkt = np.array([0.40, 0.20, 0.10, 0.20, 0.05, 0.05])

    L = np.linalg.cholesky(Sigma)
    daily_ret = mu/252 + (L @ np.random.randn(n, 500)).T

    for i, nm in enumerate(names):
        print(f"  {nm:15s}: mu={mu[i]:.1%}, vol={vols[i]:.1%}")

    # --- 2. Markowitz ---
    header("2. MARKOWITZ MEAN-VARIANCE")
    mvo = MarkowitzOptimizer(mu, Sigma, names, weight_bounds=(0, 0.40))
    mv = mvo.min_variance()
    tang = mvo.max_sharpe(risk_free_rate=0.04)
    frontier_mvo = mvo.efficient_frontier(80, 0.04)

    print(f"\n  Min Var:  ret={mv.expected_return:.2%}, vol={mv.volatility:.2%}")
    print(f"  Tangency: ret={tang.expected_return:.2%}, vol={tang.volatility:.2%}, "
          f"SR={tang.sharpe_ratio:.3f}")

    # --- 3. Black-Litterman ---
    header("3. BLACK-LITTERMAN")
    bl = BlackLittermanModel(Sigma, w_mkt, 2.5, 0.025, names)
    bl.add_absolute_view(2, 0.15, 0.7)   # EM bullish
    bl.add_relative_view(0, 1, 0.03, 0.6) # US > Int'l
    bl.add_absolute_view(4, 0.08, 0.5)    # Gold bullish
    mu_bl, Sigma_bl = bl.posterior()

    print("\n  Posterior Returns:")
    for nm, pr, po in zip(names, bl.pi, mu_bl):
        print(f"    {nm:15s}: {pr:.2%} -> {po:.2%}  ({po-pr:+.2%})")

    mvo_bl = MarkowitzOptimizer(mu_bl, Sigma_bl, names, weight_bounds=(0, 0.40))
    tang_bl = mvo_bl.max_sharpe(0.04)
    frontier_bl = mvo_bl.efficient_frontier(80, 0.04)
    print(f"\n  BL Tangency: ret={tang_bl.expected_return:.2%}, "
          f"SR={tang_bl.sharpe_ratio:.3f}")

    # --- 4. Mean-CVaR ---
    header("4. MEAN-CVaR OPTIMIZATION (95%)")
    cvar_opt = MeanCVaROptimizer(daily_ret, 0.95, names)
    cvar_res = cvar_opt.optimize(target_return=float(np.mean(mu))/252)
    print(f"\n  CVaR Portfolio:")
    print(f"    Return: {cvar_res.expected_return*252:.2%}")
    print(f"    CVaR (daily): {cvar_res.cvar:.4%}")

    # --- 5. Risk Parity ---
    header("5. RISK PARITY")
    rp = RiskParityOptimizer(Sigma, asset_names=names)
    rp_res = rp.optimize()
    print(f"\n  Portfolio Vol: {rp_res.portfolio_volatility:.2%}")
    for nm, w, rc in zip(names, rp_res.weights, rp_res.risk_contributions):
        print(f"    {nm:15s}: w={w:.2%}, RC={rc:.4f}")

    # --- 6. Visualizations ---
    header("6. GENERATING VISUALIZATIONS")
    out = "outputs/figures"
    frontiers = {"Markowitz": frontier_mvo[["return","volatility"]],
                 "Black-Litterman": frontier_bl[["return","volatility"]]}
    special = {
        "MVO Tangency": {"vol": tang.volatility, "ret": tang.expected_return},
        "BL Tangency": {"vol": tang_bl.volatility, "ret": tang_bl.expected_return},
        "Min Variance": {"vol": mv.volatility, "ret": mv.expected_return}}
    plot_frontier_comparison(frontiers, special, out)

    portfolios = {"MVO Tangency": tang.weights, "Black-Litterman": tang_bl.weights,
                  "Risk Parity": rp_res.weights}
    plot_weight_comparison(portfolios, names, out)

    header("ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()
