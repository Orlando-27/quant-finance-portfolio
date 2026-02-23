"""
VaR & CVaR Risk Engine - Main Analysis
Author: Jose Orlando Bobadilla Fuentes | CQF
"""

import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from models.var_engine import VaREngine
from models.garch_var import GARCHVaRModel
from models.backtester import VaRBacktester
from visualization.risk_plots import (
    plot_var_backtest, plot_var_comparison, plot_garch_volatility)


def header(t):
    print(f"\n{'='*70}\n  {t}\n{'='*70}")

def main():
    header("VaR & CVaR MULTI-METHOD RISK ENGINE")
    np.random.seed(42)

    # Synthetic returns with fat tails and volatility clustering
    n = 1000
    vol = np.ones(n) * 0.01
    for i in range(1, n):
        vol[i] = np.sqrt(0.00001 + 0.08 * (vol[i-1]*np.random.randn())**2
                         + 0.90 * vol[i-1]**2)
    returns = vol * np.random.standard_t(df=5, size=n) * 0.5
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    ret_series = pd.Series(returns, index=dates, name="returns")

    print(f"\n  Observations: {n}")
    print(f"  Mean return:  {np.mean(returns):.4%}")
    print(f"  Std dev:      {np.std(returns):.4%}")
    print(f"  Skewness:     {pd.Series(returns).skew():.4f}")
    print(f"  Kurtosis:     {pd.Series(returns).kurtosis():.4f}")

    # --- Multi-method VaR ---
    header("1. MULTI-METHOD VaR COMPARISON")
    engine = VaREngine(returns, portfolio_value=1_000_000)

    for conf in [0.95, 0.99]:
        print(f"\n  Confidence: {conf:.0%}")
        df = engine.compute_all_methods(conf)
        for _, row in df.iterrows():
            print(f"    {row['Method']:35s}: VaR={row['VaR']:>12,.0f}  "
                  f"CVaR={row['CVaR']:>12,.0f}  ({row['VaR_pct']:.3%})")

    comp = engine.compute_all_methods(0.99)
    plot_var_comparison(comp)

    # --- GARCH VaR ---
    header("2. GARCH(1,1) CONDITIONAL VaR")
    garch = GARCHVaRModel(ret_series, dist="normal")
    garch.fit()
    garch_res = garch.conditional_var(confidence=0.99)

    print(f"\n  GARCH Parameters:")
    for k, v in garch_res.params.items():
        if v is not None:
            print(f"    {k:15s}: {v:.6f}" if isinstance(v, float) else f"    {k:15s}: {v}")

    plot_garch_volatility(garch_res.conditional_vol, garch_res.var_series)

    # --- Backtesting ---
    header("3. BACKTESTING")
    # Use rolling historical VaR for backtest
    window = 250
    var_bt = np.array([
        -np.percentile(returns[max(0,i-window):i], 1) if i >= 20 else 0.03
        for i in range(n)
    ])
    bt = VaRBacktester(returns, var_bt, confidence=0.99)
    res = bt.run(dates=dates)

    print(f"\n  Observations:    {res.n_observations}")
    print(f"  Violations:      {res.n_violations}")
    print(f"  Violation rate:  {res.violation_rate:.2%}")
    print(f"  Expected rate:   {res.expected_rate:.2%}")
    print(f"  Kupiec LR:       {res.kupiec_statistic:.4f}")
    print(f"  Kupiec p-value:  {res.kupiec_pvalue:.4f}")
    print(f"  Reject H0:       {res.kupiec_reject}")
    print(f"  Traffic light:   {res.traffic_light}")

    plot_var_backtest(returns, var_bt, confidence=0.99)

    header("ANALYSIS COMPLETE")

if __name__ == "__main__":
    main()
