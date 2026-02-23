"""
Main Pipeline: Momentum & Mean Reversion Multi-Asset Strategy
==============================================================

Executes the full strategy pipeline:
    1. Generate synthetic multi-asset data
    2. Compute momentum signals (TSMOM + Cross-Sectional)
    3. Compute mean-reversion signals (Z-Score, RSI, Bollinger)
    4. Detect market regimes (Volatility, Dispersion, Autocorrelation)
    5. Blend signals and construct portfolio
    6. Run walk-forward backtest with transaction costs
    7. Generate publication-quality visualizations

Author: Jose Orlando Bobadilla Fuentes, CQF, MSc AI
"""

import os
import sys

# Ensure project root is in path
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from src.data_generator import generate_multi_asset_data, get_asset_class_map
from src.momentum import TimeSeriesMomentum, CrossSectionalMomentum
from src.mean_reversion import CompositeMeanReversion
from src.regime import RegimeDetector
from src.portfolio import PortfolioConstructor
from src.backtesting import BacktestEngine
from src.visualization.strategy_plots import generate_all_figures


def main():
    """Run the complete momentum & mean reversion strategy pipeline."""
    print("=" * 60)
    print("  MOMENTUM & MEAN REVERSION MULTI-ASSET STRATEGY")
    print("  Jose Orlando Bobadilla Fuentes, CQF, MSc AI")
    print("=" * 60)

    # --- Step 1: Data Generation ---
    print("\n[1/7] Generating synthetic multi-asset universe (15Y daily)...")
    prices, returns, metadata = generate_multi_asset_data(n_years=15, seed=42)
    class_map = get_asset_class_map(metadata)
    print(f"       Universe: {len(metadata)} assets across {len(class_map)} classes")
    print(f"       Date range: {returns.index[0].date()} to {returns.index[-1].date()}")

    # --- Step 2: Momentum Signals ---
    print("\n[2/7] Computing momentum signals...")
    tsmom = TimeSeriesMomentum(lookback_days=252, vol_target=0.10, skip_days=21)
    csmom = CrossSectionalMomentum(lookback_months=12, skip_months=1)

    tsmom_signal = tsmom.compute_signal(returns)
    cs_signal = csmom.compute_signal(returns)
    mom_quality = tsmom.compute_momentum_quality(returns)

    # Blend TSMOM and CS-MOM with 50/50 weight
    mom_signal = 0.5 * tsmom_signal.fillna(0) + 0.5 * cs_signal.fillna(0)
    print(f"       TSMOM avg signal: {tsmom_signal.mean().mean():.4f}")
    print(f"       CS-MOM avg signal: {cs_signal.mean().mean():.4f}")
    print(f"       Momentum quality avg: {mom_quality.mean().mean():.4f}")

    # --- Step 3: Mean Reversion Signals ---
    print("\n[3/7] Computing mean-reversion signals (Z-Score, RSI, Bollinger)...")
    mr_composite = CompositeMeanReversion(w_zscore=0.4, w_rsi=0.3, w_bollinger=0.3)
    mr_signal = mr_composite.compute_signal(prices, returns)
    print(f"       MR composite avg signal: {mr_signal.mean().mean():.4f}")

    # --- Step 4: Regime Detection ---
    print("\n[4/7] Detecting market regimes...")
    regime_detector = RegimeDetector(w_vol=0.35, w_disp=0.30, w_ac=0.35)
    regime_alpha, regime_components = regime_detector.compute_regime_scores(returns)

    avg_alpha = regime_alpha.dropna().mean()
    print(f"       Average momentum alpha: {avg_alpha:.3f}")
    print(f"       Trending periods: {(regime_alpha > 0.65).sum()} days")
    print(f"       Mean-reverting periods: {(regime_alpha < 0.35).sum()} days")

    # --- Step 5: Portfolio Construction ---
    print("\n[5/7] Constructing portfolio with risk management...")
    constructor = PortfolioConstructor(
        vol_target=0.10,
        max_position=0.20,
        max_class_weight=0.50,
        max_drawdown_threshold=0.10,
        rebalance_freq=21,
    )
    weights = constructor.construct_portfolio(
        mom_signal=mom_signal.fillna(0),
        mr_signal=mr_signal.fillna(0),
        regime_alpha=regime_alpha.fillna(0.5),
        returns=returns,
        asset_class_map=class_map,
    )
    avg_exposure = weights.abs().sum(axis=1).mean()
    print(f"       Average gross exposure: {avg_exposure:.2f}")

    # --- Step 6: Backtesting ---
    print("\n[6/7] Running walk-forward backtest (10 bps TC + 5 bps slippage)...")
    engine = BacktestEngine(transaction_cost_bps=10, slippage_bps=5)
    bt = engine.run_backtest(weights, returns)

    m = bt["metrics"]
    print("\n       === PERFORMANCE SUMMARY ===")
    print(f"       Total Return:       {m['total_return']:.2%}")
    print(f"       Annualized Return:  {m['annualized_return']:.2%}")
    print(f"       Annualized Vol:     {m['annualized_volatility']:.2%}")
    print(f"       Sharpe Ratio:       {m['sharpe_ratio']:.3f}")
    print(f"       Sortino Ratio:      {m['sortino_ratio']:.3f}")
    print(f"       Max Drawdown:       {m['max_drawdown']:.2%}")
    print(f"       Calmar Ratio:       {m['calmar_ratio']:.3f}")
    print(f"       Hit Rate:           {m['hit_rate']:.1%}")
    print(f"       Profit Factor:      {m['profit_factor']:.2f}")
    print(f"       Information Ratio:  {m['information_ratio']:.3f}")
    print(f"       Avg Ann. Turnover:  {m['avg_annual_turnover']:.1f}x")
    print(f"       Skewness:           {m['skewness']:.3f}")
    print(f"       Excess Kurtosis:    {m['excess_kurtosis']:.3f}")

    # Attribution
    attrib = engine.attribution_by_class(weights, returns, class_map)
    print("\n       === ASSET CLASS ATTRIBUTION ===")
    for ac in attrib.index:
        print(f"       {ac:12s}: Return={attrib.loc[ac, 'ann_return']:.2%}  "
              f"Vol={attrib.loc[ac, 'ann_vol']:.2%}  "
              f"Exposure={attrib.loc[ac, 'avg_gross_exposure']:.2f}")

    # --- Step 7: Visualization ---
    print("\n[7/7] Generating publication-quality figures...")
    generate_all_figures(PROJECT_DIR)

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
