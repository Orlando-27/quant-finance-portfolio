"""
Market Microstructure Analysis — Main Pipeline
================================================
Orchestrates the full analysis: data acquisition, spread estimation,
order flow analysis, illiquidity metrics, market impact modeling,
and publication-quality chart generation.

Usage:
    python main.py                                      # defaults
    python main.py --tickers SPY QQQ IWM               # custom tickers
    python main.py --start 2020-01-01 --end 2024-12-31 # custom date range
    python main.py --synthetic                          # use synthetic tick data only
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Add project root to path
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from models.spread_models import SpreadModels
from models.order_flow    import OrderFlowAnalyzer
from models.illiquidity   import IlliquidityModels
from models.market_impact import AlmgrenChrissModel, AlmgrenChrissParams
from utils.data_loader    import MarketDataLoader, SyntheticTickGenerator
from utils.helpers        import (
    compute_vwap, compute_twap, intraday_volume_profile,
    price_impact_regression, newey_west_tstat,
)
from visualization.charts import (
    plot_spread_comparison,
    plot_order_flow_dashboard,
    plot_illiquidity_dashboard,
    plot_almgren_chriss_frontier,
    plot_intraday_profile,
    save_figure,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Market Microstructure Analysis Pipeline"
    )
    p.add_argument("--tickers",    nargs="+", default=["SPY", "QQQ", "IWM"],
                   help="List of tickers to analyse")
    p.add_argument("--start",      default="2020-01-01", help="Start date YYYY-MM-DD")
    p.add_argument("--end",        default="2024-12-31", help="End date   YYYY-MM-DD")
    p.add_argument("--synthetic",  action="store_true",
                   help="Also run intraday analysis on synthetic tick data")
    p.add_argument("--output-dir", default="outputs/figures",
                   help="Directory for output figures")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _banner(msg: str) -> None:
    sep = "=" * 60
    print(f"\n{sep}\n  {msg}\n{sep}")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------
def run_spread_analysis(data: pd.DataFrame, ticker: str, out_dir: str) -> None:
    """Run all spread model estimates for one ticker."""
    _banner(f"Spread Analysis — {ticker}")

    spread_df = SpreadModels.spread_comparison(
        data["Close"], data["High"], data["Low"]
    )

    # Print summary statistics
    print("\nSpread Estimates (mean across sample):")
    summary = spread_df.describe().loc[["mean", "std", "min", "max"]] * 100
    print(summary.round(4).to_string())

    # Effective spread decomposition
    decomp = SpreadModels.effective_spread_decomposition(data["Close"], window=60)
    print(f"\nEffective Spread Decomposition (60d rolling mean):")
    print(decomp.tail(5).round(6).to_string())

    # Chart
    plot_spread_comparison(
        spread_df, ticker=ticker,
        save_path=f"{out_dir}/01_spread_models_{ticker}.png",
    )


def run_order_flow_analysis(data: pd.DataFrame, ticker: str, out_dir: str) -> None:
    """Run OFI and VPIN analysis for one ticker."""
    _banner(f"Order Flow Analysis — {ticker}")

    prices = data["Close"]
    volume = data["Volume"]

    ofi     = OrderFlowAnalyzer.order_flow_imbalance(prices, volume, window=20)
    vpin    = OrderFlowAnalyzer.vpin(prices, volume, n_buckets=50)
    ofi_acf = OrderFlowAnalyzer.ofi_autocorrelation(ofi, max_lag=20)

    # Newey-West t-stat for OFI
    t, p = newey_west_tstat(ofi.dropna(), lags=5)
    print(f"\nOFI mean: {ofi.mean():.4f} | NW t-stat: {t:.3f} | p-value: {p:.4f}")

    # Kyle OLS regression
    reg = price_impact_regression(prices.pct_change(), ofi * volume)
    print(f"Kyle OLS: λ={reg['lambda']:.2e}, R²={reg['r_squared']:.4f}, "
          f"t={reg['t_stat']:.3f}, p={reg['p_value']:.4f}")

    # VPIN exceedance rate
    exc_rate = (vpin.dropna() > 0.5).mean()
    print(f"VPIN > 0.5 frequency: {exc_rate:.2%}")

    plot_order_flow_dashboard(
        prices, ofi, vpin, ofi_acf,
        ticker=ticker,
        save_path=f"{out_dir}/02_order_flow_{ticker}.png",
    )


def run_illiquidity_analysis(data: pd.DataFrame, ticker: str, out_dir: str) -> None:
    """Compute and visualize illiquidity metrics."""
    _banner(f"Illiquidity Analysis — {ticker}")

    returns      = data["Return"]
    dollar_vol   = data["DollarVolume"]
    volume       = data["Volume"]
    prices       = data["Close"]

    amihud   = IlliquidityModels.amihud_illiq(returns, dollar_vol, window=252)
    kyle_lam = IlliquidityModels.kyle_lambda(prices, volume, window=20)
    turnover = IlliquidityModels.turnover_liquidity(volume, window=21)
    comp_liq = IlliquidityModels.composite_liquidity_score(amihud, kyle_lam, turnover)

    print(f"\nAmihud ILLIQ (mean): {amihud.mean():.4f}×10⁻⁶")
    print(f"Kyle lambda  (mean): {kyle_lam.mean():.6f}")
    print(f"Turnover     (mean): {turnover.mean():.2f}")

    plot_illiquidity_dashboard(
        amihud, kyle_lam, comp_liq, returns,
        ticker=ticker,
        save_path=f"{out_dir}/03_illiquidity_{ticker}.png",
    )


def run_market_impact_analysis(out_dir: str) -> None:
    """Almgren-Chriss optimal execution analysis."""
    _banner("Almgren-Chriss Optimal Execution")

    params = AlmgrenChrissParams(
        X=100_000, T=1.0, N=10, sigma=0.015,
        eta=2.5e-7, gamma=2.5e-9, S0=100.0,
    )
    model = AlmgrenChrissModel(params)

    frontier_df = model.efficient_frontier(n_points=200)
    traj_twap   = model.twap_trajectory()
    traj_opt    = model.optimal_trajectory(lam=1e-5)

    print("\nExecution Summary (λ=1e-5):")
    for k, v in model.execution_summary(lam=1e-5).items():
        print(f"  {k:<40}: {v}")

    print(f"\nTWAP Expected Cost: ${traj_twap['E_cost']:,.2f}")
    print(f"Optimal IS     : {traj_opt['E_cost']/(params.X*params.S0)*1e4:.2f} bps")
    print(f"TWAP IS        : {traj_twap['E_cost']/(params.X*params.S0)*1e4:.2f} bps")

    plot_almgren_chriss_frontier(
        frontier_df, traj_twap, traj_opt,
        save_path=f"{out_dir}/04_almgren_chriss_frontier.png",
    )


def run_intraday_analysis(out_dir: str) -> None:
    """Synthetic tick data intraday analysis."""
    _banner("Intraday Analysis — Synthetic Tick Data")

    gen   = SyntheticTickGenerator(S0=100.0, n_ticks=5_000, seed=42)
    ticks = gen.generate()
    bars  = gen.aggregate_to_bars(ticks, freq="5T")

    vwap_arr = compute_vwap(bars["close"], bars["volume"])
    twap_arr = compute_twap(bars["close"])
    profile  = intraday_volume_profile(bars)

    print(f"\nTotal bars     : {len(bars)}")
    print(f"VWAP (session) : ${vwap_arr.iloc[-1]:.4f}")
    print(f"TWAP (session) : ${twap_arr.iloc[-1]:.4f}")
    print(f"Peak vol bar   : {profile.loc[profile['volume_pct'].idxmax(), 'time_of_day']}")

    plot_intraday_profile(
        bars, ticker="Synthetic Asset",
        save_path=f"{out_dir}/05_intraday_profile.png",
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    _banner("MARKET MICROSTRUCTURE ANALYSIS")
    print(f"  Tickers    : {args.tickers}")
    print(f"  Period     : {args.start} → {args.end}")
    print(f"  Output dir : {args.output_dir}")

    # --- Load market data ---
    loader   = MarketDataLoader(args.tickers, args.start, args.end)
    datasets = loader.fetch_all()

    # --- Per-ticker analysis ---
    for ticker, data in datasets.items():
        print(f"\n{'─'*60}")
        print(f"  Analysing: {ticker}  |  {len(data)} trading days")
        print(f"{'─'*60}")
        try:
            run_spread_analysis    (data, ticker, args.output_dir)
            run_order_flow_analysis(data, ticker, args.output_dir)
            run_illiquidity_analysis(data, ticker, args.output_dir)
        except Exception as exc:
            print(f"  [WARNING] {ticker} — {exc}")

    # --- Market impact (asset-agnostic) ---
    run_market_impact_analysis(args.output_dir)

    # --- Intraday synthetic (optional) ---
    if args.synthetic:
        run_intraday_analysis(args.output_dir)

    _banner("PIPELINE COMPLETE")
    print(f"  Charts saved to: {args.output_dir}/")
    print("  Run: python main.py --synthetic  to include intraday profile")


if __name__ == "__main__":
    main()
