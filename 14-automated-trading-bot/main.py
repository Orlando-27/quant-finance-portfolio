"""
main.py
-------
Entry point for the Automated Trading Bot.

Usage
-----
# Paper trading (default):
    python main.py

# Dry run (no orders, just signal generation):
    DRY_RUN=true python main.py

# Live trading:
    IB_PORT=7496 PAPER_TRADING=false python main.py

# Demo mode (no IB connection, synthetic data):
    DEMO_MODE=true python main.py

Environment variables
---------------------
See src/config.py for the full list of supported env vars.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Ensure src/ is importable when running from project root
sys.path.insert(0, os.path.dirname(__file__))

from src.config  import BotConfig
from src.utils   import get_logger

log = get_logger("main")


# =============================================================================
# Demo / offline mode
# =============================================================================

def run_demo(cfg: BotConfig) -> None:
    """
    Run a full offline demonstration without an IB connection.
    Generates synthetic price data, computes signals, sizes positions,
    and produces all charts — useful for CI/CD and portfolio showcase.
    """
    from src.signal_generator import SignalGenerator
    from src.risk_manager     import RiskManager
    from src.position_monitor import PositionMonitor
    from src.notifier         import Notifier
    from src.plotter          import (
        plot_signal_dashboard, plot_equity_curve,
        plot_signal_heatmap,   plot_position_pnl,
        plot_session_dashboard,
    )

    log.info("Running in DEMO mode (no IB connection required).")

    rng      = np.random.default_rng(seed=42)
    n_bars   = 200
    symbols  = cfg.universe.symbols[:5]
    out      = os.path.join(cfg.output_dir, "charts")
    os.makedirs(out, exist_ok=True)

    signal_gen = SignalGenerator(cfg.signals)
    risk_mgr   = RiskManager(cfg.risk)
    monitor    = PositionMonitor(initial_equity=100_000.0)
    risk_mgr.initialize(100_000.0)

    # Synthetic equity curve
    equity_vals = 100_000.0 * (
        1 + np.cumsum(rng.normal(0.0003, 0.004, 80))
    )
    ts_idx = pd.date_range(end=datetime.now(), periods=80, freq="5min")
    for ts, eq in zip(ts_idx, equity_vals):
        monitor.record_equity(eq, ts)

    # Per-symbol bars and signals
    all_signals = []
    for sym in symbols:
        price0 = rng.uniform(100, 500)
        ret    = rng.normal(0.0002, 0.018, n_bars)
        prices = price0 * np.exp(np.cumsum(ret))
        dates  = pd.date_range(end=datetime.now(), periods=n_bars, freq="5min")
        bars   = pd.DataFrame({
            "open":   prices * rng.uniform(0.998, 1.0, n_bars),
            "high":   prices * rng.uniform(1.000, 1.010, n_bars),
            "low":    prices * rng.uniform(0.990, 1.000, n_bars),
            "close":  prices,
            "volume": rng.integers(100_000, 5_000_000, n_bars),
        }, index=dates)

        result = signal_gen.generate(sym, bars)
        all_signals.append({
            "symbol":     result.symbol,
            "close":      result.close,
            "signal":     result.signal,
            "score":      result.score,
            "confidence": result.confidence,
            "rsi":        result.rsi,
            "rsi_signal": result.rsi_signal,
            "macd_signal": result.macd_signal,
            "bb_signal":  result.bb_signal,
            "ema_signal": result.ema_signal,
        })

        # Per-symbol signal dashboard
        from src.signal_generator import SignalGenerator as SG
        close = bars["close"]
        rsi_s = SG._rsi(close, cfg.signals.rsi_period)
        ml, ms, mh = SG._macd(close, cfg.signals.macd_fast,
                               cfg.signals.macd_slow, cfg.signals.macd_signal)
        bu, bm, bl = SG._bollinger(close, cfg.signals.bb_period, cfg.signals.bb_std)
        es = close.ewm(span=cfg.signals.ema_short, adjust=False).mean()
        el = close.ewm(span=cfg.signals.ema_long,  adjust=False).mean()

        path = plot_signal_dashboard(
            symbol=sym, bars=bars, rsi=rsi_s,
            macd_l=ml, macd_h=mh,
            bb_upper=bu, bb_mid=bm, bb_lower=bl,
            ema_s=es, ema_l=el,
            signal=result.signal,
            output_path=os.path.join(out, f"signal_{sym}.png"),
        )
        log.info("Saved signal dashboard: %s", path)

        # Synthetic open position
        if result.signal != 0:
            shares = risk_mgr.compute_shares(sym, result.close, 100_000.0, result.signal)
            if shares > 0:
                monitor.open_position(sym, shares, result.close)

    signals_df = pd.DataFrame(all_signals)

    # Heatmap
    plot_signal_heatmap(signals_df, os.path.join(out, "signal_heatmap.png"))
    log.info("Saved signal heatmap.")

    # Position P&L
    pos = monitor.snapshot()
    if not pos.empty:
        plot_position_pnl(pos, os.path.join(out, "position_pnl.png"))
        log.info("Saved position P&L chart.")

    # Equity curve
    eq = monitor.equity_curve()
    plot_equity_curve(eq, 100_000.0, os.path.join(out, "equity_curve.png"))
    log.info("Saved equity curve.")

    # Session dashboard
    account = {
        "net_liq":       float(eq["net_liq"].iloc[-1]) if len(eq) > 0 else 100_000.0,
        "buying_power":  50_000.0,
        "unrealized_pnl": monitor.total_unrealized_pnl(),
        "realized_pnl":   0.0,
    }
    plot_session_dashboard(
        account      = account,
        signals_df   = signals_df,
        positions_df = pos if not pos.empty else pd.DataFrame(),
        orders_df    = pd.DataFrame(),
        equity_df    = eq,
        initial_eq   = 100_000.0,
        output_path  = os.path.join(out, "session_dashboard.png"),
    )
    log.info("Saved session dashboard.")

    log.info("Demo complete. Charts saved to: %s", out)
    print(f"\nSignal Results:\n{signals_df[['symbol','close','signal','score','confidence']].to_string(index=False)}")
    print(f"\nPortfolio Summary:\n{monitor.summary()}")


# =============================================================================
# Entry point
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Automated Trading Bot - Interactive Brokers API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --demo              # Offline demo, no IB needed
  python main.py --dry-run          # Connect to IB, no orders placed
  python main.py                    # Full paper trading mode
  IB_PORT=7496 python main.py       # Live trading
        """,
    )
    p.add_argument("--demo",     action="store_true", help="Offline demo mode")
    p.add_argument("--dry-run",  action="store_true", help="Connect but skip order submission")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Log verbosity")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = BotConfig()
    cfg.log_level = args.log_level

    if args.dry_run:
        cfg.dry_run = True

    log.info("=" * 60)
    log.info("  AUTOMATED TRADING BOT - INTERACTIVE BROKERS API")
    log.info("  Mode: %s", "DEMO" if args.demo else ("DRY_RUN" if cfg.dry_run else "LIVE"))
    log.info("  Paper Trading: %s", cfg.paper_trading)
    log.info("  Universe: %d symbols", len(cfg.universe.symbols))
    log.info("  Scan Interval: %ds", cfg.scan_interval)
    log.info("=" * 60)

    if args.demo or os.getenv("DEMO_MODE", "false").lower() == "true":
        run_demo(cfg)
        return

    from src.trading_bot import TradingBot
    bot = TradingBot(cfg)
    bot.start()


if __name__ == "__main__":
    main()
