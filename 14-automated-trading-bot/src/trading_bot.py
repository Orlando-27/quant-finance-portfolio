"""
trading_bot.py
--------------
Top-level orchestrator that wires together all subsystems:
  IBConnector -> SignalGenerator -> RiskManager -> OrderManager
  -> PositionMonitor -> Notifier -> Plotter

Workflow per scan cycle
-----------------------
1. Fetch account state and update RiskManager equity.
2. Download latest OHLCV bars for each symbol.
3. Generate signals for each symbol.
4. For each actionable signal:
   a. Check risk limits and size the order.
   b. Submit the order via OrderManager.
   c. Notify via email.
5. Monitor positions: apply stop-loss / take-profit rules.
6. Log and persist order journal.
"""

import os
import time
import pandas as pd
from datetime import datetime
from typing   import Dict, Optional

from src.config           import BotConfig
from src.ib_connector     import IBConnector
from src.signal_generator import SignalGenerator, SignalResult
from src.risk_manager     import RiskManager
from src.order_manager    import OrderManager
from src.position_monitor import PositionMonitor
from src.notifier         import Notifier
from src.plotter          import (
    plot_signal_dashboard,
    plot_equity_curve,
    plot_signal_heatmap,
    plot_position_pnl,
    plot_session_dashboard,
)
from src.utils            import get_logger, format_currency

log = get_logger(__name__)


class TradingBot:
    """
    End-to-end automated trading bot for Interactive Brokers.

    Parameters
    ----------
    cfg : BotConfig
        Master configuration (connection, signals, risk, orders, notifications).
    """

    def __init__(self, cfg: BotConfig):
        self.cfg       = cfg
        self.connector = IBConnector(cfg)
        self.signals   = SignalGenerator(cfg.signals)
        self.risk      = RiskManager(cfg.risk)
        self.notifier  = Notifier(cfg.notifications)
        self.monitor: Optional[PositionMonitor] = None
        self.orders:  Optional[OrderManager]    = None
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Connect to IB, initialise subsystems, and begin the scan loop."""
        log.info("Starting Trading Bot | paper=%s dry_run=%s",
                 self.cfg.paper_trading, self.cfg.dry_run)

        if not self.connector.connect():
            raise RuntimeError("Cannot start bot: IB connection failed.")

        account    = self.connector.get_account_summary()
        net_liq    = account.get("net_liq", 100_000.0)

        self.monitor = PositionMonitor(initial_equity=net_liq)
        self.orders  = OrderManager(
            self.connector.ib, self.cfg, dry_run=self.cfg.dry_run
        )
        self.risk.initialize(net_liq)
        self.monitor.record_equity(net_liq)

        log.info("Subsystems initialised | net_liq=%s", format_currency(net_liq))
        self._running = True

        try:
            self._run_loop()
        except KeyboardInterrupt:
            log.info("Bot stopped by user (KeyboardInterrupt).")
        finally:
            self._shutdown()

    def stop(self) -> None:
        """Signal the scan loop to exit cleanly."""
        self._running = False

    # ------------------------------------------------------------------
    # Main scan loop
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        """Repeatedly scan the universe until stopped."""
        interval = self.cfg.scan_interval
        log.info("Scan loop started | interval=%ds universe=%s",
                 interval, self.cfg.universe.symbols)

        while self._running:
            try:
                self._scan_cycle()
            except Exception as exc:
                log.error("Scan cycle error: %s", exc, exc_info=True)

            log.info("Sleeping %ds until next scan...", interval)
            time.sleep(interval)

    def _scan_cycle(self) -> None:
        """Execute one full scan cycle."""
        cycle_ts = datetime.now()
        log.info("--- Scan cycle %s ---", cycle_ts.strftime("%H:%M:%S"))

        # 1. Account update
        account = self.connector.get_account_summary()
        net_liq = account.get("net_liq", 0.0)
        self.risk.update_equity(net_liq)
        self.monitor.record_equity(net_liq, timestamp=cycle_ts)

        if self.risk.state.halted:
            log.warning("Bot halted. Skipping order generation.")
            self.notifier.risk_alert(self.risk.state.halt_reason)
            return

        # 2. Fetch market data
        bars_dict = self._fetch_all_bars()

        # 3. Generate signals
        signal_results = self.signals.generate_all(bars_dict)
        signals_df     = pd.DataFrame([
            {
                "symbol":     r.symbol,
                "close":      r.close,
                "signal":     r.signal,
                "score":      r.score,
                "confidence": r.confidence,
                "rsi":        r.rsi,
                "rsi_signal": r.rsi_signal,
                "macd_signal": r.macd_signal,
                "bb_signal":  r.bb_signal,
                "ema_signal": r.ema_signal,
            }
            for r in signal_results.values()
        ])

        # 4. Execute trades for actionable signals
        for sym, result in signal_results.items():
            if result.signal == 0:
                continue
            self._handle_signal(sym, result, net_liq)

        # 5. Check stop-loss / take-profit for open positions
        pos_snap = self.monitor.snapshot()
        if not pos_snap.empty:
            prices = {r.symbol: r.close for r in signal_results.values()}
            self.monitor.update_prices(prices)
            self._check_exit_rules(pos_snap, prices)

        # 6. Plot and save charts
        self._generate_charts(account, signals_df)

    # ------------------------------------------------------------------
    # Signal handling
    # ------------------------------------------------------------------

    def _handle_signal(
        self, symbol: str, result: SignalResult, net_liq: float
    ) -> None:
        """Translate a consensus signal into an order."""
        action = "BUY" if result.signal == +1 else "SELL"
        shares = self.risk.compute_shares(
            symbol, result.close, net_liq, result.signal
        )
        if shares < 1:
            return

        log.info(
            "Signal action | %s %d %s @ %s (confidence=%.2f)",
            action, shares, symbol,
            format_currency(result.close), result.confidence,
        )

        records = self.orders.submit_bracket(
            symbol   = symbol,
            action   = action,
            qty      = shares,
            price    = result.close,
            stop_pct = self.cfg.risk.stop_loss_pct,
            tp_pct   = self.cfg.risk.take_profit_pct,
        )
        self.orders.poll_fills(wait_secs=2.0)

        entry_rec = records[0]
        if entry_rec and entry_rec.status in ("Filled", "DryRun"):
            fill_px = entry_rec.filled_px or result.close
            self.monitor.open_position(symbol, shares, fill_px)
            self.risk.register_fill(symbol, shares, fill_px)
            self.notifier.trade_alert(
                symbol      = symbol,
                action      = action,
                qty         = shares,
                price       = fill_px,
                signal_info = {
                    "score":       result.score,
                    "confidence":  result.confidence,
                    "rsi":         result.rsi,
                    "macd_signal": result.macd_signal,
                    "bb_signal":   result.bb_signal,
                    "ema_signal":  result.ema_signal,
                },
            )

    # ------------------------------------------------------------------
    # Exit rules
    # ------------------------------------------------------------------

    def _check_exit_rules(
        self, pos_snap: pd.DataFrame, prices: Dict[str, float]
    ) -> None:
        """Apply stop-loss and take-profit rules to open positions."""
        for _, row in pos_snap.iterrows():
            sym   = row["symbol"]
            price = prices.get(sym)
            if price is None:
                continue

            sl = self.risk.check_stop_loss(sym, row["entry_price"], price)
            tp = self.risk.check_take_profit(sym, row["entry_price"], price)

            if sl or tp:
                direction = "SELL" if row["qty"] > 0 else "BUY"
                self.orders.submit_order(sym, direction, abs(int(row["qty"])), price)
                self.orders.poll_fills(wait_secs=2.0)
                pnl = self.monitor.close_position(sym, price)
                self.risk.release_capital(sym, abs(int(row["qty"])) * price)
                log.info(
                    "%s exit | reason=%s pnl=%s",
                    sym, "stop_loss" if sl else "take_profit",
                    format_currency(pnl or 0),
                )

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    def _fetch_all_bars(self) -> Dict[str, pd.DataFrame]:
        """Download OHLCV bars for every symbol in the universe."""
        bars = {}
        for sym in self.cfg.universe.symbols:
            try:
                df = self.connector.fetch_bars(
                    sym,
                    duration = self.cfg.universe.duration,
                    bar_size = self.cfg.universe.bar_size,
                )
                if not df.empty:
                    bars[sym] = df
            except Exception as exc:
                log.error("Failed to fetch bars for %s: %s", sym, exc)
        return bars

    # ------------------------------------------------------------------
    # Chart generation
    # ------------------------------------------------------------------

    def _generate_charts(
        self, account: dict, signals_df: pd.DataFrame
    ) -> None:
        """Save all session charts to outputs/charts/."""
        out = os.path.join(self.cfg.output_dir, "charts")
        os.makedirs(out, exist_ok=True)
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Heatmap
        if not signals_df.empty:
            plot_signal_heatmap(
                signals_df,
                os.path.join(out, f"signal_heatmap_{ts}.png"),
            )

        # Position P&L
        pos = self.monitor.snapshot()
        if not pos.empty:
            plot_position_pnl(pos, os.path.join(out, f"position_pnl_{ts}.png"))

        # Equity curve
        eq = self.monitor.equity_curve()
        if len(eq) >= 2:
            plot_equity_curve(
                eq, self.risk.state.session_start_equity,
                os.path.join(out, f"equity_curve_{ts}.png"),
            )

        # Session dashboard
        dash_path = os.path.join(out, f"session_dashboard_{ts}.png")
        plot_session_dashboard(
            account      = account,
            signals_df   = signals_df,
            positions_df = pos if not pos.empty else pd.DataFrame(),
            orders_df    = self.orders.order_log(),
            equity_df    = eq,
            initial_eq   = self.risk.state.session_start_equity,
            output_path  = dash_path,
        )

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def _shutdown(self) -> None:
        """Cleanup: cancel orders, send summary, disconnect."""
        log.info("Shutting down...")

        if self.orders:
            self.orders.cancel_all()
            order_log = self.orders.order_log()
            order_log.to_csv(
                os.path.join(self.cfg.output_dir, "logs", "orders.csv"), index=False
            )

        summary = self.monitor.summary() if self.monitor else {}
        eq      = self.monitor.equity_curve() if self.monitor else pd.DataFrame()
        pos     = self.monitor.snapshot()     if self.monitor else pd.DataFrame()

        if self.cfg.notifications.enabled:
            self.notifier.daily_summary(
                pnl          = summary.get("total_upnl", 0.0),
                orders_df    = self.orders.order_log() if self.orders else pd.DataFrame(),
                positions_df = pos,
                chart_path   = os.path.join(
                    self.cfg.output_dir, "charts", "session_dashboard_latest.png"
                ),
            )

        self.connector.disconnect()
        log.info("Bot shutdown complete. Session summary: %s", summary)
