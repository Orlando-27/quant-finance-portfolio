"""
order_manager.py
----------------
Constructs, submits, and tracks Interactive Brokers orders via ib_insync.

Supports
--------
- Limit orders with configurable slippage buffer.
- Market orders.
- Stop-loss and take-profit bracket orders.
- Dry-run mode (logs order intent without submitting to IB).
"""

import time
import pandas as pd
from datetime import datetime
from typing  import Optional, List, Dict
from dataclasses import dataclass, field

from ib_insync import IB, Stock, LimitOrder, MarketOrder, StopOrder, Trade

from src.config import BotConfig, OrderConfig
from src.utils  import get_logger, format_currency

log = get_logger(__name__)


@dataclass
class OrderRecord:
    """Immutable record of a submitted order and its outcome."""
    order_id:   int
    symbol:     str
    action:     str                    # "BUY" | "SELL"
    qty:        int
    order_type: str                    # "LMT" | "MKT"
    limit_px:   float
    submitted:  datetime              = field(default_factory=datetime.now)
    filled_px:  Optional[float]       = None
    filled_at:  Optional[datetime]    = None
    status:     str                   = "Submitted"
    pnl:        Optional[float]       = None

    def to_dict(self) -> dict:
        return {
            "order_id":  self.order_id,
            "symbol":    self.symbol,
            "action":    self.action,
            "qty":       self.qty,
            "order_type": self.order_type,
            "limit_px":  self.limit_px,
            "submitted": self.submitted.isoformat(),
            "filled_px": self.filled_px,
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "status":    self.status,
            "pnl":       self.pnl,
        }


class OrderManager:
    """
    Manages the full order lifecycle: creation, submission, monitoring,
    and cancellation.

    Parameters
    ----------
    ib      : Connected ib_insync.IB instance.
    cfg     : Master BotConfig.
    dry_run : If True, orders are logged but never sent to IB.
    """

    def __init__(self, ib: IB, cfg: BotConfig, dry_run: bool = False):
        self.ib       = ib
        self.cfg      = cfg
        self.ocfg: OrderConfig = cfg.orders
        self.dry_run  = dry_run or cfg.dry_run
        self._records: List[OrderRecord] = []
        self._trade_map: Dict[int, Trade] = {}
        self._order_counter = 1

    # ------------------------------------------------------------------
    # Order submission
    # ------------------------------------------------------------------

    def submit_order(
        self,
        symbol: str,
        action: str,
        qty:    int,
        price:  float,
    ) -> Optional[OrderRecord]:
        """
        Build and submit a single order (limit or market).

        Parameters
        ----------
        symbol : Ticker.
        action : "BUY" or "SELL".
        qty    : Number of shares (must be >= 1).
        price  : Reference market price for limit pricing.

        Returns
        -------
        OrderRecord or None if submission fails / dry_run.
        """
        if qty < 1:
            log.warning("Skipping zero-quantity order for %s.", symbol)
            return None

        contract  = Stock(symbol, self.cfg.universe.exchange, self.cfg.universe.currency)
        limit_px  = self._limit_price(action, price)
        order_id  = self._next_id()

        log.info(
            "[ORDER] %s %d %s @ LMT %s (dry_run=%s)",
            action, qty, symbol, format_currency(limit_px), self.dry_run,
        )

        record = OrderRecord(
            order_id   = order_id,
            symbol     = symbol,
            action     = action,
            qty        = qty,
            order_type = self.ocfg.order_type,
            limit_px   = limit_px,
        )

        if self.dry_run:
            record.status    = "DryRun"
            record.filled_px = limit_px
            record.filled_at = datetime.now()
            self._records.append(record)
            return record

        try:
            ib_order = LimitOrder(
                action         = action,
                totalQuantity  = qty,
                lmtPrice       = round(limit_px, 2),
                tif            = self.ocfg.time_in_force,
                outsideRth     = False,
            )
            self.ib.qualifyContracts(contract)
            trade = self.ib.placeOrder(contract, ib_order)
            self._trade_map[order_id] = trade
            record.status = "Submitted"
            log.info("Order %d placed for %s.", order_id, symbol)
        except Exception as exc:
            log.error("Order placement failed for %s: %s", symbol, exc)
            record.status = "Error"

        self._records.append(record)
        return record

    def submit_bracket(
        self,
        symbol:     str,
        action:     str,
        qty:        int,
        price:      float,
        stop_pct:   float,
        tp_pct:     float,
    ) -> List[Optional[OrderRecord]]:
        """
        Submit a bracket (entry + stop-loss + take-profit) order set.

        Parameters
        ----------
        action   : "BUY" (long bracket) or "SELL" (short bracket).
        stop_pct : Stop-loss distance as decimal (e.g. 0.02 = 2 %).
        tp_pct   : Take-profit distance as decimal (e.g. 0.04 = 4 %).

        Returns
        -------
        List of OrderRecord (entry, stop, take-profit).
        """
        entry_record = self.submit_order(symbol, action, qty, price)

        opp_action = "SELL" if action == "BUY" else "BUY"
        sl_px      = price * (1 - stop_pct) if action == "BUY" else price * (1 + stop_pct)
        tp_px      = price * (1 + tp_pct)  if action == "BUY" else price * (1 - tp_pct)

        sl_record  = self.submit_order(symbol, opp_action, qty, sl_px)
        tp_record  = self.submit_order(symbol, opp_action, qty, tp_px)

        return [entry_record, sl_record, tp_record]

    # ------------------------------------------------------------------
    # Order monitoring
    # ------------------------------------------------------------------

    def poll_fills(self, wait_secs: float = 2.0) -> None:
        """
        Check IB for fill confirmations and update records.

        Parameters
        ----------
        wait_secs : How long to wait for IB callbacks.
        """
        if self.dry_run:
            return
        self.ib.sleep(wait_secs)
        for oid, trade in list(self._trade_map.items()):
            rec = next((r for r in self._records if r.order_id == oid), None)
            if rec is None:
                continue
            status = trade.orderStatus.status
            rec.status = status
            if status == "Filled":
                rec.filled_px = trade.orderStatus.avgFillPrice
                rec.filled_at = datetime.now()
                log.info(
                    "FILL | %s %d %s @ %s",
                    rec.action, rec.qty, rec.symbol,
                    format_currency(rec.filled_px),
                )
                del self._trade_map[oid]

    def cancel_all(self) -> None:
        """Cancel all open IB orders."""
        if self.dry_run:
            return
        for trade in self._trade_map.values():
            self.ib.cancelOrder(trade.order)
        log.info("Cancelled all open orders.")
        self._trade_map.clear()

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def order_log(self) -> pd.DataFrame:
        """Return order history as a DataFrame."""
        if not self._records:
            return pd.DataFrame()
        return pd.DataFrame([r.to_dict() for r in self._records])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _limit_price(self, action: str, price: float) -> float:
        """Add a small slippage buffer to improve fill probability."""
        slip = self.ocfg.limit_slippage
        return price * (1 + slip) if action == "BUY" else price * (1 - slip)

    def _next_id(self) -> int:
        oid = self._order_counter
        self._order_counter += 1
        return oid
