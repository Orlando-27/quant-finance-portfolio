"""
position_monitor.py
-------------------
Real-time position tracking, P&L computation, and drawdown monitoring.

Maintains an internal ledger of open positions updated from order fills
and periodic market price refreshes.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional
from dataclasses import dataclass, field

from src.utils  import get_logger, format_currency, pct_change

log = get_logger(__name__)


@dataclass
class Position:
    """State of a single open position."""
    symbol:       str
    entry_price:  float
    quantity:     int           # positive=long, negative=short
    entry_time:   datetime      = field(default_factory=datetime.now)
    current_price: float        = 0.0
    unrealized_pnl: float       = 0.0
    pct_change:    float        = 0.0

    def update(self, price: float) -> None:
        self.current_price   = price
        self.unrealized_pnl  = (price - self.entry_price) * self.quantity
        self.pct_change      = pct_change(self.entry_price, price) * (
            1 if self.quantity > 0 else -1
        )

    def to_dict(self) -> dict:
        return {
            "symbol":         self.symbol,
            "qty":            self.quantity,
            "entry_price":    self.entry_price,
            "current_price":  self.current_price,
            "unrealized_pnl": self.unrealized_pnl,
            "pct_change":     self.pct_change,
            "entry_time":     self.entry_time.isoformat(),
        }


class PositionMonitor:
    """
    Maintains the real-time position book and computes portfolio-level
    statistics for risk monitoring and reporting.

    Parameters
    ----------
    initial_equity : Starting net liquidation value of the account.
    """

    def __init__(self, initial_equity: float = 0.0):
        self._positions: Dict[str, Position] = {}
        self._equity_curve: list             = []
        self._initial_equity                 = initial_equity
        self._session_high                   = initial_equity

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def open_position(
        self, symbol: str, qty: int, entry_price: float
    ) -> None:
        """Register a new position from an order fill."""
        if symbol in self._positions:
            log.warning("Overwriting existing position for %s.", symbol)
        self._positions[symbol] = Position(
            symbol      = symbol,
            entry_price = entry_price,
            quantity    = qty,
        )
        log.info(
            "Position opened | %s qty=%d @ %s",
            symbol, qty, format_currency(entry_price),
        )

    def close_position(self, symbol: str, exit_price: float) -> Optional[float]:
        """
        Close a position and return realised P&L.

        Parameters
        ----------
        symbol     : Ticker.
        exit_price : Fill price at closing.

        Returns
        -------
        float
            Realised P&L, or None if no position exists.
        """
        pos = self._positions.pop(symbol, None)
        if pos is None:
            log.warning("close_position called for %s but no position found.", symbol)
            return None
        pnl = (exit_price - pos.entry_price) * pos.quantity
        log.info(
            "Position closed | %s qty=%d entry=%s exit=%s pnl=%s",
            symbol, pos.quantity,
            format_currency(pos.entry_price),
            format_currency(exit_price),
            format_currency(pnl),
        )
        return pnl

    # ------------------------------------------------------------------
    # Market price updates
    # ------------------------------------------------------------------

    def update_prices(self, prices: Dict[str, float]) -> None:
        """
        Refresh current prices and recompute unrealised P&L for all positions.

        Parameters
        ----------
        prices : Mapping of symbol -> current market price.
        """
        for sym, pos in self._positions.items():
            if sym in prices:
                pos.update(prices[sym])
            else:
                log.debug("No price update for %s.", sym)

    # ------------------------------------------------------------------
    # Portfolio-level statistics
    # ------------------------------------------------------------------

    def snapshot(self) -> pd.DataFrame:
        """Return current position book as a DataFrame."""
        if not self._positions:
            return pd.DataFrame()
        return pd.DataFrame([p.to_dict() for p in self._positions.values()])

    def total_unrealized_pnl(self) -> float:
        """Sum of unrealised P&L across all open positions."""
        return sum(p.unrealized_pnl for p in self._positions.values())

    def record_equity(self, net_liq: float, timestamp: Optional[datetime] = None) -> None:
        """Append a point to the equity curve."""
        ts = timestamp or datetime.now()
        self._equity_curve.append({"timestamp": ts, "net_liq": net_liq})
        if net_liq > self._session_high:
            self._session_high = net_liq

    def equity_curve(self) -> pd.DataFrame:
        """Return equity curve as a DataFrame indexed by timestamp."""
        if not self._equity_curve:
            return pd.DataFrame(columns=["timestamp", "net_liq"])
        df = pd.DataFrame(self._equity_curve).set_index("timestamp")
        return df

    def current_drawdown(self) -> float:
        """Current drawdown from session high as a decimal."""
        if self._session_high == 0:
            return 0.0
        latest = self._equity_curve[-1]["net_liq"] if self._equity_curve else self._initial_equity
        return (self._session_high - latest) / self._session_high

    def summary(self) -> dict:
        """Portfolio-level summary statistics."""
        curve = self.equity_curve()
        latest_equity = float(curve["net_liq"].iloc[-1]) if len(curve) > 0 else self._initial_equity
        return {
            "open_positions":   len(self._positions),
            "total_upnl":       self.total_unrealized_pnl(),
            "session_return":   pct_change(self._initial_equity, latest_equity),
            "current_drawdown": self.current_drawdown(),
            "session_high":     self._session_high,
            "latest_equity":    latest_equity,
        }
