"""
risk_manager.py
---------------
Pre-trade risk checks and position sizing.

All sizing logic is based on the Kelly-fraction / fixed-fractional approach:
  shares = floor( (equity * max_position_pct) / price )

Risk guards implemented
-----------------------
1. Max position size per symbol (pct of net liquidation).
2. Max total deployed capital.
3. Daily loss limit (bot halts for the day if breached).
4. Max drawdown from session high (bot halts if breached).
"""

import math
from typing import Dict, Optional
from dataclasses import dataclass, field

from src.config import RiskConfig
from src.utils  import get_logger, format_currency

log = get_logger(__name__)


@dataclass
class RiskState:
    """Mutable session state tracked by the RiskManager."""
    session_start_equity: float     = 0.0
    session_high_equity:  float     = 0.0
    daily_realized_pnl:   float     = 0.0
    halted:               bool      = False
    halt_reason:          str       = ""
    deployed_capital:     float     = 0.0
    position_values:      Dict[str, float] = field(default_factory=dict)


class RiskManager:
    """
    Pre-trade risk checks and position sizing engine.

    Parameters
    ----------
    cfg : RiskConfig
        Risk limits and sizing parameters.
    """

    def __init__(self, cfg: RiskConfig):
        self.cfg   = cfg
        self.state = RiskState()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialize(self, net_liq: float) -> None:
        """Set session baseline equity."""
        self.state.session_start_equity = net_liq
        self.state.session_high_equity  = net_liq
        log.info("Risk manager initialized | net_liq=%s", format_currency(net_liq))

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def compute_shares(
        self,
        symbol:  str,
        price:   float,
        net_liq: float,
        signal:  int,
    ) -> int:
        """
        Compute how many shares to trade for a given signal.

        Parameters
        ----------
        symbol  : Ticker.
        price   : Current market price.
        net_liq : Current account net liquidation value.
        signal  : +1 (buy) or -1 (sell / short).

        Returns
        -------
        int
            Number of shares (0 if trade is blocked by risk limits).
        """
        if self.state.halted:
            log.warning("Bot halted (%s). No orders allowed.", self.state.halt_reason)
            return 0

        if price <= 0:
            return 0

        max_notional = net_liq * self.cfg.max_position_pct
        raw_shares   = math.floor(max_notional / price)
        shares       = max(raw_shares, self.cfg.max_position_pct > 0)

        # Check deployed capital
        proposed_notional = shares * price
        if (self.state.deployed_capital + proposed_notional) > (
            net_liq * self.cfg.max_portfolio_pct
        ):
            log.warning(
                "%s: order blocked — max portfolio pct would be breached "
                "(deployed=%s, proposed=%s, limit=%s%%)",
                symbol,
                format_currency(self.state.deployed_capital),
                format_currency(proposed_notional),
                self.cfg.max_portfolio_pct * 100,
            )
            return 0

        log.info(
            "%s: sized %d shares @ %s (notional=%s)",
            symbol, shares, format_currency(price),
            format_currency(proposed_notional),
        )
        return shares

    # ------------------------------------------------------------------
    # Post-trade / monitoring guards
    # ------------------------------------------------------------------

    def check_stop_loss(self, symbol: str, entry: float, current: float) -> bool:
        """Return True if stop-loss level is breached."""
        loss_pct = (current - entry) / entry
        if loss_pct <= -self.cfg.stop_loss_pct:
            log.warning(
                "%s: STOP-LOSS triggered | entry=%s current=%s loss=%.2f%%",
                symbol, format_currency(entry), format_currency(current),
                loss_pct * 100,
            )
            return True
        return False

    def check_take_profit(self, symbol: str, entry: float, current: float) -> bool:
        """Return True if take-profit level is reached."""
        gain_pct = (current - entry) / entry
        if gain_pct >= self.cfg.take_profit_pct:
            log.info(
                "%s: TAKE-PROFIT reached | entry=%s current=%s gain=%.2f%%",
                symbol, format_currency(entry), format_currency(current),
                gain_pct * 100,
            )
            return True
        return False

    def update_equity(self, current_equity: float) -> None:
        """
        Update session high watermark and check daily loss / drawdown limits.

        Parameters
        ----------
        current_equity : Current net liquidation value.
        """
        if current_equity > self.state.session_high_equity:
            self.state.session_high_equity = current_equity

        daily_pnl_pct = (
            current_equity - self.state.session_start_equity
        ) / self.state.session_start_equity

        drawdown_pct = (
            self.state.session_high_equity - current_equity
        ) / self.state.session_high_equity

        if daily_pnl_pct <= -self.cfg.max_daily_loss:
            self._halt(
                f"Daily loss limit breached: {daily_pnl_pct*100:.2f}% "
                f"(limit={self.cfg.max_daily_loss*100:.1f}%)"
            )

        if drawdown_pct >= self.cfg.max_drawdown:
            self._halt(
                f"Max drawdown breached: {drawdown_pct*100:.2f}% "
                f"(limit={self.cfg.max_drawdown*100:.1f}%)"
            )

    def register_fill(self, symbol: str, shares: int, price: float) -> None:
        """Update deployed capital after a fill."""
        notional = abs(shares) * price
        self.state.deployed_capital += notional
        self.state.position_values[symbol] = (
            self.state.position_values.get(symbol, 0) + notional
        )

    def release_capital(self, symbol: str, notional: float) -> None:
        """Release capital when a position is closed."""
        self.state.deployed_capital -= notional
        self.state.position_values.pop(symbol, None)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _halt(self, reason: str) -> None:
        if not self.state.halted:
            self.state.halted      = True
            self.state.halt_reason = reason
            log.critical("TRADING HALTED: %s", reason)
