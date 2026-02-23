"""
config.py
---------
Centralised configuration for the Automated Trading Bot.
All parameters are read from environment variables with sensible defaults,
making the bot portable across paper-trading and live environments.
"""

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class IBConfig:
    """Interactive Brokers Gateway / TWS connection parameters."""
    host: str          = os.getenv("IB_HOST",     "127.0.0.1")
    port: int          = int(os.getenv("IB_PORT",  "7497"))   # 7497=paper, 7496=live
    client_id: int     = int(os.getenv("IB_CID",   "1"))
    account: str       = os.getenv("IB_ACCOUNT",   "")        # DU-xxxxx for paper
    timeout: int       = int(os.getenv("IB_TIMEOUT","20"))


@dataclass
class UniverseConfig:
    """Tradeable universe and asset class settings."""
    symbols: List[str] = field(default_factory=lambda: [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META",
        "NVDA", "JPM",  "GS",   "SPY",  "QQQ",
    ])
    exchange:   str = "SMART"
    currency:   str = "USD"
    sec_type:   str = "STK"          # STK | ETF | OPT | FUT
    bar_size:   str = "5 mins"       # IB bar size string
    duration:   str = "2 D"          # history lookback


@dataclass
class SignalConfig:
    """Technical indicator parameters for signal generation."""
    # RSI
    rsi_period:       int   = 14
    rsi_oversold:     float = 30.0
    rsi_overbought:   float = 70.0

    # MACD
    macd_fast:        int   = 12
    macd_slow:        int   = 26
    macd_signal:      int   = 9

    # Bollinger Bands
    bb_period:        int   = 20
    bb_std:           float = 2.0

    # EMA crossover
    ema_short:        int   = 9
    ema_long:         int   = 21

    # Consensus threshold  (number of bullish signals required to trade)
    consensus_long:   int   = 3
    consensus_short:  int   = 3


@dataclass
class RiskConfig:
    """Position sizing and risk management parameters."""
    max_position_pct:  float = 0.05   # max 5 % of equity per symbol
    max_portfolio_pct: float = 0.80   # max 80 % deployed
    stop_loss_pct:     float = 0.02   # 2 % hard stop
    take_profit_pct:   float = 0.04   # 4 % take profit
    max_daily_loss:    float = 0.03   # halt if daily P&L < -3 %
    max_drawdown:      float = 0.10   # halt if drawdown > 10 %


@dataclass
class OrderConfig:
    """Order execution parameters."""
    order_type:     str   = "LMT"      # LMT | MKT
    limit_slippage: float = 0.001      # 0.1 % slippage added to market price
    min_shares:     int   = 1
    time_in_force:  str   = "DAY"


@dataclass
class NotificationConfig:
    """Email notification settings (SMTP)."""
    enabled:      bool = os.getenv("NOTIFY_EMAIL", "false").lower() == "true"
    smtp_host:    str  = os.getenv("SMTP_HOST",    "smtp.gmail.com")
    smtp_port:    int  = int(os.getenv("SMTP_PORT", "587"))
    sender:       str  = os.getenv("SMTP_SENDER",  "")
    password:     str  = os.getenv("SMTP_PASS",    "")
    recipient:    str  = os.getenv("NOTIFY_TO",    "")


@dataclass
class BotConfig:
    """Master configuration aggregating all sub-configs."""
    ib:           IBConfig             = field(default_factory=IBConfig)
    universe:     UniverseConfig       = field(default_factory=UniverseConfig)
    signals:      SignalConfig         = field(default_factory=SignalConfig)
    risk:         RiskConfig           = field(default_factory=RiskConfig)
    orders:       OrderConfig          = field(default_factory=OrderConfig)
    notifications: NotificationConfig  = field(default_factory=NotificationConfig)

    # Paths
    output_dir:   str = os.path.join(os.path.dirname(__file__), "..", "outputs")
    log_level:    str = os.getenv("LOG_LEVEL", "INFO")

    # Operational mode
    paper_trading: bool = os.getenv("PAPER_TRADING", "true").lower() == "true"
    dry_run:       bool = os.getenv("DRY_RUN",        "false").lower() == "true"
    scan_interval: int  = int(os.getenv("SCAN_SECS",  "300"))   # seconds between scans


# Singleton instance used throughout the project
CONFIG = BotConfig()
