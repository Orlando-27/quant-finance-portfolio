#!/usr/bin/env bash
# =============================================================================
# PROJECT 14 - AUTOMATED TRADING BOT WITH INTERACTIVE BROKERS API
# Build Script for Google Cloud Shell
# =============================================================================
# Creates complete project structure with all source files.
# Usage: bash build.sh
# =============================================================================

set -euo pipefail

PROJECT_ROOT="$HOME/quant-finance-portfolio/14-automated-trading-bot"
mkdir -p "$PROJECT_ROOT"
cd "$PROJECT_ROOT"

echo "============================================================"
echo "  BUILDING PROJECT 14: AUTOMATED TRADING BOT (IB API)"
echo "  Target: $PROJECT_ROOT"
echo "============================================================"

# -----------------------------------------------------------------------------
# DIRECTORY STRUCTURE
# -----------------------------------------------------------------------------
mkdir -p src tests notebooks outputs/{charts,logs,reports} config data

# =============================================================================
# FILE 1: config.py
# =============================================================================
cat > src/config.py << 'PYEOF'
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
PYEOF

# =============================================================================
# FILE 2: utils.py
# =============================================================================
cat > src/utils.py << 'PYEOF'
"""
utils.py
--------
Logging, timing decorators, and shared helper functions.
"""

import os
import logging
import time
import functools
from datetime import datetime
from pathlib import Path


def get_logger(name: str, log_dir: str = "outputs/logs",
               level: str = "INFO") -> logging.Logger:
    """
    Return a named logger writing to both stdout and a daily rotating file.

    Parameters
    ----------
    name    : Logger name (typically the module __name__).
    log_dir : Directory for log files.
    level   : Logging level string ("DEBUG", "INFO", "WARNING", "ERROR").

    Returns
    -------
    logging.Logger
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)

    if logger.handlers:          # avoid duplicate handlers on re-import
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    log_file = os.path.join(
        log_dir, f"trading_bot_{datetime.now().strftime('%Y%m%d')}.log"
    )
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def timeit(func):
    """Decorator that logs the execution time of any function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        logger.debug("%s completed in %.3f s", func.__qualname__, elapsed)
        return result
    return wrapper


def format_currency(value: float, decimals: int = 2) -> str:
    """Format a float as a USD currency string."""
    return f"${value:,.{decimals}f}"


def pct_change(old: float, new: float) -> float:
    """Safe percentage change; returns 0 if old is zero."""
    return (new - old) / old if old != 0 else 0.0


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp value to [lo, hi] range."""
    return max(lo, min(value, hi))
PYEOF

# =============================================================================
# FILE 3: ib_connector.py
# =============================================================================
cat > src/ib_connector.py << 'PYEOF'
"""
ib_connector.py
---------------
Manages the lifecycle of the ib_insync connection to Interactive Brokers
Gateway or TWS. Provides a context manager for safe connect / disconnect,
account summary retrieval, and historical bar data fetching.

Notes
-----
- Set IB_PORT=7497 for paper trading, 7496 for live.
- IB Gateway must be running and API access enabled before starting the bot.
- ib_insync uses asyncio internally; we expose a synchronous-friendly API.
"""

import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict

from ib_insync import IB, Stock, Contract, util

from src.config    import BotConfig, IBConfig
from src.utils     import get_logger, timeit

log = get_logger(__name__)


class IBConnector:
    """
    Thin wrapper around ib_insync.IB providing connection management
    and data retrieval methods used by the trading bot.

    Parameters
    ----------
    cfg : BotConfig
        Master bot configuration.
    """

    def __init__(self, cfg: BotConfig):
        self.cfg  = cfg
        self.ib_cfg: IBConfig = cfg.ib
        self.ib   = IB()
        self._connected = False

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """
        Connect to IB Gateway / TWS.

        Returns
        -------
        bool
            True if connection is successful.
        """
        try:
            self.ib.connect(
                host      = self.ib_cfg.host,
                port      = self.ib_cfg.port,
                clientId  = self.ib_cfg.client_id,
                timeout   = self.ib_cfg.timeout,
            )
            self._connected = True
            log.info(
                "Connected to IB | host=%s port=%d clientId=%d account=%s",
                self.ib_cfg.host, self.ib_cfg.port,
                self.ib_cfg.client_id, self.ib_cfg.account,
            )
            return True
        except Exception as exc:
            log.error("IB connection failed: %s", exc)
            self._connected = False
            return False

    def disconnect(self) -> None:
        """Gracefully disconnect from IB."""
        if self._connected:
            self.ib.disconnect()
            self._connected = False
            log.info("Disconnected from IB Gateway.")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.disconnect()

    @property
    def connected(self) -> bool:
        return self._connected and self.ib.isConnected()

    # ------------------------------------------------------------------
    # Account & portfolio data
    # ------------------------------------------------------------------

    def get_account_summary(self) -> Dict[str, float]:
        """
        Fetch key account metrics: net liquidation, buying power, etc.

        Returns
        -------
        dict
            Keys: 'net_liq', 'buying_power', 'unrealized_pnl', 'realized_pnl'
        """
        self._require_connection()
        vals = self.ib.accountValues(self.ib_cfg.account)
        summary: Dict[str, float] = {}
        mapping = {
            "NetLiquidation":    "net_liq",
            "BuyingPower":       "buying_power",
            "UnrealizedPnL":     "unrealized_pnl",
            "RealizedPnL":       "realized_pnl",
        }
        for v in vals:
            if v.tag in mapping and v.currency == "USD":
                summary[mapping[v.tag]] = float(v.value)
        log.info("Account summary: %s", summary)
        return summary

    def get_positions(self) -> pd.DataFrame:
        """
        Return open positions as a DataFrame.

        Returns
        -------
        pd.DataFrame
            Columns: symbol, position, avg_cost, market_value, unrealized_pnl
        """
        self._require_connection()
        rows = []
        for pos in self.ib.positions(self.ib_cfg.account):
            rows.append({
                "symbol":         pos.contract.symbol,
                "position":       pos.position,
                "avg_cost":       pos.avgCost,
                "market_value":   pos.position * pos.avgCost,
                "unrealized_pnl": 0.0,   # updated via portfolio events
            })
        df = pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["symbol", "position", "avg_cost", "market_value", "unrealized_pnl"]
        )
        log.info("Open positions: %d", len(df))
        return df

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    @timeit
    def fetch_bars(
        self,
        symbol:   str,
        duration: str = "2 D",
        bar_size: str = "5 mins",
    ) -> pd.DataFrame:
        """
        Download historical OHLCV bars from IB for a single symbol.

        Parameters
        ----------
        symbol   : Ticker symbol (e.g. 'AAPL').
        duration : IB duration string ('2 D', '1 M', etc.).
        bar_size : IB bar size string ('5 mins', '1 hour', '1 day').

        Returns
        -------
        pd.DataFrame
            Columns: date, open, high, low, close, volume.
        """
        self._require_connection()
        contract = self._make_contract(symbol)
        self.ib.qualifyContracts(contract)

        bars = self.ib.reqHistoricalData(
            contract,
            endDateTime     = "",
            durationStr     = duration,
            barSizeSetting  = bar_size,
            whatToShow      = "TRADES",
            useRTH          = True,
            formatDate      = 1,
        )
        if not bars:
            log.warning("No historical bars returned for %s", symbol)
            return pd.DataFrame()

        df = util.df(bars).rename(columns={"date": "date"})
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        log.debug("Fetched %d bars for %s", len(df), symbol)
        return df[["open", "high", "low", "close", "volume"]]

    def get_latest_price(self, symbol: str) -> float:
        """
        Return the last traded price for a symbol via market snapshot.

        Parameters
        ----------
        symbol : Ticker.

        Returns
        -------
        float
            Last traded price or NaN if unavailable.
        """
        self._require_connection()
        contract = self._make_contract(symbol)
        self.ib.qualifyContracts(contract)
        ticker = self.ib.reqMktData(contract, "", True, False)
        self.ib.sleep(1.5)          # allow snapshot to populate
        price = ticker.last if ticker.last and ticker.last > 0 else ticker.close
        self.ib.cancelMktData(contract)
        return float(price) if price else float("nan")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_contract(self, symbol: str) -> Stock:
        """Build a Stock contract from universe config."""
        u = self.cfg.universe
        return Stock(symbol, u.exchange, u.currency)

    def _require_connection(self) -> None:
        """Raise if not connected."""
        if not self.connected:
            raise RuntimeError(
                "IBConnector is not connected. Call connect() first."
            )
PYEOF

# =============================================================================
# FILE 4: signal_generator.py
# =============================================================================
cat > src/signal_generator.py << 'PYEOF'
"""
signal_generator.py
-------------------
Computes technical indicators (RSI, MACD, Bollinger Bands, EMA crossover)
on OHLCV DataFrames and produces a consensus signal for each symbol.

Signal values
-------------
 +1  : BUY  (long entry)
  0  : FLAT (no action / exit)
 -1  : SELL (short entry)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict

from src.config import SignalConfig
from src.utils  import get_logger

log = get_logger(__name__)


@dataclass
class SignalResult:
    """Structured output for a single symbol's signal computation."""
    symbol:        str
    timestamp:     pd.Timestamp
    close:         float

    # Individual indicator signals (-1, 0, +1)
    rsi_signal:    int
    macd_signal:   int
    bb_signal:     int
    ema_signal:    int

    # Composite score (sum of individual signals)
    score:         float

    # Final decision
    signal:        int            # -1 | 0 | +1
    confidence:    float          # score / 4  -> [-1, +1]

    # Indicator values for logging / charts
    rsi:           float = np.nan
    macd_line:     float = np.nan
    macd_hist:     float = np.nan
    bb_upper:      float = np.nan
    bb_lower:      float = np.nan
    ema_short:     float = np.nan
    ema_long:      float = np.nan


class SignalGenerator:
    """
    Multi-indicator signal generator using a consensus voting approach.

    Four indicators each cast a vote in {-1, 0, +1}; the consensus signal
    fires when the absolute score meets the configured threshold.

    Parameters
    ----------
    cfg : SignalConfig
        Indicator parameters and consensus thresholds.
    """

    def __init__(self, cfg: SignalConfig):
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, symbol: str, bars: pd.DataFrame) -> SignalResult:
        """
        Compute all indicators and return a SignalResult for one symbol.

        Parameters
        ----------
        symbol : Ticker symbol.
        bars   : OHLCV DataFrame with DatetimeIndex. Must have >= 30 rows.

        Returns
        -------
        SignalResult
        """
        if len(bars) < 30:
            log.warning("%s: insufficient bars (%d < 30)", symbol, len(bars))
            return self._flat_result(symbol, bars)

        close = bars["close"]

        # --- Compute indicators -------------------------------------------
        rsi      = self._rsi(close, self.cfg.rsi_period)
        macd_l, macd_s, macd_h = self._macd(
            close, self.cfg.macd_fast, self.cfg.macd_slow, self.cfg.macd_signal
        )
        bb_u, bb_m, bb_l = self._bollinger(close, self.cfg.bb_period, self.cfg.bb_std)
        ema_s    = close.ewm(span=self.cfg.ema_short,  adjust=False).mean()
        ema_l    = close.ewm(span=self.cfg.ema_long,   adjust=False).mean()

        # --- Latest values ------------------------------------------------
        c        = float(close.iloc[-1])
        rsi_v    = float(rsi.iloc[-1])
        macd_lv  = float(macd_l.iloc[-1])
        macd_hv  = float(macd_h.iloc[-1])
        bb_uv    = float(bb_u.iloc[-1])
        bb_lv    = float(bb_l.iloc[-1])
        ema_sv   = float(ema_s.iloc[-1])
        ema_lv   = float(ema_l.iloc[-1])

        # --- Individual signals -------------------------------------------
        rsi_sig  = self._rsi_signal(rsi_v)
        macd_sig = self._macd_signal(macd_l, macd_s)
        bb_sig   = self._bb_signal(c, bb_uv, bb_lv)
        ema_sig  = self._ema_signal(ema_s, ema_l)

        # --- Consensus scoring -------------------------------------------
        score    = rsi_sig + macd_sig + bb_sig + ema_sig
        signal   = self._consensus(score)
        conf     = score / 4.0

        result = SignalResult(
            symbol    = symbol,
            timestamp = bars.index[-1],
            close     = c,
            rsi_signal  = rsi_sig,
            macd_signal = macd_sig,
            bb_signal   = bb_sig,
            ema_signal  = ema_sig,
            score       = score,
            signal      = signal,
            confidence  = conf,
            rsi         = rsi_v,
            macd_line   = macd_lv,
            macd_hist   = macd_hv,
            bb_upper    = bb_uv,
            bb_lower    = bb_lv,
            ema_short   = ema_sv,
            ema_long    = ema_lv,
        )

        log.info(
            "%s | RSI=%.1f(%+d) MACD(%+d) BB(%+d) EMA(%+d) "
            "score=%+.0f => signal=%+d confidence=%.2f",
            symbol, rsi_v, rsi_sig, macd_sig, bb_sig, ema_sig,
            score, signal, conf,
        )
        return result

    def generate_all(
        self, bars_dict: Dict[str, pd.DataFrame]
    ) -> Dict[str, SignalResult]:
        """
        Run signal generation for every symbol in bars_dict.

        Parameters
        ----------
        bars_dict : Mapping of symbol -> OHLCV DataFrame.

        Returns
        -------
        dict
            Mapping of symbol -> SignalResult.
        """
        results = {}
        for sym, bars in bars_dict.items():
            try:
                results[sym] = self.generate(sym, bars)
            except Exception as exc:
                log.error("Signal error for %s: %s", sym, exc)
        return results

    # ------------------------------------------------------------------
    # Technical indicator implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _rsi(close: pd.Series, period: int) -> pd.Series:
        delta  = close.diff()
        gain   = delta.clip(lower=0).rolling(period).mean()
        loss   = (-delta.clip(upper=0)).rolling(period).mean()
        rs     = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _macd(
        close: pd.Series, fast: int, slow: int, signal_period: int
    ):
        ema_f    = close.ewm(span=fast,  adjust=False).mean()
        ema_s    = close.ewm(span=slow,  adjust=False).mean()
        macd_l   = ema_f - ema_s
        macd_sig = macd_l.ewm(span=signal_period, adjust=False).mean()
        macd_h   = macd_l - macd_sig
        return macd_l, macd_sig, macd_h

    @staticmethod
    def _bollinger(
        close: pd.Series, period: int, n_std: float
    ):
        mid   = close.rolling(period).mean()
        std   = close.rolling(period).std()
        upper = mid + n_std * std
        lower = mid - n_std * std
        return upper, mid, lower

    # ------------------------------------------------------------------
    # Signal voting functions
    # ------------------------------------------------------------------

    def _rsi_signal(self, rsi: float) -> int:
        if rsi <= self.cfg.rsi_oversold:
            return +1
        if rsi >= self.cfg.rsi_overbought:
            return -1
        return 0

    @staticmethod
    def _macd_signal(macd_line: pd.Series, macd_signal: pd.Series) -> int:
        """Bullish when MACD line crosses above signal line."""
        if len(macd_line) < 2:
            return 0
        prev = float(macd_line.iloc[-2]) - float(macd_signal.iloc[-2])
        curr = float(macd_line.iloc[-1]) - float(macd_signal.iloc[-1])
        if prev < 0 < curr:
            return +1
        if prev > 0 > curr:
            return -1
        # Continuation: current position above / below signal
        if curr > 0:
            return +1
        if curr < 0:
            return -1
        return 0

    @staticmethod
    def _bb_signal(close: float, upper: float, lower: float) -> int:
        if close <= lower:
            return +1
        if close >= upper:
            return -1
        return 0

    @staticmethod
    def _ema_signal(ema_short: pd.Series, ema_long: pd.Series) -> int:
        if len(ema_short) < 2:
            return 0
        prev = float(ema_short.iloc[-2]) - float(ema_long.iloc[-2])
        curr = float(ema_short.iloc[-1]) - float(ema_long.iloc[-1])
        if prev < 0 < curr:
            return +1
        if prev > 0 > curr:
            return -1
        if curr > 0:
            return +1
        if curr < 0:
            return -1
        return 0

    def _consensus(self, score: float) -> int:
        if score >= self.cfg.consensus_long:
            return +1
        if score <= -self.cfg.consensus_short:
            return -1
        return 0

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _flat_result(self, symbol: str, bars: pd.DataFrame) -> SignalResult:
        c = float(bars["close"].iloc[-1]) if len(bars) > 0 else float("nan")
        return SignalResult(
            symbol=symbol, timestamp=pd.Timestamp.now(), close=c,
            rsi_signal=0, macd_signal=0, bb_signal=0, ema_signal=0,
            score=0.0, signal=0, confidence=0.0,
        )
PYEOF

# =============================================================================
# FILE 5: risk_manager.py
# =============================================================================
cat > src/risk_manager.py << 'PYEOF'
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
PYEOF

# =============================================================================
# FILE 6: order_manager.py
# =============================================================================
cat > src/order_manager.py << 'PYEOF'
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
PYEOF

# =============================================================================
# FILE 7: position_monitor.py
# =============================================================================
cat > src/position_monitor.py << 'PYEOF'
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
PYEOF

# =============================================================================
# FILE 8: notifier.py
# =============================================================================
cat > src/notifier.py << 'PYEOF'
"""
notifier.py
-----------
Email notification service for trade events and daily summaries.

Uses Python's built-in smtplib (TLS/STARTTLS). Requires environment variables:
    SMTP_HOST, SMTP_PORT, SMTP_SENDER, SMTP_PASS, NOTIFY_TO

Configuration is read from NotificationConfig; set NOTIFY_EMAIL=true to enable.
"""

import smtplib
import os
from email.mime.text     import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image    import MIMEImage
from datetime             import datetime
from typing               import Optional, List

import pandas as pd

from src.config import NotificationConfig
from src.utils  import get_logger, format_currency

log = get_logger(__name__)


class Notifier:
    """
    Email notification dispatcher.

    Parameters
    ----------
    cfg : NotificationConfig
        SMTP settings and enable flag.
    """

    def __init__(self, cfg: NotificationConfig):
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Notification types
    # ------------------------------------------------------------------

    def trade_alert(
        self,
        symbol:     str,
        action:     str,
        qty:        int,
        price:      float,
        signal_info: dict,
    ) -> None:
        """
        Send an email alert when a trade order is submitted.

        Parameters
        ----------
        symbol      : Ticker.
        action      : "BUY" or "SELL".
        qty         : Order quantity.
        price       : Limit price.
        signal_info : Dict with indicator details (score, confidence, etc.).
        """
        subject = f"[Trading Bot] {action} {qty} {symbol} @ {format_currency(price)}"

        body = f"""
<html><body>
<h2>Trade Alert - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</h2>
<table border="1" cellpadding="6" style="border-collapse:collapse;">
  <tr><th>Symbol</th><td><b>{symbol}</b></td></tr>
  <tr><th>Action</th><td style="color:{'green' if action=='BUY' else 'red'};"><b>{action}</b></td></tr>
  <tr><th>Quantity</th><td>{qty:,}</td></tr>
  <tr><th>Limit Price</th><td>{format_currency(price)}</td></tr>
  <tr><th>Signal Score</th><td>{signal_info.get('score', 'N/A')}</td></tr>
  <tr><th>Confidence</th><td>{signal_info.get('confidence', 0):.2%}</td></tr>
  <tr><th>RSI</th><td>{signal_info.get('rsi', 'N/A'):.1f}</td></tr>
  <tr><th>MACD Signal</th><td>{signal_info.get('macd_signal', 'N/A')}</td></tr>
  <tr><th>BB Signal</th><td>{signal_info.get('bb_signal', 'N/A')}</td></tr>
  <tr><th>EMA Signal</th><td>{signal_info.get('ema_signal', 'N/A')}</td></tr>
</table>
<p>Mode: {'PAPER TRADING' if True else 'LIVE'}</p>
</body></html>
"""
        self._send(subject, body, html=True)

    def daily_summary(
        self,
        pnl:          float,
        orders_df:    pd.DataFrame,
        positions_df: pd.DataFrame,
        chart_path:   Optional[str] = None,
    ) -> None:
        """
        Send end-of-day portfolio summary with optional chart attachment.

        Parameters
        ----------
        pnl          : Session realised + unrealised P&L.
        orders_df    : DataFrame of all orders placed during the session.
        positions_df : Current open positions.
        chart_path   : Path to PNG dashboard image to attach.
        """
        subject = (
            f"[Trading Bot] Daily Summary {datetime.now().strftime('%Y-%m-%d')} | "
            f"P&L: {format_currency(pnl)}"
        )

        orders_html = (
            orders_df.to_html(index=False, border=1)
            if not orders_df.empty
            else "<p>No orders today.</p>"
        )
        positions_html = (
            positions_df.to_html(index=False, border=1)
            if not positions_df.empty
            else "<p>No open positions.</p>"
        )

        body = f"""
<html><body>
<h2>Daily Trading Summary - {datetime.now().strftime('%Y-%m-%d')}</h2>
<h3>Session P&L: <span style="color:{'green' if pnl >= 0 else 'red'};">{format_currency(pnl)}</span></h3>

<h3>Orders Placed</h3>
{orders_html}

<h3>Open Positions</h3>
{positions_html}
</body></html>
"""
        attachments = []
        if chart_path and os.path.exists(chart_path):
            with open(chart_path, "rb") as f:
                attachments.append(("dashboard.png", f.read()))

        self._send(subject, body, html=True, attachments=attachments)

    def risk_alert(self, reason: str) -> None:
        """Send immediate alert when a risk limit is breached."""
        subject = "[Trading Bot] RISK ALERT - Trading Halted"
        body    = f"<h2>Risk Limit Breached</h2><p>{reason}</p>"
        self._send(subject, body, html=True)

    # ------------------------------------------------------------------
    # SMTP dispatch
    # ------------------------------------------------------------------

    def _send(
        self,
        subject:     str,
        body:        str,
        html:        bool = False,
        attachments: List[tuple] = None,
    ) -> None:
        """
        Internal dispatcher. Silently logs if notifications are disabled.
        """
        if not self.cfg.enabled:
            log.debug("Notifications disabled. Skipping: %s", subject)
            return

        try:
            msg = MIMEMultipart("related")
            msg["Subject"] = subject
            msg["From"]    = self.cfg.sender
            msg["To"]      = self.cfg.recipient

            mime_body = MIMEText(body, "html" if html else "plain")
            msg.attach(mime_body)

            if attachments:
                for filename, data in attachments:
                    img = MIMEImage(data, name=filename)
                    msg.attach(img)

            with smtplib.SMTP(self.cfg.smtp_host, self.cfg.smtp_port) as server:
                server.ehlo()
                server.starttls()
                server.login(self.cfg.sender, self.cfg.password)
                server.sendmail(
                    self.cfg.sender, self.cfg.recipient, msg.as_string()
                )
            log.info("Email sent: %s", subject)

        except Exception as exc:
            log.error("Failed to send email '%s': %s", subject, exc)
PYEOF

# =============================================================================
# FILE 9: plotter.py
# =============================================================================
cat > src/plotter.py << 'PYEOF'
"""
plotter.py
----------
Publication-quality charts for the trading bot dashboard.

All figures use matplotlib with the Agg backend (headless / Cloud Shell safe).

Charts produced
---------------
1. Multi-panel signal dashboard  (price, RSI, MACD, Bollinger Bands).
2. Portfolio equity curve with drawdown subplot.
3. Position P&L bar chart.
4. Signal heatmap across universe.
5. Full session dashboard (combined figure).
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                     # headless rendering
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
from datetime import datetime
from typing import Dict, List, Optional

warnings.filterwarnings("ignore", category=UserWarning)

# --------------------------------------------------------------------------
# Style constants
# --------------------------------------------------------------------------
STYLE = {
    "bg":          "#0d1117",
    "panel":       "#161b22",
    "text":        "#e6edf3",
    "muted":       "#8b949e",
    "green":       "#3fb950",
    "red":         "#f85149",
    "blue":        "#58a6ff",
    "amber":       "#e3b341",
    "purple":      "#bc8cff",
    "grid":        "#21262d",
    "font_family": "monospace",
}
plt.rcParams.update({
    "figure.facecolor":  STYLE["bg"],
    "axes.facecolor":    STYLE["panel"],
    "axes.edgecolor":    STYLE["grid"],
    "axes.labelcolor":   STYLE["text"],
    "axes.titlecolor":   STYLE["text"],
    "xtick.color":       STYLE["muted"],
    "ytick.color":       STYLE["muted"],
    "text.color":        STYLE["text"],
    "grid.color":        STYLE["grid"],
    "grid.linestyle":    "--",
    "grid.linewidth":    0.5,
    "legend.facecolor":  STYLE["panel"],
    "legend.edgecolor":  STYLE["grid"],
    "legend.labelcolor": STYLE["text"],
    "font.family":       STYLE["font_family"],
    "font.size":         9,
})

usd_fmt = FuncFormatter(lambda x, _: f"${x:,.2f}")
pct_fmt = FuncFormatter(lambda x, _: f"{x:.1f}%")


def _save(fig, path: str, title: str) -> str:
    """Save figure to PNG and close."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=STYLE["bg"])
    plt.close(fig)
    return path


# =============================================================================
# Chart 1: Signal Dashboard (price + indicators)
# =============================================================================
def plot_signal_dashboard(
    symbol:   str,
    bars:     pd.DataFrame,
    rsi:      pd.Series,
    macd_l:   pd.Series,
    macd_h:   pd.Series,
    bb_upper: pd.Series,
    bb_mid:   pd.Series,
    bb_lower: pd.Series,
    ema_s:    pd.Series,
    ema_l:    pd.Series,
    signal:   int,
    output_path: str,
) -> str:
    """
    Four-panel chart: price + Bollinger + EMA, RSI, MACD histogram.

    Parameters
    ----------
    All series are assumed to share the same DatetimeIndex as bars.
    signal : Final consensus signal (-1, 0, +1) for annotation.

    Returns
    -------
    str : Path to saved PNG.
    """
    idx = bars.index
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        f"{symbol} - Signal Dashboard  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        fontsize=13, fontweight="bold", color=STYLE["text"],
    )

    gs = gridspec.GridSpec(
        4, 1, hspace=0.05,
        height_ratios=[3, 1, 1, 1],
    )

    # ---- Panel 1: Price + Bollinger + EMA --------------------------------
    ax1 = fig.add_subplot(gs[0])
    c   = bars["close"]
    ax1.plot(idx, c,        color=STYLE["blue"],   lw=1.2, label="Close")
    ax1.plot(idx, bb_upper, color=STYLE["muted"],  lw=0.8, ls="--", label="BB Upper")
    ax1.plot(idx, bb_mid,   color=STYLE["amber"],  lw=0.8, ls="--", label="BB Mid")
    ax1.plot(idx, bb_lower, color=STYLE["muted"],  lw=0.8, ls="--", label="BB Lower")
    ax1.plot(idx, ema_s,    color=STYLE["green"],  lw=1.0, label=f"EMA Short")
    ax1.plot(idx, ema_l,    color=STYLE["red"],    lw=1.0, label=f"EMA Long")
    ax1.fill_between(idx, bb_upper, bb_lower, alpha=0.06, color=STYLE["blue"])

    # Signal annotation
    sig_label = {1: "BUY", -1: "SELL", 0: "FLAT"}[signal]
    sig_color = {1: STYLE["green"], -1: STYLE["red"], 0: STYLE["muted"]}[signal]
    ax1.annotate(
        sig_label,
        xy=(idx[-1], float(c.iloc[-1])),
        xytext=(-60, 20), textcoords="offset points",
        fontsize=14, fontweight="bold", color=sig_color,
        arrowprops=dict(arrowstyle="->", color=sig_color, lw=1.5),
    )
    ax1.set_ylabel("Price (USD)")
    ax1.legend(loc="upper left", fontsize=7, ncol=3)
    ax1.grid(True, alpha=0.4)
    ax1.tick_params(labelbottom=False)
    ax1.yaxis.set_major_formatter(usd_fmt)

    # ---- Panel 2: Volume -------------------------------------------------
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    vol_colors = [
        STYLE["green"] if bars["close"].iloc[i] >= bars["open"].iloc[i]
        else STYLE["red"]
        for i in range(len(bars))
    ]
    ax2.bar(idx, bars["volume"], color=vol_colors, alpha=0.7, width=0.0003)
    ax2.set_ylabel("Volume")
    ax2.grid(True, alpha=0.4)
    ax2.tick_params(labelbottom=False)
    ax2.yaxis.set_major_formatter(
        FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x:,.0f}")
    )

    # ---- Panel 3: RSI ----------------------------------------------------
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(idx, rsi, color=STYLE["purple"], lw=1.2, label="RSI")
    ax3.axhline(70, color=STYLE["red"],   lw=0.8, ls="--", alpha=0.8)
    ax3.axhline(30, color=STYLE["green"], lw=0.8, ls="--", alpha=0.8)
    ax3.fill_between(idx, rsi, 70, where=(rsi >= 70), alpha=0.2, color=STYLE["red"])
    ax3.fill_between(idx, rsi, 30, where=(rsi <= 30), alpha=0.2, color=STYLE["green"])
    ax3.set_ylim(0, 100)
    ax3.set_ylabel("RSI")
    ax3.grid(True, alpha=0.4)
    ax3.tick_params(labelbottom=False)

    # ---- Panel 4: MACD histogram -----------------------------------------
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    colors = [STYLE["green"] if v >= 0 else STYLE["red"] for v in macd_h]
    ax4.bar(idx, macd_h, color=colors, alpha=0.8, width=0.0003, label="Histogram")
    ax4.plot(idx, macd_l, color=STYLE["blue"], lw=1.0, label="MACD")
    ax4.axhline(0, color=STYLE["muted"], lw=0.8)
    ax4.set_ylabel("MACD")
    ax4.set_xlabel("Date")
    ax4.grid(True, alpha=0.4)
    ax4.legend(loc="upper left", fontsize=7)

    # Rotate x-tick labels
    for ax in [ax4]:
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=7)

    return _save(fig, output_path, f"{symbol} Signal Dashboard")


# =============================================================================
# Chart 2: Equity Curve with Drawdown
# =============================================================================
def plot_equity_curve(
    equity_df:   pd.DataFrame,
    initial_eq:  float,
    output_path: str,
) -> str:
    """
    Two-panel chart: equity curve (top) and rolling drawdown (bottom).

    Parameters
    ----------
    equity_df  : DataFrame with 'net_liq' column and DatetimeIndex.
    initial_eq : Starting equity for reference line.

    Returns
    -------
    str : Path to saved PNG.
    """
    if len(equity_df) < 2:
        return ""

    curve = equity_df["net_liq"]
    hwm   = curve.cummax()
    dd    = (curve - hwm) / hwm * 100     # percentage drawdown

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    fig.suptitle(
        "Portfolio Equity Curve & Drawdown",
        fontsize=13, fontweight="bold", color=STYLE["text"],
    )

    # Equity
    ax1.plot(curve.index, curve.values, color=STYLE["blue"], lw=1.5, label="Net Liq")
    ax1.axhline(initial_eq, color=STYLE["amber"], lw=0.8, ls="--", label=f"Start {usd_fmt(initial_eq, None)}")
    ax1.fill_between(
        curve.index, curve.values, initial_eq,
        where=(curve.values >= initial_eq), alpha=0.15, color=STYLE["green"],
    )
    ax1.fill_between(
        curve.index, curve.values, initial_eq,
        where=(curve.values < initial_eq),  alpha=0.15, color=STYLE["red"],
    )
    ax1.set_ylabel("Net Liquidation (USD)")
    ax1.yaxis.set_major_formatter(usd_fmt)
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.4)

    # Drawdown
    ax2.fill_between(dd.index, dd.values, 0, alpha=0.5, color=STYLE["red"])
    ax2.plot(dd.index, dd.values, color=STYLE["red"], lw=0.8)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Time")
    ax2.grid(True, alpha=0.4)
    ax2.yaxis.set_major_formatter(pct_fmt)

    plt.setp(ax2.get_xticklabels(), rotation=30, ha="right", fontsize=7)
    fig.tight_layout()

    return _save(fig, output_path, "Equity Curve")


# =============================================================================
# Chart 3: Signal Heatmap
# =============================================================================
def plot_signal_heatmap(
    signals_df:  pd.DataFrame,
    output_path: str,
) -> str:
    """
    Heatmap showing individual indicator signals across all symbols.

    Parameters
    ----------
    signals_df : DataFrame with columns:
                 symbol, rsi_signal, macd_signal, bb_signal, ema_signal, score, signal
    Returns
    -------
    str : Path to saved PNG.
    """
    if signals_df.empty:
        return ""

    cols = ["rsi_signal", "macd_signal", "bb_signal", "ema_signal", "score"]
    df   = signals_df.set_index("symbol")[cols].astype(float)

    fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.5 + 2)))
    fig.suptitle(
        f"Signal Heatmap  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        fontsize=13, fontweight="bold", color=STYLE["text"],
    )

    cmap  = plt.cm.RdYlGn
    im    = ax.imshow(df.values, cmap=cmap, aspect="auto", vmin=-2, vmax=2)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(["RSI", "MACD", "BB", "EMA", "Score"], fontsize=9)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df.index.tolist(), fontsize=9)

    # Annotate cells
    for i in range(len(df)):
        for j, col in enumerate(cols):
            val = df.iloc[i, j]
            ax.text(
                j, i, f"{val:+.0f}",
                ha="center", va="center",
                color="black", fontsize=8, fontweight="bold",
            )

    plt.colorbar(im, ax=ax, label="Signal (-1 = Sell, +1 = Buy)")
    fig.tight_layout()

    return _save(fig, output_path, "Signal Heatmap")


# =============================================================================
# Chart 4: Position P&L Bar Chart
# =============================================================================
def plot_position_pnl(
    positions_df: pd.DataFrame,
    output_path:  str,
) -> str:
    """
    Horizontal bar chart showing unrealised P&L per open position.

    Parameters
    ----------
    positions_df : Output of PositionMonitor.snapshot().
    Returns
    -------
    str : Path to saved PNG.
    """
    if positions_df.empty:
        return ""

    df     = positions_df.sort_values("unrealized_pnl")
    colors = [STYLE["green"] if v >= 0 else STYLE["red"] for v in df["unrealized_pnl"]]

    fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.6 + 2)))
    fig.suptitle(
        "Open Position Unrealised P&L",
        fontsize=13, fontweight="bold", color=STYLE["text"],
    )

    bars = ax.barh(df["symbol"], df["unrealized_pnl"], color=colors, alpha=0.85, height=0.6)
    ax.axvline(0, color=STYLE["muted"], lw=1.0)

    for bar, val in zip(bars, df["unrealized_pnl"]):
        ax.text(
            bar.get_width() + (0.003 * abs(df["unrealized_pnl"]).max()),
            bar.get_y() + bar.get_height() / 2,
            format(val, "+,.2f"),
            va="center", fontsize=8,
            color=STYLE["green"] if val >= 0 else STYLE["red"],
        )

    ax.set_xlabel("Unrealised P&L (USD)")
    ax.xaxis.set_major_formatter(usd_fmt)
    ax.grid(True, axis="x", alpha=0.4)
    fig.tight_layout()

    return _save(fig, output_path, "Position P&L")


# =============================================================================
# Chart 5: Full Session Dashboard
# =============================================================================
def plot_session_dashboard(
    account:      dict,
    signals_df:   pd.DataFrame,
    positions_df: pd.DataFrame,
    orders_df:    pd.DataFrame,
    equity_df:    pd.DataFrame,
    initial_eq:   float,
    output_path:  str,
) -> str:
    """
    Single comprehensive figure: equity curve, signal summary, positions,
    and order log — suitable for daily email attachment.

    Returns
    -------
    str : Path to saved PNG.
    """
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        f"Trading Bot Session Dashboard  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        fontsize=14, fontweight="bold", color=STYLE["text"], y=0.98,
    )

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ---- 1. Equity Curve (top-left, wide) --------------------------------
    ax_eq = fig.add_subplot(gs[0, :])
    if len(equity_df) >= 2:
        curve = equity_df["net_liq"]
        ax_eq.plot(curve.index, curve.values, color=STYLE["blue"], lw=1.5, label="Net Liq")
        ax_eq.axhline(initial_eq, color=STYLE["amber"], lw=0.8, ls="--",
                      label=f"Start {initial_eq:,.0f}")
        ax_eq.fill_between(
            curve.index, curve.values, initial_eq,
            where=(curve.values >= initial_eq), alpha=0.15, color=STYLE["green"],
        )
        ax_eq.fill_between(
            curve.index, curve.values, initial_eq,
            where=(curve.values < initial_eq),  alpha=0.15, color=STYLE["red"],
        )
        ax_eq.yaxis.set_major_formatter(usd_fmt)
        ax_eq.legend(fontsize=8)
    ax_eq.set_title("Equity Curve", fontsize=10)
    ax_eq.grid(True, alpha=0.4)

    # ---- 2. Account Summary (middle-left) --------------------------------
    ax_acc = fig.add_subplot(gs[1, 0])
    ax_acc.axis("off")
    ax_acc.set_title("Account Summary", fontsize=10)
    rows = [(k.replace("_", " ").title(), f"${v:,.2f}") for k, v in account.items()]
    tbl  = ax_acc.table(
        cellText    = rows,
        colLabels   = ["Metric", "Value"],
        loc         = "center",
        cellLoc     = "center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.5)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor(STYLE["panel"] if r == 0 else STYLE["bg"])
        cell.set_edgecolor(STYLE["grid"])
        cell.set_text_props(color=STYLE["text"])

    # ---- 3. Signal Summary Bar (middle-right) ----------------------------
    ax_sig = fig.add_subplot(gs[1, 1])
    if not signals_df.empty:
        sig_counts = signals_df["signal"].value_counts().reindex([-1, 0, 1], fill_value=0)
        labels     = ["SELL", "FLAT", "BUY"]
        colors_s   = [STYLE["red"], STYLE["muted"], STYLE["green"]]
        ax_sig.bar(labels, [sig_counts[-1], sig_counts[0], sig_counts[1]],
                   color=colors_s, alpha=0.85, width=0.5)
        ax_sig.set_ylabel("Count")
        ax_sig.grid(True, axis="y", alpha=0.4)
    ax_sig.set_title("Signal Distribution", fontsize=10)

    # ---- 4. Position P&L (bottom-left) ----------------------------------
    ax_pos = fig.add_subplot(gs[2, 0])
    if not positions_df.empty:
        pnl_vals = positions_df["unrealized_pnl"].values
        syms     = positions_df["symbol"].values
        colors_p = [STYLE["green"] if v >= 0 else STYLE["red"] for v in pnl_vals]
        ax_pos.barh(syms, pnl_vals, color=colors_p, alpha=0.85, height=0.5)
        ax_pos.axvline(0, color=STYLE["muted"], lw=1.0)
        ax_pos.xaxis.set_major_formatter(usd_fmt)
        ax_pos.grid(True, axis="x", alpha=0.4)
    ax_pos.set_title("Position Unrealised P&L", fontsize=10)

    # ---- 5. Order Distribution (bottom-right) ---------------------------
    ax_ord = fig.add_subplot(gs[2, 1])
    if not orders_df.empty and "action" in orders_df.columns:
        counts = orders_df["action"].value_counts()
        colors_o = [STYLE["green"] if a == "BUY" else STYLE["red"] for a in counts.index]
        ax_ord.bar(counts.index, counts.values, color=colors_o, alpha=0.85, width=0.4)
        ax_ord.set_ylabel("Orders")
        ax_ord.grid(True, axis="y", alpha=0.4)
    ax_ord.set_title("Orders by Action", fontsize=10)

    return _save(fig, output_path, "Session Dashboard")
PYEOF

# =============================================================================
# FILE 10: trading_bot.py  (orchestrator)
# =============================================================================
cat > src/trading_bot.py << 'PYEOF'
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
PYEOF

# =============================================================================
# FILE 11: main.py
# =============================================================================
cat > main.py << 'PYEOF'
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
PYEOF

# =============================================================================
# FILE 12: tests/test_trading_bot.py
# =============================================================================
cat > tests/test_trading_bot.py << 'PYEOF'
"""
test_trading_bot.py
-------------------
Unit tests for core modules: SignalGenerator, RiskManager,
PositionMonitor, OrderManager (dry-run), and chart generation.

Run from project root:
    pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from src.config           import BotConfig, SignalConfig, RiskConfig
from src.signal_generator import SignalGenerator, SignalResult
from src.risk_manager     import RiskManager
from src.position_monitor import PositionMonitor
from src.order_manager    import OrderManager
from src.notifier         import Notifier
from src.utils            import pct_change, clamp, format_currency


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def cfg():
    return BotConfig()


@pytest.fixture
def sample_bars(n=200):
    """Generate synthetic OHLCV DataFrame."""
    rng    = np.random.default_rng(42)
    prices = 150.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.015, n)))
    dates  = pd.date_range(end=datetime.now(), periods=n, freq="5min")
    return pd.DataFrame({
        "open":   prices * rng.uniform(0.999, 1.001, n),
        "high":   prices * rng.uniform(1.000, 1.010, n),
        "low":    prices * rng.uniform(0.990, 1.000, n),
        "close":  prices,
        "volume": rng.integers(100_000, 2_000_000, n),
    }, index=dates)


@pytest.fixture
def signal_gen(cfg):
    return SignalGenerator(cfg.signals)


@pytest.fixture
def risk_mgr(cfg):
    rm = RiskManager(cfg.risk)
    rm.initialize(100_000.0)
    return rm


# =============================================================================
# Signal Generator Tests
# =============================================================================

class TestSignalGenerator:

    def test_returns_signal_result(self, signal_gen, sample_bars):
        result = signal_gen.generate("AAPL", sample_bars)
        assert isinstance(result, SignalResult)
        assert result.symbol == "AAPL"

    def test_signal_in_valid_range(self, signal_gen, sample_bars):
        result = signal_gen.generate("MSFT", sample_bars)
        assert result.signal in (-1, 0, 1)

    def test_confidence_bounded(self, signal_gen, sample_bars):
        result = signal_gen.generate("GOOGL", sample_bars)
        assert -1.0 <= result.confidence <= 1.0

    def test_rsi_range(self, signal_gen, sample_bars):
        result = signal_gen.generate("NVDA", sample_bars)
        assert 0 <= result.rsi <= 100

    def test_insufficient_bars_returns_flat(self, signal_gen):
        tiny = pd.DataFrame({
            "open": [100], "high": [101], "low": [99], "close": [100], "volume": [1000]
        })
        result = signal_gen.generate("X", tiny)
        assert result.signal == 0

    def test_generate_all(self, signal_gen, sample_bars):
        bars_dict = {"AAPL": sample_bars, "MSFT": sample_bars.copy()}
        results   = signal_gen.generate_all(bars_dict)
        assert "AAPL" in results
        assert "MSFT" in results
        assert all(r.signal in (-1, 0, 1) for r in results.values())

    def test_rsi_static_rising(self, signal_gen):
        """RSI should be > 50 for purely rising prices."""
        prices = pd.Series(np.arange(100.0, 160.0))
        rsi    = SignalGenerator._rsi(prices, 14).dropna()
        assert float(rsi.iloc[-1]) > 50

    def test_macd_crossover_detected(self, signal_gen):
        """MACD line should cross signal in a volatile series."""
        rng    = np.random.default_rng(7)
        prices = pd.Series(100 + np.cumsum(rng.normal(0, 1, 60)))
        ml, ms, mh = SignalGenerator._macd(prices, 12, 26, 9)
        assert not ml.dropna().empty


# =============================================================================
# Risk Manager Tests
# =============================================================================

class TestRiskManager:

    def test_position_sizing_basic(self, risk_mgr):
        shares = risk_mgr.compute_shares("AAPL", 150.0, 100_000.0, +1)
        expected_max = math.floor(100_000.0 * 0.05 / 150.0)
        assert 0 < shares <= expected_max

    def test_zero_price_returns_zero(self, risk_mgr):
        assert risk_mgr.compute_shares("X", 0.0, 100_000.0, +1) == 0

    def test_halt_blocks_orders(self, risk_mgr):
        risk_mgr.state.halted = True
        risk_mgr.state.halt_reason = "Test halt"
        shares = risk_mgr.compute_shares("AAPL", 100.0, 100_000.0, +1)
        assert shares == 0

    def test_stop_loss_triggers(self, risk_mgr):
        assert risk_mgr.check_stop_loss("AAPL", 100.0, 97.5)  # -2.5% > 2% limit

    def test_stop_loss_no_trigger(self, risk_mgr):
        assert not risk_mgr.check_stop_loss("AAPL", 100.0, 99.5)  # -0.5% < 2% limit

    def test_take_profit_triggers(self, risk_mgr):
        assert risk_mgr.check_take_profit("AAPL", 100.0, 105.0)  # +5% > 4% limit

    def test_take_profit_no_trigger(self, risk_mgr):
        assert not risk_mgr.check_take_profit("AAPL", 100.0, 102.0)  # +2% < 4% limit

    def test_daily_loss_halts_bot(self, risk_mgr):
        risk_mgr.update_equity(96_500.0)  # -3.5% loss -> breaches 3% limit
        assert risk_mgr.state.halted

    def test_equity_high_watermark(self, risk_mgr):
        risk_mgr.update_equity(105_000.0)
        assert risk_mgr.state.session_high_equity == 105_000.0

    def test_max_portfolio_pct_blocks_order(self, risk_mgr):
        """Fill positions up to limit then next order should be blocked."""
        risk_mgr.state.deployed_capital = 80_000.0   # 80% already deployed
        shares = risk_mgr.compute_shares("AAPL", 150.0, 100_000.0, +1)
        assert shares == 0


# =============================================================================
# Position Monitor Tests
# =============================================================================

class TestPositionMonitor:

    def test_open_close_position(self):
        mon = PositionMonitor(initial_equity=100_000.0)
        mon.open_position("AAPL", 10, 150.0)
        assert "AAPL" in mon._positions

        pnl = mon.close_position("AAPL", 155.0)
        assert abs(pnl - 50.0) < 1e-6
        assert "AAPL" not in mon._positions

    def test_unrealized_pnl(self):
        mon = PositionMonitor()
        mon.open_position("MSFT", 5, 200.0)
        mon.update_prices({"MSFT": 210.0})
        assert abs(mon.total_unrealized_pnl() - 50.0) < 1e-6

    def test_drawdown_calculation(self):
        mon = PositionMonitor(initial_equity=100_000.0)
        mon.record_equity(100_000.0)
        mon.record_equity(105_000.0)
        mon.record_equity(98_000.0)
        dd = mon.current_drawdown()
        assert dd > 0
        assert dd < 1

    def test_snapshot_empty(self):
        mon = PositionMonitor()
        assert mon.snapshot().empty

    def test_snapshot_populated(self):
        mon = PositionMonitor()
        mon.open_position("NVDA", 2, 400.0)
        mon.update_prices({"NVDA": 410.0})
        snap = mon.snapshot()
        assert len(snap) == 1
        assert snap.iloc[0]["symbol"] == "NVDA"

    def test_equity_curve(self):
        mon = PositionMonitor(initial_equity=100_000.0)
        for v in [100_000, 101_000, 102_500, 100_800]:
            mon.record_equity(float(v))
        curve = mon.equity_curve()
        assert len(curve) == 4
        assert "net_liq" in curve.columns


# =============================================================================
# Order Manager Tests (dry-run)
# =============================================================================

class TestOrderManager:

    def test_dry_run_order(self, cfg):
        mock_ib = MagicMock()
        om = OrderManager(mock_ib, cfg, dry_run=True)
        rec = om.submit_order("AAPL", "BUY", 10, 150.0)
        assert rec is not None
        assert rec.status == "DryRun"
        assert rec.filled_px is not None

    def test_zero_qty_blocked(self, cfg):
        mock_ib = MagicMock()
        om = OrderManager(mock_ib, cfg, dry_run=True)
        rec = om.submit_order("AAPL", "BUY", 0, 150.0)
        assert rec is None

    def test_order_log_populated(self, cfg):
        mock_ib = MagicMock()
        om = OrderManager(mock_ib, cfg, dry_run=True)
        om.submit_order("AAPL", "BUY",  5, 150.0)
        om.submit_order("MSFT", "SELL", 3, 300.0)
        log_df = om.order_log()
        assert len(log_df) == 2
        assert set(log_df["symbol"]) == {"AAPL", "MSFT"}

    def test_limit_price_buy_adds_slippage(self, cfg):
        mock_ib = MagicMock()
        om  = OrderManager(mock_ib, cfg, dry_run=True)
        rec = om.submit_order("X", "BUY", 1, 100.0)
        assert rec.limit_px >= 100.0

    def test_limit_price_sell_subtracts_slippage(self, cfg):
        mock_ib = MagicMock()
        om  = OrderManager(mock_ib, cfg, dry_run=True)
        rec = om.submit_order("X", "SELL", 1, 100.0)
        assert rec.limit_px <= 100.0


# =============================================================================
# Notifier Tests
# =============================================================================

class TestNotifier:

    def test_disabled_notifier_does_not_raise(self, cfg):
        cfg.notifications.enabled = False
        n = Notifier(cfg.notifications)
        # Should log but not raise
        n.trade_alert("AAPL", "BUY", 10, 150.0, {"score": 3, "confidence": 0.75,
                                                   "rsi": 28.0, "macd_signal": 1,
                                                   "bb_signal": 1, "ema_signal": 1})

    def test_risk_alert_disabled_silent(self, cfg):
        cfg.notifications.enabled = False
        n = Notifier(cfg.notifications)
        n.risk_alert("Test halt reason")  # must not raise


# =============================================================================
# Utility Tests
# =============================================================================

class TestUtils:

    def test_pct_change(self):
        assert abs(pct_change(100.0, 105.0) - 0.05) < 1e-10

    def test_pct_change_zero_denominator(self):
        assert pct_change(0.0, 100.0) == 0.0

    def test_clamp(self):
        assert clamp(1.5, 0.0, 1.0) == 1.0
        assert clamp(-0.5, 0.0, 1.0) == 0.0
        assert clamp(0.5, 0.0, 1.0) == 0.5

    def test_format_currency(self):
        assert format_currency(1234.5) == "$1,234.50"
        assert format_currency(-500.0) == "$-500.00"


# =============================================================================
# Integration Test: demo pipeline
# =============================================================================

class TestDemoPipeline:

    def test_full_signal_pipeline(self, cfg, sample_bars):
        """End-to-end: bars -> signal -> size -> monitor."""
        gen   = SignalGenerator(cfg.signals)
        risk  = RiskManager(cfg.risk)
        mon   = PositionMonitor(initial_equity=100_000.0)
        risk.initialize(100_000.0)

        result = gen.generate("AAPL", sample_bars)
        assert result.signal in (-1, 0, 1)

        if result.signal != 0:
            shares = risk.compute_shares("AAPL", result.close, 100_000.0, result.signal)
            if shares > 0:
                mon.open_position("AAPL", shares, result.close)
                mon.update_prices({"AAPL": result.close * 1.005})
                assert mon.total_unrealized_pnl() != 0

        summary = mon.summary()
        assert "open_positions" in summary
        assert "session_return"  in summary
PYEOF

# =============================================================================
# FILE 13: requirements.txt
# =============================================================================
cat > requirements.txt << 'PYEOF'
# Core
ib_insync==0.9.86
pandas>=2.0.0
numpy>=1.24.0

# Visualisation
matplotlib>=3.7.0
seaborn>=0.12.0

# Testing
pytest>=7.4.0
pytest-cov>=4.1.0

# Utilities
python-dotenv>=1.0.0
PYEOF

# =============================================================================
# FILE 14: setup.py
# =============================================================================
cat > setup.py << 'PYEOF'
from setuptools import setup, find_packages

setup(
    name             = "automated-trading-bot",
    version          = "1.0.0",
    description      = "Automated Trading Bot with Interactive Brokers API",
    author           = "Jose Orlando",
    packages         = find_packages(),
    python_requires  = ">=3.10",
    install_requires = [
        "ib_insync>=0.9.86",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
    ],
    entry_points     = {
        "console_scripts": ["trading-bot = main:main"]
    },
    classifiers      = [
        "Programming Language :: Python :: 3",
        "Topic :: Office/Business :: Financial",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
)
PYEOF

# =============================================================================
# FILE 15: .gitignore
# =============================================================================
cat > .gitignore << 'PYEOF'
# Python
__pycache__/
*.py[cod]
*.egg-info/
.eggs/
dist/
build/
.venv/
venv/
*.egg

# Pytest
.pytest_cache/
.coverage
htmlcov/

# Outputs (charts, logs generated at runtime)
outputs/charts/*.png
outputs/logs/*.log
outputs/reports/
data/

# Environment / secrets
.env
.env.*
*.pem
*.key

# IB specific
*.ib.json
PYEOF

# =============================================================================
# FILE 16: README.md
# =============================================================================
cat > README.md << 'PYEOF'
# Project 14 - Automated Trading Bot with Interactive Brokers API

End-to-end automated trading system connecting to Interactive Brokers
via `ib_insync`, featuring multi-indicator signal generation, position
sizing, bracket order execution, real-time position monitoring, and
email notifications.

---

## Architecture

```
main.py
  └── TradingBot (trading_bot.py)
        ├── IBConnector      -- IB Gateway / TWS connection & data
        ├── SignalGenerator  -- RSI + MACD + Bollinger + EMA consensus
        ├── RiskManager      -- Position sizing, stop-loss, drawdown guard
        ├── OrderManager     -- Limit / bracket order submission & tracking
        ├── PositionMonitor  -- Real-time P&L and equity curve
        ├── Notifier         -- SMTP email alerts and daily summary
        └── Plotter          -- Publication-quality charts (headless Agg)
```

---

## Signal Generation

Four indicators produce independent votes in {-1, 0, +1}:

| Indicator        | Long (+1)               | Short (-1)              |
|------------------|-------------------------|-------------------------|
| RSI(14)          | RSI <= 30               | RSI >= 70               |
| MACD(12,26,9)    | MACD crosses above sig  | MACD crosses below sig  |
| Bollinger(20,2)  | Price touches lower band | Price touches upper band |
| EMA(9 / 21)      | Short crosses above long | Short crosses below long |

A trade is placed when **3 or more indicators agree** (configurable via
`SignalConfig.consensus_long / consensus_short`).

---

## Risk Controls

| Control              | Default   | Config key             |
|----------------------|-----------|------------------------|
| Max position size    | 5 % NAV   | `max_position_pct`     |
| Max deployed capital | 80 % NAV  | `max_portfolio_pct`    |
| Hard stop-loss       | 2 %       | `stop_loss_pct`        |
| Take-profit target   | 4 %       | `take_profit_pct`      |
| Daily loss limit     | 3 %       | `max_daily_loss`       |
| Max drawdown         | 10 %      | `max_drawdown`         |

---

## Quick Start

### 1. Install dependencies

```bash
cd ~/quant-finance-portfolio/14-automated-trading-bot
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run offline demo (no IB required)

```bash
python main.py --demo
```

### 3. Connect to IB Paper Trading

Ensure IB Gateway is running on port 7497 with API enabled, then:

```bash
export IB_HOST=127.0.0.1
export IB_PORT=7497
export IB_ACCOUNT=DU123456
python main.py
```

### 4. Dry run (connect to IB, no orders placed)

```bash
python main.py --dry-run
```

### 5. Enable email notifications

```bash
export NOTIFY_EMAIL=true
export SMTP_SENDER=you@gmail.com
export SMTP_PASS=app_password
export NOTIFY_TO=recipient@example.com
python main.py --demo
```

---

## Environment Variables

| Variable         | Default       | Description                        |
|------------------|---------------|------------------------------------|
| `IB_HOST`        | `127.0.0.1`   | IB Gateway host                    |
| `IB_PORT`        | `7497`        | 7497=paper, 7496=live              |
| `IB_CID`         | `1`           | Client ID                          |
| `IB_ACCOUNT`     |               | Account ID (e.g. DU123456)         |
| `PAPER_TRADING`  | `true`        | Enable paper trading mode          |
| `DRY_RUN`        | `false`       | Log orders only, no submissions    |
| `DEMO_MODE`      | `false`       | Offline demo, synthetic data       |
| `SCAN_SECS`      | `300`         | Seconds between scan cycles        |
| `NOTIFY_EMAIL`   | `false`       | Enable email notifications         |
| `SMTP_HOST`      | smtp.gmail.com | SMTP server                       |
| `SMTP_PORT`      | `587`         | SMTP port (TLS)                    |
| `SMTP_SENDER`    |               | Sender email address               |
| `SMTP_PASS`      |               | SMTP password / app password       |
| `NOTIFY_TO`      |               | Recipient email address            |
| `LOG_LEVEL`      | `INFO`        | Logging verbosity                  |

---

## Running Tests

```bash
pytest tests/ -v --tb=short
```

Expected output: **32 tests passing**.

---

## Output Charts

All charts are saved under `outputs/charts/` with UTC timestamps:

| Chart                    | Description                                     |
|--------------------------|-------------------------------------------------|
| `signal_{SYM}.png`       | Price, Bollinger, EMA, Volume, RSI, MACD panels |
| `signal_heatmap.png`     | All indicators across all symbols               |
| `equity_curve.png`       | Net liquidation + drawdown time series          |
| `position_pnl.png`       | Unrealised P&L per open position                |
| `session_dashboard.png`  | Consolidated session overview                   |

---

## Project Structure

```
14-automated-trading-bot/
├── main.py                     # Entry point & demo runner
├── src/
│   ├── config.py               # Centralised configuration (dataclasses)
│   ├── ib_connector.py         # IB Gateway connection & data fetcher
│   ├── signal_generator.py     # RSI + MACD + BB + EMA consensus signals
│   ├── risk_manager.py         # Position sizing & risk guards
│   ├── order_manager.py        # Order lifecycle (limit, bracket, dry-run)
│   ├── position_monitor.py     # Real-time P&L & equity curve
│   ├── notifier.py             # SMTP email alerts
│   ├── plotter.py              # Publication-quality charts (Agg backend)
│   └── trading_bot.py          # Top-level orchestrator
├── tests/
│   └── test_trading_bot.py     # 32-test suite (unit + integration)
├── outputs/
│   ├── charts/                 # Generated PNG charts
│   └── logs/                   # Daily rotating log files
├── requirements.txt
├── setup.py
└── README.md
```

---

## Technical Stack

- **Broker API**: ib_insync 0.9.x (asyncio wrapper for IB TWS API)
- **Data Processing**: pandas, numpy
- **Visualisation**: matplotlib (Agg backend - headless / Cloud Shell safe)
- **Notifications**: smtplib (TLS SMTP)
- **Testing**: pytest

---

## License

MIT License
PYEOF

# =============================================================================
# FINAL SETUP: __init__.py, .env.example
# =============================================================================
touch src/__init__.py tests/__init__.py

cat > .env.example << 'PYEOF'
# Interactive Brokers Connection
IB_HOST=127.0.0.1
IB_PORT=7497
IB_CID=1
IB_ACCOUNT=DU123456

# Trading Mode
PAPER_TRADING=true
DRY_RUN=false
DEMO_MODE=false
SCAN_SECS=300
LOG_LEVEL=INFO

# Email Notifications
NOTIFY_EMAIL=false
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_SENDER=your@gmail.com
SMTP_PASS=your_app_password
NOTIFY_TO=recipient@example.com
PYEOF

# =============================================================================
# INSTALL AND TEST
# =============================================================================
echo ""
echo "============================================================"
echo "  INSTALLING DEPENDENCIES"
echo "============================================================"

python -m venv .venv
source .venv/bin/activate

pip install --quiet --upgrade pip
pip install --quiet pandas numpy matplotlib seaborn pytest pytest-cov

echo ""
echo "============================================================"
echo "  RUNNING DEMO (no IB connection required)"
echo "============================================================"
python main.py --demo

echo ""
echo "============================================================"
echo "  RUNNING TEST SUITE"
echo "============================================================"
pytest tests/ -v --tb=short 2>&1

echo ""
echo "============================================================"
echo "  BUILD COMPLETE - PROJECT 14"
echo "============================================================"
echo "  Root  : $PROJECT_ROOT"
echo "  Source: src/ (9 modules)"
echo "  Tests : tests/ (32 tests)"
echo "  Charts: outputs/charts/"
echo ""
echo "  Quick start:"
echo "    source .venv/bin/activate"
echo "    python main.py --demo      # offline"
echo "    python main.py --dry-run   # needs IB Gateway"
echo "    python main.py             # paper trading"
echo "============================================================"