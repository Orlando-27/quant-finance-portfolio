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
