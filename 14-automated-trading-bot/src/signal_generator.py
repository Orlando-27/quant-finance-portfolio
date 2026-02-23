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
