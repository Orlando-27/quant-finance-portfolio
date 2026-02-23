"""
================================================================================
SENTIMENT-BASED LONG-SHORT TRADING STRATEGY
================================================================================
Implements a systematic long-short equity strategy driven by sentiment signals.

Strategy Logic:
    1. Rank assets by composite sentiment signal on each rebalancing date
    2. Long top quintile (most positive sentiment)
    3. Short bottom quintile (most negative sentiment)
    4. Weight positions by inverse-volatility for risk parity within legs
    5. Apply volatility targeting to scale overall exposure
    6. Enforce position limits and turnover constraints

Risk Management:
    - Volatility targeting: scale exposure so portfolio vol = target
    - Max single-name position: 20% of leg
    - Drawdown-based deleveraging: reduce exposure if drawdown > threshold
    - Turnover penalty: penalize excessive trading

References:
    - DeMiguel, V. et al. (2009). Optimal Versus Naive Diversification.
      Review of Financial Studies.
    - Moskowitz, T. et al. (2012). Time Series Momentum.
      Journal of Financial Economics.

Author: Jose Orlando Bobadilla Fuentes, CQF | MSc AI
================================================================================
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple


class SentimentStrategy:
    """
    Long-short equity strategy driven by sentiment signals.

    Parameters
    ----------
    long_pct : float
        Percentile threshold for long leg (default 0.80 = top 20%).
    short_pct : float
        Percentile threshold for short leg (default 0.20 = bottom 20%).
    vol_target : float
        Annualized volatility target for the portfolio (default 0.10 = 10%).
    vol_lookback : int
        Days for realized volatility estimation (default 21).
    max_position : float
        Maximum weight for any single name within a leg (default 0.20).
    rebalance_freq : str
        Rebalancing frequency: 'daily', 'weekly', 'monthly'.
    transaction_cost : float
        Round-trip transaction cost in basis points (default 10 bps = 0.001).
    drawdown_threshold : float
        Maximum drawdown before deleveraging kicks in (default 0.10 = 10%).
    deleveraging_factor : float
        Exposure multiplier when drawdown exceeds threshold (default 0.5).
    """

    def __init__(
        self,
        long_pct: float = 0.80,
        short_pct: float = 0.20,
        vol_target: float = 0.10,
        vol_lookback: int = 21,
        max_position: float = 0.20,
        rebalance_freq: str = "daily",
        transaction_cost: float = 0.001,
        drawdown_threshold: float = 0.10,
        deleveraging_factor: float = 0.50,
    ):
        self.long_pct = long_pct
        self.short_pct = short_pct
        self.vol_target = vol_target
        self.vol_lookback = vol_lookback
        self.max_position = max_position
        self.rebalance_freq = rebalance_freq
        self.transaction_cost = transaction_cost
        self.drawdown_threshold = drawdown_threshold
        self.deleveraging_factor = deleveraging_factor

    def _get_rebalance_dates(self, dates: pd.DatetimeIndex) -> set:
        """Determine which dates trigger rebalancing."""
        if self.rebalance_freq == "daily":
            return set(dates)
        elif self.rebalance_freq == "weekly":
            # Rebalance on Fridays (weekday=4) or last available day of week
            return set(dates[dates.weekday == 4])
        elif self.rebalance_freq == "monthly":
            # Last business day of each month
            return set(dates.to_series().groupby(dates.to_period("M")).last())
        else:
            return set(dates)

    def _inverse_vol_weights(
        self,
        returns: pd.DataFrame,
        tickers: list,
        date: pd.Timestamp,
    ) -> Dict[str, float]:
        """
        Compute inverse-volatility weights for a set of tickers.

        w_i = (1/sigma_i) / sum(1/sigma_j)

        This ensures that each asset contributes equally to portfolio risk,
        following the risk parity principle within each leg.
        """
        lookback_start = date - pd.Timedelta(days=self.vol_lookback * 2)
        subset = returns.loc[lookback_start:date, tickers]
        vols = subset.std() * np.sqrt(252)
        vols = vols.replace(0, np.nan).dropna()

        if len(vols) == 0:
            # Equal weight fallback
            return {t: 1.0 / len(tickers) for t in tickers}

        inv_vol = 1.0 / vols
        inv_vol = inv_vol.clip(upper=inv_vol.quantile(0.95))  # Cap outliers
        weights = inv_vol / inv_vol.sum()

        # Enforce max position constraint
        weights = weights.clip(upper=self.max_position)
        weights = weights / weights.sum()

        return weights.to_dict()

    def generate_positions(
        self,
        signals: pd.DataFrame,
        returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate daily position weights from sentiment signals and price data.

        Parameters
        ----------
        signals : pd.DataFrame
            Must contain columns: 'date', 'ticker', 'composite_signal', 'cs_rank'.
        returns : pd.DataFrame
            Daily log returns matrix (dates x tickers).

        Returns
        -------
        pd.DataFrame
            Position weights matrix (dates x tickers), dollar-neutral
            (long weights sum to +1, short weights sum to -1 before
            volatility scaling).
        """
        tickers = sorted(returns.columns.tolist())
        dates = sorted(signals["date"].unique())
        rebal_dates = self._get_rebalance_dates(pd.DatetimeIndex(dates))

        # Initialize positions and tracking variables
        positions = pd.DataFrame(0.0, index=returns.index, columns=tickers)
        current_weights = pd.Series(0.0, index=tickers)
        cum_pnl = pd.Series(0.0, index=returns.index)
        peak = 0.0

        for i, date in enumerate(returns.index):
            if date not in [pd.Timestamp(d) for d in dates]:
                positions.loc[date] = current_weights
                continue

            # --- Rebalancing Logic ---
            if date in rebal_dates or pd.Timestamp(date.date()) in rebal_dates:
                day_signals = signals[signals["date"] == date.date()]
                if day_signals.empty:
                    day_signals = signals[
                        signals["date"] == pd.Timestamp(date).normalize()
                    ]
                if day_signals.empty:
                    positions.loc[date] = current_weights
                    continue

                # Identify long and short legs based on cross-sectional rank
                long_mask = day_signals["cs_rank"] >= self.long_pct
                short_mask = day_signals["cs_rank"] <= self.short_pct

                long_tickers = [
                    t for t in day_signals.loc[long_mask, "ticker"]
                    if t in tickers
                ]
                short_tickers = [
                    t for t in day_signals.loc[short_mask, "ticker"]
                    if t in tickers
                ]

                # Compute inverse-volatility weights within each leg
                new_weights = pd.Series(0.0, index=tickers)
                if long_tickers:
                    lw = self._inverse_vol_weights(returns, long_tickers, date)
                    for t, w in lw.items():
                        new_weights[t] = w
                if short_tickers:
                    sw = self._inverse_vol_weights(returns, short_tickers, date)
                    for t, w in sw.items():
                        new_weights[t] = -w

                # Volatility targeting
                port_ret = (returns.loc[:date].tail(self.vol_lookback) * current_weights).sum(axis=1)
                realized_vol = port_ret.std() * np.sqrt(252) if len(port_ret) > 5 else self.vol_target
                if realized_vol > 0:
                    vol_scalar = self.vol_target / realized_vol
                    vol_scalar = np.clip(vol_scalar, 0.25, 4.0)  # Bound scaling
                else:
                    vol_scalar = 1.0

                new_weights *= vol_scalar

                # Drawdown-based deleveraging
                if i > 0:
                    cum_pnl.iloc[i] = cum_pnl.iloc[i - 1] + (
                        returns.iloc[i] * current_weights
                    ).sum()
                    peak = max(peak, cum_pnl.iloc[i])
                    drawdown = peak - cum_pnl.iloc[i]
                    if drawdown > self.drawdown_threshold:
                        new_weights *= self.deleveraging_factor

                current_weights = new_weights

            positions.loc[date] = current_weights

        return positions

    def compute_turnover(self, positions: pd.DataFrame) -> pd.Series:
        """
        Compute daily portfolio turnover.

        Turnover_t = 0.5 * sum(|w_{t,i} - w_{t-1,i}|)
        """
        diffs = positions.diff().abs()
        return diffs.sum(axis=1) * 0.5

    def compute_transaction_costs(
        self,
        positions: pd.DataFrame,
    ) -> pd.Series:
        """
        Compute daily transaction costs from turnover.

        TC_t = turnover_t * transaction_cost_rate
        """
        turnover = self.compute_turnover(positions)
        return turnover * self.transaction_cost
