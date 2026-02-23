"""
Pairs Trading Strategy: Signal Generation & Position Management
================================================================

Implements the z-score based trading strategy for pairs/spread trading
with configurable entry, exit, and stop-loss thresholds.

Signal logic:
    z_t = (S_t - mu_rolling) / sigma_rolling

    Entry long spread:   z_t < -z_entry  (spread is "cheap")
    Entry short spread:  z_t > +z_entry  (spread is "rich")
    Exit position:       |z_t| < z_exit  (spread has reverted)
    Stop-loss:           |z_t| > z_stop  (spread is diverging)

Position sizing follows a linear z-score scaling:
    size = min(|z_t| / z_entry, max_leverage)

The strategy supports both static (OLS) and adaptive (Kalman) hedge
ratios for spread construction.

References:
    Gatev et al. (2006), Avellaneda & Lee (2010), Vidyamurthy (2004)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


class PairsTradingStrategy:
    """
    Z-score based pairs trading strategy.

    Parameters
    ----------
    z_entry : float
        Z-score threshold for entering positions (default 2.0).
    z_exit : float
        Z-score threshold for exiting positions (default 0.5).
    z_stop : float
        Z-score threshold for stop-loss (default 4.0).
    lookback : int
        Rolling window for z-score computation (default 60 days).
    max_leverage : float
        Maximum position size multiplier (default 1.0).
    hedge_ratio_method : str
        'static' for OLS or 'kalman' for adaptive hedge ratio.
    """

    def __init__(self, z_entry: float = 2.0, z_exit: float = 0.5,
                 z_stop: float = 4.0, lookback: int = 60,
                 max_leverage: float = 1.0,
                 hedge_ratio_method: str = "static"):
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.z_stop = z_stop
        self.lookback = lookback
        self.max_lev = max_leverage
        self.hr_method = hedge_ratio_method
        self.signals = None
        self.positions = None

    def compute_zscore(self, spread: pd.Series) -> pd.Series:
        """
        Compute the rolling z-score of the spread.

        z_t = (S_t - mean(S, lookback)) / std(S, lookback)

        Parameters
        ----------
        spread : pd.Series
            Time series of spread values.

        Returns
        -------
        pd.Series
            Rolling z-score series.
        """
        mu = spread.rolling(self.lookback, min_periods=20).mean()
        sigma = spread.rolling(self.lookback, min_periods=20).std()
        sigma = sigma.replace(0, np.nan)
        z = (spread - mu) / sigma
        z.name = "zscore"
        return z

    def generate_signals(self, spread: pd.Series) -> pd.DataFrame:
        """
        Generate trading signals from spread z-score.

        Signal values:
            +1 = long spread (long A, short beta*B)
            -1 = short spread (short A, long beta*B)
             0 = no position / flat

        The signal transitions follow a state machine:
            FLAT -> LONG:   z < -z_entry
            FLAT -> SHORT:  z > +z_entry
            LONG -> FLAT:   z > -z_exit (reverted) or z < -z_stop (stop)
            SHORT -> FLAT:  z < +z_exit (reverted) or z > +z_stop (stop)

        Parameters
        ----------
        spread : pd.Series
            Spread time series.

        Returns
        -------
        pd.DataFrame
            Columns: spread, zscore, raw_signal, position, entry_type.
        """
        z = self.compute_zscore(spread)
        T = len(z)
        positions = np.zeros(T)
        entry_types = ["flat"] * T

        current_pos = 0  # 0 = flat, 1 = long spread, -1 = short spread

        for t in range(1, T):
            z_t = z.iloc[t]
            if np.isnan(z_t):
                positions[t] = current_pos
                continue

            if current_pos == 0:
                # Check for entry
                if z_t < -self.z_entry:
                    current_pos = 1  # Long spread
                    entry_types[t] = "long_entry"
                elif z_t > self.z_entry:
                    current_pos = -1  # Short spread
                    entry_types[t] = "short_entry"
            elif current_pos == 1:
                # Long position: exit on reversion or stop-loss
                if z_t > -self.z_exit:
                    current_pos = 0
                    entry_types[t] = "long_exit_revert"
                elif z_t < -self.z_stop:
                    current_pos = 0
                    entry_types[t] = "long_exit_stop"
            elif current_pos == -1:
                # Short position: exit on reversion or stop-loss
                if z_t < self.z_exit:
                    current_pos = 0
                    entry_types[t] = "short_exit_revert"
                elif z_t > self.z_stop:
                    current_pos = 0
                    entry_types[t] = "short_exit_stop"

            positions[t] = current_pos

        signals_df = pd.DataFrame({
            "spread": spread,
            "zscore": z,
            "position": positions,
            "entry_type": entry_types,
        }, index=spread.index)

        self.signals = signals_df
        return signals_df

    def compute_returns(self, signals_df: pd.DataFrame,
                        price_a: pd.Series, price_b: pd.Series,
                        hedge_ratio: float,
                        transaction_cost_bps: float = 10.0) -> pd.DataFrame:
        """
        Compute strategy returns from signals and prices.

        Long spread return:  r_t = ret_A,t - beta * ret_B,t
        Short spread return: r_t = -(ret_A,t - beta * ret_B,t)

        Parameters
        ----------
        signals_df : pd.DataFrame
            Output from generate_signals().
        price_a : pd.Series
            Price series of asset A.
        price_b : pd.Series
            Price series of asset B.
        hedge_ratio : float
            Hedge ratio (beta).
        transaction_cost_bps : float
            One-way transaction cost in basis points.

        Returns
        -------
        pd.DataFrame
            Strategy returns with columns: spread_return, position,
            gross_return, transaction_cost, net_return, cumulative_return.
        """
        common = signals_df.index
        ret_a = price_a.reindex(common).pct_change()
        ret_b = price_b.reindex(common).pct_change()

        # Spread return: long A, short beta*B
        spread_ret = ret_a - hedge_ratio * ret_b

        # Strategy return = position * spread return
        positions = signals_df["position"]
        gross_ret = positions.shift(1) * spread_ret  # lag position by 1

        # Transaction costs on position changes
        pos_change = positions.diff().abs()
        tc = pos_change * (transaction_cost_bps / 10000) * (1 + abs(hedge_ratio))

        net_ret = gross_ret - tc
        net_ret = net_ret.fillna(0)

        result = pd.DataFrame({
            "spread_return": spread_ret,
            "position": positions,
            "gross_return": gross_ret,
            "transaction_cost": tc,
            "net_return": net_ret,
        }, index=common)
        result["cumulative_return"] = (1 + result["net_return"]).cumprod()

        self.positions = result
        return result

    def trade_statistics(self) -> Dict:
        """
        Compute trade-level statistics from the strategy.

        Returns
        -------
        dict
            n_trades, n_long, n_short, avg_holding_period,
            win_rate, avg_win, avg_loss, profit_factor,
            n_stop_losses, stop_loss_pct.
        """
        if self.signals is None:
            return {}

        sig = self.signals
        entries = sig[sig["entry_type"].str.contains("entry", na=False)]
        exits = sig[sig["entry_type"].str.contains("exit", na=False)]
        stops = sig[sig["entry_type"].str.contains("stop", na=False)]

        n_trades = len(entries)
        n_long = len(sig[sig["entry_type"] == "long_entry"])
        n_short = len(sig[sig["entry_type"] == "short_entry"])
        n_stops = len(stops)

        # Holding periods
        positions = sig["position"]
        in_trade = positions != 0
        trade_groups = (in_trade != in_trade.shift()).cumsum()
        trade_lengths = []
        for _, grp in sig[in_trade].groupby(trade_groups[in_trade]):
            trade_lengths.append(len(grp))

        avg_hold = np.mean(trade_lengths) if trade_lengths else 0

        # Win/loss from position returns
        if self.positions is not None:
            trade_rets = []
            for _, grp in self.positions[in_trade.reindex(
                self.positions.index, fill_value=False
            )].groupby(trade_groups.reindex(
                self.positions.index, fill_value=0
            )):
                trade_rets.append(grp["net_return"].sum())

            wins = [r for r in trade_rets if r > 0]
            losses = [r for r in trade_rets if r <= 0]
            win_rate = len(wins) / len(trade_rets) if trade_rets else 0
            avg_win = np.mean(wins) if wins else 0
            avg_loss = np.mean(losses) if losses else 0
            gross_wins = sum(wins)
            gross_losses = abs(sum(losses))
            pf = gross_wins / gross_losses if gross_losses > 0 else np.inf
        else:
            win_rate, avg_win, avg_loss, pf = 0, 0, 0, 0

        return {
            "n_trades": n_trades,
            "n_long": n_long,
            "n_short": n_short,
            "avg_holding_days": avg_hold,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": pf,
            "n_stop_losses": n_stops,
            "stop_loss_pct": n_stops / max(n_trades, 1) * 100,
        }
