"""
Walk-Forward Pairs Trading Backtest Engine
==========================================

Implements a rigorous backtesting framework with temporal separation
between formation and trading periods to prevent in-sample overfitting.

Architecture:
    1. Formation Period (F months): Identify pairs, estimate parameters.
    2. Trading Period (T months): Trade identified pairs with fixed params.
    3. Roll forward: Shift window by T months and repeat.

    This walk-forward structure ensures that pair selection and parameter
    estimation use only past data, mimicking real trading conditions.

    At each formation period:
    - Run pair selection (distance or cointegration method)
    - Estimate hedge ratios (OLS or Kalman)
    - Calibrate OU parameters (half-life, mean, volatility)
    - Set z-score thresholds

    During each trading period:
    - Generate z-score signals using formation-period parameters
    - Execute trades with position limits and stop-losses
    - Record returns, turnover, and costs

References:
    Gatev et al. (2006), Do & Faff (2010), Krauss (2017)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from src.pair_selection import PairSelector
from src.cointegration import EngleGranger
from src.ornstein_uhlenbeck import OrnsteinUhlenbeck
from src.kalman_filter import KalmanHedgeRatio
from src.strategy import PairsTradingStrategy


class PairsBacktester:
    """
    Walk-forward backtesting engine for pairs trading strategies.

    Parameters
    ----------
    formation_period : int
        Length of formation (pair selection) period in trading days.
    trading_period : int
        Length of trading period in trading days.
    n_pairs : int
        Number of pairs to trade simultaneously.
    z_entry : float
        Z-score entry threshold.
    z_exit : float
        Z-score exit threshold.
    z_stop : float
        Z-score stop-loss threshold.
    transaction_cost_bps : float
        One-way transaction cost in basis points.
    hedge_method : str
        'ols' for static or 'kalman' for adaptive hedge ratio.
    """

    def __init__(self, formation_period: int = 252,
                 trading_period: int = 126,
                 n_pairs: int = 5,
                 z_entry: float = 2.0,
                 z_exit: float = 0.5,
                 z_stop: float = 4.0,
                 transaction_cost_bps: float = 10.0,
                 hedge_method: str = "ols"):
        self.form_len = formation_period
        self.trade_len = trading_period
        self.n_pairs = n_pairs
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.z_stop = z_stop
        self.tc_bps = transaction_cost_bps
        self.hedge_method = hedge_method
        self.period_results = []
        self.portfolio_returns = None

    def run(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Execute the full walk-forward backtest.

        Parameters
        ----------
        prices : pd.DataFrame
            (T x N) price panel for the full universe.

        Returns
        -------
        pd.DataFrame
            Portfolio-level daily returns with columns:
            date, portfolio_return, n_active_pairs, period_id.
        """
        T = len(prices)
        step = self.trade_len
        min_start = self.form_len

        all_returns = []
        period_id = 0

        t = min_start
        while t + self.trade_len <= T:
            period_id += 1
            form_start = t - self.form_len
            form_end = t
            trade_start = t
            trade_end = min(t + self.trade_len, T)

            form_prices = prices.iloc[form_start:form_end]
            trade_prices = prices.iloc[trade_start:trade_end]

            # -- Formation: Select pairs --
            log_form = np.log(form_prices)
            selector = PairSelector(
                method="cointegration", significance=0.05,
                min_half_life=5, max_half_life=60,
                top_k=self.n_pairs
            )
            selected = selector.select(form_prices)

            if not selected:
                # No tradeable pairs found; flat portfolio
                flat_ret = pd.DataFrame({
                    "date": trade_prices.index,
                    "portfolio_return": 0.0,
                    "n_active_pairs": 0,
                    "period_id": period_id,
                })
                all_returns.append(flat_ret)
                t += step
                continue

            # -- Trading: Execute strategy for each pair --
            pair_returns = []

            for pair_info in selected:
                t1, t2 = pair_info["pair"]
                hr = pair_info["hedge_ratio"]

                if t1 not in trade_prices.columns or t2 not in trade_prices.columns:
                    continue

                pa = trade_prices[t1]
                pb = trade_prices[t2]

                # Construct spread
                if self.hedge_method == "kalman":
                    kf = KalmanHedgeRatio(delta=1e-4)
                    kf_res = kf.filter(np.log(pa), np.log(pb))
                    spread = kf_res["spreads"]
                    hr_final = kf_res["final_hedge_ratio"]
                else:
                    spread = np.log(pa) - hr * np.log(pb)
                    hr_final = hr

                # Generate signals
                strat = PairsTradingStrategy(
                    z_entry=self.z_entry, z_exit=self.z_exit,
                    z_stop=self.z_stop, lookback=min(60, self.form_len // 2)
                )
                signals = strat.generate_signals(spread)
                result = strat.compute_returns(
                    signals, pa, pb, hr_final, self.tc_bps
                )
                pair_returns.append(result["net_return"])

            # Equal-weight across pairs
            if pair_returns:
                pair_ret_df = pd.concat(pair_returns, axis=1).fillna(0)
                port_ret = pair_ret_df.mean(axis=1)
            else:
                port_ret = pd.Series(0.0, index=trade_prices.index)

            period_df = pd.DataFrame({
                "date": port_ret.index,
                "portfolio_return": port_ret.values,
                "n_active_pairs": len(pair_returns),
                "period_id": period_id,
            })
            all_returns.append(period_df)

            self.period_results.append({
                "period_id": period_id,
                "form_start": prices.index[form_start],
                "trade_start": prices.index[trade_start],
                "trade_end": prices.index[min(trade_end - 1, T - 1)],
                "n_pairs_selected": len(selected),
                "n_pairs_traded": len(pair_returns),
                "pairs": [p["pair"] for p in selected],
            })

            t += step

        if all_returns:
            self.portfolio_returns = pd.concat(all_returns, ignore_index=True)
            self.portfolio_returns.set_index("date", inplace=True)
            self.portfolio_returns["cumulative_return"] = (
                1 + self.portfolio_returns["portfolio_return"]
            ).cumprod()
        else:
            self.portfolio_returns = pd.DataFrame()

        return self.portfolio_returns

    def performance_summary(self) -> Dict:
        """
        Compute comprehensive performance metrics.

        Returns
        -------
        dict
            ann_return, ann_vol, sharpe, sortino, max_drawdown, calmar,
            avg_pairs_traded, total_periods, win_rate_monthly.
        """
        if self.portfolio_returns is None or self.portfolio_returns.empty:
            return {"error": "No backtest results available."}

        r = self.portfolio_returns["portfolio_return"]
        ann_ret = r.mean() * 252
        ann_vol = r.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

        downside = r[r < 0]
        dd_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else ann_vol
        sortino = ann_ret / dd_vol if dd_vol > 0 else 0

        cum = self.portfolio_returns["cumulative_return"]
        max_dd = (cum / cum.cummax() - 1).min()
        calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

        # Monthly win rate
        monthly = r.resample("M").sum()
        monthly_wr = (monthly > 0).mean() if len(monthly) > 0 else 0

        avg_pairs = self.portfolio_returns["n_active_pairs"].mean()

        return {
            "Ann. Return": ann_ret,
            "Ann. Volatility": ann_vol,
            "Sharpe Ratio": sharpe,
            "Sortino Ratio": sortino,
            "Max Drawdown": max_dd,
            "Calmar Ratio": calmar,
            "Monthly Win Rate": monthly_wr,
            "Avg Pairs Traded": avg_pairs,
            "Total Periods": len(self.period_results),
            "Final Wealth": cum.iloc[-1] if len(cum) > 0 else 1.0,
            "Skewness": r.skew(),
            "Kurtosis": r.kurtosis(),
        }
