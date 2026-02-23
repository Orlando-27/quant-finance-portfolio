"""
Walk-Forward Factor Backtesting Engine
=======================================

Implements a production-grade backtesting framework for factor strategies
with periodic rebalancing, transaction costs, turnover constraints, and
comprehensive performance analytics.

Architecture:
    The engine operates on a walk-forward basis:
    1. At each rebalancing date t:
       a. Estimate factor moments from expanding/rolling window up to t
       b. Optionally predict factor returns using ML model (fitted on data <= t)
       c. Optimize portfolio weights using selected strategy
       d. Apply turnover constraints and transaction costs
    2. Between rebalancing dates: buy-and-hold with drift
    3. Record portfolio returns, weights, and diagnostics at every step

    This ensures no lookahead bias: all decisions at time t use only
    information available up to time t.

References:
    Harvey, Liu & Zhu (2016) - Backtesting pitfalls
    Bailey, Borwein, Lopez de Prado & Zhu (2017) - Backtest overfitting
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable
from src.portfolio import FactorPortfolio


class FactorBacktester:
    """
    Walk-forward backtesting engine for factor allocation strategies.

    Parameters
    ----------
    factor_returns : pd.DataFrame
        (T x K) factor return series.
    rebalance_frequency : int
        Rebalance every N periods (default 1 = monthly).
    transaction_cost_bps : float
        One-way transaction cost in basis points (default 10 bps).
    lookback_window : int
        Rolling window for moment estimation (default 60 months).
        If None, uses expanding window.
    risk_free_rate : float
        Annual risk-free rate.
    """

    def __init__(self, factor_returns: pd.DataFrame,
                 rebalance_frequency: int = 1,
                 transaction_cost_bps: float = 10.0,
                 lookback_window: Optional[int] = 60,
                 risk_free_rate: float = 0.02):
        self.factor_returns = factor_returns
        self.rebal_freq = rebalance_frequency
        self.tc_bps = transaction_cost_bps / 10000
        self.lookback = lookback_window
        self.rf = risk_free_rate
        self.results = {}

    def run(self, strategy: str = "risk_parity",
            ml_predictions: Optional[pd.DataFrame] = None,
            min_history: int = 36) -> pd.DataFrame:
        """
        Execute walk-forward backtest for a given strategy.

        Parameters
        ----------
        strategy : str
            One of: 'equal_weight', 'inverse_vol', 'risk_parity',
            'max_sharpe', 'min_variance', 'max_diversification',
            'ml_enhanced' (uses ml_predictions for expected returns).
        ml_predictions : pd.DataFrame, optional
            (T x K) ML-predicted factor returns for 'ml_enhanced' strategy.
        min_history : int
            Minimum observations before starting backtest.

        Returns
        -------
        pd.DataFrame
            Backtest results with columns: date, portfolio_return,
            gross_return, transaction_cost, cumulative_return,
            weights (as dict), turnover.
        """
        dates = self.factor_returns.index
        K = self.factor_returns.shape[1]
        start_idx = max(min_history, self.lookback or min_history)

        records = []
        prev_weights = np.ones(K) / K
        rebal_counter = 0

        for t_idx in range(start_idx, len(dates)):
            t = dates[t_idx]
            rebal_counter += 1

            # Current period factor returns
            r_t = self.factor_returns.iloc[t_idx].values

            # Check if rebalancing
            if rebal_counter >= self.rebal_freq:
                rebal_counter = 0

                # Estimation window
                if self.lookback:
                    window_start = max(0, t_idx - self.lookback)
                else:
                    window_start = 0
                hist = self.factor_returns.iloc[window_start:t_idx]

                fp = FactorPortfolio(hist, self.rf)

                # Select strategy
                if strategy == "equal_weight":
                    new_weights = fp.equal_weight()
                elif strategy == "inverse_vol":
                    new_weights = fp.inverse_volatility()
                elif strategy == "risk_parity":
                    new_weights = fp.risk_parity()
                elif strategy == "max_sharpe":
                    new_weights = fp.mean_variance("max_sharpe")
                elif strategy == "min_variance":
                    new_weights = fp.mean_variance("min_variance")
                elif strategy == "max_diversification":
                    new_weights = fp.maximum_diversification()
                elif strategy == "ml_enhanced" and ml_predictions is not None:
                    if t in ml_predictions.index:
                        pred_ret = ml_predictions.loc[t].values
                        new_weights = fp.mean_variance("max_sharpe", pred_ret)
                    else:
                        new_weights = fp.risk_parity()
                else:
                    new_weights = fp.equal_weight()

                # Transaction costs based on turnover
                turnover = np.sum(np.abs(new_weights - prev_weights))
                tc = turnover * self.tc_bps
            else:
                new_weights = prev_weights
                turnover = 0.0
                tc = 0.0

            # Portfolio return
            gross_ret = new_weights @ r_t
            net_ret = gross_ret - tc

            records.append({
                "date": t,
                "gross_return": gross_ret,
                "transaction_cost": tc,
                "net_return": net_ret,
                "turnover": turnover,
                "weights": dict(zip(self.factor_returns.columns, new_weights)),
            })

            # Drift weights for next period
            drifted = new_weights * (1 + r_t)
            prev_weights = drifted / drifted.sum()

        df = pd.DataFrame(records).set_index("date")
        df["cumulative_return"] = (1 + df["net_return"]).cumprod()
        self.results[strategy] = df
        return df

    def run_all_strategies(self) -> Dict[str, pd.DataFrame]:
        """Run backtest for all standard strategies."""
        strategies = ["equal_weight", "inverse_vol", "risk_parity",
                      "max_sharpe", "min_variance", "max_diversification"]
        for s in strategies:
            self.run(strategy=s)
        return self.results

    def performance_summary(self) -> pd.DataFrame:
        """
        Compute comprehensive performance metrics for all backtested strategies.

        Metrics include: annualized return, volatility, Sharpe ratio,
        Sortino ratio, maximum drawdown, Calmar ratio, average turnover,
        total transaction costs, and information ratio vs equal-weight.
        """
        summaries = []
        eq_returns = None

        for name, df in self.results.items():
            r = df["net_return"]
            ann_ret = r.mean() * 12
            ann_vol = r.std() * np.sqrt(12)
            sharpe = (ann_ret - self.rf) / ann_vol if ann_vol > 0 else 0

            # Sortino: downside deviation
            downside = r[r < 0]
            dd_vol = downside.std() * np.sqrt(12) if len(downside) > 0 else ann_vol
            sortino = (ann_ret - self.rf) / dd_vol if dd_vol > 0 else 0

            # Max drawdown
            cum = (1 + r).cumprod()
            max_dd = (cum / cum.cummax() - 1).min()
            calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

            # Turnover and costs
            avg_turnover = df["turnover"].mean()
            total_tc = df["transaction_cost"].sum()

            if name == "equal_weight":
                eq_returns = r

            summaries.append({
                "Strategy": name,
                "Ann. Return": ann_ret,
                "Ann. Volatility": ann_vol,
                "Sharpe Ratio": sharpe,
                "Sortino Ratio": sortino,
                "Max Drawdown": max_dd,
                "Calmar Ratio": calmar,
                "Avg Turnover": avg_turnover,
                "Total TC": total_tc,
                "Final Wealth": cum.iloc[-1],
            })

        summary_df = pd.DataFrame(summaries).set_index("Strategy")

        # Information ratio vs equal-weight
        if eq_returns is not None:
            for name, df in self.results.items():
                if name != "equal_weight":
                    active = df["net_return"] - eq_returns.reindex(df.index).fillna(0)
                    te = active.std() * np.sqrt(12)
                    ir = active.mean() * 12 / te if te > 0 else 0
                    summary_df.loc[name, "IR vs EW"] = ir

        return summary_df
