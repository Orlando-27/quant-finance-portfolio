"""
Walk-Forward Backtesting Engine
================================

Backtests the momentum/mean-reversion strategy with realistic assumptions:
    - Walk-forward out-of-sample evaluation
    - Transaction costs (proportional to turnover)
    - Slippage modeling
    - Performance attribution by signal component and asset class
    - Comprehensive risk-return metrics

Author: Jose Orlando Bobadilla Fuentes, CQF, MSc AI
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple


class BacktestEngine:
    """
    Walk-forward backtesting engine.

    Parameters
    ----------
    transaction_cost_bps : float
        One-way transaction cost in basis points (default 10 bps).
    slippage_bps : float
        Slippage estimate in basis points (default 5 bps).
    risk_free_rate : float
        Annualized risk-free rate for Sharpe calculation (default 0.02).
    """

    def __init__(
        self,
        transaction_cost_bps: float = 10.0,
        slippage_bps: float = 5.0,
        risk_free_rate: float = 0.02,
    ):
        self.tc_bps = transaction_cost_bps
        self.slippage_bps = slippage_bps
        self.rf = risk_free_rate

    def run_backtest(
        self,
        weights: pd.DataFrame,
        returns: pd.DataFrame,
    ) -> Dict:
        """
        Execute backtest and compute performance metrics.

        Parameters
        ----------
        weights : pd.DataFrame
            Portfolio weights (dates x assets). Weights at time t
            determine position for return at time t+1.
        returns : pd.DataFrame
            Daily asset returns.

        Returns
        -------
        dict
            Comprehensive backtest results including:
            - 'portfolio_returns': daily portfolio return series
            - 'cumulative_returns': cumulative wealth curve
            - 'net_returns': returns after transaction costs
            - 'turnover': daily portfolio turnover
            - 'metrics': dict of risk-return metrics
            - 'weights': final weight matrix
        """
        # Align: weights at t determine position for return at t
        # Use shift(1) so that signal at t-1 generates return at t
        pos = weights.shift(1).fillna(0.0)

        # Gross portfolio returns
        gross_ret = (pos * returns).sum(axis=1)

        # Turnover: absolute change in weights
        turnover = pos.diff().abs().sum(axis=1)
        avg_turnover = turnover.mean() * 252  # Annualized

        # Transaction costs
        total_cost_pct = (self.tc_bps + self.slippage_bps) / 10000.0
        tc = turnover * total_cost_pct

        # Net returns
        net_ret = gross_ret - tc

        # Cumulative returns
        cum_gross = (1.0 + gross_ret).cumprod()
        cum_net = (1.0 + net_ret).cumprod()

        # Benchmark: equal-weight buy-and-hold
        bm_ret = returns.mean(axis=1)
        cum_bm = (1.0 + bm_ret).cumprod()

        # Compute metrics
        metrics = self._compute_metrics(net_ret, bm_ret)
        metrics["avg_annual_turnover"] = avg_turnover
        metrics["total_transaction_costs"] = tc.sum()

        return {
            "portfolio_returns": net_ret,
            "gross_returns": gross_ret,
            "cumulative_returns": cum_net,
            "cumulative_gross": cum_gross,
            "benchmark_cumulative": cum_bm,
            "benchmark_returns": bm_ret,
            "turnover": turnover,
            "metrics": metrics,
            "weights": pos,
        }

    def _compute_metrics(
        self, returns: pd.Series, benchmark: pd.Series
    ) -> Dict[str, float]:
        """
        Compute comprehensive risk-return metrics.

        Parameters
        ----------
        returns : pd.Series
            Strategy daily returns (net of costs).
        benchmark : pd.Series
            Benchmark daily returns.

        Returns
        -------
        dict
            Performance and risk metrics.
        """
        n_years = len(returns) / 252.0
        cum = (1.0 + returns).cumprod()

        # Annualized return
        total_ret = cum.iloc[-1] / cum.iloc[0] - 1.0 if len(cum) > 1 else 0.0
        ann_ret = (1.0 + total_ret) ** (1.0 / max(n_years, 0.01)) - 1.0

        # Annualized volatility
        ann_vol = returns.std() * np.sqrt(252)

        # Sharpe Ratio
        sharpe = (ann_ret - self.rf) / ann_vol if ann_vol > 0 else 0.0

        # Sortino Ratio (downside deviation)
        downside = returns[returns < 0]
        down_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else ann_vol
        sortino = (ann_ret - self.rf) / down_vol if down_vol > 0 else 0.0

        # Maximum Drawdown
        running_max = cum.cummax()
        drawdown = (cum - running_max) / running_max
        max_dd = drawdown.min()

        # Calmar Ratio
        calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0.0

        # Hit Rate (fraction of positive days)
        hit_rate = (returns > 0).mean()

        # Profit Factor
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

        # Information Ratio (vs benchmark)
        excess = returns - benchmark
        te = excess.std() * np.sqrt(252)
        ir = excess.mean() * 252 / te if te > 0 else 0.0

        # Skewness and Kurtosis
        skew = returns.skew()
        kurt = returns.kurtosis()

        # Best/Worst month
        monthly = returns.resample("M").sum() if hasattr(returns.index, 'freq') or len(returns) > 21 else returns
        try:
            monthly_agg = returns.groupby(pd.Grouper(freq="M")).sum()
            best_month = monthly_agg.max()
            worst_month = monthly_agg.min()
        except Exception:
            best_month = returns.max() * 21
            worst_month = returns.min() * 21

        return {
            "annualized_return": ann_ret,
            "annualized_volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_dd,
            "calmar_ratio": calmar,
            "hit_rate": hit_rate,
            "profit_factor": profit_factor,
            "information_ratio": ir,
            "tracking_error": te,
            "skewness": skew,
            "excess_kurtosis": kurt,
            "best_month": best_month,
            "worst_month": worst_month,
            "total_return": total_ret,
        }

    def attribution_by_class(
        self,
        weights: pd.DataFrame,
        returns: pd.DataFrame,
        class_map: Dict[str, list],
    ) -> pd.DataFrame:
        """
        Decompose portfolio returns by asset class.

        Parameters
        ----------
        weights : pd.DataFrame
            Portfolio weights.
        returns : pd.DataFrame
            Daily asset returns.
        class_map : dict
            Mapping from class name to list of tickers.

        Returns
        -------
        pd.DataFrame
            Annualized return contribution by asset class.
        """
        pos = weights.shift(1).fillna(0.0)
        results = {}

        for ac, tickers in class_map.items():
            cols = [c for c in tickers if c in pos.columns]
            if not cols:
                continue
            ac_ret = (pos[cols] * returns[cols]).sum(axis=1)
            results[ac] = {
                "ann_return": ac_ret.mean() * 252,
                "ann_vol": ac_ret.std() * np.sqrt(252),
                "avg_gross_exposure": pos[cols].abs().sum(axis=1).mean(),
            }

        return pd.DataFrame(results).T
