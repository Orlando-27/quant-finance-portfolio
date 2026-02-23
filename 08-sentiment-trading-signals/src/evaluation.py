"""
================================================================================
PERFORMANCE ANALYTICS & FACTOR ATTRIBUTION
================================================================================
Comprehensive evaluation of the sentiment trading strategy including:
    1. Standard performance metrics (Sharpe, Sortino, Calmar, etc.)
    2. Risk metrics (VaR, CVaR, max drawdown)
    3. Information coefficient analysis (IC, rank IC, ICIR)
    4. Fama-French factor attribution
    5. Turnover and transaction cost analysis

The Information Coefficient (IC) measures the rank correlation between
the predicted signal and subsequent realized returns:

    IC_t = corr_rank(signal_t, return_{t+1})

An IC of 0.05 is considered good for daily signals; 0.10+ is excellent.
The Information Coefficient Information Ratio (ICIR) measures consistency:

    ICIR = mean(IC_t) / std(IC_t)

ICIR > 0.5 indicates a stable, reliable signal.

Author: Jose Orlando Bobadilla Fuentes, CQF | MSc AI
================================================================================
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Optional


class PerformanceAnalyzer:
    """
    Comprehensive strategy performance analyzer.

    Parameters
    ----------
    results : dict
        Output of SentimentBacktester.run(). Must contain:
        'strategy_returns', 'benchmark_returns', 'net_returns',
        'positions', 'turnover', 'transaction_costs', 'cumulative'.
    risk_free_rate : float
        Annualized risk-free rate (default 0.04 = 4%).
    trading_days : int
        Trading days per year for annualization (default 252).
    """

    def __init__(
        self,
        results: Dict,
        risk_free_rate: float = 0.04,
        trading_days: int = 252,
    ):
        self.strat_ret = results["strategy_returns"]
        self.bench_ret = results["benchmark_returns"]
        self.net_ret = results["net_returns"]
        self.positions = results.get("positions")
        self.turnover = results.get("turnover")
        self.tc = results.get("transaction_costs")
        self.cumulative = results.get("cumulative")
        self.rf = risk_free_rate
        self.td = trading_days

    # ------------------------------------------------------------------
    # 1. Return Metrics
    # ------------------------------------------------------------------
    def annualized_return(self, returns: pd.Series) -> float:
        """Geometric annualized return."""
        total = (1 + returns).prod()
        n_years = len(returns) / self.td
        if n_years <= 0:
            return 0.0
        return total ** (1 / n_years) - 1

    def annualized_volatility(self, returns: pd.Series) -> float:
        """Annualized standard deviation of returns."""
        return returns.std() * np.sqrt(self.td)

    def sharpe_ratio(self, returns: pd.Series) -> float:
        """
        Annualized Sharpe Ratio.

        SR = (ann_return - rf) / ann_volatility
        """
        ann_ret = self.annualized_return(returns)
        ann_vol = self.annualized_volatility(returns)
        if ann_vol == 0:
            return 0.0
        return (ann_ret - self.rf) / ann_vol

    def sortino_ratio(self, returns: pd.Series) -> float:
        """
        Sortino Ratio: penalizes only downside volatility.

        Sortino = (ann_return - rf) / downside_deviation
        """
        ann_ret = self.annualized_return(returns)
        downside = returns[returns < 0]
        downside_vol = downside.std() * np.sqrt(self.td) if len(downside) > 0 else 1e-8
        return (ann_ret - self.rf) / downside_vol

    def calmar_ratio(self, returns: pd.Series) -> float:
        """
        Calmar Ratio: return relative to maximum drawdown.

        Calmar = ann_return / max_drawdown
        """
        ann_ret = self.annualized_return(returns)
        mdd = self.max_drawdown(returns)
        return ann_ret / mdd if mdd > 0 else 0.0

    # ------------------------------------------------------------------
    # 2. Risk Metrics
    # ------------------------------------------------------------------
    def max_drawdown(self, returns: pd.Series) -> float:
        """Maximum peak-to-trough drawdown."""
        cum = (1 + returns).cumprod()
        peak = cum.expanding().max()
        dd = (cum - peak) / peak
        return abs(dd.min())

    def drawdown_series(self, returns: pd.Series) -> pd.Series:
        """Full drawdown time series."""
        cum = (1 + returns).cumprod()
        peak = cum.expanding().max()
        return (cum - peak) / peak

    def var_historical(
        self, returns: pd.Series, confidence: float = 0.95
    ) -> float:
        """Historical Value-at-Risk at given confidence level."""
        return -np.percentile(returns, (1 - confidence) * 100)

    def cvar_historical(
        self, returns: pd.Series, confidence: float = 0.95
    ) -> float:
        """Conditional VaR (Expected Shortfall)."""
        var = self.var_historical(returns, confidence)
        return -returns[returns <= -var].mean()

    # ------------------------------------------------------------------
    # 3. Relative Metrics
    # ------------------------------------------------------------------
    def beta(self) -> float:
        """Portfolio beta relative to benchmark."""
        cov = np.cov(self.strat_ret.values, self.bench_ret.values)
        return cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else 0.0

    def alpha_annualized(self) -> float:
        """Jensen's alpha (annualized)."""
        b = self.beta()
        return self.annualized_return(self.strat_ret) - (
            self.rf + b * (self.annualized_return(self.bench_ret) - self.rf)
        )

    def tracking_error(self) -> float:
        """Annualized tracking error vs benchmark."""
        excess = self.strat_ret - self.bench_ret
        return excess.std() * np.sqrt(self.td)

    def information_ratio(self) -> float:
        """Information Ratio = excess return / tracking error."""
        te = self.tracking_error()
        excess_ret = (
            self.annualized_return(self.strat_ret)
            - self.annualized_return(self.bench_ret)
        )
        return excess_ret / te if te > 0 else 0.0

    # ------------------------------------------------------------------
    # 4. Hit Rate and Win/Loss
    # ------------------------------------------------------------------
    def hit_rate(self, returns: pd.Series) -> float:
        """Fraction of positive-return days."""
        return (returns > 0).mean()

    def profit_loss_ratio(self, returns: pd.Series) -> float:
        """Average win / average loss."""
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        if len(losses) == 0 or len(wins) == 0:
            return 0.0
        return abs(wins.mean() / losses.mean())

    # ------------------------------------------------------------------
    # 5. Information Coefficient Analysis
    # ------------------------------------------------------------------
    def compute_ic(
        self,
        signals: pd.DataFrame,
        returns: pd.DataFrame,
        signal_col: str = "composite_signal",
    ) -> pd.Series:
        """
        Compute daily Information Coefficient (rank correlation).

        IC_t = spearman_corr(signal_t, return_{t+1})

        Parameters
        ----------
        signals : pd.DataFrame
            Must have 'date', 'ticker', signal_col.
        returns : pd.DataFrame
            Daily returns matrix (dates x tickers).

        Returns
        -------
        pd.Series of daily IC values.
        """
        daily_ics = []
        dates = sorted(signals["date"].unique())

        for i, date in enumerate(dates[:-1]):
            next_date = dates[i + 1]
            day_sig = signals[signals["date"] == date]

            tickers = day_sig["ticker"].values
            sig_values = day_sig[signal_col].values

            # Forward returns
            if pd.Timestamp(next_date) in returns.index:
                fwd_ret = returns.loc[pd.Timestamp(next_date), tickers].values
            else:
                continue

            # Remove NaN pairs
            mask = ~(np.isnan(sig_values) | np.isnan(fwd_ret))
            if mask.sum() < 5:
                continue

            ic, _ = stats.spearmanr(sig_values[mask], fwd_ret[mask])
            daily_ics.append({"date": date, "ic": ic})

        ic_series = pd.DataFrame(daily_ics).set_index("date")["ic"]
        return ic_series

    def ic_summary(self, ic_series: pd.Series) -> Dict:
        """Compute IC summary statistics."""
        return {
            "mean_ic": ic_series.mean(),
            "std_ic": ic_series.std(),
            "icir": ic_series.mean() / ic_series.std() if ic_series.std() > 0 else 0,
            "pct_positive": (ic_series > 0).mean(),
            "t_stat": ic_series.mean() / (ic_series.std() / np.sqrt(len(ic_series)))
            if ic_series.std() > 0 else 0,
        }

    # ------------------------------------------------------------------
    # 6. Turnover Analysis
    # ------------------------------------------------------------------
    def turnover_summary(self) -> Dict:
        """Summarize portfolio turnover and transaction costs."""
        if self.turnover is None:
            return {}
        return {
            "avg_daily_turnover": self.turnover.mean(),
            "median_daily_turnover": self.turnover.median(),
            "max_daily_turnover": self.turnover.max(),
            "annualized_turnover": self.turnover.sum(),
            "total_tc_bps": self.tc.sum() * 10_000 if self.tc is not None else 0,
            "avg_daily_tc_bps": self.tc.mean() * 10_000 if self.tc is not None else 0,
        }

    # ------------------------------------------------------------------
    # 7. Summary Report
    # ------------------------------------------------------------------
    def full_report(self) -> Dict:
        """Generate comprehensive performance report."""
        report = {
            # Gross returns
            "ann_return_gross": self.annualized_return(self.strat_ret),
            "ann_vol_gross": self.annualized_volatility(self.strat_ret),
            "sharpe_gross": self.sharpe_ratio(self.strat_ret),
            "sortino_gross": self.sortino_ratio(self.strat_ret),

            # Net returns (after TC)
            "ann_return_net": self.annualized_return(self.net_ret),
            "sharpe_net": self.sharpe_ratio(self.net_ret),
            "sortino_net": self.sortino_ratio(self.net_ret),

            # Risk
            "max_drawdown": self.max_drawdown(self.strat_ret),
            "calmar": self.calmar_ratio(self.strat_ret),
            "var_95": self.var_historical(self.strat_ret, 0.95),
            "cvar_95": self.cvar_historical(self.strat_ret, 0.95),

            # Relative
            "beta": self.beta(),
            "alpha": self.alpha_annualized(),
            "tracking_error": self.tracking_error(),
            "info_ratio": self.information_ratio(),

            # Hit rate
            "hit_rate": self.hit_rate(self.strat_ret),
            "profit_loss_ratio": self.profit_loss_ratio(self.strat_ret),

            # Benchmark
            "bench_return": self.annualized_return(self.bench_ret),
            "bench_sharpe": self.sharpe_ratio(self.bench_ret),
        }

        # Add turnover if available
        report.update(self.turnover_summary())
        return report

    def print_summary(self):
        """Print formatted performance summary to console."""
        r = self.full_report()
        print("\n" + "=" * 60)
        print("  STRATEGY PERFORMANCE SUMMARY")
        print("=" * 60)

        print(f"\n  --- Return Metrics ---")
        print(f"  Ann. Return (Gross)   : {r['ann_return_gross']:>8.2%}")
        print(f"  Ann. Return (Net)     : {r['ann_return_net']:>8.2%}")
        print(f"  Ann. Volatility       : {r['ann_vol_gross']:>8.2%}")
        print(f"  Sharpe Ratio (Gross)  : {r['sharpe_gross']:>8.3f}")
        print(f"  Sharpe Ratio (Net)    : {r['sharpe_net']:>8.3f}")
        print(f"  Sortino Ratio         : {r['sortino_gross']:>8.3f}")

        print(f"\n  --- Risk Metrics ---")
        print(f"  Max Drawdown          : {r['max_drawdown']:>8.2%}")
        print(f"  Calmar Ratio          : {r['calmar']:>8.3f}")
        print(f"  VaR (95%)             : {r['var_95']:>8.4f}")
        print(f"  CVaR (95%)            : {r['cvar_95']:>8.4f}")

        print(f"\n  --- Relative Metrics ---")
        print(f"  Beta                  : {r['beta']:>8.3f}")
        print(f"  Alpha (Ann.)          : {r['alpha']:>8.2%}")
        print(f"  Tracking Error        : {r['tracking_error']:>8.2%}")
        print(f"  Information Ratio     : {r['info_ratio']:>8.3f}")

        print(f"\n  --- Trading ---")
        print(f"  Hit Rate              : {r['hit_rate']:>8.2%}")
        print(f"  Profit/Loss Ratio     : {r['profit_loss_ratio']:>8.3f}")
        if "avg_daily_turnover" in r:
            print(f"  Avg Daily Turnover    : {r['avg_daily_turnover']:>8.4f}")
            print(f"  Total TC (bps)        : {r['total_tc_bps']:>8.1f}")

        print(f"\n  --- Benchmark ({self.bench_ret.name or 'SPY'}) ---")
        print(f"  Bench. Return         : {r['bench_return']:>8.2%}")
        print(f"  Bench. Sharpe         : {r['bench_sharpe']:>8.3f}")
        print("=" * 60)
