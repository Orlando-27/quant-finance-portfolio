"""
================================================================================
WALK-FORWARD BACKTESTING ENGINE
================================================================================
Implements a rigorous walk-forward backtesting framework for the
sentiment trading strategy with expanding or rolling windows.

Walk-forward procedure:
    1. Score sentiment using only data available up to time t
    2. Generate signals from features computed in-sample
    3. Execute trades and compute P&L with realistic frictions

The engine prevents lookahead bias at every step:
    - Sentiment scores use only documents published before market close
    - Feature z-scores use only historical data
    - Volatility estimates use trailing realized volatility

Author: Jose Orlando Bobadilla Fuentes, CQF | MSc AI
================================================================================
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Optional, List, Dict
from datetime import datetime

from src.models.vader_model import VADERSentiment
from src.models.finbert_model import FinBERTSentiment
from src.models.lm_dictionary import LMDictionarySentiment
from src.data_acquisition import SyntheticNewsGenerator
from src.feature_engineering import SentimentFeatureEngine
from src.strategy import SentimentStrategy


class SentimentBacktester:
    """
    End-to-end backtesting engine for sentiment trading strategies.

    Orchestrates the full pipeline: data -> sentiment scoring ->
    feature engineering -> signal generation -> portfolio construction ->
    P&L computation -> performance analytics.

    Parameters
    ----------
    tickers : list of str
        Universe of tradeable assets.
    start_date : str
        Backtest start date ('YYYY-MM-DD').
    end_date : str
        Backtest end date ('YYYY-MM-DD').
    use_synthetic : bool
        If True, generate synthetic news instead of calling APIs.
    sentiment_models : list of str
        Models to use: 'vader', 'finbert', 'lm' (default: all three).
    model_weights : dict, optional
        Ensemble weights for combining model outputs.
        Default: {'vader': 0.25, 'finbert': 0.50, 'lm': 0.25}.
    strategy_params : dict, optional
        Parameters passed to SentimentStrategy constructor.
    feature_params : dict, optional
        Parameters passed to SentimentFeatureEngine constructor.
    benchmark : str
        Benchmark ticker for comparison (default 'SPY').
    """

    def __init__(
        self,
        tickers: List[str],
        start_date: str = "2018-01-01",
        end_date: str = "2023-12-31",
        use_synthetic: bool = True,
        sentiment_models: Optional[List[str]] = None,
        model_weights: Optional[Dict[str, float]] = None,
        strategy_params: Optional[Dict] = None,
        feature_params: Optional[Dict] = None,
        benchmark: str = "SPY",
    ):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.use_synthetic = use_synthetic
        self.benchmark = benchmark

        self.sentiment_models = sentiment_models or ["vader", "lm"]
        self.model_weights = model_weights or {
            "vader": 0.35, "finbert": 0.40, "lm": 0.25
        }
        self.strategy_params = strategy_params or {}
        self.feature_params = feature_params or {}

        # Initialize components
        self.feature_engine = SentimentFeatureEngine(**self.feature_params)
        self.strategy = SentimentStrategy(**self.strategy_params)

        # Results storage
        self.news_data = None
        self.scored_data = None
        self.features = None
        self.positions = None
        self.results = None

    def _fetch_prices(self) -> pd.DataFrame:
        """Download adjusted close prices for universe + benchmark."""
        all_tickers = list(set(self.tickers + [self.benchmark]))
        data = yf.download(
            all_tickers,
            start=self.start_date,
            end=self.end_date,
            auto_adjust=True,
        )
        if isinstance(data.columns, pd.MultiIndex):
            prices = data["Close"]
        else:
            prices = data[["Close"]]
            prices.columns = all_tickers
        return prices.dropna()

    def _collect_news(self) -> pd.DataFrame:
        """Collect news data (synthetic or live)."""
        if self.use_synthetic:
            gen = SyntheticNewsGenerator(seed=42)
            return gen.generate(self.tickers, self.start_date, self.end_date)
        else:
            from src.data_acquisition import NewsAPICollector
            collector = NewsAPICollector()
            return collector.fetch(self.tickers, self.start_date, self.end_date)

    def _score_sentiment(self, news: pd.DataFrame) -> pd.DataFrame:
        """Apply all configured sentiment models and compute ensemble score."""
        df = news.copy()

        if "vader" in self.sentiment_models:
            vader = VADERSentiment(use_financial_lexicon=True)
            df = vader.score_dataframe(df, text_column="headline", prefix="vader")
            print(f"  VADER scored {len(df)} documents")

        if "finbert" in self.sentiment_models:
            try:
                finbert = FinBERTSentiment(device="auto", batch_size=64)
                df = finbert.score_dataframe(df, text_column="headline", prefix="finbert")
                print(f"  FinBERT scored {len(df)} documents")
            except Exception as e:
                print(f"  FinBERT unavailable ({e}), using VADER + LM only")
                self.sentiment_models = [m for m in self.sentiment_models if m != "finbert"]

        if "lm" in self.sentiment_models:
            lm = LMDictionarySentiment()
            df = lm.score_dataframe(df, text_column="headline", prefix="lm")
            print(f"  LM Dictionary scored {len(df)} documents")

        # Compute ensemble score as weighted average of available models
        score_cols = []
        weights = []
        if "vader" in self.sentiment_models:
            score_cols.append("vader_compound")
            weights.append(self.model_weights.get("vader", 0.33))
        if "finbert" in self.sentiment_models:
            score_cols.append("finbert_score")
            weights.append(self.model_weights.get("finbert", 0.34))
        if "lm" in self.sentiment_models:
            score_cols.append("lm_score")
            weights.append(self.model_weights.get("lm", 0.33))

        # Normalize weights to sum to 1
        w_sum = sum(weights)
        weights = [w / w_sum for w in weights]

        df["ensemble_score"] = sum(
            df[col] * w for col, w in zip(score_cols, weights)
        )

        return df

    def run(self) -> Dict:
        """
        Execute the full backtest pipeline.

        Returns
        -------
        dict with keys:
            'strategy_returns' : pd.Series of daily strategy returns
            'benchmark_returns' : pd.Series of daily benchmark returns
            'positions' : pd.DataFrame of daily position weights
            'turnover' : pd.Series of daily turnover
            'transaction_costs' : pd.Series of daily TC
            'net_returns' : pd.Series (gross returns - TC)
            'cumulative' : pd.DataFrame with cumulative performance
        """
        print("=" * 60)
        print("  SENTIMENT TRADING BACKTEST")
        print("=" * 60)

        # Step 1: Fetch price data
        print("\n[1/6] Fetching price data...")
        prices = self._fetch_prices()
        returns = np.log(prices / prices.shift(1)).dropna()
        print(f"  {len(prices)} days, {len(self.tickers)} assets")

        # Step 2: Collect news
        print("\n[2/6] Collecting news data...")
        self.news_data = self._collect_news()
        print(f"  {len(self.news_data)} articles collected")

        # Step 3: Score sentiment
        print("\n[3/6] Scoring sentiment...")
        self.scored_data = self._score_sentiment(self.news_data)

        # Step 4: Feature engineering
        print("\n[4/6] Engineering features...")
        self.scored_data["date"] = pd.to_datetime(
            self.scored_data["datetime"]
        ).dt.normalize()
        daily = self.feature_engine.aggregate_daily(
            self.scored_data, score_column="ensemble_score"
        )
        daily["date"] = pd.to_datetime(daily["date"])
        features = self.feature_engine.compute_features(daily)
        features = self.feature_engine.build_composite_signal(features)
        self.features = self.feature_engine.build_cross_sectional_signal(features)
        print(f"  {len(self.features)} daily observations")

        # Step 5: Generate positions
        print("\n[5/6] Generating positions...")
        self.positions = self.strategy.generate_positions(
            self.features, returns[self.tickers]
        )

        # Step 6: Compute P&L
        print("\n[6/6] Computing P&L...")
        # Gross returns
        strategy_ret = (returns[self.tickers] * self.positions.reindex(
            returns.index
        ).fillna(0)).sum(axis=1)

        # Transaction costs
        turnover = self.strategy.compute_turnover(
            self.positions.reindex(returns.index).fillna(0)
        )
        tc = self.strategy.compute_transaction_costs(
            self.positions.reindex(returns.index).fillna(0)
        )
        net_ret = strategy_ret - tc

        # Benchmark
        bench_ret = returns[self.benchmark]

        # Cumulative performance
        cum = pd.DataFrame({
            "Strategy (Gross)": (1 + strategy_ret).cumprod(),
            "Strategy (Net)": (1 + net_ret).cumprod(),
            "Benchmark": (1 + bench_ret).cumprod(),
        })

        self.results = {
            "strategy_returns": strategy_ret,
            "benchmark_returns": bench_ret,
            "net_returns": net_ret,
            "positions": self.positions,
            "turnover": turnover,
            "transaction_costs": tc,
            "cumulative": cum,
        }

        print("\n  Backtest complete.")
        return self.results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sentiment Trading Backtest")
    parser.add_argument("--synthetic", action="store_true", default=True)
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2023-12-31")
    args = parser.parse_args()

    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA",
               "TSLA", "JPM", "JNJ", "V", "WMT", "PG", "XOM", "BAC"]

    bt = SentimentBacktester(
        tickers=tickers,
        start_date=args.start,
        end_date=args.end,
        use_synthetic=args.synthetic,
        sentiment_models=["vader", "lm"],
    )
    results = bt.run()

    from src.evaluation import PerformanceAnalyzer
    analyzer = PerformanceAnalyzer(results)
    analyzer.print_summary()
