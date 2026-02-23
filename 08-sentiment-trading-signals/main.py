"""
Sentiment Analysis for Trading Signals - Main Entry Point
==========================================================
Demonstrates the complete NLP-to-trading pipeline: data acquisition,
multi-model sentiment scoring (FinBERT, VADER, Loughran-McDonald),
feature engineering, signal generation, and backtesting.

Author: Jose Orlando Bobadilla Fuentes, CQF
"""

import warnings
warnings.filterwarnings("ignore")

from src.data_acquisition import NewsDataCollector, MarketDataLoader
from src.models.vader_model import VADERSentimentModel
from src.models.lm_dictionary import LoughranMcDonaldModel
from src.feature_engineering import SentimentFeatureEngineer
from src.strategy import SentimentTradingStrategy
from src.backtesting import SentimentBacktester
from src.evaluation import StrategyEvaluator
from src.utils import setup_logging, set_random_seed


def main():
    """Run the complete sentiment-to-trading pipeline."""
    print("=" * 70)
    print("SENTIMENT ANALYSIS FOR TRADING SIGNALS")
    print("=" * 70)

    setup_logging()
    set_random_seed(42)

    # --- Configuration ---
    config = {
        "ticker": "AAPL",
        "start_date": "2020-01-01",
        "end_date": "2024-12-31",
        "sentiment_models": ["vader", "loughran_mcdonald"],
        "lookback_windows": [5, 10, 20],
        "signal_threshold": 0.6,
        "transaction_cost_bps": 10,
        "initial_capital": 1_000_000,
    }
    print(f"\n  Ticker: {config['ticker']}")
    print(f"  Period: {config['start_date']} to {config['end_date']}")

    # --- Step 1: Data Acquisition ---
    print("\n[1/6] Loading market data and generating synthetic news...")
    market_loader = MarketDataLoader()
    price_data = market_loader.load(
        config["ticker"], config["start_date"], config["end_date"]
    )
    print(f"  Market data: {len(price_data)} trading days")

    news_collector = NewsDataCollector()
    news_data = news_collector.generate_synthetic(
        config["ticker"], config["start_date"], config["end_date"]
    )
    print(f"  News articles: {len(news_data)} synthetic headlines")

    # --- Step 2: Sentiment Scoring ---
    print("\n[2/6] Computing sentiment scores...")
    vader = VADERSentimentModel()
    lm = LoughranMcDonaldModel()

    vader_scores = vader.score_batch(news_data["headline"].tolist())
    lm_scores = lm.score_batch(news_data["headline"].tolist())
    print(f"  VADER mean: {vader_scores.mean():.4f} | std: {vader_scores.std():.4f}")
    print(f"  LM mean:    {lm_scores.mean():.4f} | std: {lm_scores.std():.4f}")

    # --- Step 3: Feature Engineering ---
    print("\n[3/6] Engineering sentiment features...")
    engineer = SentimentFeatureEngineer(
        lookback_windows=config["lookback_windows"]
    )
    features = engineer.create_features(price_data, news_data, vader_scores, lm_scores)
    print(f"  Features generated: {features.shape[1]} columns")
    print(f"  Date range: {features.index[0]} to {features.index[-1]}")

    # --- Step 4: Signal Generation ---
    print("\n[4/6] Generating trading signals...")
    strategy = SentimentTradingStrategy(
        threshold=config["signal_threshold"]
    )
    signals = strategy.generate_signals(features)
    long_pct = (signals == 1).mean() * 100
    short_pct = (signals == -1).mean() * 100
    flat_pct = (signals == 0).mean() * 100
    print(f"  Long:  {long_pct:.1f}% | Short: {short_pct:.1f}% | Flat: {flat_pct:.1f}%")

    # --- Step 5: Backtesting ---
    print("\n[5/6] Running backtest...")
    backtester = SentimentBacktester(
        initial_capital=config["initial_capital"],
        transaction_cost_bps=config["transaction_cost_bps"],
    )
    results = backtester.run(price_data, signals)
    print(f"  Total return: {results['total_return']:.2%}")
    print(f"  Sharpe ratio: {results['sharpe_ratio']:.3f}")
    print(f"  Max drawdown: {results['max_drawdown']:.2%}")

    # --- Step 6: Evaluation ---
    print("\n[6/6] Strategy evaluation...")
    evaluator = StrategyEvaluator()
    report = evaluator.full_report(results, price_data)
    print(f"  Annualized return: {report['ann_return']:.2%}")
    print(f"  Annualized vol:    {report['ann_vol']:.2%}")
    print(f"  Calmar ratio:      {report['calmar']:.3f}")
    print(f"  Win rate:          {report['win_rate']:.1%}")

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print("  All visualizations available via StrategyEvaluator.plot_* methods")
    print("=" * 70)


if __name__ == "__main__":
    main()
