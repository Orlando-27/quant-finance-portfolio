"""
================================================================================
UNIT TESTS -- SENTIMENT ANALYSIS FOR TRADING SIGNALS
================================================================================
Tests cover:
    1. VADER model with financial lexicon
    2. Loughran-McDonald dictionary scoring
    3. Feature engineering pipeline
    4. Strategy position generation
    5. Performance metrics computation
    6. Data acquisition (synthetic)

FinBERT tests are skipped if torch/transformers are not installed,
since model download requires ~500MB and GPU is optional.

Author: Jose Orlando Bobadilla Fuentes, CQF | MSc AI
================================================================================
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_headlines():
    """Realistic financial headlines with known sentiment polarity."""
    return [
        # Positive
        "Apple beats Q3 earnings estimates by 15%, stock surges",
        "Microsoft announces $60B share buyback program",
        "NVIDIA reports record quarterly revenue, raises guidance",
        "Goldman Sachs upgrades Tesla to Overweight",
        "Strong demand drives Amazon revenue growth of 20%",
        # Negative
        "Boeing issues profit warning citing supply chain disruptions",
        "Facebook stock drops 8% on disappointing user growth",
        "Wells Fargo faces regulatory investigation into fraud",
        "Recession fears grow as yield curve inverts",
        "Layoffs announced at Google, cutting 12,000 employees",
        # Neutral
        "Apple scheduled to report Q4 earnings next week",
        "Fed meeting minutes to be released on Wednesday",
        "Tesla CEO to present at Morgan Stanley tech conference",
        "Market trading flat ahead of jobs report",
        "Annual shareholder meeting set for March 15",
    ]


@pytest.fixture
def synthetic_news():
    """Generate synthetic news DataFrame for pipeline testing."""
    from src.data_acquisition import SyntheticNewsGenerator
    gen = SyntheticNewsGenerator(seed=42)
    tickers = ["AAPL", "MSFT", "GOOGL"]
    return gen.generate(tickers, "2022-01-01", "2022-06-30",
                        articles_per_ticker_per_day=3.0)


@pytest.fixture
def sample_returns():
    """Simulated daily returns for 5 assets over 252 days."""
    np.random.seed(42)
    dates = pd.bdate_range("2022-01-01", periods=252)
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    returns = pd.DataFrame(
        np.random.normal(0.0003, 0.015, (252, 5)),
        index=dates,
        columns=tickers,
    )
    return returns


# ============================================================================
# TEST: VADER MODEL
# ============================================================================

class TestVADER:
    """Tests for VADER sentiment with financial lexicon."""

    def test_positive_headline(self, sample_headlines):
        from src.models.vader_model import VADERSentiment
        vader = VADERSentiment(use_financial_lexicon=True)
        # "Apple beats Q3 earnings estimates by 15%, stock surges"
        result = vader.score_text(sample_headlines[0])
        assert result["compound"] > 0, "Positive headline should have positive compound"

    def test_negative_headline(self, sample_headlines):
        from src.models.vader_model import VADERSentiment
        vader = VADERSentiment(use_financial_lexicon=True)
        # "Recession fears grow as yield curve inverts"
        result = vader.score_text(sample_headlines[8])
        assert result["compound"] < 0, "Negative headline should have negative compound"

    def test_batch_scoring(self, sample_headlines):
        from src.models.vader_model import VADERSentiment
        vader = VADERSentiment()
        results = vader.score_batch(sample_headlines)
        assert len(results) == len(sample_headlines)
        assert all("compound" in r for r in results)

    def test_compound_range(self, sample_headlines):
        from src.models.vader_model import VADERSentiment
        vader = VADERSentiment()
        scores = vader.score_batch(sample_headlines, return_compound_only=True)
        assert np.all(scores >= -1.0) and np.all(scores <= 1.0)

    def test_empty_text(self):
        from src.models.vader_model import VADERSentiment
        vader = VADERSentiment()
        result = vader.score_text("")
        assert result["compound"] == 0.0
        assert result["neu"] == 1.0

    def test_financial_lexicon_override(self):
        from src.models.vader_model import VADERSentiment
        # "bull" should be positive with financial lexicon
        vader_fin = VADERSentiment(use_financial_lexicon=True)
        result = vader_fin.score_text("The market is very bull right now")
        assert result["compound"] > 0

    def test_dataframe_scoring(self, sample_headlines):
        from src.models.vader_model import VADERSentiment
        vader = VADERSentiment()
        df = pd.DataFrame({"headline": sample_headlines})
        scored = vader.score_dataframe(df)
        assert "vader_compound" in scored.columns
        assert "vader_pos" in scored.columns
        assert len(scored) == len(sample_headlines)


# ============================================================================
# TEST: LOUGHRAN-MCDONALD DICTIONARY
# ============================================================================

class TestLMDictionary:
    """Tests for Loughran-McDonald financial lexicon."""

    def test_positive_text(self):
        from src.models.lm_dictionary import LMDictionarySentiment
        lm = LMDictionarySentiment()
        result = lm.score_text("The company achieved excellent profit growth")
        assert result["score"] > 0
        assert result["n_positive"] > 0

    def test_negative_text(self):
        from src.models.lm_dictionary import LMDictionarySentiment
        lm = LMDictionarySentiment()
        result = lm.score_text("Bankruptcy filing due to fraud and default")
        assert result["score"] < 0
        assert result["n_negative"] > 0

    def test_uncertainty_detection(self):
        from src.models.lm_dictionary import LMDictionarySentiment
        lm = LMDictionarySentiment(use_uncertainty=True)
        result = lm.score_text("The forecast is uncertain and may fluctuate")
        assert result["n_uncertainty"] > 0

    def test_empty_document(self):
        from src.models.lm_dictionary import LMDictionarySentiment
        lm = LMDictionarySentiment()
        result = lm.score_text("")
        assert result["score"] == 0.0
        assert result["n_words"] == 0

    def test_score_bounded(self):
        from src.models.lm_dictionary import LMDictionarySentiment
        lm = LMDictionarySentiment()
        result = lm.score_text("Profit gain improve benefit achieve excellent")
        assert -1.0 <= result["score"] <= 1.0


# ============================================================================
# TEST: DATA ACQUISITION (SYNTHETIC)
# ============================================================================

class TestSyntheticNews:
    """Tests for synthetic news generation."""

    def test_generation(self, synthetic_news):
        assert len(synthetic_news) > 100
        assert "headline" in synthetic_news.columns
        assert "ticker" in synthetic_news.columns
        assert "datetime" in synthetic_news.columns

    def test_tickers_present(self, synthetic_news):
        tickers = synthetic_news["ticker"].unique()
        assert "AAPL" in tickers
        assert "MSFT" in tickers
        assert "GOOGL" in tickers

    def test_sentiment_distribution(self, synthetic_news):
        # True sentiment should be roughly 30/25/45
        counts = synthetic_news["true_sentiment"].value_counts(normalize=True)
        assert counts.get(1, 0) > 0.2   # Positive > 20%
        assert counts.get(-1, 0) > 0.15  # Negative > 15%
        assert counts.get(0, 0) > 0.3    # Neutral > 30%

    def test_date_range(self, synthetic_news):
        min_date = synthetic_news["datetime"].min()
        max_date = synthetic_news["datetime"].max()
        assert min_date.year == 2022
        assert max_date.year == 2022


# ============================================================================
# TEST: FEATURE ENGINEERING
# ============================================================================

class TestFeatureEngineering:
    """Tests for sentiment feature construction."""

    def test_daily_aggregation(self, synthetic_news):
        from src.models.vader_model import VADERSentiment
        from src.feature_engineering import SentimentFeatureEngine

        vader = VADERSentiment()
        scored = vader.score_dataframe(synthetic_news, text_column="headline")
        scored["ensemble_score"] = scored["vader_compound"]
        scored["date"] = pd.to_datetime(scored["datetime"]).dt.date

        engine = SentimentFeatureEngine()
        daily = engine.aggregate_daily(scored, score_column="ensemble_score")

        assert "sent_mean" in daily.columns
        assert "n_articles" in daily.columns
        assert len(daily) > 0

    def test_feature_computation(self, synthetic_news):
        from src.models.vader_model import VADERSentiment
        from src.feature_engineering import SentimentFeatureEngine

        vader = VADERSentiment()
        scored = vader.score_dataframe(synthetic_news, text_column="headline")
        scored["ensemble_score"] = scored["vader_compound"]
        scored["date"] = pd.to_datetime(scored["datetime"]).dt.date

        engine = SentimentFeatureEngine(ewma_halflife=3, zscore_window=21)
        daily = engine.aggregate_daily(scored, score_column="ensemble_score")
        daily["date"] = pd.to_datetime(daily["date"])
        features = engine.compute_features(daily)

        assert "sent_ewma" in features.columns
        assert "sent_momentum" in features.columns
        assert "sent_dispersion" in features.columns
        assert "news_volume" in features.columns

    def test_composite_signal_bounded(self, synthetic_news):
        from src.models.vader_model import VADERSentiment
        from src.feature_engineering import SentimentFeatureEngine

        vader = VADERSentiment()
        scored = vader.score_dataframe(synthetic_news, text_column="headline")
        scored["ensemble_score"] = scored["vader_compound"]
        scored["date"] = pd.to_datetime(scored["datetime"]).dt.date

        engine = SentimentFeatureEngine(zscore_window=21)
        daily = engine.aggregate_daily(scored, score_column="ensemble_score")
        daily["date"] = pd.to_datetime(daily["date"])
        features = engine.compute_features(daily)
        features = engine.build_composite_signal(features)

        # Composite signal is winsorized at +/- 3
        assert features["composite_signal"].max() <= 3.0
        assert features["composite_signal"].min() >= -3.0


# ============================================================================
# TEST: PERFORMANCE METRICS
# ============================================================================

class TestPerformanceMetrics:
    """Tests for performance evaluation."""

    def test_sharpe_ratio(self):
        from src.evaluation import PerformanceAnalyzer
        np.random.seed(42)
        n = 252
        ret = pd.Series(np.random.normal(0.0005, 0.01, n))
        bench = pd.Series(np.random.normal(0.0003, 0.015, n))
        results = {
            "strategy_returns": ret,
            "benchmark_returns": bench,
            "net_returns": ret,
            "cumulative": None,
        }
        analyzer = PerformanceAnalyzer(results)
        sr = analyzer.sharpe_ratio(ret)
        assert isinstance(sr, float)
        assert not np.isnan(sr)

    def test_max_drawdown(self):
        from src.evaluation import PerformanceAnalyzer
        # Create a series with known drawdown
        ret = pd.Series([0.10, -0.20, 0.05, -0.10, 0.15])
        results = {
            "strategy_returns": ret,
            "benchmark_returns": ret,
            "net_returns": ret,
        }
        analyzer = PerformanceAnalyzer(results)
        mdd = analyzer.max_drawdown(ret)
        assert mdd > 0  # Must be positive
        assert mdd <= 1.0

    def test_hit_rate(self):
        from src.evaluation import PerformanceAnalyzer
        ret = pd.Series([0.01, -0.005, 0.02, 0.005, -0.01])
        results = {
            "strategy_returns": ret,
            "benchmark_returns": ret,
            "net_returns": ret,
        }
        analyzer = PerformanceAnalyzer(results)
        hr = analyzer.hit_rate(ret)
        assert hr == 0.6  # 3 positive out of 5

    def test_var_cvar(self):
        from src.evaluation import PerformanceAnalyzer
        np.random.seed(42)
        ret = pd.Series(np.random.normal(0, 0.01, 1000))
        results = {
            "strategy_returns": ret,
            "benchmark_returns": ret,
            "net_returns": ret,
        }
        analyzer = PerformanceAnalyzer(results)
        var = analyzer.var_historical(ret, 0.95)
        cvar = analyzer.cvar_historical(ret, 0.95)
        assert var > 0
        assert cvar >= var  # CVaR >= VaR by definition


# ============================================================================
# TEST: STRATEGY (BASIC)
# ============================================================================

class TestStrategy:
    """Tests for strategy position generation."""

    def test_inverse_vol_weights(self, sample_returns):
        from src.strategy import SentimentStrategy
        strategy = SentimentStrategy()
        tickers = sample_returns.columns.tolist()
        date = sample_returns.index[-1]
        weights = strategy._inverse_vol_weights(sample_returns, tickers, date)
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        assert all(w > 0 for w in weights.values())

    def test_turnover_computation(self):
        from src.strategy import SentimentStrategy
        strategy = SentimentStrategy()
        positions = pd.DataFrame({
            "A": [0.5, 0.3, 0.6],
            "B": [-0.5, -0.3, -0.6],
        })
        turnover = strategy.compute_turnover(positions)
        assert len(turnover) == 3
        assert turnover.iloc[0] == 0  # First day: no prior


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
