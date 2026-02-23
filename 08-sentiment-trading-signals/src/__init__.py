"""
Sentiment Analysis for Trading Signals
=======================================
NLP pipeline for extracting tradeable signals from financial news
and social media using FinBERT, VADER, and Loughran-McDonald methods.

Modules:
    data_acquisition      - Multi-source news and social media collection
    models.vader_model    - VADER sentiment with financial context tuning
    models.finbert_model  - FinBERT transformer-based sentiment inference
    models.lm_dictionary  - Loughran-McDonald financial lexicon scoring
    feature_engineering   - Signal aggregation, EWMA, momentum, dispersion
    strategy              - Long-short portfolio with volatility targeting
    backtesting           - Walk-forward engine with transaction costs
    evaluation            - Performance analytics and factor attribution

Author: Jose Orlando Bobadilla Fuentes, CQF | MSc AI
"""

__version__ = "1.0.0"
__author__ = "Jose Orlando Bobadilla Fuentes"
