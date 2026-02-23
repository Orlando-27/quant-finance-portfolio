"""
Setup for Sentiment Analysis for Trading Signals.

Author: Jose Orlando Bobadilla Fuentes, CQF | MSc AI
"""
from setuptools import setup, find_packages

setup(
    name="sentiment-trading-signals",
    version="1.0.0",
    author="Jose Orlando Bobadilla Fuentes",
    description=(
        "NLP-driven trading strategy framework combining FinBERT, VADER, "
        "and Loughran-McDonald sentiment with systematic backtesting."
    ),
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.11.0",
        "transformers>=4.30.0",
        "torch>=2.0.0",
        "vaderSentiment>=3.3.2",
        "yfinance>=0.2.28",
        "matplotlib>=3.7.0",
        "plotly>=5.16.0",
    ],
    keywords=[
        "sentiment-analysis", "finbert", "nlp", "trading",
        "quantitative-finance", "alternative-data",
    ],
)
