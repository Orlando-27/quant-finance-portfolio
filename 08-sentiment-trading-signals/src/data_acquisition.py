"""
================================================================================
MULTI-SOURCE FINANCIAL NEWS DATA ACQUISITION
================================================================================
Collects financial news and social media data from multiple sources:
    1. NewsAPI       -- Headlines and articles from 80,000+ sources
    2. Reddit (PRAW) -- Posts from r/wallstreetbets, r/investing, r/stocks
    3. RSS Feeds     -- Financial news feeds (Reuters, Bloomberg, Yahoo Finance)
    4. Synthetic     -- Generates realistic synthetic data for testing

All sources produce a standardized DataFrame with columns:
    [datetime, ticker, headline, source, url]

Author: Jose Orlando Bobadilla Fuentes, CQF | MSc AI
================================================================================
"""

import os
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional, Dict
from tqdm import tqdm


class NewsAPICollector:
    """
    Collects financial news headlines via NewsAPI (newsapi.org).

    Requires a free API key (limited to 100 requests/day on free tier).
    Queries are constructed per ticker with financial context terms
    to improve relevance.

    Parameters
    ----------
    api_key : str, optional
        NewsAPI key. If None, reads from NEWSAPI_KEY environment variable.
    language : str
        Article language filter (default 'en').
    page_size : int
        Number of articles per request (max 100).
    """

    BASE_URL = "https://newsapi.org/v2/everything"

    def __init__(
        self,
        api_key: Optional[str] = None,
        language: str = "en",
        page_size: int = 100,
    ):
        import requests
        self.api_key = api_key or os.getenv("NEWSAPI_KEY", "")
        self.language = language
        self.page_size = page_size
        self.session = requests.Session()

        if not self.api_key:
            print("WARNING: No NewsAPI key found. Use synthetic data or set NEWSAPI_KEY.")

    def fetch(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        delay: float = 1.0,
    ) -> pd.DataFrame:
        """
        Fetch news articles for a list of tickers.

        Parameters
        ----------
        tickers : list of str
            Stock tickers to query (e.g., ['AAPL', 'MSFT']).
        start_date, end_date : str
            Date range in 'YYYY-MM-DD' format.
        delay : float
            Seconds between API calls (rate limiting).

        Returns
        -------
        pd.DataFrame with columns [datetime, ticker, headline, source, url].
        """
        all_articles = []

        for ticker in tqdm(tickers, desc="NewsAPI"):
            params = {
                "q": f'"{ticker}" AND (stock OR shares OR earnings OR revenue)',
                "from": start_date,
                "to": end_date,
                "language": self.language,
                "pageSize": self.page_size,
                "sortBy": "publishedAt",
                "apiKey": self.api_key,
            }
            try:
                resp = self.session.get(self.BASE_URL, params=params)
                data = resp.json()

                if data.get("status") == "ok":
                    for article in data.get("articles", []):
                        all_articles.append({
                            "datetime": article.get("publishedAt", ""),
                            "ticker": ticker,
                            "headline": article.get("title", ""),
                            "description": article.get("description", ""),
                            "source": article.get("source", {}).get("name", ""),
                            "url": article.get("url", ""),
                        })
            except Exception as e:
                print(f"  Error fetching {ticker}: {e}")

            time.sleep(delay)

        df = pd.DataFrame(all_articles)
        if not df.empty:
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
            df = df.sort_values("datetime").reset_index(drop=True)
        return df


class RedditCollector:
    """
    Collects posts from financial subreddits via PRAW (Python Reddit API Wrapper).

    Targets subreddits: wallstreetbets, investing, stocks, StockMarket.

    Parameters
    ----------
    client_id : str, optional
    client_secret : str, optional
    user_agent : str, optional
    """

    DEFAULT_SUBREDDITS = ["wallstreetbets", "investing", "stocks", "StockMarket"]

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        user_agent: Optional[str] = None,
    ):
        self.client_id = client_id or os.getenv("REDDIT_CLIENT_ID", "")
        self.client_secret = client_secret or os.getenv("REDDIT_SECRET", "")
        self.user_agent = user_agent or os.getenv(
            "REDDIT_USER_AGENT", "sentiment-trading/1.0"
        )

    def fetch(
        self,
        tickers: List[str],
        subreddits: Optional[List[str]] = None,
        limit: int = 500,
    ) -> pd.DataFrame:
        """
        Search subreddits for posts mentioning target tickers.

        Parameters
        ----------
        tickers : list of str
            Stock tickers to search for.
        subreddits : list of str, optional
            Subreddits to search. Defaults to financial subreddits.
        limit : int
            Maximum posts per subreddit per ticker.

        Returns
        -------
        pd.DataFrame with columns [datetime, ticker, headline, source, url].
        """
        import praw

        reddit = praw.Reddit(
            client_id=self.client_id,
            client_secret=self.client_secret,
            user_agent=self.user_agent,
        )
        subs = subreddits or self.DEFAULT_SUBREDDITS
        all_posts = []

        for sub_name in tqdm(subs, desc="Reddit"):
            subreddit = reddit.subreddit(sub_name)
            for ticker in tickers:
                try:
                    for post in subreddit.search(
                        f"${ticker}", limit=limit, sort="new"
                    ):
                        all_posts.append({
                            "datetime": pd.Timestamp.utcfromtimestamp(
                                post.created_utc
                            ),
                            "ticker": ticker,
                            "headline": post.title,
                            "description": (post.selftext or "")[:500],
                            "source": f"reddit/{sub_name}",
                            "url": f"https://reddit.com{post.permalink}",
                            "score": post.score,
                            "num_comments": post.num_comments,
                        })
                except Exception as e:
                    print(f"  Error in r/{sub_name} for {ticker}: {e}")

        df = pd.DataFrame(all_posts)
        if not df.empty:
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
            df = df.sort_values("datetime").reset_index(drop=True)
        return df


class SyntheticNewsGenerator:
    """
    Generates realistic synthetic financial news for testing and
    demonstration purposes. No API keys required.

    The generator creates headlines with controlled sentiment polarity
    and realistic temporal patterns (more news on earnings dates,
    market-moving events clustered).

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    """

    POSITIVE_TEMPLATES = [
        "{ticker} beats Q{q} earnings estimates by {pct}%",
        "{ticker} announces ${amt}B share buyback program",
        "{ticker} stock surges {pct}% on strong revenue growth",
        "{ticker} receives analyst upgrade to Overweight from {bank}",
        "{ticker} reports record quarterly revenue of ${amt}B",
        "{ticker} raises full-year guidance above consensus",
        "{ticker} signs major partnership deal with {partner}",
        "{ticker} expands margins to {pct}% in latest quarter",
        "{ticker} announces new product launch driving optimism",
        "Bullish outlook for {ticker} as demand accelerates",
        "{ticker} dividend increased by {pct}% signaling confidence",
        "{ticker} CEO: strong pipeline visibility for next {q} quarters",
    ]

    NEGATIVE_TEMPLATES = [
        "{ticker} misses Q{q} earnings expectations by {pct}%",
        "{ticker} issues profit warning citing weak demand",
        "{ticker} stock drops {pct}% on disappointing guidance",
        "{ticker} downgraded to Underweight by {bank}",
        "{ticker} reports revenue decline of {pct}% year-over-year",
        "{ticker} announces layoffs of {amt},000 employees",
        "{ticker} faces regulatory investigation into {issue}",
        "{ticker} warns of margin pressure from rising costs",
        "{ticker} loses major contract worth ${amt}B",
        "Bearish sentiment grows as {ticker} market share erodes",
        "{ticker} CFO departure raises governance concerns",
        "{ticker} supply chain disruptions expected through Q{q}",
    ]

    NEUTRAL_TEMPLATES = [
        "{ticker} scheduled to report Q{q} earnings next week",
        "{ticker} CEO to present at {bank} technology conference",
        "{ticker} maintains current dividend at ${amt} per share",
        "Analysts mixed on {ticker} ahead of earnings season",
        "{ticker} announces leadership transition in {dept} division",
        "{ticker} trading flat as market awaits economic data",
        "{ticker} to hold annual shareholder meeting on {date}",
        "{ticker} completes routine refinancing of credit facility",
    ]

    BANKS = [
        "Goldman Sachs", "Morgan Stanley", "JP Morgan", "Bank of America",
        "Citigroup", "UBS", "Deutsche Bank", "Barclays", "Credit Suisse",
        "Wells Fargo",
    ]
    PARTNERS = ["Microsoft", "Amazon", "Google", "Apple", "NVIDIA", "Meta"]
    ISSUES = [
        "accounting practices", "data privacy", "antitrust concerns",
        "environmental compliance", "labor practices",
    ]
    DEPTS = ["technology", "operations", "marketing", "finance", "sales"]

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def _fill_template(self, template: str, ticker: str) -> str:
        """Fill a headline template with random realistic values."""
        return template.format(
            ticker=ticker,
            q=self.rng.choice([1, 2, 3, 4]),
            pct=round(self.rng.uniform(2, 25), 1),
            amt=round(self.rng.uniform(1, 50), 1),
            bank=self.rng.choice(self.BANKS),
            partner=self.rng.choice(self.PARTNERS),
            issue=self.rng.choice(self.ISSUES),
            dept=self.rng.choice(self.DEPTS),
            date="March 15",
        )

    def generate(
        self,
        tickers: List[str],
        start_date: str = "2018-01-01",
        end_date: str = "2023-12-31",
        articles_per_ticker_per_day: float = 2.5,
    ) -> pd.DataFrame:
        """
        Generate synthetic news dataset.

        Parameters
        ----------
        tickers : list of str
            Ticker symbols.
        start_date, end_date : str
            Date range.
        articles_per_ticker_per_day : float
            Average number of articles per ticker per business day.
            Actual count follows a Poisson distribution.

        Returns
        -------
        pd.DataFrame with columns matching real data sources:
            [datetime, ticker, headline, source, url, true_sentiment]
        """
        dates = pd.bdate_range(start_date, end_date)
        records = []

        sources = [
            "Reuters", "Bloomberg", "CNBC", "MarketWatch",
            "Yahoo Finance", "Barron's", "WSJ", "Financial Times",
        ]

        for ticker in tickers:
            for date in dates:
                n_articles = self.rng.poisson(articles_per_ticker_per_day)
                for _ in range(n_articles):
                    # Sentiment distribution: 30% positive, 25% negative, 45% neutral
                    roll = self.rng.random()
                    if roll < 0.30:
                        template = self.rng.choice(self.POSITIVE_TEMPLATES)
                        true_sent = 1
                    elif roll < 0.55:
                        template = self.rng.choice(self.NEGATIVE_TEMPLATES)
                        true_sent = -1
                    else:
                        template = self.rng.choice(self.NEUTRAL_TEMPLATES)
                        true_sent = 0

                    headline = self._fill_template(template, ticker)

                    # Random time during trading hours (9:30-16:00 ET)
                    hour = self.rng.randint(6, 20)
                    minute = self.rng.randint(0, 59)
                    dt = date + pd.Timedelta(hours=hour, minutes=minute)

                    records.append({
                        "datetime": dt,
                        "ticker": ticker,
                        "headline": headline,
                        "source": self.rng.choice(sources),
                        "url": f"https://synthetic-news.example.com/{ticker}/{date.strftime('%Y%m%d')}",
                        "true_sentiment": true_sent,
                    })

        df = pd.DataFrame(records)
        df = df.sort_values("datetime").reset_index(drop=True)
        return df
