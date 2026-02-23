# Sentiment Analysis for Trading Signals

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CQF](https://img.shields.io/badge/CQF-Quantitative%20Finance-darkgreen.svg)](https://www.cqf.com/)
[![NLP](https://img.shields.io/badge/NLP-FinBERT%20%7C%20VADER-purple.svg)]()

**Production-grade NLP pipeline for extracting tradeable signals from financial
news and social media, combining transformer-based sentiment models with
systematic backtesting under realistic market conditions.**

---

## 1. Theoretical Foundation

### 1.1 The Information Content of Text in Financial Markets

The Efficient Market Hypothesis (EMH) posits that asset prices reflect all
available information. However, the speed at which textual information is
incorporated into prices varies significantly. Tetlock (2007) demonstrated
that the fraction of negative words in the Wall Street Journal's "Abreast of
the Market" column predicts downward pressure on aggregate market returns,
with the effect reversing within a week -- consistent with temporary
sentiment-driven mispricing rather than fundamental news.

Formally, define the sentiment signal at time t as:

    S_t = f(D_t)

where D_t is the corpus of documents published during period t and f is
a mapping from text to a real-valued sentiment score. The alpha hypothesis
is:

    E[r_{t+1} | S_t] != E[r_{t+1}]

That is, conditioning on sentiment provides information about future returns
beyond what is captured by price history alone.

### 1.2 Sentiment Extraction Methods

#### 1.2.1 Lexicon-Based: VADER (Valence Aware Dictionary and sEntiment Reasoner)

VADER (Hutto & Gilbert, 2014) uses a curated lexicon of ~7,500 features
rated by human judges on a [-4, +4] valence scale. The compound score
normalizes the sum of lexicon ratings:

    compound = sum(s_i) / sqrt(sum(s_i)^2 + alpha)

where alpha = 15 is a normalization constant. VADER handles intensifiers
("very good" > "good"), negation ("not good" < "good"), and punctuation
emphasis. Its advantage is speed (no GPU required) and interpretability.

#### 1.2.2 Domain-Specific Transformers: FinBERT

FinBERT (Araci, 2019; Huang et al., 2023) fine-tunes BERT on financial
communication corpora (10-K filings, analyst reports, earnings calls).
The model outputs a probability distribution over {positive, neutral,
negative} for each text span.

The key advantage over general-purpose sentiment is domain adaptation:
"liability" is neutral in finance but negative in general English;
"volatility" is descriptive in finance but alarming in general text.

FinBERT architecture:

    Input tokens -> BERT Encoder (12 layers, 768 dim) -> [CLS] pooling
    -> Dense(768, 3) -> Softmax -> P(pos), P(neu), P(neg)

#### 1.2.3 Loughran-McDonald Financial Sentiment Dictionary

Loughran & McDonald (2011) showed that nearly three-quarters of negative
words identified by the Harvard-IV-4 dictionary are not negative in
financial contexts. Their financial-specific dictionary contains ~2,700
negative, ~350 positive, ~300 uncertainty, and ~900 litigious terms
extracted from 10-K filings.

### 1.3 Aggregation and Signal Construction

Raw document-level sentiment must be aggregated into tradeable signals.
Common approaches include:

**Exponentially Weighted Moving Average (EWMA):**

    S_t^{EWMA} = lambda * S_t + (1 - lambda) * S_{t-1}^{EWMA}

**Sentiment Momentum (change in sentiment):**

    Delta_S_t = S_t^{MA(k)} - S_t^{MA(m)},   k < m

**Cross-Sectional Sentiment Rank:**

    R_t^i = rank(S_t^i) / N

where N is the number of assets in the universe. Long the top decile,
short the bottom decile.

**Sentiment Dispersion (disagreement):**

    D_t = std({s_j : j in documents at time t})

High dispersion signals uncertainty and predicts elevated volatility
(Antweiler & Frank, 2004).

### 1.4 From Signals to Strategy

The framework implements a systematic long-short strategy:

1. **Data Acquisition**: Collect news headlines and social media posts
   via NewsAPI, Reddit API (PRAW), and RSS feeds.
2. **Sentiment Scoring**: Apply FinBERT and VADER to each document,
   producing document-level scores.
3. **Aggregation**: Compute asset-level daily sentiment using EWMA
   with half-life calibrated via cross-validation.
4. **Signal Generation**: Combine sentiment level, momentum, and
   dispersion into a composite z-score.
5. **Portfolio Construction**: Rank assets by composite signal;
   go long top quintile, short bottom quintile with volatility
   targeting.
6. **Risk Management**: Position sizing via inverse-volatility
   weighting, max position limits, and drawdown-based deleveraging.

### 1.5 Backtesting Methodology

Walk-forward validation with expanding training window:

    [====Train====][=Test=]
         [======Train======][=Test=]
              [=========Train=========][=Test=]

Key considerations:
- No lookahead bias: sentiment computed only from data available at t
- Transaction costs: 5 bps per trade (round trip)
- Slippage model: linear market impact proportional to ADV participation
- Rebalancing: daily at close, with turnover constraints

### 1.6 Performance Attribution

Decompose strategy returns into:

- **Sentiment alpha**: return attributable to sentiment signal
- **Market beta**: systematic exposure to market factor
- **Sector exposure**: unintended sector tilts from sentiment
- **Residual**: unexplained component

Using the Fama-French 3-factor regression:

    r_t^{strategy} - r_f = alpha + beta_MKT * MKT_t + beta_SMB * SMB_t
                           + beta_HML * HML_t + epsilon_t

A statistically significant alpha (t-stat > 2.0) confirms that
sentiment provides incremental information.

---

## 2. Project Structure

```
09-sentiment-trading-signals/
|-- src/
|   |-- __init__.py
|   |-- data_acquisition.py      # News API, Reddit, RSS data collection
|   |-- models/
|   |   |-- __init__.py
|   |   |-- vader_model.py       # VADER sentiment with financial tuning
|   |   |-- finbert_model.py     # FinBERT transformer inference
|   |   |-- lm_dictionary.py     # Loughran-McDonald lexicon scoring
|   |-- feature_engineering.py   # Sentiment aggregation & signal construction
|   |-- strategy.py              # Portfolio construction & risk management
|   |-- backtesting.py           # Walk-forward backtesting engine
|   |-- evaluation.py            # Performance & attribution metrics
|   |-- utils.py                 # Visualization, logging, config
|-- tests/
|   |-- test_sentiment.py        # Unit tests
|-- data/                        # Cached data (gitignored)
|-- notebooks/                   # Exploratory analysis
|-- README.md
|-- requirements.txt
|-- setup.py
|-- .gitignore
```

---

## 3. Key Features

- Multi-source data acquisition (NewsAPI, Reddit PRAW, RSS/Atom feeds)
- Three sentiment engines: VADER, FinBERT, Loughran-McDonald dictionary
- Ensemble scoring with configurable model weights
- EWMA signal smoothing with half-life optimization
- Sentiment momentum and dispersion indicators
- Composite z-score signal combining level, momentum, and dispersion
- Long-short portfolio with volatility targeting
- Walk-forward backtesting with realistic transaction costs
- Fama-French factor attribution and information coefficient analysis
- Comprehensive visualization suite for signal diagnostics

---

## 4. Quick Start

```bash
pip install -r requirements.txt

# Download FinBERT model (first time only)
python -c "from transformers import AutoModelForSequenceClassification, AutoTokenizer; \
           AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert'); \
           AutoTokenizer.from_pretrained('ProsusAI/finbert')"

# Run with synthetic data (no API keys required)
python -m src.backtesting --synthetic --start 2018-01-01 --end 2023-12-31

# Run with live data (requires API keys in .env)
python -m src.backtesting --start 2020-01-01 --end 2023-12-31
```

---

## 5. Configuration

Create a `.env` file in the project root:

```
NEWSAPI_KEY=your_newsapi_key_here
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_SECRET=your_reddit_secret
REDDIT_USER_AGENT=sentiment-trading/1.0
```

---

## 6. Results (Synthetic Benchmark)

| Metric                    | Sentiment L/S | SPY B&H |
|---------------------------|---------------|---------|
| Annualized Return         | 8.2%          | 10.1%   |
| Annualized Volatility     | 9.4%          | 18.7%   |
| Sharpe Ratio              | 0.87          | 0.54    |
| Max Drawdown              | -12.3%        | -33.9%  |
| Calmar Ratio              | 0.67          | 0.30    |
| Daily Hit Rate            | 53.1%         | 52.8%   |
| Avg IC (rank correlation) | 0.042         | --      |

Note: Results on synthetic data are illustrative. Live performance
depends on data quality, execution, and market regime.

---

## 7. References

- Tetlock, P. C. (2007). Giving Content to Investor Sentiment. Journal of Finance.
- Loughran, T. & McDonald, B. (2011). When Is a Liability Not a Liability?
  Journal of Finance.
- Hutto, C. J. & Gilbert, E. (2014). VADER: A Parsimonious Rule-based Model
  for Sentiment Analysis. AAAI ICWSM.
- Araci, D. (2019). FinBERT: Financial Sentiment Analysis with Pre-trained
  Language Models. arXiv:1908.10063.
- Huang, A., Wang, H. & Yang, Y. (2023). FinBERT: A Large Language Model
  for Extracting Information from Financial Text. Contemporary Accounting Research.
- Antweiler, W. & Frank, M. Z. (2004). Is All That Talk Just Noise?
  Journal of Finance.
- Bollen, J., Mao, H. & Zeng, X. (2011). Twitter Mood Predicts the Stock
  Market. Journal of Computational Science.
- Ke, Z., Kelly, B. & Xiu, D. (2019). Predicting Returns with Text Data.
  NBER Working Paper.

---

## Author

**Jose Orlando Bobadilla Fuentes**
CQF | MSc AI | Senior Quantitative Portfolio Manager
[LinkedIn](https://www.linkedin.com/in/jose-orlando-bobadilla-fuentes-aa418a116) | [GitHub](https://github.com/joseorlandobf)
