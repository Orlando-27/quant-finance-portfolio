#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULE 44 -- NLP SENTIMENT ANALYSIS OF FINANCIAL HEADLINES
=============================================================================
CQF Concepts Explained: Interactive Jupyter Notebooks
Project 19 of 20 -- Quantitative Finance Portfolio
Author  : Jose Orlando Bobadilla Fuentes | CQF | MSc AI Applied to Fin. Markets
Role    : Senior Quantitative Portfolio Manager & Lead Data Scientist
Firm    : Colombian Pension Fund -- Vicepresidencia de Inversiones

ACADEMIC OVERVIEW
-----------------
Natural Language Processing (NLP) applies computational methods to extract
structured information from unstructured text.  In quantitative finance, the
primary task is *sentiment analysis*: mapping a news headline or report into
a scalar signal s in [-1, +1] that proxies the market impact of the text.

THREE LAYERS OF SENTIMENT
--------------------------
1. Lexicon-based (rule-based):
   For each token t in document d, look up a pre-defined polarity score p(t).
   Aggregate:  s = (1/|d|) * sum_t p(t)
   Examples: VADER, Harvard IV-4, Loughran-McDonald (finance-specific).

2. Statistical (bag-of-words + classifier):
   Represent document as TF-IDF vector, train logistic regression or naive
   Bayes classifier on labelled examples.
   TF-IDF(t, d) = tf(t,d) * log(N / df(t))

3. Transformer-based (contextual embeddings):
   Pre-trained models (BERT, FinBERT) produce context-aware token embeddings.
   Fine-tuned on financial corpora for superior performance.
   (Implemented conceptually here; full inference requires GPU environment.)

LOUGHRAN-MCDONALD FINANCIAL LEXICON
-------------------------------------
Loughran & McDonald (2011) show that general-purpose lexicons misclassify
many finance terms.  Words like "liability", "risk", "loss" are negative in
finance but neutral in Harvard IV.  The LM lexicon provides:
    - Negative (2355 words)   -- e.g. "default", "impairment", "restatement"
    - Positive (354 words)    -- e.g. "profitable", "exceed", "record"
    - Uncertainty (297 words) -- e.g. "approximate", "contingent"
    - Litigious (903 words)   -- legal risk language

SIGNAL CONSTRUCTION PIPELINE
------------------------------
Step 1: Parse headline text into tokens (lowercase, strip punctuation)
Step 2: Score each token against lexicon -> raw score
Step 3: Aggregate to document-level polarity: s_d = (pos - neg) / (pos + neg + 1)
Step 4: Smooth over time: S_t = EWM(s_d, span=5)
Step 5: Compute IC = corr(S_{t-1}, r_t) over rolling 60-day window
Step 6: Construct sentiment-based trading signal and backtest

INFORMATION COEFFICIENT (IC)
------------------------------
IC_t = Spearman_rho(S_{t-1}, r_t)

IC > 0 means sentiment predicts next-day return direction.
IC > 0.05 is considered economically significant in factor research.
ICIR = mean(IC) / std(IC) -- analogous to Sharpe ratio of the signal.

REFERENCES
----------
[1] Loughran, T. & McDonald, B. (2011). "When Is a Liability Not a Liability?"
    Journal of Finance 66(1):35-65.
[2] Tetlock, P. (2007). "Giving Content to Investor Sentiment." JF 62(3):1139-68.
[3] Malo, P. et al. (2014). "Good Debt or Bad Debt: Detecting Semantic
    Orientations in Economic Texts." JASIST 65(4):782-796.
[4] Hutto, C. & Gilbert, E. (2014). "VADER: A Parsimonious Rule-based Model
    for Sentiment Analysis." ICWSM.
=============================================================================
Usage (Cloud Shell):
    cd ~/quant-finance-portfolio/19-cqf-concepts-explained
    python src/m44_nlp_sentiment/m44_nlp_sentiment.py
=============================================================================
"""

import os
import re
import string
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import spearmanr
import yfinance as yf

warnings.filterwarnings("ignore")
np.random.seed(42)

# =============================================================================
# PATHS
# =============================================================================
BASE  = os.path.dirname(os.path.abspath(__file__))
ROOT  = os.path.abspath(os.path.join(BASE, "..", ".."))
FIGS  = os.path.join(ROOT, "outputs", "figures", "m44")
os.makedirs(FIGS, exist_ok=True)

# =============================================================================
# DARK THEME
# =============================================================================
DARK   = "#0d1117"
PANEL  = "#161b22"
GRID   = "#21262d"
TEXT   = "#e6edf3"
ACCENT = "#58a6ff"
GREEN  = "#3fb950"
RED    = "#f85149"
AMBER  = "#d29922"
VIOLET = "#bc8cff"
TEAL   = "#39d353"

plt.rcParams.update({
    "figure.facecolor" : DARK,  "axes.facecolor"  : PANEL,
    "axes.edgecolor"   : GRID,  "axes.labelcolor" : TEXT,
    "axes.titlecolor"  : TEXT,  "xtick.color"     : TEXT,
    "ytick.color"      : TEXT,  "text.color"      : TEXT,
    "grid.color"       : GRID,  "grid.linestyle"  : "--",
    "grid.alpha"       : 0.5,   "legend.facecolor": PANEL,
    "legend.edgecolor" : GRID,  "font.family"     : "monospace",
    "font.size"        : 9,     "axes.titlesize"  : 10,
})

def section(n, msg): print(f"  [{n:02d}] {msg}")

# =============================================================================
# PRINT HEADER
# =============================================================================
print()
print("=" * 65)
print("  MODULE 44: NLP SENTIMENT ANALYSIS")
print("  Lexicon | TF-IDF | IC | Signal Construction | Backtest")
print("=" * 65)

# =============================================================================
# 1.  MARKET DATA
# =============================================================================
raw   = yf.download("SPY", start="2018-01-01", end="2023-12-31",
                    auto_adjust=True, progress=False)
close = raw["Close"].squeeze().dropna()
ret   = np.log(close / close.shift(1)).dropna()
dates = ret.index
N     = len(ret)

section(1, f"SPY returns: {N} days  [{dates[0].date()} -- {dates[-1].date()}]")

# =============================================================================
# 2.  LOUGHRAN-MCDONALD FINANCIAL LEXICON (representative subset)
#     Full lexicon: https://sraf.nd.edu/loughranmcdonald-master-dictionary/
#     We use a curated subset sufficient for demonstration.
# =============================================================================
LM_NEGATIVE = {
    "loss","losses","decline","declines","declined","decrease","decreased",
    "decreases","default","defaults","defaulted","risk","risks","risky",
    "volatile","volatility","impairment","impairments","restatement",
    "restatements","lawsuit","lawsuits","litigation","litigious","penalty",
    "penalties","fail","failed","failure","failures","deficit","deficits",
    "downturn","downturns","weak","weakened","weakness","weaknesses",
    "concern","concerns","uncertainty","uncertain","miss","missed","below",
    "unfavorable","adverse","adversely","layoff","layoffs","restructuring",
    "write-off","write-offs","writedown","writedowns","fraud","fraudulent",
    "investigation","investigations","suspend","suspended","suspension",
    "downgrade","downgraded","downfall","shortfall","shortfalls","warning",
    "warnings","negative","negatively","slowdown","slowdowns","recession",
    "recessionary","bankrupt","bankruptcy","insolvency","insolvent",
    "deterioration","deteriorate","deteriorated","challenge","challenges",
    "difficult","difficulties","difficult","headwind","headwinds",
}

LM_POSITIVE = {
    "profit","profits","profitable","profitability","gain","gains","grew",
    "grow","growth","increase","increased","increases","exceed","exceeded",
    "exceeds","record","strong","strength","strengthen","strengthened",
    "improve","improved","improvement","improvements","positive","positively",
    "favorable","outperform","outperformed","outperformance","beat","beats",
    "advance","advanced","advances","recovery","recover","recovered",
    "expansion","expand","expanded","opportunity","opportunities","robust",
    "solid","exceed","milestone","milestones","innovative","innovation",
    "breakthrough","ahead","upbeat","upgrade","upgraded","accelerate",
    "accelerated","acceleration","optimistic","confidence","confident",
    "surpass","surpassed","exceed","dividend","dividends","buyback",
    "buybacks","momentum","efficient","efficiency","streamline","streamlined",
}

LM_UNCERTAINTY = {
    "approximately","approximate","contingent","contingency","uncertain",
    "uncertainty","unclear","unresolved","pending","preliminary","estimate",
    "estimated","estimates","projected","projection","may","might","could",
    "possibly","possibly","potential","potentially","subject","depends",
    "dependent","unknown","variable","variables","unpredictable",
}

section(2, f"LM lexicon loaded  neg={len(LM_NEGATIVE)}  pos={len(LM_POSITIVE)}  "
           f"unc={len(LM_UNCERTAINTY)}")

# =============================================================================
# 3.  SYNTHETIC HEADLINE CORPUS
#     We generate a realistic synthetic corpus of financial headlines
#     calibrated to SPY return signs. Each headline is constructed by
#     sampling tokens from positive, negative, and neutral vocabularies,
#     with the mix driven by the actual return on that day.
# =============================================================================
POSITIVE_PHRASES = [
    "earnings beat expectations",
    "revenue growth accelerates",
    "profit outlook upgraded",
    "record quarterly earnings",
    "strong consumer spending",
    "GDP growth exceeds forecast",
    "Fed signals policy easing",
    "jobs report beats forecast",
    "tech sector rally continues",
    "markets advance on optimism",
    "buyback program expanded",
    "dividend increased by board",
    "inflation falls below target",
    "manufacturing index improves",
    "trade deal progress reported",
]

NEGATIVE_PHRASES = [
    "earnings miss estimates",
    "revenue decline reported",
    "profit warning issued",
    "recession fears intensify",
    "Fed hikes rates aggressively",
    "unemployment rises sharply",
    "defaults surge in credit market",
    "banks face writedown pressure",
    "geopolitical tensions escalate",
    "supply chain disruption worsens",
    "inflation remains elevated risk",
    "growth slowdown concerns mount",
    "tech sector weakness deepens",
    "investigation launched by SEC",
    "trade war uncertainty weighs",
]

NEUTRAL_PHRASES = [
    "markets open mixed",
    "earnings season begins",
    "economic data released",
    "Fed meeting minutes published",
    "analysts revise estimates",
    "quarterly results due",
    "index rebalancing scheduled",
    "trading volume moderate",
    "sector rotation observed",
    "macro data awaited",
]

rng = np.random.default_rng(77)

def make_headline(ret_val: float, noise: float = 0.3) -> str:
    """
    Generate a synthetic headline whose sentiment is loosely correlated
    with the sign of `ret_val` plus Gaussian noise for realism.
    """
    signal = ret_val / (ret.std() + 1e-9)   # standardised return
    noisy  = signal + rng.normal(0, noise)
    if noisy > 0.5:
        phrase = rng.choice(POSITIVE_PHRASES)
    elif noisy < -0.5:
        phrase = rng.choice(NEGATIVE_PHRASES)
    else:
        phrase = rng.choice(NEUTRAL_PHRASES)
    # Add a random company/ticker context
    tickers = ["S&P 500", "equities", "US markets", "Wall Street",
               "tech stocks", "financials", "SPY", "blue chips"]
    ctx = rng.choice(tickers)
    return f"{ctx}: {phrase}"

headlines = [make_headline(r) for r in ret.values]
section(3, f"Synthetic corpus: {len(headlines)} headlines generated")

# =============================================================================
# 4.  TOKENISATION & LEXICON SCORING
# =============================================================================
def tokenize(text: str) -> list:
    """Lowercase, remove punctuation, split on whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.split()

def lm_score(text: str) -> dict:
    """
    Score a document against the LM financial lexicon.

    Returns
    -------
    dict with keys: pos_count, neg_count, unc_count, polarity, subjectivity
    """
    tokens = tokenize(text)
    n_tok  = max(len(tokens), 1)
    pos = sum(1 for t in tokens if t in LM_POSITIVE)
    neg = sum(1 for t in tokens if t in LM_NEGATIVE)
    unc = sum(1 for t in tokens if t in LM_UNCERTAINTY)
    # Polarity: normalised net sentiment
    polarity     = (pos - neg) / (pos + neg + 1)
    # Subjectivity: fraction of sentiment-laden tokens
    subjectivity = (pos + neg) / n_tok
    return {"pos": pos, "neg": neg, "unc": unc,
            "polarity": polarity, "subjectivity": subjectivity}

scores    = [lm_score(h) for h in headlines]
polarity  = np.array([s["polarity"]     for s in scores])
subjective= np.array([s["subjectivity"] for s in scores])
unc_score = np.array([s["unc"]          for s in scores])

section(4, f"LM scoring done  mean_polarity={polarity.mean():.4f}  "
           f"std={polarity.std():.4f}")

# =============================================================================
# 5.  TF-IDF + LOGISTIC REGRESSION CLASSIFIER
# =============================================================================
# Binary label: 1 if next-day return > 0, else 0
y_class = (ret.values > 0).astype(int)

# Walk-forward splits (financial time series -- no look-ahead)
tss     = TimeSeriesSplit(n_splits=5)
tfidf   = TfidfVectorizer(ngram_range=(1, 2), max_features=500,
                           sublinear_tf=True)
X_tfidf = tfidf.fit_transform(headlines)

auc_folds, acc_folds = [], []
lr_model = None
for train_idx, test_idx in tss.split(X_tfidf):
    Xtr = X_tfidf[train_idx]; Xte = X_tfidf[test_idx]
    ytr = y_class[train_idx]; yte = y_class[test_idx]
    clf = LogisticRegression(C=1.0, max_iter=200, random_state=42)
    clf.fit(Xtr, ytr)
    prob = clf.predict_proba(Xte)[:, 1]
    auc_folds.append(roc_auc_score(yte, prob))
    acc_folds.append((clf.predict(Xte) == yte).mean())
    lr_model = clf

mean_auc = np.mean(auc_folds)
mean_acc = np.mean(acc_folds)
section(5, f"TF-IDF + LR  mean_AUC={mean_auc:.3f}  mean_Acc={mean_acc:.3f}  "
           f"(5-fold walk-forward)")

# =============================================================================
# 6.  INFORMATION COEFFICIENT (IC) ANALYSIS
# =============================================================================
# Smooth polarity with EWM (span=5)
pol_ewm = np.zeros(N)
alpha   = 2 / (5 + 1)
pol_ewm[0] = polarity[0]
for i in range(1, N):
    pol_ewm[i] = alpha * polarity[i] + (1 - alpha) * pol_ewm[i-1]

# Rolling 60-day IC: Spearman(S_{t-1}, r_t)
win_ic = 60
ic_series = np.full(N, np.nan)
for i in range(win_ic, N):
    sig_w = pol_ewm[i-win_ic:i-1]
    ret_w = ret.values[i-win_ic+1:i]
    if len(sig_w) == len(ret_w) and len(sig_w) > 5:
        ic_series[i], _ = spearmanr(sig_w, ret_w)

ic_valid = ic_series[~np.isnan(ic_series)]
ic_mean  = ic_valid.mean()
ic_std   = ic_valid.std()
icir     = ic_mean / (ic_std + 1e-9)

section(6, f"IC analysis  mean_IC={ic_mean:.4f}  std_IC={ic_std:.4f}  ICIR={icir:.3f}")

# =============================================================================
# 7.  SENTIMENT-BASED TRADING SIGNAL + BACKTEST
# =============================================================================
# Signal: go long if pol_ewm > threshold, short if < -threshold, else flat
threshold = 0.01
signal    = np.zeros(N)
signal[pol_ewm >  threshold] =  1.0
signal[pol_ewm < -threshold] = -1.0

# Strategy returns (signal at t-1, return at t)
strat_ret = np.zeros(N)
strat_ret[1:] = signal[:-1] * ret.values[1:]

# Cumulative returns
cum_spy   = np.exp(np.cumsum(ret.values)) - 1
cum_strat = np.exp(np.cumsum(strat_ret))  - 1

# Performance metrics
ann       = 252
sharpe    = (strat_ret.mean() / (strat_ret.std() + 1e-9)) * np.sqrt(ann)
spy_sharpe= (ret.values.mean() / (ret.values.std() + 1e-9)) * np.sqrt(ann)
dd        = np.maximum.accumulate(np.exp(np.cumsum(strat_ret)))
max_dd    = ((dd - np.exp(np.cumsum(strat_ret))) / (dd + 1e-9)).max()
hit_rate  = (strat_ret[strat_ret != 0] > 0).mean()

section(7, f"Sentiment strategy  Sharpe={sharpe:.3f}  MaxDD={max_dd:.3f}  "
           f"HitRate={hit_rate:.3f}  (SPY Sharpe={spy_sharpe:.3f})")

# =============================================================================
# 8.  WORD FREQUENCY ANALYSIS
# =============================================================================
all_tokens  = [t for h in headlines for t in tokenize(h)]
token_freq  = Counter(all_tokens)
# Separate positive and negative token frequencies
pos_tokens  = [t for t in all_tokens if t in LM_POSITIVE]
neg_tokens  = [t for t in all_tokens if t in LM_NEGATIVE]
pos_freq    = Counter(pos_tokens).most_common(10)
neg_freq    = Counter(neg_tokens).most_common(10)

section(8, f"Token analysis  vocab={len(token_freq)}  "
           f"pos_hits={len(pos_tokens)}  neg_hits={len(neg_tokens)}")

# =============================================================================
# FIGURE 1: SENTIMENT SIGNAL + IC + CUMULATIVE RETURNS
# =============================================================================
fig = plt.figure(figsize=(16, 12), facecolor=DARK)
fig.suptitle("Module 44 -- NLP Sentiment: Signal, IC & Strategy Performance",
             fontsize=12, color=TEXT, y=0.99)
gs = gridspec.GridSpec(3, 2, figure=fig, wspace=0.35, hspace=0.45)

# 1A: Raw polarity vs EWM
ax = fig.add_subplot(gs[0, 0])
t  = np.arange(N)
ax.plot(t, polarity, color=ACCENT, lw=0.5, alpha=0.4, label="Raw polarity")
ax.plot(t, pol_ewm,  color=AMBER,  lw=1.2, label="EWM(span=5)")
ax.axhline(0, color=GRID, lw=0.8)
ax.axhline( threshold, color=GREEN, lw=0.8, ls="--", alpha=0.7)
ax.axhline(-threshold, color=RED,   lw=0.8, ls="--", alpha=0.7)
ax.set_title("LM Polarity Score Over Time")
ax.set_xlabel("Day index"); ax.set_ylabel("Polarity")
ax.legend(fontsize=7); ax.grid(True)

# 1B: Rolling IC
ax = fig.add_subplot(gs[0, 1])
ic_t = np.where(~np.isnan(ic_series), ic_series, 0)
ax.fill_between(t, ic_t, 0, where=ic_t > 0, color=GREEN, alpha=0.5, label="IC > 0")
ax.fill_between(t, ic_t, 0, where=ic_t < 0, color=RED,   alpha=0.5, label="IC < 0")
ax.plot(t, ic_t, color=ACCENT, lw=0.6, alpha=0.6)
ax.axhline(ic_mean, color=AMBER, lw=1.2, ls="--",
           label=f"mean IC={ic_mean:.4f}  ICIR={icir:.3f}")
ax.axhline(0, color=GRID, lw=0.8)
ax.set_title(f"Rolling 60-day IC (Spearman)")
ax.set_xlabel("Day index"); ax.set_ylabel("IC")
ax.legend(fontsize=7); ax.grid(True)

# 1C: Signal position
ax = fig.add_subplot(gs[1, 0])
long_mask  = signal == 1
short_mask = signal == -1
flat_mask  = signal == 0
ax.scatter(t[long_mask],  ret.values[long_mask],  color=GREEN, s=3, alpha=0.5, label="Long")
ax.scatter(t[short_mask], ret.values[short_mask], color=RED,   s=3, alpha=0.5, label="Short")
ax.scatter(t[flat_mask],  ret.values[flat_mask],  color=GRID,  s=2, alpha=0.3, label="Flat")
ax.axhline(0, color=GRID, lw=0.8)
ax.set_title("Signal Position vs Realised Return")
ax.set_xlabel("Day index"); ax.set_ylabel("Log-return")
ax.legend(fontsize=7, markerscale=3); ax.grid(True)

# 1D: Cumulative performance
ax = fig.add_subplot(gs[1, 1])
ax.plot(t, cum_strat * 100, color=ACCENT, lw=1.5, label=f"Sentiment  Sharpe={sharpe:.2f}")
ax.plot(t, cum_spy   * 100, color=GREEN,  lw=1.5, label=f"SPY B&H    Sharpe={spy_sharpe:.2f}")
ax.axhline(0, color=GRID, lw=0.8)
ax.set_title("Cumulative Return (%)")
ax.set_xlabel("Day index"); ax.set_ylabel("Return (%)")
ax.legend(fontsize=7); ax.grid(True)

# 1E: IC distribution
ax = fig.add_subplot(gs[2, 0])
ax.hist(ic_valid, bins=40, color=ACCENT, alpha=0.8, edgecolor=DARK, density=True)
ax.axvline(ic_mean, color=AMBER, lw=1.5, ls="--", label=f"mean={ic_mean:.4f}")
ax.axvline(0,       color=RED,   lw=1.0, ls="--", label="zero")
ax.set_title("IC Distribution"); ax.set_xlabel("IC"); ax.set_ylabel("Density")
ax.legend(fontsize=7); ax.grid(True)

# 1F: AUC walk-forward folds
ax = fig.add_subplot(gs[2, 1])
folds = np.arange(1, len(auc_folds) + 1)
bars = ax.bar(folds, auc_folds, color=VIOLET, edgecolor=DARK, width=0.6)
ax.axhline(0.5,      color=RED,  lw=1.0, ls="--", label="Random")
ax.axhline(mean_auc, color=AMBER, lw=1.2, ls="--", label=f"mean={mean_auc:.3f}")
for bar, v in zip(bars, auc_folds):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.003, f"{v:.3f}",
            ha="center", va="bottom", fontsize=7, color=TEXT)
ax.set_ylim(0.4, max(auc_folds) * 1.1)
ax.set_title("TF-IDF + LR: Walk-Forward AUC")
ax.set_xlabel("Fold"); ax.set_ylabel("ROC-AUC")
ax.legend(fontsize=7); ax.grid(True, axis="y")

fig.savefig(os.path.join(FIGS, "m44_fig1_sentiment_signal_performance.png"),
            dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close(fig)

# =============================================================================
# FIGURE 2: LEXICON ANALYSIS + WORD FREQUENCY
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor=DARK)
fig.suptitle("Module 44 -- Lexicon Analysis & Token Frequencies",
             fontsize=12, color=TEXT, y=1.01)

# 2A: Polarity vs next-day return scatter
ax = axes[0, 0]
ax.scatter(pol_ewm[:-1], ret.values[1:], s=5, alpha=0.3, color=ACCENT)
# Overlay binned means
bins_   = np.linspace(pol_ewm.min(), pol_ewm.max(), 12)
dig     = np.digitize(pol_ewm[:-1], bins_)
bin_ret = [ret.values[1:][dig == b].mean() for b in range(1, len(bins_))]
bin_mid = 0.5 * (bins_[:-1] + bins_[1:])
ax.plot(bin_mid, bin_ret, color=AMBER, lw=2.0, marker="o", ms=5, label="Binned mean")
ax.axhline(0, color=GRID, lw=0.8); ax.axvline(0, color=GRID, lw=0.8)
ax.set_title("Sentiment EWM vs Next-Day Return")
ax.set_xlabel("Polarity (EWM)"); ax.set_ylabel("Log-return t+1")
ax.legend(fontsize=7); ax.grid(True)

# 2B: Subjectivity distribution
ax = axes[0, 1]
ax.hist(subjective, bins=30, color=TEAL, alpha=0.8, edgecolor=DARK, density=True)
ax.axvline(subjective.mean(), color=AMBER, lw=1.5, ls="--",
           label=f"mean={subjective.mean():.4f}")
ax.set_title("Subjectivity Score Distribution")
ax.set_xlabel("Subjectivity"); ax.set_ylabel("Density")
ax.legend(fontsize=7); ax.grid(True)

# 2C: Top positive tokens
ax = axes[1, 0]
if pos_freq:
    words_p, counts_p = zip(*pos_freq)
    bars = ax.barh(list(words_p), list(counts_p), color=GREEN, edgecolor=DARK)
    for bar, v in zip(bars, counts_p):
        ax.text(v + 0.5, bar.get_y() + bar.get_height()/2,
                str(v), va="center", fontsize=7, color=TEXT)
ax.set_title("Top 10 Positive LM Tokens")
ax.set_xlabel("Frequency"); ax.invert_yaxis(); ax.grid(True, axis="x")

# 2D: Top negative tokens
ax = axes[1, 1]
if neg_freq:
    words_n, counts_n = zip(*neg_freq)
    bars = ax.barh(list(words_n), list(counts_n), color=RED, edgecolor=DARK)
    for bar, v in zip(bars, counts_n):
        ax.text(v + 0.5, bar.get_y() + bar.get_height()/2,
                str(v), va="center", fontsize=7, color=TEXT)
ax.set_title("Top 10 Negative LM Tokens")
ax.set_xlabel("Frequency"); ax.invert_yaxis(); ax.grid(True, axis="x")

for ax in axes.flat:
    ax.set_facecolor(PANEL)

fig.tight_layout()
fig.savefig(os.path.join(FIGS, "m44_fig2_lexicon_word_frequency.png"),
            dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close(fig)

# =============================================================================
# FIGURE 3: SENTIMENT PIPELINE DIAGRAM
# =============================================================================
fig, ax = plt.subplots(figsize=(14, 6), facecolor=DARK)
ax.set_facecolor(PANEL); ax.axis("off")
fig.suptitle("Module 44 -- Sentiment Analysis Pipeline",
             fontsize=12, color=TEXT, y=0.98)

pipeline = [
    ("Raw\nHeadline", 0.05, ACCENT),
    ("Tokenise\n+ Lowercase", 0.20, VIOLET),
    ("LM Lexicon\nLookup", 0.35, AMBER),
    ("Polarity\nScore", 0.50, GREEN),
    ("EWM\nSmoothing", 0.65, TEAL),
    ("IC\nEvaluation", 0.80, ACCENT),
    ("Trading\nSignal", 0.95, GREEN),
]

for label, x, color in pipeline:
    ax.add_patch(plt.Rectangle((x - 0.06, 0.35), 0.10, 0.30,
                                color=color, alpha=0.25,
                                transform=ax.transAxes, clip_on=False))
    ax.text(x, 0.50, label, transform=ax.transAxes,
            ha="center", va="center", fontsize=9,
            color=TEXT, fontfamily="monospace", fontweight="bold")
    if x < 0.95:
        ax.annotate("", xy=(x + 0.09, 0.50), xytext=(x + 0.06, 0.50),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color=TEXT, lw=1.5))

# Metrics box
metrics = (
    f"LM Polarity:    mean = {polarity.mean():.4f}\n"
    f"EWM Polarity:   std  = {pol_ewm.std():.4f}\n"
    f"Rolling IC:     mean = {ic_mean:.4f}  ICIR = {icir:.3f}\n"
    f"TF-IDF AUC:     {mean_auc:.3f}\n"
    f"Sentiment Sharpe: {sharpe:.3f}  (SPY: {spy_sharpe:.3f})\n"
    f"Max Drawdown:   {max_dd:.3f}  Hit Rate: {hit_rate:.3f}"
)
ax.text(0.50, 0.12, metrics, transform=ax.transAxes,
        ha="center", va="top", fontsize=9, color=TEXT,
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor=DARK,
                  edgecolor=ACCENT, alpha=0.9))

fig.savefig(os.path.join(FIGS, "m44_fig3_pipeline_diagram.png"),
            dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close(fig)

# =============================================================================
# SUMMARY
# =============================================================================
print()
print("  MODULE 44 COMPLETE -- 3 figures saved")
print("  Key Concepts:")
print("  [1] LM lexicon: finance-specific pos/neg/uncertainty word lists")
print("  [2] Polarity = (pos - neg) / (pos + neg + 1)  in [-1, +1]")
print("  [3] EWM smoothing reduces noise in raw token-level scores")
print(f"  [4] IC = Spearman(S_{{t-1}}, r_t)  mean={ic_mean:.4f}  ICIR={icir:.3f}")
print("  [5] IC > 0.05 considered economically significant in factor research")
print(f"  [6] TF-IDF + LR: mean walk-forward AUC = {mean_auc:.3f}")
print(f"  [7] Sentiment strategy Sharpe = {sharpe:.3f}  MaxDD = {max_dd:.3f}")
print(f"  [8] Loughran-McDonald outperforms Harvard-IV for financial text")
print(f"  NEXT: M45 -- Word2Vec Financial Embeddings")
print()
