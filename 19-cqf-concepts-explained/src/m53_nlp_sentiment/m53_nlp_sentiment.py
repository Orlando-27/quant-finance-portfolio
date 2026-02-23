"""
M53 -- Natural Language Processing for Finance: Sentiment Analysis
===================================================================
CQF Concepts Explained | Project 19 | Quantitative Finance Portfolio

Theory
------
Financial NLP extracts quantitative signals from unstructured text.
Three core paradigms:

1. Bag-of-Words (BoW)
   Document d is represented as a frequency vector over vocabulary V:
       d_i = count(w_i in d),  i = 1...|V|

2. TF-IDF (Term Frequency - Inverse Document Frequency)
   Downweights common words, upweights discriminative terms:
       tfidf(w,d,D) = tf(w,d) * idf(w,D)
       tf(w,d)      = count(w,d) / |d|
       idf(w,D)     = log( |D| / (1 + df(w,D)) )

3. Naive Bayes Classifier
   Assumes conditional independence of words given class c:
       P(c|d) propto P(c) * prod_w P(w|c)^count(w,d)
   Log-space with Laplace smoothing (alpha):
       log P(w|c) = log( (count(w,c) + alpha) /
                         (sum_w count(w,c) + alpha*|V|) )

Loughran-McDonald (LM) Financial Sentiment Lexicon
---------------------------------------------------
Domain-specific word lists calibrated for SEC filings and earnings calls.
General-purpose lexicons (Harvard IV) misclassify ~73% of financial negatives.
LM categories relevant here: Negative, Positive, Uncertainty, Constraining.

Sentiment Signal -> Trading
---------------------------
Daily sentiment score:  sent_t = (pos_t - neg_t) / (pos_t + neg_t + eps)
Position signal:        p_t    = sign( EMA(sent_t, span=3) )
Reward:                 r_t    = p_t * ret_t - TC * |p_t - p_{t-1}|

References
----------
Loughran & McDonald (2011) "When is a Liability not a Liability?", JF 66(1)
Blei, Ng & Jordan (2003) "Latent Dirichlet Allocation", JMLR 3:993-1022
Pang & Lee (2008) "Opinion Mining and Sentiment Analysis", FT&IR 2(1-2)
"""

import os
import re
import math
import string
import warnings
from collections import Counter, defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

warnings.filterwarnings("ignore")

# =============================================================================
# STYLE
# =============================================================================
DARK   = "#0d1117"
PANEL  = "#161b22"
TEXT   = "#c9d1d9"
GREEN  = "#3fb950"
RED    = "#f85149"
ACCENT = "#58a6ff"
GOLD   = "#d29922"
PURPLE = "#bc8cff"
ORANGE = "#f0883e"

plt.rcParams.update({
    "figure.facecolor":  DARK,
    "axes.facecolor":    PANEL,
    "axes.edgecolor":    TEXT,
    "axes.labelcolor":   TEXT,
    "xtick.color":       TEXT,
    "ytick.color":       TEXT,
    "text.color":        TEXT,
    "grid.color":        "#21262d",
    "grid.linestyle":    "--",
    "grid.linewidth":    0.5,
    "font.family":       "monospace",
    "font.size":         8,
    "legend.facecolor":  PANEL,
    "legend.edgecolor":  TEXT,
})

FIGS = os.path.join(os.path.dirname(__file__), "..", "..", "figures", "m53_nlp_sentiment")
os.makedirs(FIGS, exist_ok=True)

SEED = 42
np.random.seed(SEED)

print()
print("=" * 65)
print("  MODULE 53: NLP FOR FINANCE -- SENTIMENT ANALYSIS")
print("  BoW | TF-IDF | Naive Bayes | LM Lexicon | Signal Backtest")
print("=" * 65)

# =============================================================================
# 1. SYNTHETIC CORPUS -- FINANCIAL HEADLINES
# =============================================================================
# Representative positive and negative financial headline templates.
# Synthetic data avoids dependency on external corpora while preserving
# realistic vocabulary distributions found in earnings press releases.

POS_TEMPLATES = [
    "company reports record quarterly earnings exceeding analyst expectations",
    "strong revenue growth drives profit margins to multi-year highs",
    "firm raises full-year guidance amid robust demand outlook",
    "earnings beat consensus estimates as operating leverage improves",
    "board approves dividend increase signalling confidence in cash generation",
    "management reaffirms bullish outlook citing improving macro conditions",
    "net income surges driven by cost efficiencies and top-line momentum",
    "company achieves record free cash flow underpinning valuation upside",
    "loan book expansion accelerates supported by favourable credit environment",
    "operating margins expand as pricing power offsets input cost pressures",
    "revenue accelerates above expectations with strong forward indicators",
    "profit growth exceeds forecasts aided by operational improvements",
    "solid performance supports optimistic assessment of business trajectory",
    "recurring revenues grow reflecting durable competitive advantages",
    "capital returns increase through buyback programme and higher dividend",
    "management confident in sustained earnings power through economic cycle",
    "beat on both top and bottom line with raised full-year profit outlook",
    "exceptional cash generation enables accelerated deleveraging",
    "customer additions ahead of plan reinforcing revenue visibility",
    "return on equity improves materially reflecting disciplined capital deployment",
]

NEG_TEMPLATES = [
    "company misses quarterly earnings amid deteriorating demand conditions",
    "revenue declines sharply as competitive pressures intensify across segments",
    "management slashes guidance citing macroeconomic headwinds and margin compression",
    "net loss widens as restructuring costs weigh on profitability",
    "credit losses surge raising concerns over asset quality deterioration",
    "earnings disappoint as elevated costs erode operating leverage",
    "firm warns of significant shortfall versus prior year consensus estimates",
    "default risk rises following covenant breach and liquidity deterioration",
    "profit warning triggers sharp selloff amid weak forward indicators",
    "operating cash flow impaired by working capital build and capex overruns",
    "writedown of goodwill reflects impairment of strategic acquisition value",
    "debt burden increases as leverage ratio breaches internal target threshold",
    "declining volumes and adverse pricing dynamics compress gross margins",
    "regulatory investigation creates material uncertainty around business outlook",
    "layoffs and plant closures signal structural deterioration in core markets",
    "cash burn accelerates raising near-term liquidity and solvency concerns",
    "loss provision increase signals rising credit deterioration in loan book",
    "market share erosion accelerates amid aggressive competitor pricing strategy",
    "capital raise at distressed valuation dilutes existing shareholder value",
    "earnings guidance cut for third consecutive quarter amid persistent headwinds",
]

def augment(templates, n_docs, noise=0.15):
    """
    Generate n_docs headlines by sampling templates and randomly
    inserting/dropping words to simulate vocabulary variation.
    """
    filler = ["the", "a", "an", "its", "our", "their", "this", "that",
              "fiscal", "annual", "quarterly", "reported", "noted", "said",
              "indicated", "showed", "posted", "delivered", "achieved"]
    docs = []
    for _ in range(n_docs):
        base = np.random.choice(templates)
        words = base.split()
        # random word insertion
        if np.random.rand() < noise:
            pos_ins = np.random.randint(0, len(words))
            words.insert(pos_ins, np.random.choice(filler))
        # random word drop (never reduce below 5 words)
        if np.random.rand() < noise and len(words) > 5:
            words.pop(np.random.randint(0, len(words)))
        docs.append(" ".join(words))
    return docs

N_POS = 400
N_NEG = 400
pos_docs = augment(POS_TEMPLATES, N_POS)
neg_docs = augment(NEG_TEMPLATES, N_NEG)
all_docs  = pos_docs + neg_docs
all_labels = [1] * N_POS + [0] * N_NEG   # 1=positive, 0=negative

idx_shuf = np.random.permutation(len(all_docs))
all_docs   = [all_docs[i]   for i in idx_shuf]
all_labels = [all_labels[i] for i in idx_shuf]

print(f"  [01] Corpus: {len(all_docs)} documents  "
      f"Positive={N_POS}  Negative={N_NEG}")

# =============================================================================
# 2. TEXT PREPROCESSING
# =============================================================================
STOPWORDS = {
    "the","a","an","and","or","but","in","on","at","to","for","of","with",
    "as","by","is","are","was","were","be","been","being","have","has","had",
    "do","does","did","will","would","could","should","may","might","shall",
    "its","our","their","this","that","these","those","it","he","she","we",
    "they","i","you","all","both","each","few","more","most","other","some",
    "such","into","through","during","before","after","above","below","from",
}

def preprocess(text: str) -> list:
    """Lowercase, remove punctuation, tokenise, remove stopwords."""
    text  = text.lower()
    text  = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]

tokenised = [preprocess(d) for d in all_docs]

# Build vocabulary (min frequency = 2)
freq_all = Counter(w for doc in tokenised for w in doc)
vocab    = sorted([w for w, f in freq_all.items() if f >= 2])
w2i      = {w: i for i, w in enumerate(vocab)}
V        = len(vocab)
print(f"  [02] Vocabulary: {V} terms  (min_freq=2, stopwords removed)")

# =============================================================================
# 3. BAG-OF-WORDS MATRIX
# =============================================================================
def bow_matrix(docs_tok: list, w2i: dict) -> np.ndarray:
    n, v = len(docs_tok), len(w2i)
    X = np.zeros((n, v), dtype=np.int16)
    for i, doc in enumerate(docs_tok):
        for w in doc:
            if w in w2i:
                X[i, w2i[w]] += 1
    return X

X_bow = bow_matrix(tokenised, w2i)
print(f"  [03] BoW matrix: {X_bow.shape}  "
      f"sparsity={100*(X_bow==0).mean():.1f}%")

# =============================================================================
# 4. TF-IDF
# =============================================================================
def tfidf_matrix(X_bow: np.ndarray) -> np.ndarray:
    """Compute TF-IDF from BoW count matrix."""
    n, v = X_bow.shape
    # TF: row-normalised counts
    row_sum = X_bow.sum(axis=1, keepdims=True).astype(float)
    row_sum[row_sum == 0] = 1
    tf = X_bow / row_sum
    # IDF: log( N / (1 + df) )
    df  = (X_bow > 0).sum(axis=0).astype(float)
    idf = np.log(n / (1.0 + df))
    return tf * idf[np.newaxis, :]

X_tfidf = tfidf_matrix(X_bow)
print(f"  [04] TF-IDF matrix: {X_tfidf.shape}  "
      f"max={X_tfidf.max():.4f}  mean_nz={X_tfidf[X_tfidf>0].mean():.4f}")

# =============================================================================
# 5. NAIVE BAYES CLASSIFIER (from scratch)
# =============================================================================
class NaiveBayesText:
    """
    Multinomial Naive Bayes with Laplace smoothing.
    Trained on raw count matrices (BoW).
    """
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.classes_ = np.unique(y)
        n, v = X.shape
        self.log_prior_  = {}
        self.log_likeli_ = {}
        for c in self.classes_:
            mask = (y == c)
            n_c  = mask.sum()
            self.log_prior_[c] = math.log(n_c / n)
            # Word counts in class c
            wc = X[mask].sum(axis=0).astype(float)
            total = wc.sum() + self.alpha * v
            self.log_likeli_[c] = np.log((wc + self.alpha) / total)
        return self

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        n   = X.shape[0]
        nc  = len(self.classes_)
        out = np.zeros((n, nc))
        for j, c in enumerate(self.classes_):
            out[:, j] = self.log_prior_[c] + X @ self.log_likeli_[c]
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        lp = self.predict_log_proba(X)
        return self.classes_[np.argmax(lp, axis=1)]

# Train / test split (80/20, stratified by class order)
y = np.array(all_labels)
n_train = int(0.8 * len(all_docs))
X_tr, X_te = X_bow[:n_train], X_bow[n_train:]
y_tr, y_te = y[:n_train],     y[n_train:]

nb = NaiveBayesText(alpha=1.0)
nb.fit(X_tr, y_tr)
y_pred = nb.predict(X_te)

acc = float(np.mean(y_pred == y_te))
tp  = int(np.sum((y_pred==1) & (y_te==1)))
fp  = int(np.sum((y_pred==1) & (y_te==0)))
fn  = int(np.sum((y_pred==0) & (y_te==1)))
tn  = int(np.sum((y_pred==0) & (y_te==0)))
prec   = tp / (tp + fp + 1e-9)
recall = tp / (tp + fn + 1e-9)
f1     = 2 * prec * recall / (prec + recall + 1e-9)

print(f"  [05] Naive Bayes (Laplace alpha=1): "
      f"Acc={acc:.3f}  Prec={prec:.3f}  Rec={recall:.3f}  F1={f1:.3f}")
print(f"       Confusion: TP={tp} FP={fp} FN={fn} TN={tn}")

# =============================================================================
# 6. LOUGHRAN-MCDONALD LEXICON (domain-specific subset)
# =============================================================================
# Representative subset of LM Financial Sentiment Lexicon word lists.
# Full LM lexicon: ~86k entries across 6 sentiment categories.

LM_NEGATIVE = {
    "loss","losses","decline","declines","declining","deficit","deficits",
    "impairment","impairments","writedown","writedowns","shortfall","shortfalls",
    "miss","misses","missed","disappointing","disappoints","disappoint",
    "deterioration","deteriorate","deteriorating","weak","weakness","weakening",
    "default","defaults","delinquency","delinquencies","restructuring",
    "headwinds","adverse","adversely","unfavorable","unfavourable",
    "compression","erode","erodes","eroding","pressures","pressure",
    "distressed","distress","liquidation","bankruptcy","insolvency",
    "breach","breaches","violation","violations","penalty","penalties",
    "warning","warnings","concern","concerns","risk","risks","uncertainty",
    "cut","cuts","reduced","reduction","reductions","slash","slashes",
}

LM_POSITIVE = {
    "growth","growing","grew","increase","increases","increased","gain","gains",
    "improvement","improvements","improved","improve","beat","beats","exceeded",
    "exceeds","exceed","record","records","strong","strength","robust",
    "outperform","outperforms","outperformed","momentum","accelerating",
    "acceleration","expand","expands","expanding","expansion","profitable",
    "profitability","confident","confidence","optimistic","optimism",
    "reaffirm","reaffirms","raised","raises","upgrade","upgraded","upgrades",
    "surged","surge","surges","solid","sustained","durable","efficient",
    "efficiencies","upside","visibility","advantage","advantages",
}

LM_UNCERTAINTY = {
    "uncertain","uncertainty","uncertain","volatile","volatility","fluctuate",
    "unpredictable","unclear","ambiguous","contingent","dependent","may",
    "might","could","possibly","potentially","approximate","estimated",
}

def lm_score(tokens: list) -> dict:
    """Return LM sentiment counts for a token list."""
    pos = sum(1 for t in tokens if t in LM_POSITIVE)
    neg = sum(1 for t in tokens if t in LM_NEGATIVE)
    unc = sum(1 for t in tokens if t in LM_UNCERTAINTY)
    n   = len(tokens) + 1e-9
    return {
        "pos": pos,
        "neg": neg,
        "unc": unc,
        "score": (pos - neg) / n,   # net sentiment per token
        "polarity": (pos - neg) / (pos + neg + 1e-9),  # [-1,1]
    }

lm_scores = [lm_score(tok) for tok in tokenised]
pol   = np.array([s["polarity"] for s in lm_scores])
label = np.array(all_labels)

# LM lexicon accuracy as binary classifier (polarity > 0 -> positive)
lm_pred = (pol > 0).astype(int)
lm_acc  = float(np.mean(lm_pred == label))
lm_tp   = int(np.sum((lm_pred==1) & (label==1)))
lm_fp   = int(np.sum((lm_pred==1) & (label==0)))
lm_fn   = int(np.sum((lm_pred==0) & (label==1)))
lm_tn   = int(np.sum((lm_pred==0) & (label==0)))
lm_prec = lm_tp / (lm_tp + lm_fp + 1e-9)
lm_rec  = lm_tp / (lm_tp + lm_fn + 1e-9)
lm_f1   = 2 * lm_prec * lm_rec / (lm_prec + lm_rec + 1e-9)

print(f"  [06] LM Lexicon classifier: "
      f"Acc={lm_acc:.3f}  Prec={lm_prec:.3f}  Rec={lm_rec:.3f}  F1={lm_f1:.3f}")

# =============================================================================
# 7. SENTIMENT SIGNAL -> TRADING STRATEGY BACKTEST
# =============================================================================
# Simulate daily returns correlated with document sentiment labels.
# Documents are ordered in time; sentiment polarity drives a trading signal.

N = len(all_docs)
dt = 1 / 252

# Returns: positive label days have positive drift, negative have negative
drift = np.where(label == 1, 0.12 * dt, -0.08 * dt)
vol   = 0.15 * np.sqrt(dt)
ret   = drift + vol * np.random.randn(N)

# EMA of LM polarity score (span=3 days)
def ema(x: np.ndarray, span: int = 3) -> np.ndarray:
    alpha = 2.0 / (span + 1)
    out   = np.zeros_like(x)
    out[0] = x[0]
    for t in range(1, len(x)):
        out[t] = alpha * x[t] + (1 - alpha) * out[t-1]
    return out

sent_ema = ema(pol, span=3)

# NB predicted probability as signal
nb_all = NaiveBayesText(alpha=1.0).fit(X_bow, y)
log_proba = nb_all.predict_log_proba(X_bow)
# softmax to convert log-probs to probs
log_proba -= log_proba.max(axis=1, keepdims=True)
proba = np.exp(log_proba)
proba /= proba.sum(axis=1, keepdims=True)
nb_signal = proba[:, 1] - 0.5   # center around 0

TC = 0.0005  # 5 bps one-way

def backtest(signal: np.ndarray, ret: np.ndarray,
             tc: float = TC, threshold: float = 0.0) -> np.ndarray:
    """Signal-based strategy: long if signal > threshold, else short."""
    pos   = np.sign(signal - threshold)
    pos[pos == 0] = 0
    chg   = np.abs(np.diff(np.concatenate([[0], pos])))
    r_strat = pos * ret - tc * chg
    return np.cumsum(r_strat)

pnl_lm   = backtest(pol,        ret)
pnl_nb   = backtest(nb_signal,  ret)
pnl_bnh  = np.cumsum(ret)

def sharpe(pnl, freq=252):
    d = np.diff(pnl)
    return float(np.mean(d) / (np.std(d) + 1e-9) * np.sqrt(freq))

def max_dd(pnl):
    peak = np.maximum.accumulate(pnl)
    return float(np.max(peak - pnl))

print(f"  [07] Backtest Results:")
print(f"       {'Strategy':<18} {'Total PnL':>10} {'Sharpe':>8} {'Max DD':>8}")
for name, pnl in [("LM Lexicon", pnl_lm), ("Naive Bayes", pnl_nb),
                  ("Buy-and-Hold", pnl_bnh)]:
    print(f"       {name:<18} {pnl[-1]:>10.4f} "
          f"{sharpe(pnl):>8.3f} {max_dd(pnl):>8.4f}")

# =============================================================================
# 8. FIGURE 1 -- TF-IDF Top Terms & Naive Bayes Log-Likelihood
# =============================================================================
fig = plt.figure(figsize=(15, 9), facecolor=DARK)
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.4)
fig.suptitle("M53 -- NLP for Finance: TF-IDF, Naive Bayes, LM Lexicon",
             color=TEXT, fontsize=11)

# 8A: Top TF-IDF terms (positive documents)
ax = fig.add_subplot(gs[0, 0])
pos_mask = y == 1
tfidf_pos_mean = X_tfidf[pos_mask].mean(axis=0)
top_idx  = np.argsort(tfidf_pos_mean)[-20:]
ax.barh([vocab[i] for i in top_idx],
        tfidf_pos_mean[top_idx],
        color=GREEN, edgecolor=DARK, linewidth=0.4)
ax.set_title("Top 20 TF-IDF Terms\n(Positive Documents)")
ax.set_xlabel("Mean TF-IDF Weight")
ax.set_facecolor(PANEL)
ax.grid(True, axis="x")

# 8B: Top TF-IDF terms (negative documents)
ax = fig.add_subplot(gs[0, 1])
neg_mask = y == 0
tfidf_neg_mean = X_tfidf[neg_mask].mean(axis=0)
top_idx_n = np.argsort(tfidf_neg_mean)[-20:]
ax.barh([vocab[i] for i in top_idx_n],
        tfidf_neg_mean[top_idx_n],
        color=RED, edgecolor=DARK, linewidth=0.4)
ax.set_title("Top 20 TF-IDF Terms\n(Negative Documents)")
ax.set_xlabel("Mean TF-IDF Weight")
ax.set_facecolor(PANEL)
ax.grid(True, axis="x")

# 8C: Naive Bayes log-likelihood ratio (top discriminative terms)
ax = fig.add_subplot(gs[0, 2])
ll_ratio = nb.log_likeli_[1] - nb.log_likeli_[0]
top_pos_nb = np.argsort(ll_ratio)[-15:]
top_neg_nb = np.argsort(ll_ratio)[:15]
idx_show   = np.concatenate([top_neg_nb, top_pos_nb])
colors_nb  = [RED if ll_ratio[i] < 0 else GREEN for i in idx_show]
ax.barh([vocab[i] for i in idx_show],
        ll_ratio[idx_show],
        color=colors_nb, edgecolor=DARK, linewidth=0.3)
ax.axvline(0, color=TEXT, lw=0.8, ls="--")
ax.set_title("NB Log-Likelihood Ratio\nlog P(w|pos) - log P(w|neg)")
ax.set_xlabel("Log-Likelihood Ratio")
ax.set_facecolor(PANEL)
ax.grid(True, axis="x")

# 8D: Confusion matrix NB
ax = fig.add_subplot(gs[1, 0])
cm = np.array([[tn, fp], [fn, tp]])
im = ax.imshow(cm, cmap="Blues")
for (r, c), v in np.ndenumerate(cm):
    ax.text(c, r, str(v), ha="center", va="center",
            color=TEXT, fontsize=12, fontweight="bold")
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(["Pred Neg", "Pred Pos"])
ax.set_yticklabels(["True Neg", "True Pos"])
ax.set_title(f"Naive Bayes Confusion Matrix\nAcc={acc:.3f}  F1={f1:.3f}")
ax.set_facecolor(PANEL)

# 8E: LM polarity score distribution
ax = fig.add_subplot(gs[1, 1])
pos_pol = pol[label == 1]
neg_pol = pol[label == 0]
bins = np.linspace(-1, 1, 30)
ax.hist(pos_pol, bins=bins, color=GREEN, alpha=0.6, label="Positive", density=True)
ax.hist(neg_pol, bins=bins, color=RED,   alpha=0.6, label="Negative", density=True)
ax.axvline(0, color=TEXT, lw=0.8, ls="--")
ax.set_title(f"LM Polarity Score Distribution\nAcc={lm_acc:.3f}  F1={lm_f1:.3f}")
ax.set_xlabel("Polarity Score")
ax.set_ylabel("Density")
ax.legend(fontsize=7)
ax.set_facecolor(PANEL)
ax.grid(True)

# 8F: NB probability score distribution
ax = fig.add_subplot(gs[1, 2])
prob_pos = proba[label == 1, 1]
prob_neg = proba[label == 0, 1]
bins2 = np.linspace(0, 1, 30)
ax.hist(prob_pos, bins=bins2, color=GREEN, alpha=0.6, label="True Pos", density=True)
ax.hist(prob_neg, bins=bins2, color=RED,   alpha=0.6, label="True Neg", density=True)
ax.axvline(0.5, color=TEXT, lw=0.8, ls="--", label="Threshold=0.5")
ax.set_title("Naive Bayes P(positive|doc)")
ax.set_xlabel("Predicted Probability")
ax.set_ylabel("Density")
ax.legend(fontsize=7)
ax.set_facecolor(PANEL)
ax.grid(True)

fig.savefig(os.path.join(FIGS, "m53_fig1_features_classifier.png"),
            dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close(fig)
print("  [08] Fig 1 saved: TF-IDF terms, NB diagnostics, LM distribution")

# =============================================================================
# 9. FIGURE 2 -- Backtest PnL & Sentiment Signal
# =============================================================================
t = np.arange(N)

fig, axes = plt.subplots(3, 1, figsize=(15, 10), facecolor=DARK)
fig.suptitle("M53 -- Sentiment Signal: Backtest vs Buy-and-Hold",
             color=TEXT, fontsize=11)

# 9A: Cumulative PnL
ax = axes[0]
ax.plot(t, pnl_nb,  color=ACCENT, lw=1.5, label="Naive Bayes")
ax.plot(t, pnl_lm,  color=GREEN,  lw=1.5, label="LM Lexicon")
ax.plot(t, pnl_bnh, color=RED,    lw=1.2, label="Buy-and-Hold", ls="--")
ax.axhline(0, color=TEXT, lw=0.5, ls=":")
ax.set_title("Cumulative PnL: Sentiment Strategies vs Buy-and-Hold")
ax.set_ylabel("Cumulative Log-Return")
ax.legend(fontsize=8)
ax.set_facecolor(PANEL)
ax.grid(True)

# 9B: LM polarity signal over time
ax = axes[1]
ax.fill_between(t, pol,  0, where=pol > 0,  color=GREEN, alpha=0.5, label="Positive")
ax.fill_between(t, pol,  0, where=pol <= 0, color=RED,   alpha=0.5, label="Negative")
ax.plot(t, ema(pol, 10), color=GOLD, lw=1.2, label="EMA(10)")
ax.axhline(0, color=TEXT, lw=0.5, ls="--")
ax.set_title("LM Polarity Score (Daily) with EMA(10)")
ax.set_ylabel("Polarity Score")
ax.legend(fontsize=7)
ax.set_facecolor(PANEL)
ax.grid(True)

# 9C: NB probability signal
ax = axes[2]
nb_prob_smooth = ema(proba[:, 1], 5)
ax.plot(t, proba[:, 1], color=ACCENT, lw=0.5, alpha=0.4, label="P(pos|doc)")
ax.plot(t, nb_prob_smooth, color=ACCENT, lw=1.5, label="EMA(5)")
ax.axhline(0.5, color=TEXT, lw=0.8, ls="--", label="Threshold=0.5")
ax.fill_between(t, nb_prob_smooth, 0.5,
                where=nb_prob_smooth > 0.5, color=GREEN, alpha=0.3)
ax.fill_between(t, nb_prob_smooth, 0.5,
                where=nb_prob_smooth <= 0.5, color=RED, alpha=0.3)
ax.set_title("Naive Bayes P(positive|doc) Signal")
ax.set_xlabel("Document / Day")
ax.set_ylabel("P(positive)")
ax.legend(fontsize=7)
ax.set_facecolor(PANEL)
ax.grid(True)

fig.tight_layout()
fig.savefig(os.path.join(FIGS, "m53_fig2_backtest_signal.png"),
            dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close(fig)
print("  [09] Fig 2 saved: backtest PnL & sentiment signal")

# =============================================================================
# 10. FIGURE 3 -- NLP Pipeline Architecture & Word Co-occurrence
# =============================================================================
fig = plt.figure(figsize=(15, 8), facecolor=DARK)
gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)
fig.suptitle("M53 -- NLP Pipeline Architecture & Word Co-occurrence Matrix",
             color=TEXT, fontsize=11)

# 10A: Pipeline diagram
ax = fig.add_subplot(gs[0, 0])
ax.set_xlim(0, 10)
ax.set_ylim(0, 14)
ax.axis("off")
ax.set_facecolor(PANEL)
ax.set_title("NLP Text-to-Signal Pipeline", color=TEXT)

steps = [
    (5, 13, "Raw Financial Text\n(Headlines / Filings)", GOLD),
    (5, 11, "Preprocessing\n(lowercase, strip punctuation,\nremove stopwords)", ACCENT),
    (5,  9, "Feature Extraction\n(BoW  |  TF-IDF)", ACCENT),
    (5,  7, "Classification\n(Naive Bayes  |  LM Lexicon)", ACCENT),
    (5,  5, "Sentiment Score\n(polarity in [-1, +1])", PURPLE),
    (5,  3, "Signal Smoothing\n(EMA, rolling window)", PURPLE),
    (5,  1, "Trading Position\n(long / flat / short)", GREEN),
]
for x, y_pos, label, col in steps:
    box = FancyBboxPatch((x-3.5, y_pos-0.7), 7, 1.4,
                         boxstyle="round,pad=0.1",
                         facecolor=DARK, edgecolor=col, linewidth=1.5)
    ax.add_patch(box)
    ax.text(x, y_pos, label, ha="center", va="center",
            color=TEXT, fontsize=7.5, linespacing=1.4)
    if y_pos > 1:
        ax.annotate("", xy=(x, y_pos - 0.7 - 0.2),
                    xytext=(x, y_pos - 0.7),
                    arrowprops=dict(arrowstyle="-|>", color=col,
                                   lw=1.2))

# 10B: Word co-occurrence heatmap (top 20 terms)
ax = fig.add_subplot(gs[0, 1])
top20_idx = np.argsort(
    X_bow.sum(axis=0)
)[-20:]
top20_words = [vocab[i] for i in top20_idx]
X_sub = (X_bow[:, top20_idx] > 0).astype(float)
cooc  = X_sub.T @ X_sub
# normalise by sqrt of marginal products
denom = np.outer(np.sqrt(cooc.diagonal()), np.sqrt(cooc.diagonal())) + 1e-9
cooc_norm = cooc / denom
np.fill_diagonal(cooc_norm, 0)

im = ax.imshow(cooc_norm, cmap="YlOrRd", aspect="auto",
               vmin=0, vmax=cooc_norm.max())
ax.set_xticks(range(20))
ax.set_yticks(range(20))
ax.set_xticklabels(top20_words, rotation=45, ha="right", fontsize=6)
ax.set_yticklabels(top20_words, fontsize=6)
ax.set_title("Word Co-occurrence (Top 20 Terms)\nNormalised Pointwise Mutual Information")
fig.colorbar(im, ax=ax, label="Normalised PMI", fraction=0.046, pad=0.04)
ax.set_facecolor(PANEL)

fig.tight_layout()
fig.savefig(os.path.join(FIGS, "m53_fig3_pipeline_cooccurrence.png"),
            dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close(fig)
print("  [10] Fig 3 saved: NLP pipeline & word co-occurrence")

# =============================================================================
# SUMMARY
# =============================================================================
nb_sharpe  = sharpe(pnl_nb)
lm_sharpe  = sharpe(pnl_lm)
bnh_sharpe = sharpe(pnl_bnh)

print()
print("  MODULE 53 COMPLETE -- 3 figures saved")
print("  Key Concepts:")
print("  [1] BoW: document as word frequency vector d_i = count(w_i, d)")
print("  [2] TF-IDF: tf*log(N/(1+df)) -- upweights rare discriminative terms")
print("  [3] Naive Bayes: P(c|d) propto P(c)*prod P(w|c)^count(w,d)")
print("  [4] Laplace smoothing: P(w|c)=(count+alpha)/(total+alpha*V)")
print(f"  [5] NB classifier:  Acc={acc:.3f}  F1={f1:.3f}")
print(f"  [6] LM lexicon:     Acc={lm_acc:.3f}  F1={lm_f1:.3f}")
print(f"  [7] NB strategy:    Sharpe={nb_sharpe:.3f}  MDD={max_dd(pnl_nb):.4f}")
print(f"  [8] LM strategy:    Sharpe={lm_sharpe:.3f}  MDD={max_dd(pnl_lm):.4f}")
print(f"  [9] Buy-and-Hold:   Sharpe={bnh_sharpe:.3f}  MDD={max_dd(pnl_bnh):.4f}")
print("  NEXT: M54 -- Extreme Value Theory & Tail Risk")
print()
