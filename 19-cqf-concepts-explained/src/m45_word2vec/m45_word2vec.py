#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULE 45 -- WORD2VEC FINANCIAL EMBEDDINGS
=============================================================================
CQF Concepts Explained: Interactive Jupyter Notebooks
Project 19 of 20 -- Quantitative Finance Portfolio
Author  : Jose Orlando Bobadilla Fuentes | CQF | MSc AI Applied to Fin. Markets
Role    : Senior Quantitative Portfolio Manager & Lead Data Scientist
Firm    : Colombian Pension Fund -- Vicepresidencia de Inversiones

ACADEMIC OVERVIEW
-----------------
Word2Vec (Mikolov et al., 2013) learns dense vector representations of words
by training a shallow neural network to predict context from a target word
(Skip-gram) or a target from its context (CBOW).  The key insight is the
*distributional hypothesis*: words appearing in similar contexts have similar
meanings.  This is captured geometrically -- semantically related words have
high cosine similarity in embedding space.

MATHEMATICAL FORMULATION (SKIP-GRAM)
--------------------------------------
Given a corpus W = {w_1, ..., w_T}, the Skip-gram objective maximises:

    L = (1/T) * sum_{t=1}^T  sum_{-c <= j <= c, j!=0}  log P(w_{t+j} | w_t)

where c is the context window size.  The conditional probability is:

    P(w_O | w_I) = exp(v'_{w_O} . v_{w_I}) / sum_{w=1}^V exp(v'_w . v_{w_I})

v_{w_I} : input embedding of the centre word  (V x d matrix W_in)
v'_{w_O}: output embedding of the context word (V x d matrix W_out)

NEGATIVE SAMPLING (NEG)
-----------------------
Computing the full softmax over vocabulary V is O(V) -- expensive.
Negative sampling approximates the objective:

    L_NEG = log sigma(v'_{w_O} . v_{w_I})
           + sum_{k=1}^K E_{w_k ~ P_n(w)} [ log sigma(-v'_{w_k} . v_{w_I}) ]

where sigma is the sigmoid function and K negative samples are drawn from
the unigram distribution raised to the 3/4 power:

    P_n(w) = f(w)^{3/4} / sum_w f(w)^{3/4}

The 3/4 exponent smooths the distribution, giving rare words more probability.

FINANCIAL WORD ANALOGIES
-------------------------
Word2Vec supports linear algebraic reasoning:
    v("king") - v("man") + v("woman") ~ v("queen")

In finance:
    v("equity") - v("stock") + v("bond") ~ v("debt")
    v("bullish") - v("bull") + v("bear") ~ v("bearish")
    v("profit") - v("loss") + v("gain")  ~ v("revenue")

APPLICATIONS IN QUANTITATIVE FINANCE
--------------------------------------
1. Document embeddings: mean-pool word vectors -> headline vector -> signal
2. Semantic clustering: group companies/sectors by textual similarity
3. Analogical reasoning: transfer learned finance relationships
4. Feature augmentation: word embeddings as features in downstream classifiers
5. Knowledge graph construction: map financial entity relationships

REFERENCES
----------
[1] Mikolov, T. et al. (2013). "Efficient Estimation of Word Representations
    in Vector Space." ICLR.
[2] Mikolov, T. et al. (2013). "Distributed Representations of Words and
    Phrases and their Compositionality." NeurIPS.
[3] Levy, O. & Goldberg, Y. (2014). "Neural Word Embedding as Implicit
    Matrix Factorization." NeurIPS.
[4] Lopez de Prado, M. (2018). Advances in Financial Machine Learning. Wiley.
=============================================================================
Usage (Cloud Shell):
    cd ~/quant-finance-portfolio/19-cqf-concepts-explained
    python src/m45_word2vec/m45_word2vec.py
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
from collections import Counter, defaultdict
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import yfinance as yf

warnings.filterwarnings("ignore")
np.random.seed(42)

# =============================================================================
# PATHS
# =============================================================================
BASE  = os.path.dirname(os.path.abspath(__file__))
ROOT  = os.path.abspath(os.path.join(BASE, "..", ".."))
FIGS  = os.path.join(ROOT, "outputs", "figures", "m45")
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
print("  MODULE 45: WORD2VEC FINANCIAL EMBEDDINGS")
print("  Skip-gram | Neg Sampling | Cosine Sim | Analogies | t-SNE")
print("=" * 65)

# =============================================================================
# 1.  FINANCIAL CORPUS CONSTRUCTION
#     We build a synthetic but realistic financial text corpus by combining:
#     (a) A fixed vocabulary of finance terms with known semantic groups
#     (b) Structured sentences that encode financial relationships
#     This ensures Word2Vec learns meaningful embeddings we can analyse.
# =============================================================================

# Semantic groups -- used to colour embeddings and verify clustering
GROUPS = {
    "equity":     ["stock", "equity", "share", "shares", "dividend",
                   "earnings", "buyback", "ipo", "listing"],
    "debt":       ["bond", "debt", "credit", "yield", "coupon", "maturity",
                   "duration", "spread", "default", "rating"],
    "macro":      ["gdp", "inflation", "unemployment", "fed", "rate",
                   "monetary", "fiscal", "growth", "recession", "recovery"],
    "risk":       ["volatility", "risk", "drawdown", "var", "hedging",
                   "correlation", "exposure", "downside", "tail"],
    "sentiment":  ["bullish", "bearish", "optimistic", "pessimistic",
                   "rally", "selloff", "momentum", "trend", "signal"],
    "valuation":  ["pe", "pb", "ev", "ebitda", "dcf", "multiples",
                   "discount", "premium", "intrinsic", "fair"],
    "derivatives":["option", "futures", "swap", "forward", "delta",
                   "gamma", "vega", "theta", "hedge", "expiry"],
}

# Sentence templates that encode financial co-occurrence relationships
TEMPLATES = [
    # Equity context
    "the {equity} market shows strong {equity} {sentiment}",
    "{equity} {sentiment} driven by {macro} outlook",
    "{equity} valuations measured by {valuation} metrics",
    "investors buy {equity} on {sentiment} signals",
    "{equity} earnings beat expectations boosting {sentiment}",
    # Debt context
    "{debt} yields rise as {macro} data surprises",
    "{debt} {risk} increases with credit {debt} spreads",
    "central bank {macro} policy impacts {debt} duration",
    "{debt} default {risk} elevated in recession {macro}",
    # Risk context
    "{risk} management uses {derivatives} for hedging",
    "portfolio {risk} measured by {risk} var metrics",
    "{sentiment} {risk} drives {equity} volatility",
    "{derivatives} delta {derivatives} hedge reduces {risk}",
    # Valuation
    "{valuation} pe multiples expand in {sentiment} markets",
    "{valuation} dcf discount rates linked to {macro} rates",
    "{equity} fair value estimated by {valuation} models",
    # Macro
    "{macro} gdp growth supports {equity} {sentiment}",
    "{macro} inflation impacts {debt} yields and {equity} pe",
    "{macro} fed rate decision affects {debt} and {derivatives}",
    # Cross-group
    "{sentiment} rally in {equity} reduces {risk} premium",
    "{derivatives} options hedge {equity} downside {risk}",
    "{macro} recession increases {debt} default and {risk}",
    "{valuation} discount on {equity} signals {sentiment} opportunity",
]

rng = np.random.default_rng(42)

def fill_template(template: str) -> str:
    """Replace {group} placeholders with a random word from that group."""
    words = {}
    for grp in GROUPS:
        if "{" + grp + "}" in template:
            words[grp] = rng.choice(GROUPS[grp])
    result = template
    for grp, word in words.items():
        result = result.replace("{" + grp + "}", word)
    return result

# Generate corpus: repeat templates many times for sufficient co-occurrence
N_SENTENCES = 8000
corpus_sentences = []
for _ in range(N_SENTENCES):
    tmpl = rng.choice(TEMPLATES)
    corpus_sentences.append(fill_template(tmpl))

# Tokenise
def tokenize(text: str) -> list:
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return text.split()

tokenized = [tokenize(s) for s in corpus_sentences]
all_tokens = [t for sent in tokenized for t in sent]

section(1, f"Corpus: {N_SENTENCES} sentences  {len(all_tokens)} tokens")

# =============================================================================
# 2.  VOCABULARY CONSTRUCTION
# =============================================================================
MIN_FREQ = 3
freq = Counter(all_tokens)
vocab = {w: i for i, (w, c) in enumerate(freq.most_common()) if c >= MIN_FREQ}
idx2word = {i: w for w, i in vocab.items()}
V = len(vocab)

# Subsampling probability (discard frequent words):
# P_discard(w) = 1 - sqrt(t / f(w))   where t = 1e-4
t_sub  = 1e-4
total  = sum(freq[w] for w in vocab)
f_w    = {w: freq[w] / total for w in vocab}
p_keep = {w: min(1.0, np.sqrt(t_sub / (f_w[w] + 1e-9)) + t_sub / (f_w[w] + 1e-9))
          for w in vocab}

# Negative sampling distribution P_n(w) = f(w)^{3/4}
counts_arr  = np.array([freq[idx2word[i]] for i in range(V)], dtype=float)
neg_weights = counts_arr ** 0.75
neg_weights /= neg_weights.sum()

section(2, f"Vocabulary: V={V}  (min_freq={MIN_FREQ})")

# =============================================================================
# 3.  SKIP-GRAM WITH NEGATIVE SAMPLING (numpy implementation)
# =============================================================================
class Word2Vec:
    """
    Skip-gram Word2Vec with negative sampling, implemented in NumPy.

    Architecture:
        W_in  : (V, d)  input  embeddings  -- one row per word
        W_out : (V, d)  output embeddings  -- one row per context word

    Training objective (per (centre, context) pair):
        L = log sigma(v_out[c] . v_in[w])
           + sum_{k} log sigma(-v_out[neg_k] . v_in[w])

    Gradient update (SGD):
        grad_v_in[w]   = (sigma(score_c) - 1) * v_out[c]
                        + sum_k sigma(score_neg_k) * v_out[neg_k]
        grad_v_out[c]  = (sigma(score_c) - 1) * v_in[w]
        grad_v_out[nk] = sigma(score_nk) * v_in[w]
    """

    def __init__(self, vocab_size: int, dim: int = 50,
                 neg_samples: int = 5, window: int = 3, lr: float = 0.025):
        self.V   = vocab_size
        self.d   = dim
        self.K   = neg_samples
        self.c   = window
        self.lr  = lr
        # Uniform initialisation in [-0.5/d, 0.5/d]
        scale = 0.5 / dim
        self.W_in  = (np.random.rand(vocab_size, dim) - 0.5) * 2 * scale
        self.W_out = np.zeros((vocab_size, dim))

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))

    def _negative_sample(self, exclude: set) -> list:
        """Draw K negative samples excluding the positive context words."""
        negs = []
        while len(negs) < self.K:
            idx = np.searchsorted(self._neg_cdf,
                                  np.random.rand(self.K * 2))
            for i in idx:
                i = min(i, self.V - 1)
                if i not in exclude:
                    negs.append(i)
                if len(negs) == self.K:
                    break
        return negs[:self.K]

    def _build_neg_cdf(self, neg_weights: np.ndarray):
        self._neg_cdf = np.cumsum(neg_weights)

    def train_pair(self, centre_idx: int, context_idx: int) -> float:
        """
        Update embeddings for one (centre, context) pair.
        Returns the loss contribution for monitoring.
        """
        # Positive sample
        v_in    = self.W_in[centre_idx]                   # (d,)
        exclude = {centre_idx, context_idx}
        negs    = self._negative_sample(exclude)           # list of K indices

        # Gradient accumulator for v_in
        grad_in = np.zeros(self.d)

        # Positive pair
        score_pos = np.dot(v_in, self.W_out[context_idx])
        sig_pos   = self.sigmoid(score_pos)
        err_pos   = sig_pos - 1.0                          # target = 1
        grad_in  += err_pos * self.W_out[context_idx]
        self.W_out[context_idx] -= self.lr * err_pos * v_in

        # Negative pairs
        loss = -np.log(sig_pos + 1e-9)
        for neg_idx in negs:
            score_neg = np.dot(v_in, self.W_out[neg_idx])
            sig_neg   = self.sigmoid(score_neg)
            err_neg   = sig_neg                            # target = 0
            grad_in  += err_neg * self.W_out[neg_idx]
            self.W_out[neg_idx] -= self.lr * err_neg * v_in
            loss -= np.log(1.0 - sig_neg + 1e-9)

        self.W_in[centre_idx] -= self.lr * grad_in
        return loss

    def fit(self, tokenized_corpus: list, vocab: dict,
            neg_weights: np.ndarray, p_keep: dict,
            epochs: int = 5, verbose_every: int = 1) -> list:
        """
        Full training loop over the tokenised corpus.

        Parameters
        ----------
        tokenized_corpus : list of list of str
        vocab            : word -> index mapping
        neg_weights      : V-length array for negative sampling distribution
        p_keep           : word -> subsampling keep probability
        epochs           : training epochs
        verbose_every    : print loss every N epochs

        Returns
        -------
        List of epoch-level mean losses.
        """
        self._build_neg_cdf(neg_weights)
        epoch_losses = []
        for ep in range(1, epochs + 1):
            total_loss = 0.0; n_pairs = 0
            # Shuffle sentences each epoch
            order = np.random.permutation(len(tokenized_corpus))
            for si in order:
                sent = tokenized_corpus[si]
                # Convert to indices with subsampling
                idxs = []
                for w in sent:
                    if w in vocab and np.random.rand() < p_keep.get(w, 1.0):
                        idxs.append(vocab[w])
                if len(idxs) < 2:
                    continue
                # Dynamic window: uniformly sample c' in [1, window]
                for pos, centre in enumerate(idxs):
                    c_actual = np.random.randint(1, self.c + 1)
                    lo = max(0, pos - c_actual)
                    hi = min(len(idxs), pos + c_actual + 1)
                    for ctx_pos in range(lo, hi):
                        if ctx_pos == pos:
                            continue
                        context = idxs[ctx_pos]
                        total_loss += self.train_pair(centre, context)
                        n_pairs    += 1
            mean_loss = total_loss / max(n_pairs, 1)
            epoch_losses.append(mean_loss)
            if ep % verbose_every == 0:
                print(f"       epoch {ep}/{epochs}  loss={mean_loss:.4f}  pairs={n_pairs:,}")
        return epoch_losses

    def get_vector(self, word: str, vocab: dict) -> np.ndarray:
        """Return L2-normalised input embedding for a word."""
        idx = vocab.get(word)
        if idx is None:
            return None
        v = self.W_in[idx]
        return v / (np.linalg.norm(v) + 1e-9)

    def most_similar(self, word: str, vocab: dict, idx2word: dict,
                     topn: int = 8) -> list:
        """Return topn most similar words by cosine similarity."""
        v = self.get_vector(word, vocab)
        if v is None:
            return []
        # Normalise all embeddings
        norms  = np.linalg.norm(self.W_in, axis=1, keepdims=True) + 1e-9
        W_norm = self.W_in / norms
        sims   = W_norm @ v
        sims[vocab[word]] = -1.0   # exclude self
        top_idx = np.argsort(sims)[::-1][:topn]
        return [(idx2word[i], float(sims[i])) for i in top_idx]

    def analogy(self, pos1: str, neg1: str, pos2: str,
                vocab: dict, idx2word: dict, topn: int = 5) -> list:
        """
        Solve: pos1 - neg1 + pos2 = ?
        Classic Word2Vec analogy: king - man + woman = queen
        """
        exclude = {pos1, neg1, pos2}
        vecs = []
        for w, sign in [(pos1, 1), (neg1, -1), (pos2, 1)]:
            v = self.get_vector(w, vocab)
            if v is None:
                return []
            vecs.append(sign * v)
        query = sum(vecs)
        query /= np.linalg.norm(query) + 1e-9
        norms  = np.linalg.norm(self.W_in, axis=1, keepdims=True) + 1e-9
        W_norm = self.W_in / norms
        sims   = W_norm @ query
        for w in exclude:
            if w in vocab:
                sims[vocab[w]] = -1.0
        top_idx = np.argsort(sims)[::-1][:topn]
        return [(idx2word[i], float(sims[i])) for i in top_idx]


# Train
DIM     = 50
EPOCHS  = 8
WINDOW  = 3
K_NEG   = 5
LR      = 0.025

print(f"  Training Word2Vec  V={V}  d={DIM}  window={WINDOW}  K={K_NEG}  epochs={EPOCHS}")
w2v = Word2Vec(vocab_size=V, dim=DIM, neg_samples=K_NEG, window=WINDOW, lr=LR)
loss_curve = w2v.fit(tokenized, vocab, neg_weights, p_keep,
                     epochs=EPOCHS, verbose_every=2)

section(3, f"Training complete  final_loss={loss_curve[-1]:.4f}")

# =============================================================================
# 4.  SIMILARITY ANALYSIS
# =============================================================================
query_words = ["stock", "bond", "volatility", "bullish", "pe", "option"]
print()
for qw in query_words:
    sims = w2v.most_similar(qw, vocab, idx2word, topn=5)
    sim_str = "  ".join([f"{w}({s:.2f})" for w, s in sims])
    print(f"       {qw:12s} -> {sim_str}")
print()

# Semantic similarity matrix for selected words
PROBE = ["stock", "equity", "bond", "debt", "bullish", "bearish",
         "volatility", "risk", "option", "delta", "gdp", "inflation"]
probe_in_vocab = [w for w in PROBE if w in vocab]
probe_vecs = np.array([w2v.get_vector(w, vocab) for w in probe_in_vocab])
sim_matrix  = cosine_similarity(probe_vecs)

section(4, f"Similarity matrix computed for {len(probe_in_vocab)} probe words")

# =============================================================================
# 5.  ANALOGY EVALUATION
# =============================================================================
analogies = [
    ("bullish", "bull",  "bear",    "bearish analogy"),
    ("equity",  "stock", "bond",    "equity-debt analogy"),
    ("profit",  "gain",  "loss",    "profit-loss analogy"),
    ("delta",   "option","futures", "derivatives analogy"),
    ("gdp",     "growth","recession","macro analogy"),
]

print("  Analogy evaluation:")
for pos1, neg1, pos2, label in analogies:
    if all(w in vocab for w in [pos1, neg1, pos2]):
        result = w2v.analogy(pos1, neg1, pos2, vocab, idx2word, topn=3)
        res_str = "  ".join([f"{w}({s:.2f})" for w, s in result])
        print(f"       {pos1}-{neg1}+{pos2} = {res_str}  [{label}]")

section(5, "Analogy evaluation complete")

# =============================================================================
# 6.  t-SNE VISUALISATION OF EMBEDDINGS
# =============================================================================
# Collect all finance group words present in vocab
group_words, group_labels, group_colors = [], [], []
color_map = {
    "equity": GREEN, "debt": AMBER, "macro": ACCENT,
    "risk": RED, "sentiment": VIOLET, "valuation": TEAL, "derivatives": "#ff7b72",
}
for grp, words in GROUPS.items():
    for w in words:
        if w in vocab:
            group_words.append(w)
            group_labels.append(grp)
            group_colors.append(color_map[grp])

vecs_grp = np.array([w2v.get_vector(w, vocab) for w in group_words])

# t-SNE to 2D
tsne  = TSNE(n_components=2, perplexity=min(20, len(group_words)//2),
             random_state=42, max_iter=1000, learning_rate="auto", init="pca")
vecs_2d = tsne.fit_transform(vecs_grp)

section(6, f"t-SNE 2D projection of {len(group_words)} finance terms computed")

# =============================================================================
# 7.  DOCUMENT EMBEDDINGS AS RETURN SIGNAL
# =============================================================================
# Mean-pool word vectors for each sentence -> document vector
# Use as feature in a simple directional predictor

# First, get market data to align
raw   = yf.download("SPY", start="2020-01-01", end="2023-12-31",
                    auto_adjust=True, progress=False)
close = raw["Close"].squeeze().dropna()
ret   = np.log(close / close.shift(1)).dropna()
N_mkt = len(ret)

# Ensure we have enough sentences
n_use = min(N_mkt, len(tokenized))
doc_vecs = []
for sent in tokenized[:n_use]:
    idxs = [vocab[w] for w in sent if w in vocab]
    if idxs:
        doc_vecs.append(w2v.W_in[idxs].mean(axis=0))
    else:
        doc_vecs.append(np.zeros(DIM))
doc_vecs = np.array(doc_vecs)               # (n_use, DIM)

# Project to scalar signal via first principal component
pca1   = PCA(n_components=1)
signal = pca1.fit_transform(doc_vecs).ravel()

# Align with returns and compute IC
ret_vals = ret.values[:n_use]
from scipy.stats import spearmanr
ic, pval = spearmanr(signal[:-1], ret_vals[1:])

# Cumulative signal strategy
sign_signal = np.sign(signal[:-1])
strat_ret   = sign_signal * ret_vals[1:]
sharpe      = (strat_ret.mean() / (strat_ret.std() + 1e-9)) * np.sqrt(252)

section(7, f"Doc-embedding signal  IC={ic:.4f}  p={pval:.3f}  Sharpe={sharpe:.3f}")

# =============================================================================
# FIGURE 1: TRAINING CURVE + SIMILARITY MATRIX + t-SNE
# =============================================================================
fig = plt.figure(figsize=(16, 12), facecolor=DARK)
fig.suptitle("Module 45 -- Word2Vec: Training, Embeddings & Semantic Structure",
             fontsize=12, color=TEXT, y=0.99)
gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.38, hspace=0.45)

# 1A: Loss curve
ax = fig.add_subplot(gs[0, 0])
ax.plot(range(1, len(loss_curve)+1), loss_curve, color=ACCENT, lw=2.0, marker="o", ms=5)
ax.set_title("Skip-gram Training Loss")
ax.set_xlabel("Epoch"); ax.set_ylabel("Mean Loss (NEG)")
ax.grid(True)

# 1B: Cosine similarity heatmap
ax = fig.add_subplot(gs[0, 1])
im = ax.imshow(sim_matrix, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(len(probe_in_vocab)))
ax.set_yticks(range(len(probe_in_vocab)))
ax.set_xticklabels(probe_in_vocab, rotation=45, ha="right", fontsize=7)
ax.set_yticklabels(probe_in_vocab, fontsize=7)
for i in range(len(probe_in_vocab)):
    for j in range(len(probe_in_vocab)):
        ax.text(j, i, f"{sim_matrix[i,j]:.2f}",
                ha="center", va="center", fontsize=6,
                color="black" if abs(sim_matrix[i,j]) < 0.5 else "white")
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
cb.set_label("Cosine Sim", color=TEXT, fontsize=7)
cb.ax.yaxis.set_tick_params(color=TEXT)
ax.set_title("Cosine Similarity Matrix")

# 1C: t-SNE scatter
ax = fig.add_subplot(gs[0, 2])
for grp in GROUPS:
    mask = [l == grp for l in group_labels]
    xs = vecs_2d[mask, 0]; ys = vecs_2d[mask, 1]
    ax.scatter(xs, ys, c=color_map[grp], s=40, label=grp, zorder=3, alpha=0.9)
    for xi, yi, w in zip(xs, ys,
                          [group_words[i] for i,m in enumerate(mask) if m]):
        ax.annotate(w, (xi, yi), fontsize=6, color=TEXT, alpha=0.8,
                    xytext=(3, 3), textcoords="offset points")
ax.set_title("t-SNE Embedding Space"); ax.legend(fontsize=6, ncol=2)
ax.grid(True)

# 1D: Document-embedding signal vs returns
ax = fig.add_subplot(gs[1, 0])
t_ = np.arange(len(signal))
ax.plot(t_, signal, color=VIOLET, lw=0.8, alpha=0.7, label="Doc-embed signal")
ax2 = ax.twinx()
ax2.plot(t_, ret_vals, color=AMBER, lw=0.5, alpha=0.4)
ax2.set_ylabel("Log-return", color=AMBER); ax2.tick_params(axis="y", colors=AMBER)
ax.set_title(f"Doc Embedding Signal  IC={ic:.4f}")
ax.set_xlabel("Day"); ax.set_ylabel("Signal (PC1)")
ax.legend(fontsize=7); ax.grid(True)

# 1E: Similar words bar charts for 2 query words
for col, qw in enumerate(["stock", "volatility"]):
    ax = fig.add_subplot(gs[1, col+1])
    sims_qw = w2v.most_similar(qw, vocab, idx2word, topn=8)
    if sims_qw:
        words_, vals_ = zip(*sims_qw)
        colors_ = [GREEN if v > 0 else RED for v in vals_]
        ax.barh(list(words_), list(vals_), color=colors_, edgecolor=DARK)
        ax.axvline(0, color=GRID, lw=0.8)
        ax.set_xlim(-0.2, 1.05)
        for i, (w_, v_) in enumerate(zip(words_, vals_)):
            ax.text(max(v_, 0) + 0.01, i, f"{v_:.2f}",
                    va="center", fontsize=7, color=TEXT)
    ax.set_title(f'Most Similar to "{qw}"')
    ax.set_xlabel("Cosine Similarity"); ax.invert_yaxis(); ax.grid(True, axis="x")

fig.savefig(os.path.join(FIGS, "m45_fig1_embeddings_similarity_tsne.png"),
            dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close(fig)

# =============================================================================
# FIGURE 2: ARCHITECTURE DIAGRAM + NEGATIVE SAMPLING DISTRIBUTION
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=DARK)
fig.suptitle("Module 45 -- Skip-gram Architecture & Negative Sampling",
             fontsize=12, color=TEXT, y=1.01)

ax = axes[0]
ax.set_facecolor(PANEL); ax.axis("off")
arch = (
    "SKIP-GRAM WITH NEGATIVE SAMPLING\n"
    "==================================\n\n"
    "  Input:  one-hot centre word  e_w  in {{0,1}}^V\n"
    "       |\n"
    "  W_in  (V x d)  --  embedding lookup\n"
    "       |  v_w = W_in[centre_idx]\n"
    "  Centre embedding:  v_w  in  R^{d}          d = {d}\n"
    "       |\n"
    "  For each context word c and K negatives nk:\n\n"
    "  Positive:  score_c  = v_w . W_out[c]\n"
    "  Negative:  score_nk = v_w . W_out[nk]\n\n"
    "  NEG loss:\n"
    "    L = -log sig(score_c)\n"
    "       - sum_k log sig(-score_nk)\n\n"
    "  SGD update:\n"
    "    W_in[w]    -= lr * grad_v_w\n"
    "    W_out[c]   -= lr * (sig(score_c)-1) * v_w\n"
    "    W_out[nk]  -= lr * sig(score_nk)   * v_w\n\n"
    "  Hyperparameters:\n"
    "    V={V}  d={d}  window={w}  K={k}  lr={lr}  epochs={ep}"
).format(d=DIM, V=V, w=WINDOW, k=K_NEG, lr=LR, ep=EPOCHS)

ax.text(0.05, 0.95, arch, transform=ax.transAxes,
        fontsize=8.5, va="top", fontfamily="monospace",
        color=TEXT, linespacing=1.7)
ax.set_title("Architecture & Gradient Update")

ax = axes[1]
# Plot the negative sampling distribution (top 30 words)
top30 = [(idx2word[i], neg_weights[i])
         for i in np.argsort(neg_weights)[::-1][:30] if i in idx2word]
top30_words, top30_vals = zip(*top30)
ax.bar(range(len(top30_words)), top30_vals,
       color=ACCENT, edgecolor=DARK, alpha=0.85)
ax.set_xticks(range(len(top30_words)))
ax.set_xticklabels(top30_words, rotation=70, ha="right", fontsize=7)
ax.set_title("Negative Sampling Distribution P_n(w) = f(w)^0.75 (top 30)")
ax.set_ylabel("Sampling probability"); ax.grid(True, axis="y")

fig.tight_layout()
fig.savefig(os.path.join(FIGS, "m45_fig2_architecture_negsampling.png"),
            dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close(fig)

# =============================================================================
# FIGURE 3: ANALOGY VISUALISATION IN 2D PCA SPACE
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 8), facecolor=DARK)
ax.set_facecolor(PANEL)
fig.suptitle("Module 45 -- Word2Vec Analogy Geometry in 2D PCA Space",
             fontsize=12, color=TEXT)

# PCA 2D of all vocab embeddings for context
pca2 = PCA(n_components=2)
pca2.fit(w2v.W_in)

analogy_vis = [
    ("bullish", "bull",  "bear",  "bearish"),
    ("equity",  "stock", "bond",  "debt"),
    ("profit",  "gain",  "loss",  "default"),
]
arrow_colors = [GREEN, ACCENT, AMBER]

for (pos1, neg1, pos2, expected), col in zip(analogy_vis, arrow_colors):
    words_av = [pos1, neg1, pos2, expected]
    if not all(w in vocab for w in words_av):
        continue
    pts = pca2.transform(np.array([w2v.get_vector(w, vocab) for w in words_av]))
    for w, pt in zip(words_av, pts):
        ax.scatter(*pt, color=col, s=80, zorder=4)
        ax.annotate(w, pt, fontsize=8, color=TEXT,
                    xytext=(5, 5), textcoords="offset points")
    # Draw parallelogram: pos1->neg1 and pos2->expected
    ax.annotate("", xy=pts[1], xytext=pts[0],
                arrowprops=dict(arrowstyle="->", color=col, lw=1.5, alpha=0.7))
    ax.annotate("", xy=pts[3], xytext=pts[2],
                arrowprops=dict(arrowstyle="->", color=col, lw=1.5, alpha=0.7,
                                linestyle="dashed"))
    ax.annotate("", xy=pts[2], xytext=pts[0],
                arrowprops=dict(arrowstyle="-", color=col, lw=0.8, alpha=0.4))
    ax.annotate("", xy=pts[3], xytext=pts[1],
                arrowprops=dict(arrowstyle="-", color=col, lw=0.8, alpha=0.4))

ax.set_title("Analogy Geometry: A - B + C = D  (parallelogram law)")
ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.grid(True)

fig.tight_layout()
fig.savefig(os.path.join(FIGS, "m45_fig3_analogy_geometry.png"),
            dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close(fig)

# =============================================================================
# SUMMARY
# =============================================================================
print()
print("  MODULE 45 COMPLETE -- 3 figures saved")
print("  Key Concepts:")
print("  [1] Skip-gram: predict context from centre word")
print("  [2] Neg sampling: K negatives drawn from P_n(w) = f(w)^0.75")
print("  [3] Distributional hypothesis: similar context -> similar meaning")
print(f"  [4] Training: V={V}  d={DIM}  epochs={EPOCHS}  final_loss={loss_curve[-1]:.4f}")
print("  [5] Cosine similarity encodes semantic proximity in R^d")
print("  [6] Analogy: pos1 - neg1 + pos2 = target (parallelogram law)")
print(f"  [7] Doc-embedding IC={ic:.4f}  Sharpe={sharpe:.3f}")
print("  [8] t-SNE reveals cluster structure by financial semantic group")
print(f"  NEXT: M46 -- Vectorised Backtesting: Moving Average Crossover")
print()
