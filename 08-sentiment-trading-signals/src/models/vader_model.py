"""
================================================================================
VADER SENTIMENT MODEL WITH FINANCIAL CONTEXT TUNING
================================================================================
Implements VADER (Valence Aware Dictionary and sEntiment Reasoner) with
optional financial lexicon extensions. VADER is rule-based and requires
no GPU, making it ideal for high-throughput scoring of large corpora.

The compound score is computed as:

    compound = normalize(sum(valence_i * modifier_i))

where modifier_i accounts for intensifiers, negation, and punctuation.
The normalization maps the raw sum to [-1, +1] via:

    compound = x / sqrt(x^2 + alpha),  alpha = 15

Reference:
    Hutto, C. J. & Gilbert, E. (2014). VADER: A Parsimonious Rule-based
    Model for Sentiment Analysis of Social Media Text. AAAI ICWSM.

Author: Jose Orlando Bobadilla Fuentes, CQF | MSc AI
================================================================================
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class VADERSentiment:
    """
    VADER-based sentiment scorer with financial domain extensions.

    The base VADER lexicon contains ~7,500 features. We extend it with
    a curated financial lexicon that corrects domain-specific misclassifications.
    For example, "liability" is negative in general English but neutral in
    financial reporting; "bull" is animal-related in VADER but positive in
    finance.

    Parameters
    ----------
    use_financial_lexicon : bool
        If True, augment VADER with financial domain corrections.
    custom_lexicon : dict, optional
        Additional {word: valence} pairs to inject into the lexicon.
        Valence should be in [-4, +4].
    """

    # -------------------------------------------------------------------------
    # Financial domain corrections: {word: valence}
    # Positive valence = bullish/positive; Negative = bearish/negative
    # These override VADER defaults where financial meaning differs
    # -------------------------------------------------------------------------
    FINANCIAL_LEXICON = {
        # --- Words misclassified by general VADER ---
        "bull": 2.0,          # VADER: animal -> Finance: bullish
        "bear": -2.0,         # VADER: animal -> Finance: bearish
        "liability": 0.0,     # VADER: negative -> Finance: neutral (balance sheet)
        "short": -1.0,        # VADER: neutral -> Finance: bearish position
        "long": 1.0,          # VADER: neutral -> Finance: bullish position
        "call": 0.5,          # VADER: neutral -> Finance: mildly bullish
        "put": -0.5,          # VADER: neutral -> Finance: mildly bearish
        "volatile": -0.5,     # VADER: negative -> Finance: mildly negative
        "debt": -0.3,         # VADER: negative -> Finance: context-dependent
        "leverage": 0.0,      # VADER: neutral -> Finance: neutral (can be good/bad)
        "risk": -0.3,         # VADER: very negative -> Finance: mildly negative

        # --- Financial positive terms ---
        "upgrade": 2.5,
        "outperform": 2.5,
        "overweight": 1.5,
        "beat": 2.0,
        "beats": 2.0,
        "exceeded": 2.0,
        "upside": 2.0,
        "rally": 2.5,
        "bullish": 3.0,
        "breakout": 2.0,
        "dividend": 1.0,
        "buyback": 1.5,
        "guidance raised": 3.0,
        "consensus beat": 2.5,
        "all-time high": 2.5,
        "record revenue": 2.5,
        "strong demand": 2.0,

        # --- Financial negative terms ---
        "downgrade": -2.5,
        "underperform": -2.5,
        "underweight": -1.5,
        "miss": -2.0,
        "missed": -2.0,
        "misses": -2.0,
        "downside": -2.0,
        "selloff": -2.5,
        "sell-off": -2.5,
        "bearish": -3.0,
        "bankruptcy": -3.5,
        "default": -3.0,
        "layoffs": -2.0,
        "guidance cut": -3.0,
        "profit warning": -3.0,
        "revenue miss": -2.5,
        "margin compression": -2.0,
        "writedown": -2.5,
        "impairment": -2.0,
        "recession": -2.5,
        "stagflation": -2.5,
        "hawkish": -1.0,     # Negative for equities (tighter policy)
        "dovish": 1.0,       # Positive for equities (easier policy)
        "taper": -1.5,
        "inverted yield curve": -2.0,
    }

    def __init__(
        self,
        use_financial_lexicon: bool = True,
        custom_lexicon: Optional[Dict[str, float]] = None,
    ):
        self.analyzer = SentimentIntensityAnalyzer()

        # Inject financial lexicon into VADER
        if use_financial_lexicon:
            self.analyzer.lexicon.update(self.FINANCIAL_LEXICON)

        # Inject user-provided custom lexicon
        if custom_lexicon is not None:
            self.analyzer.lexicon.update(custom_lexicon)

        self.model_name = "VADER-Financial" if use_financial_lexicon else "VADER"

    def score_text(self, text: str) -> Dict[str, float]:
        """
        Score a single text and return full VADER output.

        Parameters
        ----------
        text : str
            Input text (headline, article body, social media post).

        Returns
        -------
        dict with keys:
            'compound' : float in [-1, 1] -- overall sentiment
            'pos'      : float in [0, 1]  -- positive proportion
            'neg'      : float in [0, 1]  -- negative proportion
            'neu'      : float in [0, 1]  -- neutral proportion
        """
        if not isinstance(text, str) or len(text.strip()) == 0:
            return {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0}
        return self.analyzer.polarity_scores(text)

    def score_batch(
        self,
        texts: List[str],
        return_compound_only: bool = False,
    ) -> Union[List[Dict[str, float]], np.ndarray]:
        """
        Score a batch of texts.

        Parameters
        ----------
        texts : list of str
            Documents to score.
        return_compound_only : bool
            If True, return numpy array of compound scores only.

        Returns
        -------
        List of score dicts, or np.ndarray of compound scores.
        """
        results = [self.score_text(t) for t in texts]
        if return_compound_only:
            return np.array([r["compound"] for r in results])
        return results

    def score_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "headline",
        prefix: str = "vader",
    ) -> pd.DataFrame:
        """
        Add sentiment columns to a DataFrame.

        Adds columns: {prefix}_compound, {prefix}_pos, {prefix}_neg, {prefix}_neu.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain a text column.
        text_column : str
            Name of the column containing text.
        prefix : str
            Prefix for new sentiment columns.

        Returns
        -------
        pd.DataFrame with additional sentiment columns.
        """
        df = df.copy()
        scores = self.score_batch(df[text_column].fillna("").tolist())
        for key in ["compound", "pos", "neg", "neu"]:
            df[f"{prefix}_{key}"] = [s[key] for s in scores]
        return df

    def __repr__(self) -> str:
        n_terms = len(self.analyzer.lexicon)
        return f"{self.model_name}(lexicon_size={n_terms})"
