"""
================================================================================
LOUGHRAN-MCDONALD FINANCIAL SENTIMENT DICTIONARY
================================================================================
Implements the Loughran & McDonald (2011) financial sentiment lexicon,
which was specifically developed for 10-K filings and financial text.

Key insight: nearly 75% of words classified as negative by the
Harvard-IV-4 dictionary (commonly used in early finance NLP research)
are NOT negative in financial contexts. For example:
    - "tax", "cost", "capital", "liability" -> negative in H-IV, neutral in finance
    - "mine", "crude" -> negative in H-IV, industry terms in finance

The L-M dictionary defines six tone categories:
    1. Negative (~2,700 words)
    2. Positive (~350 words)
    3. Uncertainty (~300 words)
    4. Litigious (~900 words)
    5. Strong Modal (~20 words)
    6. Weak Modal (~30 words)

Sentiment score for document d:

    S(d) = (N_pos - N_neg) / (N_pos + N_neg + 1)

where N_pos and N_neg are counts of positive and negative words.
The +1 in the denominator avoids division by zero and acts as
a Bayesian pseudo-count pulling scores toward zero for short documents.

Reference:
    Loughran, T. & McDonald, B. (2011). When Is a Liability Not a Liability?
    Textual Analysis, Dictionaries, and 10-Ks. Journal of Finance, 66(1).

Author: Jose Orlando Bobadilla Fuentes, CQF | MSc AI
================================================================================
"""

import re
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union
from collections import Counter


class LMDictionarySentiment:
    """
    Loughran-McDonald financial dictionary sentiment scorer.

    Uses a curated subset of the most impactful L-M terms. In production,
    the full dictionary can be loaded from the official CSV at:
    https://sraf.nd.edu/loughranmcdonald-master-dictionary/

    Parameters
    ----------
    use_uncertainty : bool
        Include uncertainty words as a separate signal dimension.
    use_litigious : bool
        Include litigious words as a separate signal dimension.
    normalize_by_length : bool
        If True, normalize word counts by total document word count.
    """

    # -------------------------------------------------------------------------
    # Curated subsets of the Loughran-McDonald dictionary
    # Full dictionary has ~2,700 negative and ~350 positive terms
    # We include the 150 most frequent and impactful terms per category
    # -------------------------------------------------------------------------
    NEGATIVE_WORDS = {
        "abandon", "abdicate", "aberrant", "abolish", "abuse", "accident",
        "accusation", "adverse", "against", "aggravate", "allegation",
        "annul", "argue", "arrest", "assault", "attrition", "aversion",
        "backdating", "bail", "bailout", "bankrupt", "bankruptcy", "bribe",
        "burden", "catastrophe", "cease", "censure", "claim", "closure",
        "collusion", "complaint", "concern", "condemn", "confiscate",
        "controversy", "conviction", "corrupt", "crisis", "critical",
        "curtail", "damages", "deadlock", "debar", "decline", "default",
        "deferral", "deficiency", "deficit", "defraud", "delay", "deleterious",
        "delinquency", "demotion", "denial", "deplete", "depreciate",
        "depress", "deprive", "destabilize", "detain", "deter", "deteriorate",
        "detriment", "devastate", "diminish", "disadvantage", "disapproval",
        "disclose", "discontinue", "discrepancy", "dismiss", "displace",
        "dispute", "disqualify", "dissent", "dissolution", "distort",
        "distress", "divest", "downgrade", "downsize", "downturn",
        "embargo", "encumber", "erode", "erroneous", "evade", "eviction",
        "exacerbate", "exaggerate", "excessive", "exclude", "exhaust",
        "exploit", "expose", "expropriate", "fail", "failure", "false",
        "fatality", "fault", "felony", "fine", "forbid", "force", "forfeit",
        "fraud", "freeze", "grave", "grievance", "guilty", "halt", "hamper",
        "harass", "hardship", "harm", "hazard", "hinder", "hostile",
        "illegal", "impair", "impede", "implicate", "impose", "inability",
        "inadequate", "incapacity", "incompetent", "indict", "infringe",
        "insolvent", "instability", "insufficient", "interfere", "invalid",
        "investigate", "jeopardize", "lapse", "layoff", "litigate", "loss",
        "malfeasance", "malpractice", "misappropriate", "misconduct",
        "mislead", "misrepresent", "misstate", "monopolize", "negligence",
        "noncompliance", "nullify", "obstruct", "offense", "omission",
        "oppose", "oust", "overdue", "overstate", "penalize", "peril",
        "perjury", "plummet", "prohibit", "prosecute", "protest", "punish",
        "reassess", "recall", "recession", "reckless", "recoup", "redact",
        "reject", "relinquish", "reluctant", "remediate", "repeal", "resign",
        "restrain", "retaliate", "retract", "revoke", "sabotage", "sanction",
        "scandal", "scarcity", "scrutinize", "seize", "setback", "severe",
        "shareholder activism", "shortfall", "shutdown", "slander", "slump",
        "stagnate", "subpoena", "sue", "suffer", "suppress", "suspend",
        "terminate", "threat", "trespass", "turmoil", "undermine",
        "underperform", "underpay", "unfavorable", "unpaid", "unprofitable",
        "unresolved", "unstable", "unsuccessful", "usurp", "vandalism",
        "violate", "volatile", "vulnerability", "warn", "weaken",
        "whistleblower", "worsen", "writedown", "writeoff",
    }

    POSITIVE_WORDS = {
        "able", "accomplish", "achieve", "adequate", "advance", "advantage",
        "affirm", "agree", "approve", "assure", "attain", "attractive",
        "augment", "award", "benefit", "bolster", "boost", "breakthrough",
        "brilliant", "champion", "collaborate", "commend", "complement",
        "compliment", "comply", "confident", "constructive", "cooperate",
        "creative", "deliver", "dependable", "diligent", "distinguish",
        "diversify", "earn", "effective", "efficient", "empower", "enable",
        "encourage", "endorse", "enhance", "enjoy", "enrich", "enthusiasm",
        "entrepreneurial", "excel", "excellent", "exceptional", "exclusive",
        "exemplary", "expand", "favorable", "flourish", "fulfill", "gain",
        "generate", "good", "great", "guarantee", "honor", "ideal",
        "improve", "incentive", "increase", "influence", "innovative",
        "instrumental", "integrity", "leadership", "lucrative", "maximize",
        "meritorious", "milestone", "momentum", "notable", "opportunity",
        "optimal", "optimism", "outpace", "outperform", "outstanding",
        "overcome", "paramount", "partnership", "pioneer", "pleasant",
        "pledge", "popular", "positive", "premier", "premium", "prestigious",
        "proactive", "proficiency", "profit", "profitability", "progress",
        "prominent", "prosper", "prudent", "rebound", "recommend", "recover",
        "refund", "reinforce", "reliable", "remarkable", "resolve", "restore",
        "retain", "revitalize", "reward", "robust", "satisfy", "secure",
        "simplify", "solvent", "stabilize", "stellar", "stimulate",
        "strategic", "streamline", "strength", "strong", "succeed",
        "superior", "support", "surpass", "sustain", "synergy", "thrive",
        "top", "transform", "transparency", "triumph", "trustworthy",
        "unmatched", "unprecedented", "upgrade", "upside", "upturn",
        "valuable", "vibrant", "win",
    }

    UNCERTAINTY_WORDS = {
        "almost", "ambiguity", "ambiguous", "anticipate", "appear",
        "approximate", "assumption", "believe", "conceivable", "conditional",
        "contingency", "contingent", "depend", "doubt", "equivocal",
        "estimate", "expect", "expose", "fluctuate", "forecast", "hesitant",
        "hypothetical", "imprecise", "indefinite", "indeterminate",
        "inexact", "instability", "intangible", "likelihood", "may",
        "might", "nearly", "noncommittal", "obscure", "pending", "perhaps",
        "possibility", "possible", "precaution", "predict", "preliminary",
        "presume", "probability", "probable", "project", "provisional",
        "random", "reassess", "reconsider", "risky", "roughly", "seem",
        "sometime", "somewhat", "speculate", "suggest", "susceptible",
        "tentative", "uncertain", "unclear", "unconfirmed", "undecided",
        "undefined", "undetermined", "unforeseeable", "unknown",
        "unpredictable", "unproven", "unquantifiable", "unsettled",
        "unspecified", "unsure", "untested", "variable", "variability",
        "vary", "volatile",
    }

    def __init__(
        self,
        use_uncertainty: bool = True,
        use_litigious: bool = False,
        normalize_by_length: bool = True,
    ):
        self.use_uncertainty = use_uncertainty
        self.use_litigious = use_litigious
        self.normalize_by_length = normalize_by_length

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        Simple whitespace tokenizer with lowercasing and punctuation removal.
        For production, consider spaCy or NLTK word_tokenize.
        """
        text = text.lower()
        text = re.sub(r"[^a-z\s]", " ", text)
        return text.split()

    def score_text(self, text: str) -> Dict[str, float]:
        """
        Score a single document using Loughran-McDonald word counts.

        Returns
        -------
        dict with keys:
            'score'      : float in [-1, 1] -- (pos - neg) / (pos + neg + 1)
            'n_positive' : int -- count of positive words
            'n_negative' : int -- count of negative words
            'n_uncertainty' : int -- count of uncertainty words (if enabled)
            'pct_positive' : float -- positive words / total words
            'pct_negative' : float -- negative words / total words
            'n_words'    : int -- total word count
        """
        if not isinstance(text, str) or len(text.strip()) == 0:
            return {
                "score": 0.0, "n_positive": 0, "n_negative": 0,
                "n_uncertainty": 0, "pct_positive": 0.0,
                "pct_negative": 0.0, "n_words": 0,
            }

        tokens = self._tokenize(text)
        n_words = len(tokens)
        if n_words == 0:
            return {
                "score": 0.0, "n_positive": 0, "n_negative": 0,
                "n_uncertainty": 0, "pct_positive": 0.0,
                "pct_negative": 0.0, "n_words": 0,
            }

        word_counts = Counter(tokens)

        n_pos = sum(word_counts.get(w, 0) for w in self.POSITIVE_WORDS)
        n_neg = sum(word_counts.get(w, 0) for w in self.NEGATIVE_WORDS)
        n_unc = sum(word_counts.get(w, 0) for w in self.UNCERTAINTY_WORDS)

        # Sentiment score with pseudo-count to handle short documents
        score = (n_pos - n_neg) / (n_pos + n_neg + 1)

        result = {
            "score": score,
            "n_positive": n_pos,
            "n_negative": n_neg,
            "n_uncertainty": n_unc,
            "pct_positive": n_pos / n_words if self.normalize_by_length else n_pos,
            "pct_negative": n_neg / n_words if self.normalize_by_length else n_neg,
            "n_words": n_words,
        }
        return result

    def score_batch(
        self,
        texts: List[str],
        return_scores_only: bool = False,
    ) -> Union[List[Dict[str, float]], np.ndarray]:
        """Score a batch of texts."""
        results = [self.score_text(t) for t in texts]
        if return_scores_only:
            return np.array([r["score"] for r in results])
        return results

    def score_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "headline",
        prefix: str = "lm",
    ) -> pd.DataFrame:
        """Add L-M sentiment columns to a DataFrame."""
        df = df.copy()
        texts = df[text_column].fillna("").tolist()
        scores = self.score_batch(texts)
        df[f"{prefix}_score"] = [s["score"] for s in scores]
        df[f"{prefix}_n_pos"] = [s["n_positive"] for s in scores]
        df[f"{prefix}_n_neg"] = [s["n_negative"] for s in scores]
        df[f"{prefix}_pct_pos"] = [s["pct_positive"] for s in scores]
        df[f"{prefix}_pct_neg"] = [s["pct_negative"] for s in scores]
        if self.use_uncertainty:
            df[f"{prefix}_n_unc"] = [s["n_uncertainty"] for s in scores]
        return df

    def __repr__(self) -> str:
        return (
            f"LMDictionarySentiment(positive={len(self.POSITIVE_WORDS)}, "
            f"negative={len(self.NEGATIVE_WORDS)}, "
            f"uncertainty={len(self.UNCERTAINTY_WORDS)})"
        )
