"""
================================================================================
FINBERT SENTIMENT MODEL
================================================================================
Transformer-based sentiment analysis fine-tuned on financial text.

FinBERT uses the BERT architecture (Devlin et al., 2019) fine-tuned on
a corpus of financial communications including 10-K filings, analyst reports,
and earnings call transcripts. It outputs a probability distribution over
three classes: {positive, neutral, negative}.

Architecture:
    Input text -> WordPiece tokenization (max 512 tokens)
    -> BERT Encoder (12 layers, 768 hidden dim, 12 attention heads)
    -> [CLS] token pooling -> Dense(768, 3) -> Softmax
    -> P(positive), P(neutral), P(negative)

The sentiment score is computed as:
    score = P(positive) - P(negative)

which yields a continuous value in [-1, +1].

Reference:
    Araci, D. (2019). FinBERT: Financial Sentiment Analysis with
    Pre-trained Language Models. arXiv:1908.10063.
    Huang, A. et al. (2023). FinBERT: A Large Language Model for
    Extracting Information from Financial Text. Contemporary Accounting Research.

Author: Jose Orlando Bobadilla Fuentes, CQF | MSc AI
================================================================================
"""

import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Optional, Union
from tqdm import tqdm


class FinBERTSentiment:
    """
    FinBERT-based sentiment scorer for financial text.

    Loads ProsusAI/finbert from HuggingFace Hub (or a local cache).
    Supports CPU and GPU inference with automatic batching for
    memory-efficient processing of large corpora.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier. Default is 'ProsusAI/finbert'.
    device : str, optional
        'cuda', 'cpu', or 'auto' (auto-detect GPU availability).
    max_length : int
        Maximum token length for BERT input (truncated beyond this).
    batch_size : int
        Inference batch size. Larger = faster but more memory.
    """

    # Class labels as defined by ProsusAI/finbert
    LABELS = ["positive", "negative", "neutral"]

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        device: Optional[str] = "auto",
        max_length: int = 512,
        batch_size: int = 32,
    ):
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        # Resolve device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()  # Inference mode (disable dropout)

        self.max_length = max_length
        self.batch_size = batch_size
        self.model_name = model_name

    @torch.no_grad()
    def _predict_batch(self, texts: List[str]) -> np.ndarray:
        """
        Run inference on a single batch.

        Returns
        -------
        np.ndarray of shape (len(texts), 3)
            Columns: [P(positive), P(negative), P(neutral)]
        """
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encodings = {k: v.to(self.device) for k, v in encodings.items()}

        outputs = self.model(**encodings)
        probabilities = torch.softmax(outputs.logits, dim=-1)
        return probabilities.cpu().numpy()

    def score_text(self, text: str) -> Dict[str, float]:
        """
        Score a single text.

        Returns
        -------
        dict with keys:
            'score'    : float in [-1, 1] -- P(pos) - P(neg)
            'positive' : float -- probability of positive class
            'negative' : float -- probability of negative class
            'neutral'  : float -- probability of neutral class
            'label'    : str   -- most probable class label
        """
        if not isinstance(text, str) or len(text.strip()) == 0:
            return {
                "score": 0.0,
                "positive": 0.0,
                "negative": 0.0,
                "neutral": 1.0,
                "label": "neutral",
            }
        probs = self._predict_batch([text])[0]
        p_pos, p_neg, p_neu = probs[0], probs[1], probs[2]
        label = self.LABELS[np.argmax(probs)]
        return {
            "score": float(p_pos - p_neg),
            "positive": float(p_pos),
            "negative": float(p_neg),
            "neutral": float(p_neu),
            "label": label,
        }

    def score_batch(
        self,
        texts: List[str],
        return_scores_only: bool = False,
        show_progress: bool = True,
    ) -> Union[List[Dict[str, float]], np.ndarray]:
        """
        Score a list of texts with automatic batching.

        Parameters
        ----------
        texts : list of str
            Documents to score.
        return_scores_only : bool
            If True, return np.ndarray of composite scores (P(pos)-P(neg)).
        show_progress : bool
            Show tqdm progress bar.

        Returns
        -------
        List of score dicts, or np.ndarray of float scores.
        """
        all_probs = []
        n_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        iterator = range(0, len(texts), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, total=n_batches, desc="FinBERT")

        for i in iterator:
            batch_texts = [
                t if isinstance(t, str) and len(t.strip()) > 0 else "neutral"
                for t in texts[i : i + self.batch_size]
            ]
            probs = self._predict_batch(batch_texts)
            all_probs.append(probs)

        all_probs = np.vstack(all_probs)  # (N, 3)

        if return_scores_only:
            # score = P(positive) - P(negative)
            return all_probs[:, 0] - all_probs[:, 1]

        results = []
        for j in range(len(texts)):
            p_pos, p_neg, p_neu = all_probs[j]
            label = self.LABELS[np.argmax(all_probs[j])]
            results.append({
                "score": float(p_pos - p_neg),
                "positive": float(p_pos),
                "negative": float(p_neg),
                "neutral": float(p_neu),
                "label": label,
            })
        return results

    def score_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "headline",
        prefix: str = "finbert",
    ) -> pd.DataFrame:
        """
        Add FinBERT sentiment columns to a DataFrame.

        Adds: {prefix}_score, {prefix}_pos, {prefix}_neg, {prefix}_neu, {prefix}_label.
        """
        df = df.copy()
        texts = df[text_column].fillna("").tolist()
        scores = self.score_batch(texts, show_progress=True)
        df[f"{prefix}_score"] = [s["score"] for s in scores]
        df[f"{prefix}_pos"] = [s["positive"] for s in scores]
        df[f"{prefix}_neg"] = [s["negative"] for s in scores]
        df[f"{prefix}_neu"] = [s["neutral"] for s in scores]
        df[f"{prefix}_label"] = [s["label"] for s in scores]
        return df

    def __repr__(self) -> str:
        return f"FinBERTSentiment(model={self.model_name}, device={self.device})"
