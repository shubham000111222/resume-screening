"""
embeddings.py — TF-IDF, Word2Vec-aggregated, and BERT-based embeddings
"""
from __future__ import annotations

import logging
import warnings
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# TF-IDF
# ──────────────────────────────────────────────────────────────────────────────

def build_tfidf(
    corpus: list[str],
    max_features: int = 10_000,
    ngram_range: tuple[int, int] = (1, 2),
) -> tuple[np.ndarray, TfidfVectorizer]:
    """
    Fit a TF-IDF vectorizer on *corpus* and return (matrix, fitted_vectorizer).
    """
    vec = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    matrix = vec.fit_transform(corpus).toarray()
    return matrix.astype(np.float32), vec


def tfidf_embed(
    texts: list[str],
    vectorizer: TfidfVectorizer,
) -> np.ndarray:
    """Transform *texts* using an already-fitted TF-IDF vectorizer."""
    return vectorizer.transform(texts).toarray().astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# BERT / Sentence-Transformers
# ──────────────────────────────────────────────────────────────────────────────

_ST_MODEL: Optional[object] = None
_ST_MODEL_NAME = "all-MiniLM-L6-v2"   # small, fast, good quality


def _get_st_model():
    global _ST_MODEL
    if _ST_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            _ST_MODEL = SentenceTransformer(_ST_MODEL_NAME)
            logger.info("Loaded SentenceTransformer: %s", _ST_MODEL_NAME)
        except ImportError:
            logger.error(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )
            raise
    return _ST_MODEL


def bert_embed(
    texts: list[str],
    batch_size: int = 32,
    show_progress: bool = False,
) -> np.ndarray:
    """
    Encode texts using Sentence-Transformers (all-MiniLM-L6-v2).

    Returns L2-normalized embeddings of shape (N, 384).
    """
    model = _get_st_model()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
    return normalize(embeddings.astype(np.float32))


# ──────────────────────────────────────────────────────────────────────────────
# Ensemble: weighted combination of TF-IDF + BERT
# ──────────────────────────────────────────────────────────────────────────────

def ensemble_embed(
    texts: list[str],
    vectorizer: TfidfVectorizer,
    tfidf_weight: float = 0.3,
    bert_weight: float = 0.7,
) -> np.ndarray:
    """
    Weighted concatenation of L2-normalized TF-IDF and BERT embeddings.

    The combined vector is L2-normalized again before returning.
    """
    tfidf_mat = normalize(tfidf_embed(texts, vectorizer))
    bert_mat  = bert_embed(texts)
    combined  = np.hstack(
        [tfidf_weight * tfidf_mat, bert_weight * bert_mat]
    )
    return normalize(combined)
