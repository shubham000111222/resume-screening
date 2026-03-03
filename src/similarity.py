"""
similarity.py — Cosine similarity computation and candidate ranking
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# ──────────────────────────────────────────────────────────────────────────────
# Core similarity
# ──────────────────────────────────────────────────────────────────────────────

def cosine_scores(
    jd_vector: np.ndarray,
    resume_vectors: np.ndarray,
) -> np.ndarray:
    """
    Compute cosine similarity between a single JD vector and N resume vectors.

    Parameters
    ----------
    jd_vector      : shape (D,) or (1, D)
    resume_vectors : shape (N, D)

    Returns
    -------
    np.ndarray of shape (N,) with values in [0, 1]
    """
    jd = jd_vector.reshape(1, -1)
    scores = cosine_similarity(jd, resume_vectors)[0]
    return np.clip(scores, 0, 1)


# ──────────────────────────────────────────────────────────────────────────────
# Ranking
# ──────────────────────────────────────────────────────────────────────────────

def rank_candidates(
    filenames: list[str],
    tfidf_scores: np.ndarray,
    bert_scores: np.ndarray,
    skill_overlaps: list[float],
    tfidf_weight: float = 0.25,
    bert_weight: float = 0.55,
    skill_weight: float = 0.20,
) -> pd.DataFrame:
    """
    Build a ranked DataFrame of candidates using a weighted composite score.

    Parameters
    ----------
    filenames      : list of resume file names
    tfidf_scores   : cosine similarities from TF-IDF embeddings (N,)
    bert_scores    : cosine similarities from BERT embeddings (N,)
    skill_overlaps : skill match percentages / 100 (N,)  → in [0, 1]
    *_weight       : weights (must sum to 1.0)

    Returns
    -------
    pd.DataFrame sorted by composite_score descending
    """
    so = np.array(skill_overlaps, dtype=np.float32)
    composite = (
        tfidf_weight * tfidf_scores
        + bert_weight * bert_scores
        + skill_weight * so
    )

    df = pd.DataFrame(
        {
            "Rank":              range(1, len(filenames) + 1),  # placeholder
            "Candidate":        filenames,
            "BERT Score":       (bert_scores * 100).round(1),
            "TF-IDF Score":     (tfidf_scores * 100).round(1),
            "Skill Match %":    (so * 100).round(1),
            "Composite Score":  (composite * 100).round(1),
        }
    )
    df = df.sort_values("Composite Score", ascending=False).reset_index(drop=True)
    df["Rank"] = range(1, len(df) + 1)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ──────────────────────────────────────────────────────────────────────────────

def precision_at_k(
    ranked_df: pd.DataFrame,
    relevant_labels: list[str],
    k: int = 5,
) -> float:
    """
    Precision@K: fraction of top-K candidates that are relevant.

    Parameters
    ----------
    ranked_df       : output of rank_candidates()
    relevant_labels : ground-truth list of relevant candidate names
    k               : cutoff
    """
    top_k = set(ranked_df.head(k)["Candidate"].tolist())
    relevant = set(relevant_labels)
    return len(top_k & relevant) / k


def mean_reciprocal_rank(
    ranked_df: pd.DataFrame,
    relevant_labels: list[str],
) -> float:
    """Mean Reciprocal Rank (MRR) for the ranked list."""
    relevant = set(relevant_labels)
    for i, name in enumerate(ranked_df["Candidate"].tolist(), start=1):
        if name in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(
    ranked_df: pd.DataFrame,
    relevant_labels: list[str],
    k: int = 5,
) -> float:
    """Normalised Discounted Cumulative Gain @ K (binary relevance)."""
    relevant = set(relevant_labels)
    top_k = ranked_df.head(k)["Candidate"].tolist()
    dcg  = sum(int(c in relevant) / np.log2(i + 2) for i, c in enumerate(top_k))
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(relevant))))
    return round(dcg / idcg, 4) if idcg > 0 else 0.0
