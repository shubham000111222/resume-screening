"""
tests/test_pipeline.py — Unit tests for each module
"""
from __future__ import annotations

import numpy as np
import pytest


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessor
# ──────────────────────────────────────────────────────────────────────────────
from src.preprocessor import (
    normalize_unicode,
    remove_noise,
    tokenize,
    remove_stopwords,
    lemmatize,
    preprocess,
)


def test_normalize_unicode():
    assert isinstance(normalize_unicode("café résumé"), str)


def test_remove_noise_removes_url():
    cleaned = remove_noise("Visit https://example.com for details")
    assert "http" not in cleaned


def test_remove_noise_removes_email():
    cleaned = remove_noise("Contact john@example.com")
    assert "@" not in cleaned


def test_tokenize_returns_lowercase_alpha():
    tokens = tokenize("Python Developer 2024!")
    assert all(t.isalpha() and t == t.lower() for t in tokens)


def test_remove_stopwords():
    tokens = ["the", "python", "developer", "is", "great"]
    filtered = remove_stopwords(tokens)
    assert "the" not in filtered
    assert "python" in filtered


def test_lemmatize_verbs():
    assert "run" in lemmatize(["running", "ran", "run"])


def test_preprocess_returns_string():
    result = preprocess("Senior Python Engineer with 5 years of ML experience")
    assert isinstance(result, str)
    assert len(result) > 0


def test_preprocess_returns_list_when_join_false():
    result = preprocess("Senior Python Engineer", join=False)
    assert isinstance(result, list)


# ──────────────────────────────────────────────────────────────────────────────
# Skill Extractor
# ──────────────────────────────────────────────────────────────────────────────
from src.skill_extractor import extract_skills_keyword, skill_overlap


def test_extract_skills_keyword_finds_python():
    text = "We require Python, TensorFlow, and Docker experience."
    result = extract_skills_keyword(text)
    all_skills = [s for skills in result.values() for s in skills]
    assert "python" in all_skills


def test_skill_overlap_matched():
    ov = skill_overlap(["python", "docker", "sql"], ["python", "docker", "java"])
    assert ov["matched"] == ["docker", "python"]
    assert "java" in ov["missing"]


def test_skill_overlap_percentage():
    ov = skill_overlap(["python", "docker"], ["python", "docker", "aws"])
    assert ov["match_pct"] == pytest.approx(66.7, abs=0.2)


# ──────────────────────────────────────────────────────────────────────────────
# Embeddings
# ──────────────────────────────────────────────────────────────────────────────
from src.embeddings import build_tfidf, tfidf_embed


def test_tfidf_shape():
    corpus = ["python machine learning", "data science nlp", "deep learning pytorch"]
    matrix, vec = build_tfidf(corpus)
    assert matrix.shape[0] == 3


def test_tfidf_embed_single():
    corpus = ["python machine learning", "data science nlp"]
    _, vec = build_tfidf(corpus)
    emb = tfidf_embed(["deep learning python"], vec)
    assert emb.shape[0] == 1


# ──────────────────────────────────────────────────────────────────────────────
# Similarity
# ──────────────────────────────────────────────────────────────────────────────
from src.similarity import cosine_scores, rank_candidates


def test_cosine_scores_range():
    jd  = np.random.rand(128).astype(np.float32)
    res = np.random.rand(5, 128).astype(np.float32)
    scores = cosine_scores(jd, res)
    assert scores.shape == (5,)
    assert all(0 <= s <= 1 for s in scores)


def test_rank_candidates_sorted():
    names    = ["A", "B", "C"]
    tfidf_sc = np.array([0.8, 0.5, 0.6])
    bert_sc  = np.array([0.9, 0.4, 0.7])
    skill_sc = [0.8, 0.3, 0.6]
    ranked   = rank_candidates(names, tfidf_sc, bert_sc, skill_sc)
    scores   = ranked["Composite Score"].tolist()
    assert scores == sorted(scores, reverse=True)


def test_rank_candidates_columns():
    names    = ["X", "Y"]
    ranked   = rank_candidates(
        names,
        np.array([0.7, 0.5]),
        np.array([0.8, 0.6]),
        [0.5, 0.4],
    )
    assert "Rank" in ranked.columns
    assert "Composite Score" in ranked.columns
