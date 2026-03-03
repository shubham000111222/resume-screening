"""
skill_extractor.py — NER-based and keyword-based skill extraction
"""
from __future__ import annotations

import re
from typing import Optional

import spacy

# ──────────────────────────────────────────────────────────────────────────────
# Built-in skill taxonomy (expandable)
# ──────────────────────────────────────────────────────────────────────────────
SKILL_TAXONOMY: dict[str, list[str]] = {
    "programming_languages": [
        "python", "r", "java", "scala", "c++", "c#", "javascript", "typescript",
        "go", "rust", "julia", "matlab", "sql", "bash", "shell", "perl", "ruby",
    ],
    "ml_frameworks": [
        "tensorflow", "pytorch", "keras", "scikit-learn", "sklearn", "xgboost",
        "lightgbm", "catboost", "huggingface", "transformers", "fastai", "jax",
        "mxnet", "caffe", "onnx",
    ],
    "nlp": [
        "nlp", "bert", "gpt", "llm", "spacy", "nltk", "gensim", "word2vec",
        "glove", "fasttext", "sentence-transformers", "ner", "text classification",
        "sentiment analysis", "information extraction", "named entity recognition",
        "question answering", "summarization", "machine translation",
    ],
    "data_tools": [
        "pandas", "numpy", "scipy", "matplotlib", "seaborn", "plotly", "bokeh",
        "spark", "pyspark", "hadoop", "hive", "kafka", "dbt", "airflow", "luigi",
        "prefect", "dagster",
    ],
    "cloud_mlops": [
        "aws", "gcp", "azure", "sagemaker", "vertex ai", "databricks", "mlflow",
        "kubeflow", "docker", "kubernetes", "terraform", "ci/cd", "jenkins",
        "github actions", "fastapi", "flask", "django", "ray", "dask",
    ],
    "databases": [
        "postgresql", "mysql", "mongodb", "redis", "elasticsearch", "cassandra",
        "bigquery", "redshift", "snowflake", "pinecone", "weaviate", "chromadb",
    ],
    "soft_skills": [
        "communication", "leadership", "teamwork", "problem solving",
        "project management", "agile", "scrum", "mentoring",
    ],
}

# Flat set for fast lookup
ALL_SKILLS: set[str] = {s for skills in SKILL_TAXONOMY.values() for s in skills}

# ──────────────────────────────────────────────────────────────────────────────
# spaCy model (lazy-loaded)
# ──────────────────────────────────────────────────────────────────────────────
_NLP: Optional[spacy.language.Language] = None


def _get_nlp() -> spacy.language.Language:
    global _NLP
    if _NLP is None:
        try:
            _NLP = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess, sys
            subprocess.run(
                [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                check=True,
            )
            _NLP = spacy.load("en_core_web_sm")
    return _NLP


# ──────────────────────────────────────────────────────────────────────────────
# Extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_skills_keyword(text: str) -> dict[str, list[str]]:
    """
    Keyword-match based skill extraction against the taxonomy.

    Returns a dict of category → matched skills.
    """
    text_lower = text.lower()
    matched: dict[str, list[str]] = {}
    for category, skills in SKILL_TAXONOMY.items():
        found = [s for s in skills if re.search(rf"\b{re.escape(s)}\b", text_lower)]
        if found:
            matched[category] = found
    return matched


def extract_skills_ner(text: str) -> list[str]:
    """
    spaCy NER-based entity extraction (ORG, PRODUCT, WORK_OF_ART tags often
    capture technology names).
    """
    nlp = _get_nlp()
    doc = nlp(text[:100_000])  # cap at 100K chars for performance
    entities = [
        ent.text.lower()
        for ent in doc.ents
        if ent.label_ in ("ORG", "PRODUCT", "WORK_OF_ART", "GPE")
    ]
    # Keep only those that intersect our taxonomy
    return list({e for e in entities if e in ALL_SKILLS})


def extract_skills(text: str) -> dict[str, object]:
    """
    Combined skill extraction.

    Returns
    -------
    dict with keys:
      - by_category : dict[str, list[str]]
      - all_skills  : list[str]   (flat, deduplicated)
      - ner_skills  : list[str]
    """
    by_cat  = extract_skills_keyword(text)
    ner_sk  = extract_skills_ner(text)
    all_sk  = list({s for skills in by_cat.values() for s in skills} | set(ner_sk))
    return {"by_category": by_cat, "all_skills": sorted(all_sk), "ner_skills": ner_sk}


def skill_overlap(
    resume_skills: list[str], jd_skills: list[str]
) -> dict[str, object]:
    """
    Compute overlap statistics between a resume's skills and JD's skills.
    """
    rs = set(resume_skills)
    js = set(jd_skills)
    matched   = sorted(rs & js)
    missing   = sorted(js - rs)
    extra     = sorted(rs - js)
    pct = round(len(matched) / len(js) * 100, 1) if js else 0.0
    return {
        "matched": matched,
        "missing": missing,
        "extra":   extra,
        "match_pct": pct,
        "n_matched": len(matched),
        "n_required": len(js),
    }
