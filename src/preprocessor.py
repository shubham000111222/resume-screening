"""
preprocessor.py — Text cleaning, tokenization, lemmatization, stopword removal
"""
from __future__ import annotations

import re
import string
import unicodedata

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK data (safe to call multiple times)
for _pkg in ("punkt", "wordnet", "stopwords", "averaged_perceptron_tagger", "punkt_tab"):
    try:
        nltk.download(_pkg, quiet=True)
    except Exception:
        pass

_LEMMATIZER = WordNetLemmatizer()
_STOP_WORDS = set(stopwords.words("english"))

# Patterns
_URL_RE    = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE  = re.compile(r"\S+@\S+\.\S+")
_PHONE_RE  = re.compile(r"[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]")
_BULLET_RE = re.compile(r"^[\u2022\u2023\u25E6\u2043\u2219\-\*•·]+\s*", re.MULTILINE)
_WS_RE     = re.compile(r"\s+")


def normalize_unicode(text: str) -> str:
    """Normalize unicode characters to ASCII equivalents where possible."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def remove_noise(text: str) -> str:
    """Remove URLs, emails, phone numbers, bullets, and extra whitespace."""
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    text = _PHONE_RE.sub(" ", text)
    text = _BULLET_RE.sub(" ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = _WS_RE.sub(" ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    """Word-tokenize text and lowercase all tokens."""
    return [t.lower() for t in word_tokenize(text) if t.isalpha()]


def remove_stopwords(tokens: list[str]) -> list[str]:
    """Remove NLTK English stopwords."""
    return [t for t in tokens if t not in _STOP_WORDS]


def lemmatize(tokens: list[str]) -> list[str]:
    """Lemmatize tokens using WordNet lemmatizer."""
    return [_LEMMATIZER.lemmatize(t) for t in tokens]


def preprocess(text: str, join: bool = True) -> str | list[str]:
    """
    Full preprocessing pipeline.

    Steps:
      1. Unicode normalization
      2. Noise removal (URLs, emails, phones, bullets, punctuation)
      3. Tokenization
      4. Stopword removal
      5. Lemmatization

    Parameters
    ----------
    text : str
        Raw extracted text.
    join : bool
        If True, return a cleaned string; else return list of tokens.
    """
    text = normalize_unicode(text)
    text = remove_noise(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)
    return " ".join(tokens) if join else tokens


def preprocess_batch(texts: list[str]) -> list[str]:
    """Preprocess a list of texts."""
    return [preprocess(t) for t in texts]
