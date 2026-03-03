# 🎯 ResumeRank AI — Resume Screening & Candidate Ranking System

> **NLP-powered resume screening** that ranks candidates against a job description using TF-IDF + BERT embeddings, skill extraction, and cosine similarity.

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40-red?logo=streamlit)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Overview

| Metric | Value |
|--------|-------|
| Embedding Models | TF-IDF (10K features) + BERT (all-MiniLM-L6-v2) |
| Skill Categories | 7 (80+ skills) |
| Supported Formats | PDF, DOCX, TXT |
| Ranking Metric | Weighted Composite Score (BERT 55% · TF-IDF 25% · Skills 20%) |
| Evaluation | Precision@K, MRR, NDCG@K |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Streamlit UI                         │
│  Upload Resumes → Upload JD → Configure → Analyse        │
└────────────────────────┬────────────────────────────────┘
                         │
          ┌──────────────▼──────────────┐
          │       src/extractor.py       │  PDF / DOCX / TXT → raw text
          └──────────────┬──────────────┘
                         │
          ┌──────────────▼──────────────┐
          │     src/preprocessor.py      │  Unicode → clean → tokenise
          │                              │  → stopwords → lemmatise
          └──────────┬──────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
┌────────▼────────┐   ┌──────────▼────────┐
│ src/embeddings  │   │ src/skill_extractor│
│ TF-IDF + BERT   │   │ NER + keyword match│
└────────┬────────┘   └──────────┬────────┘
         │                       │
         └───────────┬───────────┘
                     │
          ┌──────────▼──────────┐
          │  src/similarity.py   │  Cosine similarity → Composite score
          └──────────┬──────────┘
                     │
          ┌──────────▼──────────┐
          │  src/evaluator.py    │  Charts + Precision@K + NDCG@K
          └─────────────────────┘
```

---

## 📁 Project Structure

```
resume-screening/
├── app/
│   └── app.py                  # Streamlit UI
├── src/
│   ├── __init__.py
│   ├── extractor.py            # PDF / DOCX / TXT text extraction
│   ├── preprocessor.py         # Cleaning, tokenisation, lemmatisation
│   ├── embeddings.py           # TF-IDF + BERT embeddings
│   ├── similarity.py           # Cosine similarity + ranking + metrics
│   ├── skill_extractor.py      # NER + keyword skill extraction
│   └── evaluator.py            # Visualisations + evaluation plots
├── tests/
│   └── test_pipeline.py        # pytest unit tests
├── data/                       # (gitignored) local resume/JD files
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### 1. Clone & install

```bash
git clone https://github.com/shubham000111222/resume-screening.git
cd resume-screening
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download spaCy model

```bash
python -m spacy download en_core_web_sm
```

### 3. Run

```bash
streamlit run app/app.py
```

Open `http://localhost:8501`

---

## 🐳 Docker

```bash
# Build & run
docker compose up --build

# Or single command
docker run -p 8501:8501 resume-screening:latest
```

---

## 🧠 How It Works

### Text Extraction
- **PDF** → `pdfplumber` (table-aware, multi-page)
- **DOCX** → `docx2txt`
- **TXT**  → encoding-aware decode (UTF-8 / Latin-1 / CP1252)

### Preprocessing Pipeline
1. Unicode normalisation (NFKD → ASCII)
2. URL / email / phone / bullet removal
3. Punctuation stripping
4. NLTK word tokenisation
5. Stopword removal (English)
6. WordNet lemmatisation

### Skill Extraction
- **Keyword matching**: regex against 80+ skill taxonomy across 7 categories
- **NER**: spaCy `en_core_web_sm` — ORG / PRODUCT / WORK_OF_ART entities filtered to taxonomy

### Embeddings
| Method | Model | Dim |
|--------|-------|-----|
| TF-IDF | sklearn, 10K features, 1-2 ngrams | 10K |
| BERT   | `all-MiniLM-L6-v2` (Sentence-Transformers) | 384 |

### Composite Score

$$\text{Score} = w_{\text{bert}} \cdot \cos(\hat{e}_{jd}, \hat{e}_{r}) + w_{\text{tfidf}} \cdot \cos(\hat{t}_{jd}, \hat{t}_{r}) + w_{\text{skill}} \cdot \frac{|\text{skills}_{r} \cap \text{skills}_{jd}|}{|\text{skills}_{jd}|}$$

Default weights: BERT=0.55, TF-IDF=0.25, Skill=0.20 (configurable via sidebar).

### Evaluation Metrics
- **Precision@K** — fraction of top-K results that are relevant
- **MRR** — Mean Reciprocal Rank
- **NDCG@K** — Normalised Discounted Cumulative Gain

---

## 📊 Features

| Feature | Details |
|---------|---------|
| Multi-resume upload | PDF / DOCX / TXT |
| JD upload or paste | Flexible input |
| Configurable weights | BERT / TF-IDF / Skill sliders |
| Ranking table | Color-coded, downloadable CSV |
| Skill heatmap | Candidate × Skill binary matrix |
| Radar chart | Per-candidate score profile |
| Score distribution | Histogram of similarity scores |
| Grouped bar chart | BERT / TF-IDF / Skill breakdown |
| Evaluation tab | Precision@K, MRR, NDCG@K with custom ground truth |

---

## 🧪 Tests

```bash
pytest tests/ -v
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| UI | Streamlit 1.40 |
| BERT Embeddings | Sentence-Transformers (all-MiniLM-L6-v2) |
| TF-IDF | scikit-learn |
| NLP Preprocessing | NLTK (tokenise, lemmatise, stopwords) |
| NER | spaCy en_core_web_sm |
| PDF Extraction | pdfplumber, PyMuPDF |
| Visualisation | Plotly |
| Containerisation | Docker / Docker Compose |

---

## 👤 Author

**Shubham Kumar** — Data Science & AI  
[GitHub](https://github.com/shubham000111222) · [Portfolio](https://shubham-psi-navy.vercel.app) · [LinkedIn](https://linkedin.com/in/shubham-kumar-288b7437b)

---

## 📄 License

MIT © 2026 Shubham Kumar
