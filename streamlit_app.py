"""
streamlit_app.py — Root-level entry point for Streamlit Cloud
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import nltk
for _p in ("punkt", "wordnet", "stopwords", "averaged_perceptron_tagger", "punkt_tab"):
    nltk.download(_p, quiet=True)

from src.extractor       import extract_text
from src.preprocessor    import preprocess
from src.embeddings      import build_tfidf, tfidf_embed
from src.similarity      import cosine_scores, rank_candidates
from src.skill_extractor import extract_skills, skill_overlap
from src.evaluator       import (
    plot_similarity_distribution,
    plot_score_breakdown,
    plot_radar,
    plot_skill_heatmap,
    plot_composite_bar,
)

@st.cache_resource(show_spinner="Loading BERT model (first run only)...")
def load_bert_model():
    from sentence_transformers import SentenceTransformer
    from sklearn.preprocessing import normalize
    import numpy as np
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model

@st.cache_resource(show_spinner="Loading spaCy model...")
def load_spacy_model():
    import spacy
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        import subprocess, sys
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
        return spacy.load("en_core_web_sm")

def bert_embed(texts, show_progress=False):
    from sklearn.preprocessing import normalize
    import numpy as np
    model = load_bert_model()
    emb = model.encode(texts, show_progress_bar=show_progress, convert_to_numpy=True)
    return normalize(emb.astype(np.float32))

# Pre-load models on startup
load_bert_model()
load_spacy_model()

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ResumeRank AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.metric-card {
    background: rgba(99,102,241,0.1);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 12px;
    padding: 1rem 1.5rem;
    text-align: center;
}
.rank-badge { font-size:1.8rem; font-weight:800; color:#6366f1; }
.skill-chip {
    display:inline-block; background:rgba(99,102,241,0.2);
    border:1px solid #6366f1; border-radius:20px;
    padding:2px 10px; margin:2px; font-size:0.75rem; color:#a5b4fc;
}
.missing-chip { background:rgba(239,68,68,0.15); border-color:#ef4444; color:#fca5a5; }
</style>
""", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("# 🎯 ResumeRank AI")
st.markdown(
    "**NLP-powered resume screening** · TF-IDF + BERT embeddings · "
    "Skill extraction · Candidate ranking · Built by Shubham Kumar"
)
st.divider()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    bert_weight  = st.slider("BERT weight",   0.0, 1.0, 0.55, 0.05)
    tfidf_weight = st.slider("TF-IDF weight", 0.0, 1.0, 0.25, 0.05)
    skill_weight = st.slider("Skill weight",  0.0, 1.0, 0.20, 0.05)
    total = bert_weight + tfidf_weight + skill_weight
    if abs(total - 1.0) > 0.01:
        st.warning(f"Weights sum to {total:.2f} — will be auto-normalised.")
    top_n = st.slider("Top-N to display", 3, 20, 10)
    st.divider()
    st.caption("**Model**: all-MiniLM-L6-v2\n\n**Skills DB**: 80+ skills, 7 categories")

# Normalise weights
total = bert_weight + tfidf_weight + skill_weight
if total > 0:
    bert_weight /= total; tfidf_weight /= total; skill_weight /= total

# ─── Upload ───────────────────────────────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("📄 Upload Resumes")
    resume_files = st.file_uploader(
        "Upload resumes (PDF, DOCX, TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        key="resumes",
    )

with col_right:
    st.subheader("📋 Job Description")
    jd_mode = st.radio("Input mode", ["Paste text", "Upload file"], horizontal=True)
    jd_text_raw = ""
    if jd_mode == "Paste text":
        jd_text_raw = st.text_area(
            "Paste job description",
            height=250,
            placeholder="We are looking for a Senior Data Scientist with experience in Python, machine learning...",
        )
    else:
        jd_file = st.file_uploader("Upload JD (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], key="jd")
        if jd_file:
            jd_text_raw = extract_text(jd_file.name, jd_file.read())

# ─── Analyse button ───────────────────────────────────────────────────────────
ready = bool(resume_files and jd_text_raw.strip())
if not resume_files:
    st.info("👆 Upload at least one resume and provide a job description to get started.")

if st.button("🚀 Analyse & Rank Candidates", type="primary", disabled=not ready) and ready:

    with st.spinner("📥 Extracting text..."):
        raw_texts = {f.name: extract_text(f.name, f.read()) for f in resume_files}

    names    = list(raw_texts.keys())
    raw_list = list(raw_texts.values())

    with st.spinner("🧹 Preprocessing..."):
        clean_resumes = [preprocess(t) for t in raw_list]
        clean_jd      = preprocess(jd_text_raw)

    with st.spinner("🔍 Extracting skills..."):
        jd_skills          = extract_skills(jd_text_raw)["all_skills"]
        resume_skills_list = [extract_skills(t)["all_skills"] for t in raw_list]
        overlaps           = [skill_overlap(rs, jd_skills) for rs in resume_skills_list]
        skill_pcts         = [o["match_pct"] / 100.0 for o in overlaps]

    with st.spinner("📐 TF-IDF embeddings..."):
        _, vectorizer = build_tfidf(clean_resumes + [clean_jd])
        jd_tfidf  = tfidf_embed([clean_jd], vectorizer)[0]
        res_tfidf = tfidf_embed(clean_resumes, vectorizer)
        tfidf_sc  = cosine_scores(jd_tfidf, res_tfidf)

    with st.spinner("🤖 BERT embeddings (this may take ~30s on first run)..."):
        all_emb = bert_embed(raw_list + [jd_text_raw])
        bert_sc = cosine_scores(all_emb[-1], all_emb[:-1])

    ranked = rank_candidates(
        names, tfidf_sc, bert_sc, skill_pcts,
        tfidf_weight=tfidf_weight, bert_weight=bert_weight, skill_weight=skill_weight,
    )

    # ── Results ── #
    st.divider()
    st.markdown("## 📊 Results")

    top3 = ranked.head(min(3, len(ranked)))
    medals = ["🥇", "🥈", "🥉"]
    kpi_cols = st.columns(len(top3))
    for i, (_, row) in enumerate(top3.iterrows()):
        with kpi_cols[i]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="rank-badge">{medals[i]}</div>
                <div style="font-size:1rem;font-weight:700;margin:4px 0">{row['Candidate']}</div>
                <div style="font-size:1.6rem;font-weight:800;color:#a5b4fc">{row['Composite Score']:.1f}%</div>
                <div style="font-size:0.75rem;color:#888">
                    BERT {row['BERT Score']:.0f}% · TF-IDF {row['TF-IDF Score']:.0f}% · Skills {row['Skill Match %']:.0f}%
                </div>
            </div>""", unsafe_allow_html=True)

    st.divider()
    tab_rank, tab_skills, tab_charts, tab_detail, tab_eval = st.tabs(
        ["🏆 Ranking", "🧠 Skills", "📈 Charts", "🔬 Detail", "📏 Evaluation"]
    )

    with tab_rank:
        def _color(val):
            if val >= 75: return "background-color:rgba(34,197,94,0.25);color:#86efac"
            elif val >= 50: return "background-color:rgba(234,179,8,0.2);color:#fde047"
            return "background-color:rgba(239,68,68,0.15);color:#fca5a5"

        styled = (ranked.style
            .map(_color, subset=["Composite Score"])
            .format({"BERT Score":"{:.1f}%","TF-IDF Score":"{:.1f}%",
                     "Skill Match %":"{:.1f}%","Composite Score":"{:.1f}%"}))
        st.dataframe(styled, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download CSV", ranked.to_csv(index=False),
                           "ranking.csv", "text/csv")

    with tab_skills:
        chips = " ".join(f'<span class="skill-chip">{s}</span>' for s in sorted(jd_skills))
        st.markdown("**JD Required Skills:** " + (chips or "<em>none detected</em>"),
                    unsafe_allow_html=True)
        st.divider()
        for i, name in enumerate(names):
            ov = overlaps[i]
            with st.expander(f"**{name}** — {ov['match_pct']:.0f}% skill match"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Matched", ov["n_matched"])
                c2.metric("Required", ov["n_required"])
                c3.metric("Match %", f"{ov['match_pct']:.0f}%")
                mc = " ".join(f'<span class="skill-chip">{s}</span>' for s in ov["matched"]) or "<em>none</em>"
                mg = " ".join(f'<span class="skill-chip missing-chip">{s}</span>' for s in ov["missing"]) or "<em>none</em>"
                st.markdown(f"✅ **Matched:** {mc}", unsafe_allow_html=True)
                st.markdown(f"❌ **Missing:** {mg}", unsafe_allow_html=True)
        st.plotly_chart(plot_skill_heatmap(names, resume_skills_list, jd_skills),
                        use_container_width=True)

    with tab_charts:
        st.plotly_chart(plot_composite_bar(ranked), use_container_width=True)
        st.plotly_chart(plot_score_breakdown(ranked, top_n=min(top_n, len(ranked))),
                        use_container_width=True)
        c1, c2 = st.columns(2)
        c1.plotly_chart(plot_similarity_distribution((bert_sc*100).tolist(), "BERT Distribution"),
                        use_container_width=True)
        c2.plotly_chart(plot_similarity_distribution((tfidf_sc*100).tolist(), "TF-IDF Distribution"),
                        use_container_width=True)

    with tab_detail:
        sel = st.selectbox("Select candidate", ranked["Candidate"].tolist())
        idx = names.index(sel)
        row = ranked[ranked["Candidate"] == sel].iloc[0]
        c1, c2 = st.columns(2)
        c1.plotly_chart(plot_radar(sel, float(row["BERT Score"]), float(row["TF-IDF Score"]),
                                   float(row["Skill Match %"]), float(row["Composite Score"])),
                        use_container_width=True)
        with c2:
            st.metric("Composite Score", f"{row['Composite Score']:.1f}%")
            st.metric("BERT Score",      f"{row['BERT Score']:.1f}%")
            st.metric("TF-IDF Score",    f"{row['TF-IDF Score']:.1f}%")
            st.metric("Skill Match",     f"{row['Skill Match %']:.1f}%")
            st.metric("Rank",            f"#{int(row['Rank'])} of {len(ranked)}")
        with st.expander("📄 Resume text"):
            st.text(raw_list[idx][:3000])

    with tab_eval:
        st.info("Select ground-truth relevant candidates to measure ranking quality.")
        rel = st.multiselect("Relevant candidates (ground truth)", options=names)
        k   = st.slider("K", 1, min(10, len(names)), min(5, len(names)))
        if rel:
            from src.similarity import precision_at_k, mean_reciprocal_rank, ndcg_at_k
            m1, m2, m3 = st.columns(3)
            m1.metric(f"Precision@{k}", f"{precision_at_k(ranked, rel, k):.3f}")
            m2.metric("MRR",            f"{mean_reciprocal_rank(ranked, rel):.3f}")
            m3.metric(f"NDCG@{k}",      f"{ndcg_at_k(ranked, rel, k):.4f}")
