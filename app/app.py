"""
app.py — Resume Screening & Candidate Ranking System (Streamlit UI)
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import streamlit as st

from src.extractor      import extract_text
from src.preprocessor   import preprocess
from src.embeddings     import build_tfidf, tfidf_embed, bert_embed
from src.similarity     import cosine_scores, rank_candidates
from src.skill_extractor import extract_skills, skill_overlap
from src.evaluator      import (
    plot_similarity_distribution,
    plot_score_breakdown,
    plot_radar,
    plot_skill_heatmap,
    plot_composite_bar,
)

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ResumeRank AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .main { background: #0f0f1a; }
    .metric-card {
        background: rgba(99,102,241,0.1);
        border: 1px solid rgba(99,102,241,0.3);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        text-align: center;
    }
    .rank-badge {
        font-size: 1.8rem;
        font-weight: 800;
        color: #6366f1;
    }
    .skill-chip {
        display: inline-block;
        background: rgba(99,102,241,0.2);
        border: 1px solid #6366f1;
        border-radius: 20px;
        padding: 2px 10px;
        margin: 2px;
        font-size: 0.75rem;
        color: #a5b4fc;
    }
    .missing-chip {
        background: rgba(239,68,68,0.15);
        border-color: #ef4444;
        color: #fca5a5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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

    bert_weight  = st.slider("BERT weight",     0.0, 1.0, 0.55, 0.05)
    tfidf_weight = st.slider("TF-IDF weight",   0.0, 1.0, 0.25, 0.05)
    skill_weight = st.slider("Skill weight",    0.0, 1.0, 0.20, 0.05)

    total = bert_weight + tfidf_weight + skill_weight
    if abs(total - 1.0) > 0.01:
        st.warning(f"Weights sum to {total:.2f}. They will be auto-normalised.")

    top_n = st.slider("Top-N results to display", 3, 20, 10)

    st.divider()
    st.caption(
        "**Model**: all-MiniLM-L6-v2\n\n"
        "**TF-IDF**: max 10K features, 1-2 ngrams\n\n"
        "**Skills DB**: 80+ skills across 7 categories"
    )

# ─── Normalize weights ────────────────────────────────────────────────────────
total = bert_weight + tfidf_weight + skill_weight
if total > 0:
    bert_weight  /= total
    tfidf_weight /= total
    skill_weight /= total

# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 — Upload inputs
# ──────────────────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("📄 Upload Resumes")
    resume_files = st.file_uploader(
        "Upload one or more resumes (PDF, DOCX, TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        key="resumes",
    )

with col_right:
    st.subheader("📋 Job Description")
    jd_mode = st.radio("Input mode", ["Upload file", "Paste text"], horizontal=True)
    if jd_mode == "Upload file":
        jd_file = st.file_uploader(
            "Upload job description (PDF, DOCX, TXT)",
            type=["pdf", "docx", "txt"],
            key="jd_file",
        )
        jd_text_raw = ""
        if jd_file:
            jd_text_raw = extract_text(jd_file.name, jd_file.read())
    else:
        jd_text_raw = st.text_area(
            "Paste job description here",
            height=280,
            placeholder="We are looking for a Senior Data Scientist with experience in...",
        )

# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 — Run analysis
# ──────────────────────────────────────────────────────────────────────────────
analyze_btn = st.button(
    "🚀 Analyse & Rank Candidates",
    type="primary",
    disabled=(not resume_files or not jd_text_raw.strip()),
)

if not resume_files:
    st.info("👆 Upload at least one resume and a job description to get started.")

if analyze_btn and resume_files and jd_text_raw.strip():
    # ── Extract raw text ── #
    with st.spinner("📥 Extracting text from resumes..."):
        raw_texts: dict[str, str] = {}
        for f in resume_files:
            raw_texts[f.name] = extract_text(f.name, f.read())

    names     = list(raw_texts.keys())
    raw_list  = list(raw_texts.values())

    # ── Preprocess ── #
    with st.spinner("🧹 Preprocessing text (tokenise · lemmatise · stopwords)..."):
        clean_resumes = [preprocess(t) for t in raw_list]
        clean_jd      = preprocess(jd_text_raw)

    # ── Skill extraction ── #
    with st.spinner("🔍 Extracting skills via NER + keyword matching..."):
        jd_skills_data     = extract_skills(jd_text_raw)
        jd_skills          = jd_skills_data["all_skills"]
        resume_skills_data = [extract_skills(t) for t in raw_list]
        resume_skills_list = [d["all_skills"] for d in resume_skills_data]
        overlaps           = [
            skill_overlap(rs, jd_skills)
            for rs in resume_skills_list
        ]
        skill_pcts = [o["match_pct"] / 100.0 for o in overlaps]

    # ── TF-IDF embeddings ── #
    with st.spinner("📐 Building TF-IDF embeddings..."):
        corpus = clean_resumes + [clean_jd]
        tfidf_matrix, vectorizer = build_tfidf(corpus)
        jd_tfidf  = tfidf_embed([clean_jd], vectorizer)[0]
        res_tfidf = tfidf_embed(clean_resumes, vectorizer)
        tfidf_sc  = cosine_scores(jd_tfidf, res_tfidf)

    # ── BERT embeddings ── #
    with st.spinner("🤖 Computing BERT embeddings (all-MiniLM-L6-v2)..."):
        all_emb  = bert_embed(raw_list + [jd_text_raw], show_progress=False)
        jd_emb   = all_emb[-1]
        res_emb  = all_emb[:-1]
        bert_sc  = cosine_scores(jd_emb, res_emb)

    # ── Rank ── #
    ranked = rank_candidates(
        filenames=names,
        tfidf_scores=tfidf_sc,
        bert_scores=bert_sc,
        skill_overlaps=skill_pcts,
        tfidf_weight=tfidf_weight,
        bert_weight=bert_weight,
        skill_weight=skill_weight,
    )

    # ====================================================================
    # RESULTS
    # ====================================================================
    st.divider()
    st.markdown("## 📊 Results")

    # ── Top 3 KPI cards ── #
    top3 = ranked.head(3)
    kpi_cols = st.columns(3)
    medals = ["🥇", "🥈", "🥉"]
    for i, (_, row) in enumerate(top3.iterrows()):
        with kpi_cols[i]:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="rank-badge">{medals[i]}</div>
                    <div style="font-size:1rem;font-weight:700;margin:4px 0">{row['Candidate']}</div>
                    <div style="font-size:1.6rem;font-weight:800;color:#a5b4fc">{row['Composite Score']:.1f}%</div>
                    <div style="font-size:0.75rem;color:#888">
                        BERT {row['BERT Score']:.0f}% · TF-IDF {row['TF-IDF Score']:.0f}% · Skills {row['Skill Match %']:.0f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Tabs ── #
    tab_rank, tab_skills, tab_charts, tab_detail, tab_eval = st.tabs(
        ["🏆 Ranking Table", "🧠 Skill Analysis", "📈 Charts", "🔬 Candidate Detail", "📏 Evaluation"]
    )

    # ────────────────────────────────────────────────────────────────────
    # Tab 1 — Ranking Table
    # ────────────────────────────────────────────────────────────────────
    with tab_rank:
        st.subheader("Full Candidate Ranking")

        def _color_composite(val):
            if val >= 75:
                return "background-color: rgba(34,197,94,0.25); color:#86efac"
            elif val >= 50:
                return "background-color: rgba(234,179,8,0.2); color:#fde047"
            else:
                return "background-color: rgba(239,68,68,0.15); color:#fca5a5"

        styled = (
            ranked.style
            .applymap(_color_composite, subset=["Composite Score"])
            .format({
                "BERT Score":    "{:.1f}%",
                "TF-IDF Score":  "{:.1f}%",
                "Skill Match %": "{:.1f}%",
                "Composite Score": "{:.1f}%",
            })
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)

        csv = ranked.to_csv(index=False)
        st.download_button(
            "⬇️ Download Ranking CSV",
            csv,
            file_name="candidate_ranking.csv",
            mime="text/csv",
        )

    # ────────────────────────────────────────────────────────────────────
    # Tab 2 — Skill Analysis
    # ────────────────────────────────────────────────────────────────────
    with tab_skills:
        st.subheader("JD Required Skills")
        jd_chip_html = " ".join(
            f'<span class="skill-chip">{s}</span>' for s in sorted(jd_skills)
        ) or "<em>No skills detected</em>"
        st.markdown(jd_chip_html, unsafe_allow_html=True)

        st.divider()

        # Per-candidate overlap
        for i, name in enumerate(names):
            ov    = overlaps[i]
            score = ov["match_pct"]
            color = "#22c55e" if score >= 60 else "#f59e0b" if score >= 30 else "#ef4444"
            with st.expander(f"**{name}** — {score:.0f}% skill match"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Matched",  ov["n_matched"])
                c2.metric("Required", ov["n_required"])
                c3.metric("Match %",  f"{score:.0f}%")

                mc = " ".join(f'<span class="skill-chip">{s}</span>' for s in ov["matched"]) or "<em>none</em>"
                mg = " ".join(f'<span class="skill-chip missing-chip">{s}</span>' for s in ov["missing"]) or "<em>none</em>"
                st.markdown(f"**✅ Matched:** {mc}", unsafe_allow_html=True)
                st.markdown(f"**❌ Missing:** {mg}", unsafe_allow_html=True)

        st.divider()
        st.subheader("Skill Heatmap — All Candidates")
        fig_heat = plot_skill_heatmap(names, resume_skills_list, jd_skills)
        st.plotly_chart(fig_heat, use_container_width=True)

    # ────────────────────────────────────────────────────────────────────
    # Tab 3 — Charts
    # ────────────────────────────────────────────────────────────────────
    with tab_charts:
        st.plotly_chart(plot_composite_bar(ranked), use_container_width=True)
        st.plotly_chart(plot_score_breakdown(ranked, top_n=min(top_n, len(ranked))), use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(
                plot_similarity_distribution(
                    (bert_sc * 100).tolist(), "BERT Score Distribution"
                ),
                use_container_width=True,
            )
        with col_b:
            st.plotly_chart(
                plot_similarity_distribution(
                    (tfidf_sc * 100).tolist(), "TF-IDF Score Distribution"
                ),
                use_container_width=True,
            )

    # ────────────────────────────────────────────────────────────────────
    # Tab 4 — Candidate Detail
    # ────────────────────────────────────────────────────────────────────
    with tab_detail:
        selected = st.selectbox("Select candidate", ranked["Candidate"].tolist())
        idx = names.index(selected)
        row = ranked[ranked["Candidate"] == selected].iloc[0]

        radar = plot_radar(
            selected,
            float(row["BERT Score"]),
            float(row["TF-IDF Score"]),
            float(row["Skill Match %"]),
            float(row["Composite Score"]),
        )
        c1, c2 = st.columns([1, 1])
        with c1:
            st.plotly_chart(radar, use_container_width=True)
        with c2:
            st.metric("Composite Score", f"{row['Composite Score']:.1f}%")
            st.metric("BERT Score",      f"{row['BERT Score']:.1f}%")
            st.metric("TF-IDF Score",    f"{row['TF-IDF Score']:.1f}%")
            st.metric("Skill Match",     f"{row['Skill Match %']:.1f}%")
            st.metric("Overall Rank",    f"#{int(row['Rank'])} of {len(ranked)}")

        with st.expander("📄 Extracted Resume Text"):
            st.text(raw_list[idx][:3000] + ("..." if len(raw_list[idx]) > 3000 else ""))

    # ────────────────────────────────────────────────────────────────────
    # Tab 5 — Evaluation
    # ────────────────────────────────────────────────────────────────────
    with tab_eval:
        st.info(
            "Provide ground-truth relevant candidates to compute Precision@K, MRR, and NDCG@K."
        )
        relevant_input = st.multiselect(
            "Select ground-truth relevant candidates",
            options=names,
        )
        k_val = st.slider("K", 1, min(10, len(names)), min(5, len(names)))

        if relevant_input:
            from src.similarity import precision_at_k, mean_reciprocal_rank, ndcg_at_k

            p_at_k  = precision_at_k(ranked, relevant_input, k=k_val)
            mrr     = mean_reciprocal_rank(ranked, relevant_input)
            ndcg    = ndcg_at_k(ranked, relevant_input, k=k_val)

            m1, m2, m3 = st.columns(3)
            m1.metric(f"Precision@{k_val}", f"{p_at_k:.3f}")
            m2.metric("MRR",                f"{mrr:.3f}")
            m3.metric(f"NDCG@{k_val}",      f"{ndcg:.4f}")

            st.caption(
                "**Precision@K**: fraction of top-K results that are relevant. "
                "**MRR**: reciprocal rank of first relevant result. "
                "**NDCG@K**: ranking quality accounting for position."
            )
        else:
            st.caption("Select at least one ground-truth candidate above.")
