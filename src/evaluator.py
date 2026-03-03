"""
evaluator.py — Ranking quality metrics and similarity distribution utilities
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_similarity_distribution(
    scores: list[float],
    title: str = "Similarity Score Distribution",
) -> go.Figure:
    """Histogram of cosine similarity scores."""
    fig = px.histogram(
        x=scores,
        nbins=20,
        labels={"x": "Similarity Score (%)"},
        title=title,
        color_discrete_sequence=["#6366f1"],
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        bargap=0.05,
    )
    return fig


def plot_score_breakdown(ranked_df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Grouped bar chart: BERT / TF-IDF / Skill scores for top-N candidates."""
    df = ranked_df.head(top_n)
    fig = go.Figure()
    for col, color in [
        ("BERT Score",    "#6366f1"),
        ("TF-IDF Score",  "#22d3ee"),
        ("Skill Match %", "#f59e0b"),
    ]:
        fig.add_trace(
            go.Bar(name=col, x=df["Candidate"], y=df[col], marker_color=color)
        )
    fig.update_layout(
        barmode="group",
        title=f"Score Breakdown — Top {top_n} Candidates",
        xaxis_title="Candidate",
        yaxis_title="Score (%)",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_radar(
    candidate_name: str,
    bert: float,
    tfidf: float,
    skill: float,
    composite: float,
) -> go.Figure:
    """Radar chart for a single candidate's score profile."""
    categories = ["BERT", "TF-IDF", "Skill Match", "Composite", "BERT"]
    values     = [bert, tfidf, skill, composite, bert]
    fig = go.Figure(
        go.Scatterpolar(
            r=values,
            theta=categories,
            fill="toself",
            line_color="#6366f1",
            fillcolor="rgba(99,102,241,0.25)",
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title=f"Score Radar — {candidate_name}",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_skill_heatmap(
    candidates: list[str],
    skill_sets: list[list[str]],
    all_skills: list[str],
) -> go.Figure:
    """Binary skill heatmap: candidates × skills."""
    sk_set = sorted(set(s for sk in skill_sets for s in sk))[:50]   # cap at 50
    matrix = [
        [1 if s in set(sk) else 0 for s in sk_set]
        for sk in skill_sets
    ]
    fig = px.imshow(
        matrix,
        x=sk_set,
        y=candidates,
        color_continuous_scale=["#1e1b4b", "#6366f1"],
        aspect="auto",
        title="Candidate Skill Heatmap",
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(tickangle=-45),
    )
    return fig


def plot_composite_bar(ranked_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of composite scores for all candidates."""
    df = ranked_df.sort_values("Composite Score")
    fig = px.bar(
        df,
        x="Composite Score",
        y="Candidate",
        orientation="h",
        color="Composite Score",
        color_continuous_scale=["#3730a3", "#6366f1", "#a5b4fc"],
        text="Composite Score",
        title="Overall Candidate Ranking",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,
        xaxis_range=[0, 110],
    )
    return fig
