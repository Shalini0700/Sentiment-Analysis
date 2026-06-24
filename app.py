"""
app.py
------
Streamlit sentiment-analysis web app.

Compares two models side-by-side:
  • Scikit-learn  — TF-IDF + Logistic Regression trained on IMDB (local)
  • DistilBERT    — distilbert-base-uncased-finetuned-sst-2-english (HuggingFace)

Run:
    streamlit run app.py
"""

import os
import json

import joblib
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from transformers import pipeline

# ---------------------------------------------------------------------------
# Page config — must be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Custom CSS — light card styling and confidence bar colour
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .model-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.5rem;
    }
    .positive { color: #27ae60; font-weight: 700; font-size: 1.2rem; }
    .negative { color: #e74c3c; font-weight: 700; font-size: 1.2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Model loaders (cached so they only run once per session)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_sklearn_model():
    """Load the TF-IDF vectorizer and Logistic Regression saved by train_model.py."""
    vectorizer = joblib.load("model/tfidf_vectorizer.joblib")
    clf        = joblib.load("model/logistic_regression.joblib")
    return vectorizer, clf


@st.cache_resource(show_spinner=False)
def load_bert_pipeline():
    """Download (or use cached) DistilBERT fine-tuned on SST-2."""
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        truncation=True,   # auto-truncate inputs longer than 512 tokens
        max_length=512,
    )


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def predict_sklearn(text: str, vectorizer, clf) -> tuple[str, float, list[float]]:
    """
    Returns (sentiment_label, confidence, [neg_prob, pos_prob]).
    Label: 0 = Negative, 1 = Positive.
    """
    X         = vectorizer.transform([text])
    label     = int(clf.predict(X)[0])
    proba     = clf.predict_proba(X)[0].tolist()   # [P(neg), P(pos)]
    sentiment = "Positive" if label == 1 else "Negative"
    confidence = proba[label]
    return sentiment, confidence, proba


def predict_bert(text: str, bert_pipe) -> tuple[str, float, list[float]]:
    """
    Returns (sentiment_label, confidence, [neg_prob, pos_prob]).
    Pipeline returns the winning-class score only, so we infer the other.
    """
    result     = bert_pipe(text)[0]
    is_pos     = result["label"] == "POSITIVE"
    confidence = result["score"]
    sentiment  = "Positive" if is_pos else "Negative"
    proba      = [1 - confidence, confidence] if is_pos else [confidence, 1 - confidence]
    return sentiment, confidence, proba


# ---------------------------------------------------------------------------
# UI — Header
# ---------------------------------------------------------------------------
st.title("🎬 Sentiment Analysis Web App")
st.markdown(
    "Compare **Scikit-learn (TF-IDF + Logistic Regression)** "
    "vs **DistilBERT** on movie reviews or any text you like."
)
st.divider()

# ---------------------------------------------------------------------------
# Load models (show a spinner while they initialise)
# ---------------------------------------------------------------------------
sklearn_ready = False
with st.spinner("Loading models — first run downloads DistilBERT (~250 MB)…"):
    try:
        vectorizer, clf = load_sklearn_model()
        sklearn_ready = True
    except FileNotFoundError:
        st.warning(
            "Scikit-learn model not found. "
            "Run `python train_model.py` first, then reload the app."
        )

    bert_pipe = load_bert_pipeline()

# ---------------------------------------------------------------------------
# UI — Text input
# ---------------------------------------------------------------------------
st.subheader("Enter text to analyse")
user_text = st.text_area(
    label="Review or sentence:",
    placeholder=(
        "e.g.  This film was an absolute masterpiece — "
        "the direction, acting and soundtrack were all exceptional!"
    ),
    height=140,
    label_visibility="collapsed",
)

analyze = st.button(
    "Analyse Sentiment ✨",
    type="primary",
    disabled=not user_text.strip(),
    use_container_width=True,
)

# ---------------------------------------------------------------------------
# UI — Results
# ---------------------------------------------------------------------------
if analyze and user_text.strip():

    # Run both models
    bert_sentiment, bert_conf, bert_proba = predict_bert(user_text, bert_pipe)

    if sklearn_ready:
        sk_sentiment, sk_conf, sk_proba = predict_sklearn(user_text, vectorizer, clf)
    else:
        sk_sentiment, sk_conf, sk_proba = "N/A", 0.0, [0.0, 0.0]

    st.divider()
    st.subheader("Results")

    # --- Side-by-side prediction cards ---
    col_sk, col_bert = st.columns(2)

    with col_sk:
        st.markdown("#### 🔵 Scikit-learn")
        if sklearn_ready:
            label_class = "positive" if sk_sentiment == "Positive" else "negative"
            icon        = "✅" if sk_sentiment == "Positive" else "❌"
            st.markdown(
                f'<p class="{label_class}">{icon} {sk_sentiment}</p>',
                unsafe_allow_html=True,
            )
            st.markdown(f"Confidence: **{sk_conf:.1%}**")
            st.progress(sk_conf)
        else:
            st.info("Model not loaded.")

    with col_bert:
        st.markdown("#### 🟠 DistilBERT")
        label_class = "positive" if bert_sentiment == "Positive" else "negative"
        icon        = "✅" if bert_sentiment == "Positive" else "❌"
        st.markdown(
            f'<p class="{label_class}">{icon} {bert_sentiment}</p>',
            unsafe_allow_html=True,
        )
        st.markdown(f"Confidence: **{bert_conf:.1%}**")
        st.progress(bert_conf)

    # --- Grouped bar chart ---
    st.divider()
    st.subheader("📊 Confidence Comparison")

    fig = go.Figure(data=[
        go.Bar(
            name="Negative",
            x=["Scikit-learn", "DistilBERT"],
            y=[sk_proba[0], bert_proba[0]],
            marker_color="#e74c3c",
            text=[f"{sk_proba[0]:.1%}", f"{bert_proba[0]:.1%}"],
            textposition="outside",
        ),
        go.Bar(
            name="Positive",
            x=["Scikit-learn", "DistilBERT"],
            y=[sk_proba[1], bert_proba[1]],
            marker_color="#27ae60",
            text=[f"{sk_proba[1]:.1%}", f"{bert_proba[1]:.1%}"],
            textposition="outside",
        ),
    ])
    fig.update_layout(
        barmode="group",
        yaxis=dict(
            tickformat=".0%",
            range=[0, 1.15],
            title="Confidence",
        ),
        xaxis_title="Model",
        legend_title="Sentiment",
        height=380,
        margin=dict(t=30, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# UI — Model accuracy section
# ---------------------------------------------------------------------------
st.divider()
st.subheader("📈 Model Accuracy on IMDB Test Set (25 000 reviews)")

metrics_path = "model/metrics.json"
if os.path.exists(metrics_path):
    with open(metrics_path) as f:
        metrics = json.load(f)
    sk_acc = metrics.get("sklearn_accuracy", None)
else:
    sk_acc = None

col1, col2 = st.columns(2)

with col1:
    if sk_acc is not None:
        st.metric(
            label="🔵 Scikit-learn (TF-IDF + LR)",
            value=f"{sk_acc:.2%}",
        )
    else:
        st.metric(
            label="🔵 Scikit-learn (TF-IDF + LR)",
            value="—",
            help="Run `python train_model.py` to compute accuracy.",
        )

with col2:
    st.metric(
        label="🟠 DistilBERT (pre-trained SST-2)",
        value="≈ 93.0 %",
        help="Published accuracy on the SST-2 benchmark (Hugging Face model card).",
    )

with st.expander("ℹ️ About these numbers"):
    st.markdown(
        """
        | Model | Dataset | Notes |
        |---|---|---|
        | TF-IDF + Logistic Regression | IMDB (train/test split) | Trained locally via `train_model.py` |
        | DistilBERT | SST-2 (Stanford Sentiment Treebank) | Pre-trained — no local training needed |

        Both models perform well on short, clear reviews.
        DistilBERT generally handles nuanced or negated language better because
        it understands word context, while TF-IDF treats each word independently.
        """
    )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "Built with [Streamlit](https://streamlit.io) · "
    "[Scikit-learn](https://scikit-learn.org) · "
    "[Hugging Face Transformers](https://huggingface.co/docs/transformers)"
)
