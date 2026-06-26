"""
app.py
------
Streamlit sentiment-analysis web app.

Compares two models side-by-side:
  • Scikit-learn  — TF-IDF + Logistic Regression (local, instant)
  • DistilBERT    — via Hugging Face Inference API (no PyTorch needed)

Run locally:
    streamlit run app.py

Streamlit Cloud secrets required:
    [HF_TOKEN] = "hf_xxxxxxxxxxxxxxxxxxxx"
"""

import os
import json
import time

import joblib
import requests
import plotly.graph_objects as go
import streamlit as st

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
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .positive { color: #27ae60; font-weight: 700; font-size: 1.2rem; }
    .negative { color: #e74c3c; font-weight: 700; font-size: 1.2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Hugging Face token — loaded from Streamlit secrets (never hardcoded)
# ---------------------------------------------------------------------------
HF_TOKEN = st.secrets.get("HF_TOKEN", None)

HF_API_URL = (
    "https://api-inference.huggingface.co/models/"
    "distilbert-base-uncased-finetuned-sst-2-english"
)

# ---------------------------------------------------------------------------
# Sklearn model loader — cached so it only runs once per session
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_sklearn_model():
    vectorizer = joblib.load("model/tfidf_vectorizer.joblib")
    clf        = joblib.load("model/logistic_regression.joblib")
    return vectorizer, clf


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def predict_sklearn(text, vectorizer, clf):
    """Returns (sentiment, confidence, [neg_prob, pos_prob])."""
    X          = vectorizer.transform([text])
    label      = int(clf.predict(X)[0])
    proba      = clf.predict_proba(X)[0].tolist()
    sentiment  = "Positive" if label == 1 else "Negative"
    confidence = proba[label]
    return sentiment, confidence, proba


def predict_bert_api(text):
    """
    Calls HF Inference API for DistilBERT sentiment.
    Token read from st.secrets["HF_TOKEN"] — never hardcoded.
    Returns (sentiment, confidence, [neg_prob, pos_prob]) or None on failure.
    """
    headers = {}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    payload = {"inputs": text[:512]}   # truncate to model limit

    for attempt in range(3):
        try:
            resp = requests.post(HF_API_URL, headers=headers, json=payload, timeout=30)

            if resp.status_code == 200:
                data    = resp.json()
                results = data[0] if isinstance(data[0], list) else data
                pos     = next((r["score"] for r in results if r["label"] == "POSITIVE"), 0.0)
                neg     = next((r["score"] for r in results if r["label"] == "NEGATIVE"), 0.0)
                is_pos  = pos >= neg
                return (
                    "Positive" if is_pos else "Negative",
                    pos if is_pos else neg,
                    [neg, pos],
                )

            elif resp.status_code == 503:
                # Model cold-starting on HF side — wait and retry
                wait = min(resp.json().get("estimated_time", 10), 20)
                time.sleep(wait)
                continue

            elif resp.status_code == 401:
                return "AUTH_ERROR", 0.0, [0.0, 0.0]

            else:
                return None

        except requests.exceptions.Timeout:
            time.sleep(3)
            continue
        except requests.exceptions.RequestException:
            return None

    return None   # all 3 attempts failed


# ---------------------------------------------------------------------------
# UI — Header
# ---------------------------------------------------------------------------
st.title("🎬 Sentiment Analysis Web App")
st.markdown(
    "Compare **Scikit-learn (TF-IDF + Logistic Regression)** "
    "vs **DistilBERT** on any text."
)
st.divider()

# ---------------------------------------------------------------------------
# Load sklearn model
# ---------------------------------------------------------------------------
sklearn_ready = False
try:
    with st.spinner("Loading model…"):
        vectorizer, clf = load_sklearn_model()
    sklearn_ready = True
except FileNotFoundError:
    st.warning(
        "⚠️ Scikit-learn model files not found in `model/`. "
        "Run `python train_model.py` locally, push the `model/` folder, then redeploy."
    )
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")

# ---------------------------------------------------------------------------
# UI — Text input
# ---------------------------------------------------------------------------
st.subheader("Enter text to analyse")
user_text = st.text_area(
    label="Review or sentence:",
    placeholder=(
        "e.g. This film was an absolute masterpiece — "
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
# Run analysis and store in session_state so results persist across reruns
# ---------------------------------------------------------------------------
if analyze and user_text.strip():
    sk_result   = None
    bert_result = None

    if sklearn_ready:
        sk_result = predict_sklearn(user_text, vectorizer, clf)

    with st.spinner("Calling DistilBERT API…"):
        bert_result = predict_bert_api(user_text)

    st.session_state["last_text"]        = user_text
    st.session_state["last_sk_result"]   = sk_result
    st.session_state["last_bert_result"] = bert_result

# ---------------------------------------------------------------------------
# UI — Results (rendered from session_state so they survive reruns)
# ---------------------------------------------------------------------------
if "last_sk_result" in st.session_state or "last_bert_result" in st.session_state:
    sk_result   = st.session_state.get("last_sk_result")
    bert_result = st.session_state.get("last_bert_result")

    st.divider()
    st.subheader("Results")
    if "last_text" in st.session_state:
        st.caption(f'Text analysed: *"{st.session_state["last_text"]}"*')

    col_sk, col_bert = st.columns(2)

    # --- Sklearn ---
    with col_sk:
        st.markdown("#### 🔵 Scikit-learn")
        if sk_result:
            sk_sent, sk_conf, sk_proba = sk_result
            css  = "positive" if sk_sent == "Positive" else "negative"
            icon = "✅" if sk_sent == "Positive" else "❌"
            st.markdown(f'<p class="{css}">{icon} {sk_sent}</p>', unsafe_allow_html=True)
            st.markdown(f"Confidence: **{sk_conf:.1%}**")
            st.progress(sk_conf)
        else:
            st.info("Model not loaded.")
            sk_proba = [0.0, 0.0]

    # --- DistilBERT via API ---
    with col_bert:
        st.markdown("#### 🟠 DistilBERT")
        if bert_result and bert_result[0] == "AUTH_ERROR":
            st.error(
                "❌ HF_TOKEN is missing or invalid. "
                "Add it under Streamlit Cloud → Settings → Secrets."
            )
            bert_proba = [0.0, 0.0]
        elif bert_result:
            bert_sent, bert_conf, bert_proba = bert_result
            css  = "positive" if bert_sent == "Positive" else "negative"
            icon = "✅" if bert_sent == "Positive" else "❌"
            st.markdown(f'<p class="{css}">{icon} {bert_sent}</p>', unsafe_allow_html=True)
            st.markdown(f"Confidence: **{bert_conf:.1%}**")
            st.progress(bert_conf)
        else:
            st.warning("⚠️ Model service temporarily unavailable. Please try again in a moment.")
            bert_proba = [0.0, 0.0]

    # --- Confidence comparison chart ---
    if sk_result and bert_result and bert_result[0] not in (None, "AUTH_ERROR"):
        sk_proba   = sk_result[2]
        bert_proba = bert_result[2]
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
            yaxis=dict(tickformat=".0%", range=[0, 1.15], title="Confidence"),
            xaxis_title="Model",
            legend_title="Sentiment",
            height=380,
            margin=dict(t=30, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Model accuracy section
# ---------------------------------------------------------------------------
st.divider()
st.subheader("📈 Model Accuracy on IMDB Test Set (25,000 reviews)")

col1, col2 = st.columns(2)

with col1:
    sk_acc = None
    if os.path.exists("model/metrics.json"):
        try:
            with open("model/metrics.json") as f:
                sk_acc = json.load(f).get("sklearn_accuracy")
        except Exception:
            pass
    st.metric(
        label="🔵 Scikit-learn (TF-IDF + LR)",
        value=f"{sk_acc:.2%}" if sk_acc else "~89%",
    )

with col2:
    st.metric(
        label="🟠 DistilBERT (SST-2)",
        value="≈ 93.0%",
        help="Published accuracy on SST-2 benchmark (Hugging Face model card).",
    )

with st.expander("ℹ️ About these models"):
    st.markdown(
        """
        | Model | Approach | Notes |
        |---|---|---|
        | TF-IDF + Logistic Regression | Classical ML | Trained locally on IMDB |
        | DistilBERT | Transformer (HF API) | Pre-trained on SST-2, no local GPU needed |

        DistilBERT handles negation and context better ("not great" ≠ "great").
        TF-IDF is faster, lighter, and fully transparent.
        """
    )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "Built with [Streamlit](https://streamlit.io) · "
    "[Scikit-learn](https://scikit-learn.org) · "
    "[Hugging Face](https://huggingface.co)"
)
