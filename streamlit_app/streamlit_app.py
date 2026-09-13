
import os
import re
import html
import unicodedata
import joblib
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import shap
import textstat

from textblob import TextBlob
from scipy import sparse


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TFIDF_PATH = os.path.join(
    BASE_DIR,
    "features",
    "tfidf",
    "tfidf_vectorizer.joblib"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "features",
    "combined",
    "auxiliary_scaler.joblib"
)

SELECTOR_PATH = os.path.join(
    BASE_DIR,
    "features",
    "combined",
    "tfidf_selector.joblib"
)

W2V_PATH = os.path.join(
    BASE_DIR,
    "features",
    "word2vec",
    "word2vec_model.model"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "results",
    "day_06",
    "day_06_xgboost_best_tuned.joblib"
)


# =========================================================
# LOAD MODELS
# =========================================================

@st.cache_resource
def load_models():

    tfidf = joblib.load(TFIDF_PATH)
    scaler = joblib.load(SCALER_PATH)
    selector = joblib.load(SELECTOR_PATH)
    w2v = __import__("gensim").models.Word2Vec.load(W2V_PATH)
    model = joblib.load(MODEL_PATH)

    explainer = shap.TreeExplainer(model)

    return tfidf, scaler, selector, w2v, model, explainer


tfidf, scaler, selector, w2v, model, explainer = load_models()


# =========================================================
# FEATURE NAMES
# =========================================================

tfidf_feature_names = tfidf.get_feature_names_out()

selected_mask = selector.get_support()

selected_tfidf_names = tfidf_feature_names[selected_mask]

feature_names = list(selected_tfidf_names)

feature_names += [
    f"W2V_{i:03d}"
    for i in range(100)
]

feature_names += [
    "Flesch Reading Ease",
    "Flesch-Kincaid Grade",
    "Gunning Fog",
    "ARI",
    "Sentiment Polarity",
    "Sentiment Subjectivity",
    "Word Count",
    "Character Count",
    "Sentence Count",
    "Avg Word Length",
    "Avg Sentence Length",
    "Unique Word Ratio",
    "Digit Count",
    "Uppercase Count",
    "Punctuation Count"
]


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text):

    text = html.unescape(str(text))
    text = unicodedata.normalize("NFKC", text)

    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        " EMAIL ",
        text
    )

    text = text.lower()

    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", text)

    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    text = text.replace("‘", "'")
    text = text.replace("’", "'")

    text = re.sub(
        r"[^a-z0-9\s.,!?;:'\"%()-]",
        " ",
        text
    )

    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        text = "empty"

    return text


# =========================================================
# WORD2VEC
# =========================================================

def get_word2vec_vector(text):

    tokens = text.split()

    vectors = []

    for token in tokens:

        if token in w2v.wv:
            vectors.append(w2v.wv[token])

    if vectors:
        return np.mean(vectors, axis=0)

    return np.zeros(100)


# =========================================================
# AUXILIARY FEATURES
# =========================================================

def get_auxiliary_features(original_text, cleaned_text):

    words = cleaned_text.split()

    word_count = len(words)
    char_count = len(cleaned_text)

    sentences = re.split(r"[.!?]+", cleaned_text)
    sentences = [s for s in sentences if s.strip()]
    sentence_count = max(len(sentences), 1)

    avg_word_length = (
        np.mean([len(w) for w in words])
        if words else 0
    )

    avg_sentence_length = (
        word_count / sentence_count
    )

    unique_word_ratio = (
        len(set(words)) / word_count
        if word_count > 0 else 0
    )

    digit_count = sum(c.isdigit() for c in original_text)

    # IMPORTANT:
    # uppercase count is calculated from ORIGINAL text
    uppercase_count = sum(c.isupper() for c in original_text)

    punctuation_count = sum(
        c in ".,!?;:'\"%-()"
        for c in original_text
    )

    readability = [
        textstat.flesch_reading_ease(cleaned_text),
        textstat.flesch_kincaid_grade(cleaned_text),
        textstat.gunning_fog(cleaned_text),
        textstat.automated_readability_index(cleaned_text)
    ]

    blob = TextBlob(cleaned_text)

    sentiment = [
        blob.sentiment.polarity,
        blob.sentiment.subjectivity
    ]

    w2v_vector = get_word2vec_vector(cleaned_text)

    linguistic = [
        word_count,
        char_count,
        sentence_count,
        avg_word_length,
        avg_sentence_length,
        unique_word_ratio,
        digit_count,
        uppercase_count,
        punctuation_count
    ]

    auxiliary = np.concatenate([
        w2v_vector,
        np.array(readability),
        np.array(sentiment),
        np.array(linguistic)
    ])

    return auxiliary.astype(np.float32)


# =========================================================
# BUILD FINAL FEATURE VECTOR
# =========================================================

def build_feature_vector(text):

    cleaned = clean_text(text)

    tfidf_vector = tfidf.transform([cleaned])

    selected_tfidf = selector.transform(tfidf_vector)

    auxiliary = get_auxiliary_features(
        text,
        cleaned
    ).reshape(1, -1)

    scaled_auxiliary = scaler.transform(auxiliary)

    final_features = sparse.hstack([
        selected_tfidf,
        sparse.csr_matrix(scaled_auxiliary)
    ])

    return final_features


# =========================================================
# PREDICTION
# =========================================================

def predict_news(text):

    X = build_feature_vector(text)

    probabilities = model.predict_proba(X)[0]

    fake_probability = float(probabilities[0])
    real_probability = float(probabilities[1])

    prediction = (
        "Real"
        if real_probability >= fake_probability
        else "Fake"
    )

    credibility_score = real_probability * 100

    X_dense = X.toarray()

    shap_values = explainer.shap_values(X_dense)

    if isinstance(shap_values, list):
        shap_row = shap_values[1][0]
    else:
        shap_row = shap_values[0]

    top_indices = np.argsort(
        np.abs(shap_row)
    )[::-1][:10]

    top_shap_features = []

    for idx in top_indices:

        top_shap_features.append({
            "feature": feature_names[idx],
            "shap_value": round(
                float(shap_row[idx]),
                6
            )
        })

    return {
        "prediction": prediction,
        "fake_probability": round(
            fake_probability,
            6
        ),
        "real_probability": round(
            real_probability,
            6
        ),
        "credibility_score": round(
            credibility_score,
            2
        ),
        "top_shap_features": top_shap_features
    }


# =========================================================
# STREAMLIT UI
# =========================================================

st.set_page_config(
    page_title="NewsGuard",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ NewsGuard")
st.subheader(
    "Fake News Detection & Credibility Scoring System"
)

st.write(
    "Enter a news article or news headline to analyze its credibility."
)

news_text = st.text_area(
    "News Article",
    height=250,
    placeholder="Paste news article text here..."
)


if st.button(
    "🔍 Analyze News",
    type="primary"
):

    if not news_text.strip():

        st.warning("Please enter some news text.")

    else:

        with st.spinner("Analyzing news..."):

            result = predict_news(news_text)

        prediction = result["prediction"]
        credibility = result["credibility_score"]

        col1, col2 = st.columns(2)

        with col1:

            if prediction == "Real":
                st.success(
                    f"Prediction: {prediction}"
                )
            else:
                st.error(
                    f"Prediction: {prediction}"
                )

            st.metric(
                "Fake Probability",
                f"{result['fake_probability'] * 100:.2f}%"
            )

            st.metric(
                "Real Probability",
                f"{result['real_probability'] * 100:.2f}%"
            )

        with col2:

            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=credibility,
                    title={
                        "text": "Credibility Score"
                    },
                    gauge={
                        "axis": {
                            "range": [0, 100]
                        },
                        "bar": {
                            "color": "#2563eb"
                        },
                        "steps": [
                            {
                                "range": [0, 40],
                                "color": "#ef4444"
                            },
                            {
                                "range": [40, 70],
                                "color": "#facc15"
                            },
                            {
                                "range": [70, 100],
                                "color": "#22c55e"
                            }
                        ]
                    }
                )
            )

            fig.update_layout(
                height=300,
                margin=dict(
                    l=20,
                    r=20,
                    t=50,
                    b=20
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.divider()

        st.subheader(
            "🔎 Top SHAP Feature Contributions"
        )

        for item in result["top_shap_features"]:

            value = item["shap_value"]

            if value >= 0:
                direction = "⬆️"
            else:
                direction = "⬇️"

            st.write(
                f"{direction} **{item['feature']}** — "
                f"`{value:+.6f}`"
            )

        st.divider()

        st.caption(
            "NewsGuard uses TF-IDF, Word2Vec, readability, sentiment, "
            "and linguistic features with an XGBoost classifier."
        )
