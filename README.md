# 🛡️ NewsGuard — Fake News Detection & Credibility Scoring System

> **An end-to-end NLP and Machine Learning system for detecting fake news, estimating credibility, and explaining model predictions using SHAP.**

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange?logo=xgboost)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Deployment-Streamlit-red?logo=streamlit)](https://streamlit.io/)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-purple)](https://shap.readthedocs.io/)
[![Git LFS](https://img.shields.io/badge/Large%20Files-Git%20LFS-green)](https://git-lfs.com/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

## 🚀 Live Demo

### 🌐 Streamlit App

**[Open NewsGuard →](https://newsguardfakenewsdetection-ysisqwbjklyyl7jpylzx7v.streamlit.app/)**

Paste a news article or headline into the application and NewsGuard returns:

* 🟢 **Real / Fake prediction**
* 📊 **Fake probability**
* 📈 **Real probability**
* 🎯 **Credibility score (0–100)**
* 🔎 **Top SHAP feature contributions**

---

## 📌 Project Overview

The rapid growth of online news and social media has made misinformation increasingly difficult to identify manually.

**NewsGuard** is an end-to-end Fake News Detection system designed to analyze the linguistic and semantic characteristics of a news article and classify it as **Fake** or **Real**.

The system combines:

* TF-IDF features
* Word2Vec embeddings
* Readability metrics
* Sentiment analysis
* Linguistic/statistical features
* XGBoost classification
* SHAP explainability
* Flask REST API
* Streamlit interactive interface

The final system provides not only a classification but also an interpretable **credibility score** and feature-level explanation.

---

# 🧠 System Architecture

```text
                    ┌─────────────────────┐
                    │    News Article     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Text Preprocessing  │
                    │ Cleaning + NLP      │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
       ┌─────────────┐ ┌─────────────┐ ┌────────────────┐
       │   TF-IDF    │ │  Word2Vec   │ │ Auxiliary NLP  │
       │ 500 features│ │ 100 features│ │    15 features │
       └──────┬──────┘ └──────┬──────┘ └───────┬────────┘
              │                │                 │
              └────────────────┼─────────────────┘
                               ▼
                    ┌─────────────────────┐
                    │  Feature Pipeline   │
                    │     615 Features    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   XGBoost Model     │
                    └──────────┬──────────┘
                               │
                 ┌─────────────┴─────────────┐
                 ▼                           ▼
        ┌─────────────────┐         ┌─────────────────┐
        │ Fake / Real     │         │ SHAP Explainability│
        │ Prediction      │         │ Top Contributions │
        └────────┬────────┘         └────────┬────────┘
                 │                           │
                 └─────────────┬─────────────┘
                               ▼
                    ┌─────────────────────┐
                    │ Credibility Score   │
                    │       0–100         │
                    └─────────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Streamlit UI      │
                    └─────────────────────┘
```

---

# ✨ Key Features

## 📰 Fake News Classification

Classifies an article into:

```text
Fake
Real
```

using a tuned XGBoost classifier.

## 🎯 Credibility Score

The system converts the predicted probability of the **Real** class into a 0–100 score:

```text
Credibility Score = P(Real) × 100
```

For example:

```text
Real Probability = 0.9857

Credibility Score = 98.57 / 100
```

> The credibility score represents the model's confidence in the **Real class**, not a factual guarantee that the article is true.

## 🔎 SHAP Explainability

NewsGuard uses **SHAP TreeExplainer** to identify the features contributing most strongly to each prediction.

Example:

```text
⬇️ read more          -1.581756
⬇️ featured image     -0.973254
⬇️ century wire       -0.968295
⬆️ washington reuters +0.850923
⬆️ nov                +0.549894
```

This makes the prediction more interpretable instead of treating the model as a black box.

## 📊 Interactive Credibility Gauge

The Streamlit application provides a visual credibility gauge from:

```text
0 ───────────────────────────── 100
Fake                          Real
```

along with the predicted probabilities.

---

# 🧹 NLP Pipeline

NewsGuard performs multiple stages of text processing.

### Text Cleaning

The preprocessing pipeline handles:

* HTML entities
* HTML tags
* URLs
* Email addresses
* Unicode normalization
* Control characters
* Quotes and dashes
* Case normalization
* Punctuation normalization
* Whitespace normalization

### Feature Engineering

The final model uses **615 features**:

| Feature Type    | Features |
| --------------- | -------: |
| Selected TF-IDF |      500 |
| Word2Vec        |      100 |
| Readability     |        4 |
| Sentiment       |        2 |
| Linguistic      |        9 |
| **Total**       |  **615** |

---

# 🔤 TF-IDF

TF-IDF is used to capture important words and phrases within the news articles.

Configuration:

```text
max_features = 5000
ngram_range = (1, 2)
min_df = 2
max_df = 0.95
sublinear_tf = True
```

The pipeline initially generates 5,000 TF-IDF features.

For the final XGBoost model, **SelectKBest with chi-square scoring** selects the top 500 TF-IDF features.

---

# 🧬 Word2Vec

Word2Vec is used to capture semantic information that cannot be represented effectively through simple word frequency.

Configuration:

```text
vector_size = 100
window = 5
min_count = 2
sg = 1
epochs = 2
seed = 42
```

A 100-dimensional sentence representation is generated using the mean of known word vectors.

---

# 📖 Readability Features

NewsGuard extracts four readability measurements:

* Flesch Reading Ease
* Flesch-Kincaid Grade
* Gunning Fog Index
* Automated Readability Index (ARI)

These features help capture differences in writing complexity.

---

# ❤️ Sentiment Features

Two TextBlob-based features are extracted:

* Polarity
* Subjectivity

---

# 🔤 Linguistic Features

The system additionally extracts:

* Word Count
* Character Count
* Sentence Count
* Average Word Length
* Average Sentence Length
* Unique Word Ratio
* Digit Count
* Uppercase Count
* Punctuation Count

---

# 🤖 Model Comparison

Multiple machine learning algorithms were evaluated.

| Model               |     F1 Score |
| ------------------- | -----------: |
| Logistic Regression |     0.993899 |
| Naive Bayes         |     0.957115 |
| SVM                 |     0.996700 |
| Random Forest       |     0.991537 |
| Gradient Boosting   |     0.996174 |
| **XGBoost**         | **0.997701** |

XGBoost achieved the strongest cross-validation performance and was selected as the final model.

---

# 🏆 Final Model Performance

## Internal Test Set

The final tuned XGBoost model achieved:

| Metric    |      Score |
| --------- | ---------: |
| Accuracy  | **99.72%** |
| Precision | **99.58%** |
| Recall    | **99.91%** |
| F1 Score  | **99.74%** |
| ROC-AUC   | **99.98%** |

Confusion Matrix:

```text
                Predicted
              Fake    Real
Actual Fake   1782      9
       Real      2   2118
```

The project target of **F1 > 0.88** was achieved on the internal held-out test set.

---

# 🌍 External Validation

To evaluate generalization beyond the original dataset, NewsGuard was additionally tested on a balanced sample from the **WELFake** dataset.

### WELFake External Results

| Metric    |      Score |
| --------- | ---------: |
| Accuracy  | **79.15%** |
| Precision | **98.34%** |
| Recall    | **59.30%** |
| F1 Score  | **73.99%** |
| ROC-AUC   | **92.83%** |

This result demonstrates an important limitation:

> High performance on the internal dataset does not automatically guarantee equivalent performance on completely different news sources and distributions.

NewsGuard therefore **should not be treated as a universal truth detector**.

---

# ⚠️ Dataset Bias & Generalization

During development, source/template-related features were found to have a noticeable influence on predictions.

Examples include tokens such as:

```text
Reuters
Washington Reuters
Getty
Featured Image
Century Wire
```

Additional experiments showed that removing suspicious source-related features changed predictions substantially.

This suggests that the model can learn **dataset-specific patterns and source characteristics**, rather than purely learning whether a claim is factually true.

Therefore, NewsGuard should be viewed as:

> **A machine-learning based credibility classification system, not a replacement for professional fact-checking.**

---

# 🔬 Explainability with SHAP

NewsGuard uses:

```text
SHAP TreeExplainer
```

on the final XGBoost classifier.

For each prediction, the system identifies the most influential features.

Example output:

```text
Feature                 SHAP Value
-----------------------------------
read more               -1.581756
featured image          -0.973254
century wire            -0.968295
washington reuters      +0.850923
W2V_058                  -0.603700
nov                      +0.549894
getty                   -0.527997
W2V_001                 -0.449337
W2V_057                 +0.433148
W2V_009                 +0.386796
```

### SHAP Interpretation

```text
Positive SHAP → pushes prediction toward Real
Negative SHAP → pushes prediction toward Fake
```

The magnitude represents the strength of the contribution.

---

# 🌐 REST API

NewsGuard also includes a Flask REST API implementation.

### Endpoint

```http
POST /predict
```

### Request

```json
{
  "text": "Your news article goes here..."
}
```

### Response

```json
{
  "status": "success",
  "result": {
    "prediction": "Real",
    "fake_probability": 0.013039,
    "real_probability": 0.986961,
    "credibility_score": 98.7,
    "top_shap_features": []
  }
}
```

The API validates malformed and missing JSON requests and returns appropriate HTTP errors.

---

# 🖥️ Streamlit Application

The production-facing interface is built using **Streamlit**.

The application provides:

1. News article input
2. Fake/Real prediction
3. Fake probability
4. Real probability
5. 0–100 credibility gauge
6. Top SHAP feature contributions
7. Human-readable explanation of model output

### Live Application

**[🚀 Launch NewsGuard](https://newsguardfakenewsdetection-ysisqwbjklyyl7jpylzx7v.streamlit.app/)**

---

# 🗂️ Project Structure

```text
NewsGuard_Fake_News_Detection/
│
├── data/
│   └── processed/
│       ├── train_preprocessed.csv
│       ├── validation_preprocessed.csv
│       └── test_preprocessed.csv
│
├── features/
│   ├── tfidf/
│   │   ├── tfidf_vectorizer.joblib
│   │   ├── X_train_tfidf.npz
│   │   ├── X_validation_tfidf.npz
│   │   └── X_test_tfidf.npz
│   │
│   ├── word2vec/
│   │   ├── word2vec_model.model
│   │   ├── word2vec_model.model.wv.vectors.npy
│   │   └── word2vec_model.model.syn1neg.npy
│   │
│   └── combined/
│       ├── auxiliary_scaler.joblib
│       └── tfidf_selector.joblib
│
├── notebooks/
│   ├── NewsGuard_Day_01_Dataset_Setup_EDA.ipynb
│   ├── NewsGuard_Day_02_Text_Preprocessing.ipynb
│   ├── NewsGuard_Day_03_TFIDF_Feature_Engineering.ipynb
│   ├── NewsGuard_Day_04_Auxiliary_NLP_Features.ipynb
│   ├── NewsGuard_Day_05_Feature_Union_Baseline_Models.ipynb
│   ├── NewsGuard_Day_06_Advanced_Models_and_Hyperparameter_Tuning.ipynb
│   ├── NewsGuard_Day_07_Final_Model_Evaluation.ipynb
│   ├── NewsGuard_Day_08_SHAP_Explainability.ipynb
│   └── NewsGuard_Day_09_Flask_API_Gradio_UI.ipynb
│
├── results/
│   ├── day_05/
│   ├── day_06/
│   ├── day_07/
│   ├── day_08/
│   ├── day_09/
│   └── day_10/
│
├── streamlit_app/
│   └── streamlit_app.py
│
├── tests/
│   └── test_newsguard.py
│
├── requirements.txt
├── .python-version
├── .gitattributes
└── README.md
```

Large Word2Vec files are managed using **Git LFS**.

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/SagarSharma131006/NewsGuard_Fake_News_Detection.git
cd NewsGuard_Fake_News_Detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Streamlit Locally

```bash
streamlit run streamlit_app/streamlit_app.py
```

The application will open locally in your browser.

---

# 🧪 Testing

The project includes automated tests using `pytest`.

Run:

```bash
pytest -q
```

The test suite validates important components of the NewsGuard pipeline, including prediction and credibility scoring behaviour.

---

# 📈 MLflow

Model experiments and final metrics are tracked using **MLflow**.

Tracked information includes:

* Model parameters
* Validation metrics
* Test metrics
* ROC-AUC
* F1 score
* Final model information

MLflow artifacts are stored under:

```text
results/day_10/mlflow/
```

---

# 📋 Model Card

A detailed Model Card is included in:

```text
results/day_10/miruns/MODEL_CARD.md
```

It documents:

* Intended use
* Dataset information
* Model architecture
* Evaluation results
* Limitations
* Ethical considerations
* External validation

---

# 🧪 Reproducibility

The project follows a reproducible pipeline:

```text
Raw Dataset
     ↓
Data Cleaning
     ↓
Train / Validation / Test Split
     ↓
Text Preprocessing
     ↓
TF-IDF + Word2Vec + Auxiliary Features
     ↓
Feature Selection
     ↓
Model Training
     ↓
Hyperparameter Tuning
     ↓
Final Evaluation
     ↓
SHAP Explainability
     ↓
API / Streamlit Deployment
```

Important preprocessing and feature transformations are fitted using training data to reduce data leakage.

---

# ⚠️ Limitations

NewsGuard has several important limitations.

### 1. Dataset Bias

The model can learn patterns specific to the training dataset, including source and formatting characteristics.

### 2. Domain Shift

Performance can decrease when the model encounters news from a different distribution.

### 3. Not a Fact Checker

A high credibility score does **not** prove that a statement is factually correct.

### 4. Source Bias

Certain publisher/template-related terms can strongly influence predictions.

### 5. External Performance

Performance on WELFake was substantially lower than the internal test performance, highlighting the need for broader datasets and continued validation.

---

# 🔮 Future Improvements

Potential improvements include:

* 🌍 Training on multiple independent fake-news datasets
* 🧠 Transformer-based models such as BERT/RoBERTa
* 🔎 Claim-level fact verification
* 🌐 Trusted-source verification
* 📰 Publisher/source credibility analysis
* 🔗 External fact-checking APIs
* 📚 Larger and more diverse datasets
* 🧪 Cross-domain evaluation
* 🔐 Production API authentication
* 📊 Advanced monitoring and drift detection
* 🤖 LLM-assisted explanation
* ☁️ Scalable cloud deployment

---

# 🎯 Project Goals Achieved

| Goal                  | Status |
| --------------------- | ------ |
| Dataset preparation   | ✅      |
| NLP preprocessing     | ✅      |
| TF-IDF                | ✅      |
| Word2Vec              | ✅      |
| Readability analysis  | ✅      |
| Sentiment analysis    | ✅      |
| Linguistic features   | ✅      |
| Feature selection     | ✅      |
| 5+ ML models          | ✅      |
| Hyperparameter tuning | ✅      |
| XGBoost final model   | ✅      |
| SHAP explainability   | ✅      |
| Flask API             | ✅      |
| Interactive UI        | ✅      |
| MLflow tracking       | ✅      |
| Automated testing     | ✅      |
| Model Card            | ✅      |
| Git LFS               | ✅      |
| Public deployment     | ✅      |

---

# 👨‍💻 Author

**Sagar Sharma**

B.Tech CSE — Artificial Intelligence & Machine Learning
Panipat Institute of Engineering and Technology (PIET)

### Connect

* **GitHub:** [SagarSharma131006](https://github.com/SagarSharma131006)
* **Repository:** [NewsGuard_Fake_News_Detection](https://github.com/SagarSharma131006/NewsGuard_Fake_News_Detection)

---

# ⭐ If You Find This Project Useful

If you find NewsGuard interesting, consider giving the repository a ⭐ on GitHub.

---

## 📜 License

This project is licensed under the MIT License.
