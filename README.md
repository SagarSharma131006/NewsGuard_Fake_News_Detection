# 🛡️ NewsGuard — Fake News Detection & Credibility Scoring

## 📌 Project Status

**Day 2 / 10 — Dataset Setup, EDA & Text Preprocessing**

NewsGuard is an end-to-end NLP system for detecting fake news and generating a credibility score.

---

## 🎯 Objective

The project aims to build a complete fake news detection pipeline that will:

- Preprocess news text
- Extract TF-IDF features
- Generate Word2Vec embeddings
- Calculate readability features
- Calculate sentiment features
- Include article length features
- Compare multiple machine learning classifiers
- Evaluate models using stratified cross-validation
- Explain predictions using SHAP
- Generate a credibility score
- Deploy the best model through a Flask REST API
- Provide an interactive Gradio interface

---

## 📊 Dataset

### LIAR Dataset

The project uses the official LIAR dataset.

### Original Dataset

| Split | Samples |
|---|---:|
| Train | 10,240 |
| Validation | 1,284 |
| Test | 1,267 |
| **Total** | **12,791** |

### After Duplicate Removal

26 duplicate statements were removed.

| Class | Samples | Percentage |
|---|---:|---:|
| Fake | 8,263 | 64.73% |
| Real | 4,502 | 35.27% |
| **Total** | **12,765** | **100%** |

---

## 🔄 Binary Label Mapping

The original LIAR dataset contains six truthfulness labels.

For this project, they are converted into two classes.

### Fake

- `pants-fire`
- `false`
- `barely-true`
- `half-true`

### Real

- `mostly-true`
- `true`

> This binary mapping is a project-level design choice and is not the native LIAR classification scheme.

---

## 🧹 Day 2 — Text Preprocessing

The following preprocessing pipeline has been implemented:

```text
Raw Statement
      ↓
Lowercase Conversion
      ↓
URL Removal
      ↓
Punctuation Removal
      ↓
Tokenization
      ↓
Stopword Removal
      ↓
SpaCy Lemmatization
      ↓
Cleaned Statement
