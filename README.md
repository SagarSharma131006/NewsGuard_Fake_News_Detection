# NewsGuard — Fake News Detection & Credibility Scoring System

NewsGuard is an NLP-based fake news detection system designed to classify news articles as **Real** or **Fake** and eventually provide a **credibility score** along with explainable predictions.

The project combines multiple NLP feature-engineering techniques including **TF-IDF, Word2Vec, readability, sentiment, and linguistic features**.

> **Project Status:** Day 1–Day 4 completed. Model training, evaluation, SHAP explainability, API, UI, MLflow, testing, and final documentation are still in progress.

---

## 🎯 Project Objective

The goal of NewsGuard is to build an end-to-end machine learning pipeline that can:

* Detect whether a news article is Real or Fake
* Combine multiple NLP feature types
* Compare multiple machine learning models
* Provide model explainability using SHAP
* Generate a credibility score
* Provide predictions through a Flask REST API
* Provide an interactive user interface
* Maintain a reproducible ML workflow

---

## 📊 Dataset

The project uses the **Fake and Real News Dataset** from Kaggle.

**Dataset:** Fake and Real News Dataset
**Source:** https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

The original dataset contains:

* `Fake.csv`
* `True.csv`

The dataset files are **not stored in this GitHub repository** because of their size.

### Dataset Processing

After cleaning duplicate articles and preparing the data:

* Total cleaned articles: **39,103**
* Real articles: **21,196**
* Fake articles: **17,907**

### Data Split

A stratified split was created:

| Split      | Samples |
| ---------- | ------: |
| Training   |  31,282 |
| Validation |   3,910 |
| Test       |   3,911 |

The final label distribution is approximately:

* **Real:** 54.2%
* **Fake:** 45.8%

No content overlap was found between train, validation, and test sets.

---

## 🛠️ Technology Stack

* Python
* Pandas
* NumPy
* SciPy
* Scikit-learn
* NLTK
* Gensim
* TextStat
* TextBlob
* XGBoost
* SHAP
* Flask
* Gradio
* MLflow
* PyTest
* Jupyter / Google Colab

---

# 📅 Development Progress

## Day 1 — Dataset Setup & EDA ✅

Completed:

* Dataset extraction and loading
* Dataset structure inspection
* Missing-value analysis
* Duplicate analysis
* Subject/category leakage analysis
* Removal of leakage-prone columns
* Duplicate content removal
* Article length analysis
* Stratified train/validation/test split
* Cross-split overlap verification
* Dataset integrity checks

Final cleaned dataset:

**39,103 articles**

---

## Day 2 — Text Preprocessing ✅

Implemented text preprocessing including:

* HTML entity decoding
* Unicode normalization
* HTML removal
* URL and email handling
* Lowercasing
* Control-character removal
* Punctuation normalization
* Whitespace normalization
* Markdown/raw URL handling
* Empty-text handling

Final preprocessing verification:

* Empty texts: **0**
* Null cleaned texts: **0**
* Remaining raw URLs: **0**

The original text and cleaned text were retained for reproducibility and feature engineering.

---

## Day 3 — TF-IDF Feature Engineering ✅

Implemented TF-IDF feature extraction using:

* Unigrams + bigrams
* Maximum features: **5,000**
* `min_df = 2`
* `max_df = 0.95`
* Sublinear TF scaling
* Train-only vocabulary fitting

Feature matrix sizes:

| Split      | TF-IDF Shape   |
| ---------- | -------------- |
| Train      | 31,282 × 5,000 |
| Validation | 3,910 × 5,000  |
| Test       | 3,911 × 5,000  |

The TF-IDF vocabulary was learned only from the training data and reused for validation and test sets.

---

## Day 4 — Auxiliary NLP Features ✅

Additional NLP features were engineered using the training data.

### Word2Vec

Configuration:

* Vector size: **100**
* Window: **5**
* Minimum word count: **2**
* Skip-gram: **Yes**
* Epochs: **2**
* Seed: **42**

Document-level Word2Vec vectors:

**100 features**

### Readability Features

Four readability metrics:

* Flesch Reading Ease
* Flesch-Kincaid Grade
* Gunning Fog Index
* Automated Readability Index

**4 features**

### Sentiment Features

Using TextBlob:

* Polarity
* Subjectivity

**2 features**

### Linguistic Features

Nine linguistic features:

1. Word Count
2. Character Count
3. Sentence Count
4. Average Word Length
5. Average Sentence Length
6. Unique Word Ratio
7. Digit Count
8. Uppercase Count
9. Punctuation Count

**9 features**

### Total Auxiliary Features

**100 + 4 + 2 + 9 = 115 features**

Final auxiliary matrices:

| Split      | Shape        |
| ---------- | ------------ |
| Train      | 31,282 × 115 |
| Validation | 3,910 × 115  |
| Test       | 3,911 × 115  |

All generated feature artifacts were saved and successfully reloaded for integrity verification.

---

# 🔄 Upcoming Work

The remaining development roadmap is:

### Day 5

* FeatureUnion / combined feature pipeline
* Combine TF-IDF + auxiliary features
* Baseline model training
* Logistic Regression
* SVM
* Random Forest
* Gradient Boosting
* XGBoost

### Day 6

* Advanced models
* Hyperparameter tuning
* GridSearchCV
* Cross-validation

### Day 7

* Final model evaluation
* Validation and held-out test performance
* Confusion matrix
* Precision / Recall / F1
* ROC-AUC
* Best model selection

### Day 8

* SHAP explainability
* Global feature importance
* Individual prediction explanations
* Word-level importance

### Day 9

* Flask REST API
* `/predict` endpoint
* Credibility score
* Gradio interface

### Day 10

* MLflow tracking
* PyTest
* Model card
* Final README
* Documentation
* Screenshots
* Reproducibility checks
* Final project cleanup

---

# 📁 Current Repository Structure

```text
NewsGuard_Fake_News_Detection/
│
├── .gitignore
├── LICENSE
├── requirements.txt
│
└── notebooks/
    ├── NewsGuard_Day_01_Dataset_Setup_EDA.ipynb
    ├── NewsGuard_Day_02_Text_Preprocessing.ipynb
    ├── NewsGuard_Day_03_TFIDF_Feature_Engineering.ipynb
    └── NewsGuard_Day_04_Auxiliary_NLP_Features.ipynb
```

Additional folders such as `src/`, `app/`, `tests/`, `models/`, `results/`, and `reports/` will be added progressively as the project reaches the corresponding milestones.

---

# 🔬 Reproducibility

The project follows these principles:

* Fixed random seeds where applicable
* Stratified train/validation/test splitting
* No fitting of learned feature transformers on validation/test data
* Train-only TF-IDF vocabulary
* Train-only Word2Vec training
* Saved feature artifacts
* Artifact reload verification
* Dataset integrity checks after each major stage

---

# 👨‍💻 Author

**Sagar Sharma**

B.Tech CSE — Artificial Intelligence & Machine Learning

Panipat Institute of Engineering and Technology (PIET)

---

## ⚠️ Project Status

This README is a **temporary Day 1–Day 4 version**.

Final results, model performance, SHAP explanations, API usage, UI screenshots, MLflow results, and final project structure will be added after completion of the remaining milestones.
