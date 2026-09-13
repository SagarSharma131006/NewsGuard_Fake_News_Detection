# NewsGuard — Fake News Detection & Credibility Scoring System

NewsGuard is an NLP-based fake news detection system designed to classify news articles as **Real** or **Fake** and eventually provide a **credibility score** along with explainable predictions.

The project combines multiple NLP feature-engineering techniques including **TF-IDF, Word2Vec, readability, sentiment, and linguistic features**, followed by comparison and tuning of multiple machine learning models.

> **Project Status:** Day 1–Day 6 completed. Final evaluation, SHAP explainability, API, UI, MLflow tracking, testing, and final documentation are in progress.

---

## 🎯 Project Objective

The goal of NewsGuard is to build an end-to-end machine learning pipeline that can:

* Detect whether a news article is Real or Fake
* Combine multiple NLP feature types
* Compare multiple machine learning models
* Tune the best-performing models
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

After duplicate removal and data preparation:

* Total cleaned articles: **39,103**
* Real articles: **21,196**
* Fake articles: **17,907**

### Data Split

A stratified 80/10/10 split was created:

| Split      | Samples |
| ---------- | ------: |
| Training   |  31,282 |
| Validation |   3,910 |
| Test       |   3,911 |

Final label distribution:

* **Real:** 54.2%
* **Fake:** 45.8%

No content overlap was found between the train, validation, and test sets.

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

Implemented:

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

Final verification:

* Empty texts: **0**
* Null cleaned texts: **0**
* Remaining raw URLs: **0**

Both original and cleaned text were retained for reproducibility and feature engineering.

---

## Day 3 — TF-IDF Feature Engineering ✅

Implemented TF-IDF using:

* Unigrams + bigrams
* Maximum features: **5,000**
* `min_df = 2`
* `max_df = 0.95`
* Sublinear TF scaling
* Train-only vocabulary fitting

### TF-IDF Matrix

| Split      | Shape          |
| ---------- | -------------- |
| Train      | 31,282 × 5,000 |
| Validation | 3,910 × 5,000  |
| Test       | 3,911 × 5,000  |

The vocabulary was learned only from the training data and reused for validation and test sets.

---

## Day 4 — Auxiliary NLP Features ✅

Additional NLP features were engineered using Word2Vec, readability, sentiment, and linguistic statistics.

### Word2Vec

Configuration:

* Vector size: **100**
* Window: **5**
* Minimum word count: **2**
* Skip-gram: **Yes**
* Epochs: **2**
* Seed: **42**

Document-level Word2Vec representation:

**100 features**

### Readability Features

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

| Split      | Shape        |
| ---------- | ------------ |
| Train      | 31,282 × 115 |
| Validation | 3,910 × 115  |
| Test       | 3,911 × 115  |

All generated feature artifacts were saved and successfully reloaded for integrity verification.

---

## Day 5 — FeatureUnion & Baseline Models ✅

Combined:

* **5,000 TF-IDF features**
* **115 auxiliary NLP features**

Final combined representation:

**5,115 features**

A Scikit-Learn `FeatureUnion` was created to represent the combined feature pipeline.

### Baseline Models

#### Logistic Regression

5-fold stratified cross-validation:

* Accuracy: **99.34%**
* Precision: **99.35%**
* Recall: **99.43%**
* **F1: 99.39%**
* ROC-AUC: **99.94%**

#### Naive Bayes

5-fold stratified cross-validation:

* Accuracy: **95.33%**
* Precision: **95.28%**
* Recall: **96.15%**
* **F1: 95.71%**
* ROC-AUC: **99.02%**

### Best Baseline

**Logistic Regression — F1: 99.39%**

---

## Day 6 — Advanced Models & Hyperparameter Tuning ✅

Advanced classification models were evaluated using stratified cross-validation.

### Model Comparison

| Model               |   Accuracy |  Precision |     Recall |         F1 |     ROC-AUC |
| ------------------- | ---------: | ---------: | ---------: | ---------: | ----------: |
| **XGBoost**         | **99.75%** | **99.71%** | **99.83%** | **99.77%** | **99.995%** |
| Linear SVM          |     99.64% |     99.61% |     99.73% |     99.67% |     99.966% |
| Gradient Boosting   |     99.58% |     99.43% |     99.81% |     99.62% |     99.972% |
| Logistic Regression |     99.34% |     99.35% |     99.43% |     99.39% |     99.941% |
| Random Forest       |     99.08% |     98.82% |     99.49% |     99.15% |     99.962% |
| Naive Bayes         |     95.33% |     95.28% |     96.15% |     95.71% |     99.024% |

### XGBoost Hyperparameter Tuning

`GridSearchCV` was used to tune XGBoost.

Best parameters:

```text
learning_rate = 0.1
max_depth = 3
n_estimators = 120
```

Best tuned cross-validation F1:

**99.77%**

The tuned XGBoost model was saved as a reusable artifact for subsequent evaluation.

### Current Best Model

🏆 **XGBoost**

Cross-validation F1:

**0.9977**

---

# 🔬 Upcoming Work

## Day 7 — Final Model Evaluation

* Evaluate candidate models on validation set
* Evaluate final selected model on held-out test set
* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion matrix
* Classification report
* Final model selection

## Day 8 — SHAP Explainability

* SHAP TreeExplainer
* Global feature importance
* Feature contribution analysis
* Individual prediction explanations
* Word-level importance analysis
* SHAP visualizations

## Day 9 — Flask API & User Interface

* Flask REST API
* `/predict` endpoint
* Real/Fake prediction
* Credibility score
* SHAP-based explanation
* Gradio interface

## Day 10 — Finalization

* MLflow experiment tracking
* PyTest test suite
* Model card
* Final README
* Documentation
* Screenshots
* Reproducibility verification
* Project cleanup

---

# 📁 Repository Structure

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
    ├── NewsGuard_Day_04_Auxiliary_NLP_Features.ipynb
    ├── NewsGuard_Day_05_Feature_Union_Baseline_Models.ipynb
    └── NewsGuard_Day_06_Advanced_Models_and_Hyperparameter_Tuning.ipynb
```

Additional folders such as:

```text
src/
app/
tests/
models/
results/
reports/
```

will be added progressively as the project reaches the corresponding milestones.

---

# 🔁 Reproducibility

The project follows these principles:

* Fixed random seeds where applicable
* Stratified train/validation/test splitting
* Train-only TF-IDF vocabulary fitting
* Train-only Word2Vec training
* Saved feature artifacts
* Artifact reload verification
* Dataset integrity checks
* Stratified cross-validation
* Hyperparameter tuning using GridSearchCV
* Held-out validation and test sets reserved for final evaluation

> **Note:** Some Day 5–Day 6 cross-validation experiments use precomputed training features and reduced feature representations. These experiments are documented as development-stage model comparisons; final held-out evaluation will be performed separately on the untouched validation/test sets.

---

# 📌 Current Best Result

After Day 6:

**Best Model:** XGBoost

**Cross-Validation F1:** **99.77%**

**Cross-Validation ROC-AUC:** **99.995%**

The final held-out test performance will be reported after Day 7 evaluation.

---

# 👨‍💻 Author

**Sagar Sharma**

B.Tech CSE — Artificial Intelligence & Machine Learning

Panipat Institute of Engineering and Technology (PIET)

---

## ⚠️ Project Status

NewsGuard is currently in the **model evaluation and explainability phase**.

Final held-out test results, SHAP explanations, credibility scoring, Flask API, Gradio UI, MLflow tracking, automated tests, and final documentation will be added as the remaining milestones are completed.
