# NewsGuard — Fake News Detection & Credibility Scoring System

NewsGuard is an NLP-based fake news detection system designed to classify news articles as **Real** or **Fake**, with future support for credibility scoring and explainable predictions.

The project combines multiple NLP feature-engineering techniques including **TF-IDF, Word2Vec, readability, sentiment, and linguistic features**, followed by comparison and hyperparameter tuning of multiple machine learning models.

> **Project Status:** Day 1–Day 7 completed. SHAP explainability, credibility scoring, Flask API, Gradio UI, MLflow tracking, automated testing, model card, and final documentation are in progress.

---

## 🎯 Project Objective

The goal of NewsGuard is to build an end-to-end machine learning pipeline that can:

* Detect whether a news article is Real or Fake
* Combine multiple NLP feature types
* Compare multiple machine learning models
* Tune the best-performing model
* Evaluate the final model on unseen data
* Provide model explainability using SHAP
* Generate a credibility score
* Provide predictions through a Flask REST API
* Provide an interactive user interface
* Maintain a reproducible ML workflow

---

## 📊 Dataset

The project uses the **Fake and Real News Dataset** from Kaggle.

**Dataset:** Fake and Real News Dataset

**Source:**
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

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

| Split      |          Shape |
| ---------- | -------------: |
| Train      | 31,282 × 5,000 |
| Validation |  3,910 × 5,000 |
| Test       |  3,911 × 5,000 |

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

Document-level representation:

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

| Split      |        Shape |
| ---------- | -----------: |
| Train      | 31,282 × 115 |
| Validation |  3,910 × 115 |
| Test       |  3,911 × 115 |

All generated feature artifacts were saved and successfully reloaded for integrity verification.

---

## Day 5 — FeatureUnion & Baseline Models ✅

Combined:

* **5,000 TF-IDF features**
* **115 auxiliary NLP features**

Final combined representation:

**5,115 features**

A Scikit-Learn `FeatureUnion` was created to represent the combined feature pipeline.

### Logistic Regression

5-fold stratified cross-validation:

| Metric    |      Score |
| --------- | ---------: |
| Accuracy  |     99.34% |
| Precision |     99.35% |
| Recall    |     99.43% |
| **F1**    | **99.39%** |
| ROC-AUC   |     99.94% |

### Naive Bayes

5-fold stratified cross-validation:

| Metric    |      Score |
| --------- | ---------: |
| Accuracy  |     95.33% |
| Precision |     95.28% |
| Recall    |     96.15% |
| **F1**    | **95.71%** |
| ROC-AUC   |     99.02% |

### Best Baseline

🏆 **Logistic Regression — F1: 99.39%**

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

The tuned XGBoost model was saved as an artifact for subsequent evaluation.

---

## Day 7 — Final Model Evaluation ✅

The tuned XGBoost model was evaluated on the validation set and the held-out test set.

### Validation Performance

| Metric    |      Score |
| --------- | ---------: |
| Accuracy  | **99.67%** |
| Precision | **99.48%** |
| Recall    | **99.91%** |
| **F1**    | **99.69%** |
| ROC-AUC   | **99.98%** |

### Held-Out Test Performance

| Metric    |      Score |
| --------- | ---------: |
| Accuracy  | **99.72%** |
| Precision | **99.58%** |
| Recall    | **99.91%** |
| **F1**    | **99.74%** |
| ROC-AUC   | **99.98%** |

### Classification Report

| Class | Precision | Recall |     F1 |
| ----- | --------: | -----: | -----: |
| Fake  |    99.89% | 99.50% | 99.69% |
| Real  |    99.58% | 99.91% | 99.74% |

Test set:

* Fake samples: **1,791**
* Real samples: **2,120**
* Total: **3,911**

### Confusion Matrix

```text
                 Predicted
              Fake     Real
Actual Fake    1782       9
Actual Real       2    2118
```

### Final Model

🏆 **XGBoost**

Held-out test F1:

**99.74%**

Held-out test ROC-AUC:

**99.98%**

### Assignment Target

Required F1:

**> 0.88**

Achieved test F1:

**0.9974**

✅ **Target achieved**

### Day 7 Artifacts

```text
results/day_07/
├── confusion_matrix_xgboost_test.png
├── roc_curve_xgboost_test.png
├── model_comparison_f1.png
├── day_07_final_results.csv
├── day_07_final_results.joblib
└── day_07_confusion_matrix.csv
```

---

# 📈 Model Performance Overview

The best cross-validation F1 scores obtained so far are:

| Model               |         F1 |
| ------------------- | ---------: |
| Naive Bayes         |     95.71% |
| Random Forest       |     99.15% |
| Logistic Regression |     99.39% |
| Gradient Boosting   |     99.62% |
| Linear SVM          |     99.67% |
| **XGBoost**         | **99.77%** |

XGBoost is currently the best-performing model based on cross-validation F1.

---

# 🔬 Upcoming Work

## Day 8 — SHAP Explainability

Planned:

* SHAP TreeExplainer
* Global feature importance
* Feature contribution analysis
* Individual prediction explanations
* Word-level importance analysis
* SHAP visualizations

## Day 9 — Flask API & User Interface

Planned:

* Flask REST API
* `/predict` endpoint
* Real/Fake prediction
* Credibility score
* SHAP-based explanation
* Gradio interface

## Day 10 — Finalization

Planned:

* MLflow experiment tracking
* PyTest test suite
* Model card
* Final documentation
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
├── README.md
├── requirements.txt
│
├── notebooks/
│   ├── NewsGuard_Day_01_Dataset_Setup_EDA.ipynb
│   ├── NewsGuard_Day_02_Text_Preprocessing.ipynb
│   ├── NewsGuard_Day_03_TFIDF_Feature_Engineering.ipynb
│   ├── NewsGuard_Day_04_Auxiliary_NLP_Features.ipynb
│   ├── NewsGuard_Day_05_Feature_Union_Baseline_Models.ipynb
│   ├── NewsGuard_Day_06_Advanced_Models_and_Hyperparameter_Tuning.ipynb
│   └── NewsGuard_Day_07_Final_Model_Evaluation.ipynb
│
└── results/
    ├── day_05/
    │   ├── day_05_baseline_results.csv
    │   └── day_05_baseline_results.joblib
    │
    ├── day_06/
    │   ├── day_06_model_comparison.csv
    │   ├── day_06_model_comparison.joblib
    │   ├── day_06_xgboost_gridsearch.csv
    │   └── day_06_xgboost_best_tuned.joblib
    │
    └── day_07/
        ├── confusion_matrix_xgboost_test.png
        ├── roc_curve_xgboost_test.png
        ├── model_comparison_f1.png
        ├── day_07_final_results.csv
        ├── day_07_final_results.joblib
        └── day_07_confusion_matrix.csv
```

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
* Held-out validation and test evaluation
* Separate final evaluation on unseen data

> **Development note:** Some Day 5–Day 6 cross-validation experiments use precomputed training features and reduced feature representations. These experiments are documented as development-stage model comparisons. The final validation/test evaluation was performed separately using the held-out splits.

---

# 📌 Current Project Result

### Best Model

**XGBoost**

### Best Cross-Validation F1

**99.77%**

### Held-Out Test F1

**99.74%**

### Held-Out Test ROC-AUC

**99.98%**

### Required Target

**F1 > 0.88**

### Target Status

✅ **Achieved**

---

# 👨‍💻 Author

**Sagar Sharma**

B.Tech CSE — Artificial Intelligence & Machine Learning

Panipat Institute of Engineering and Technology (PIET)

---

## ⚠️ Project Status

NewsGuard has successfully completed:

**Day 1 → Dataset & EDA**
**Day 2 → Text Preprocessing**
**Day 3 → TF-IDF Feature Engineering**
**Day 4 → Auxiliary NLP Features**
**Day 5 → FeatureUnion & Baseline Models**
**Day 6 → Advanced Models & Hyperparameter Tuning**
**Day 7 → Final Model Evaluation**

🚧 **Next:** SHAP Explainability
