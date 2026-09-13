# NewsGuard — Fake News Detection & Credibility Scoring System

NewsGuard is an NLP-based fake news detection system designed to classify news articles as **Real** or **Fake** using multiple textual, linguistic, readability, sentiment, and semantic features.

The project follows an end-to-end machine learning workflow covering dataset preparation, preprocessing, feature engineering, baseline modeling, advanced model comparison, hyperparameter tuning, held-out evaluation, and SHAP-based explainability.

> **Project Status:** Day 1–Day 8 completed. Flask API, Gradio UI, MLflow tracking, automated testing, model card, and final documentation are planned next.

---

## 🎯 Project Objective

The goal of NewsGuard is to build a reproducible fake-news detection pipeline that can:

* Detect whether a news article is Real or Fake
* Combine multiple NLP feature types
* Compare different machine learning models
* Tune the best-performing model
* Evaluate the final model on unseen data
* Explain predictions using SHAP
* Generate a credibility-oriented score
* Provide predictions through an API and interactive UI
* Maintain reproducible ML artifacts throughout development

---

## 📊 Dataset

The project uses the **Fake and Real News Dataset** from Kaggle.

**Dataset:** Fake and Real News Dataset

**Source:**
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

The original dataset contains:

* `Fake.csv`
* `True.csv`

The raw dataset is **not stored in this repository** because of its size.

### Cleaned Dataset

After duplicate removal and preparation:

* Total articles: **39,103**
* Real articles: **21,196**
* Fake articles: **17,907**

### Data Split

A stratified 80/10/10 split was created:

| Split      | Samples |
| ---------- | ------: |
| Training   |  31,282 |
| Validation |   3,910 |
| Test       |   3,911 |

The train, validation, and test sets were checked for content overlap.

---

# 🛠️ Technology Stack

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
* Jupyter
* Google Colab

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

Both original and cleaned text were retained for reproducibility.

---

## Day 3 — TF-IDF Feature Engineering ✅

TF-IDF configuration:

* Unigrams + bigrams
* Maximum features: **5,000**
* `min_df = 2`
* `max_df = 0.95`
* Sublinear TF scaling
* Vocabulary fitted only on training data

### TF-IDF Matrix

| Split      |          Shape |
| ---------- | -------------: |
| Train      | 31,282 × 5,000 |
| Validation |  3,910 × 5,000 |
| Test       |  3,911 × 5,000 |

---

## Day 4 — Auxiliary NLP Features ✅

Additional NLP features were engineered using semantic, readability, sentiment, and linguistic information.

### Word2Vec

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

* Word Count
* Character Count
* Sentence Count
* Average Word Length
* Average Sentence Length
* Unique Word Ratio
* Digit Count
* Uppercase Count
* Punctuation Count

**9 features**

### Total Auxiliary Features

**100 + 4 + 2 + 9 = 115 features**

| Split      |        Shape |
| ---------- | -----------: |
| Train      | 31,282 × 115 |
| Validation |  3,910 × 115 |
| Test       |  3,911 × 115 |

---

## Day 5 — FeatureUnion & Baseline Models ✅

Combined representation:

* **5,000 TF-IDF features**
* **115 auxiliary features**
* **5,115 total features**

A Scikit-learn `FeatureUnion` pipeline was created.

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

Advanced models were evaluated using stratified cross-validation.

### Model Comparison

| Model               |   Accuracy |  Precision |     Recall |     **F1** |     ROC-AUC |
| ------------------- | ---------: | ---------: | ---------: | ---------: | ----------: |
| **XGBoost**         | **99.75%** | **99.71%** | **99.83%** | **99.77%** | **99.995%** |
| Linear SVM          |     99.64% |     99.61% |     99.73% |     99.67% |     99.966% |
| Gradient Boosting   |     99.58% |     99.43% |     99.81% |     99.62% |     99.972% |
| Logistic Regression |     99.34% |     99.35% |     99.43% |     99.39% |     99.941% |
| Random Forest       |     99.08% |     98.82% |     99.49% |     99.15% |     99.962% |
| Naive Bayes         |     95.33% |     95.28% |     96.15% |     95.71% |     99.024% |

### XGBoost Hyperparameter Tuning

`GridSearchCV` was used to tune the XGBoost model.

Best parameters:

```text
learning_rate = 0.1
max_depth = 3
n_estimators = 120
```

Best tuned CV F1:

**99.77%**

The tuned model was saved as a reusable artifact.

---

## Day 7 — Final Model Evaluation ✅

The tuned XGBoost model was evaluated separately on validation and held-out test data.

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

### Test Classification Report

| Class | Precision | Recall |     F1 |
| ----- | --------: | -----: | -----: |
| Fake  |    99.89% | 99.50% | 99.69% |
| Real  |    99.58% | 99.91% | 99.74% |

Test set:

* Fake: **1,791**
* Real: **2,120**
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

### Target

Required F1:

**> 0.88**

Achieved:

**0.9974**

✅ **Target achieved**

---

## Day 8 — SHAP Explainability ✅

Day 8 focused on understanding the decisions made by the final XGBoost model.

### SHAP Setup

* XGBoost `TreeExplainer`
* Exact **615-feature** model space recreated
* 500 held-out test samples used for SHAP analysis
* Random seed: **42**

### SHAP Feature Space

```text
500 selected TF-IDF features
+
115 auxiliary NLP features
=
615 total features
```

### Top SHAP Features

| Rank | Feature              | Mean Absolute SHAP |
| ---: | -------------------- | -----------------: |
|    1 | `read more`          |           1.689772 |
|    2 | `reuters`            |           1.528391 |
|    3 | `washington reuters` |           1.046988 |
|    4 | `featured image`     |           1.027853 |
|    5 | `century wire`       |           0.977140 |
|    6 | `getty`              |           0.536068 |
|    7 | `said`               |           0.518828 |
|    8 | `via`                |           0.418443 |
|    9 | `nov`                |           0.405618 |
|   10 | `W2V_044`            |           0.356393 |

The SHAP analysis provides global feature importance as well as individual prediction explanations.

### Individual Explanation Example

Three held-out test samples were analyzed.

One example was intentionally captured as a misclassified case:

```text
Actual: Real
Predicted: Fake
Real Probability: 0.257711
Credibility Score: 25.77 / 100
```

### Day 8 Artifacts

```text
results/day_08/
├── shap_feature_importance.csv
├── shap_summary_bar.png
├── shap_summary_beeswarm.png
├── shap_force_plot_1.html
├── shap_force_plot_2.html
├── shap_force_plot_3.html
├── shap_individual_explanations.csv
└── day_08_shap_summary.json
```

---

# 📈 Overall Model Performance

Best cross-validation F1 scores:

| Model               |      CV F1 |
| ------------------- | ---------: |
| Naive Bayes         |     95.71% |
| Random Forest       |     99.15% |
| Logistic Regression |     99.39% |
| Gradient Boosting   |     99.62% |
| Linear SVM          |     99.67% |
| **XGBoost**         | **99.77%** |

### Current Best Model

**XGBoost**

* Best CV F1: **99.77%**
* Held-out Test F1: **99.74%**
* Held-out Test ROC-AUC: **99.98%**
* Required F1 target: **> 88%**
* Target: ✅ **Achieved**

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
│   ├── NewsGuard_Day_07_Final_Model_Evaluation.ipynb
│   └── NewsGuard_Day_08_SHAP_Explainability.ipynb
│
└── results/
    ├── day_05/
    ├── day_06/
    ├── day_07/
    └── day_08/
```

Raw datasets and large generated feature files are intentionally excluded from GitHub.

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
* Hyperparameter tuning with GridSearchCV
* Held-out validation and test evaluation
* Separate final evaluation on unseen data
* SHAP explainability using the saved best model
* Reproducible result artifacts

> **Development note:** Some Day 5–Day 6 experiments use precomputed or reduced feature representations for model comparison. These are documented as development-stage experiments. Final validation/test performance was measured separately on held-out data.

---

# 🚀 Upcoming Work

## Day 9 — Deployment & User Interface

Planned:

* Flask REST API
* `/predict` endpoint
* Real/Fake prediction
* Credibility score
* SHAP-based explanation
* Gradio interface

## Day 10 — ML Finalization

Planned:

* MLflow experiment tracking
* PyTest test suite
* Model card
* Final documentation
* Screenshots
* Reproducibility verification
* Project cleanup

---

# 👨‍💻 Author

**Sagar Sharma**

B.Tech CSE — Artificial Intelligence & Machine Learning

Panipat Institute of Engineering and Technology (PIET)

---

# 📌 Current Project Status

### Completed

**Day 1 → Dataset Setup & EDA** ✅
**Day 2 → Text Preprocessing** ✅
**Day 3 → TF-IDF Feature Engineering** ✅
**Day 4 → Auxiliary NLP Features** ✅
**Day 5 → FeatureUnion & Baseline Models** ✅
**Day 6 → Advanced Models & Hyperparameter Tuning** ✅
**Day 7 → Final Model Evaluation** ✅
**Day 8 → SHAP Explainability** ✅

### Current Best Result

**XGBoost — 99.74% Held-Out Test F1**

🎯 **F1 Target > 0.88: ACHIEVED**

🚧 **Next: Day 9 — Deployment & User Interface**
