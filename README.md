'''
# NewsGuard — Fake News Detection & Credibility Scoring System

## Project Overview

NewsGuard is an end-to-end NLP-based fake news detection and credibility scoring system.

The system combines:

- TF-IDF text features
- Word2Vec embeddings
- Readability features
- Sentiment analysis
- Linguistic features
- Multiple machine learning models
- XGBoost hyperparameter tuning
- SHAP explainability
- Flask REST API
- Gradio interface
- MLflow experiment tracking
- Automated testing
- External validation on WELFake

The system predicts whether an article is Fake or Real and generates a credibility score from 0–100 based on the predicted probability of the Real class.

---

## Dataset

Primary dataset:

Kaggle — Fake and Real News Dataset

After duplicate/content cleaning:

- Total articles: 39,103
- Training: 31,282
- Validation: 3,910
- Test: 3,911

The dataset was split using stratified sampling with an 80/10/10 ratio.

The subject and date columns were removed because they introduced strong source/label leakage.

---

# Development Progress

## Day 1 — Dataset Setup & EDA

Completed:

- Dataset loading
- Missing-value analysis
- Duplicate analysis
- Class distribution
- Subject/source leakage investigation
- Content construction
- Exact duplicate removal
- Stratified train/validation/test split
- Split integrity verification

Final dataset:

39,103 articles

---

## Day 2 — Text Preprocessing

Implemented:

- HTML removal
- Unicode normalization
- URL handling
- Email handling
- Lowercasing
- Control-character removal
- Punctuation normalization
- Whitespace normalization
- Empty-text handling

Final preprocessing integrity:

- Empty cleaned texts: 0
- Null cleaned texts: 0

---

## Day 3 — TF-IDF Feature Engineering

TF-IDF configuration:

- Maximum features: 5,000
- N-grams: unigram + bigram
- min_df = 2
- max_df = 0.95
- Sublinear TF enabled

Feature matrices:

- Train: 31,282 × 5,000
- Validation: 3,910 × 5,000
- Test: 3,911 × 5,000

---

## Day 4 — Auxiliary NLP Features

Created 115 auxiliary features:

### Word2Vec

- 100-dimensional sentence vectors
- Vocabulary: 129,860

### Readability

- Flesch Reading Ease
- Flesch-Kincaid Grade
- Gunning Fog
- Automated Readability Index

### Sentiment

- TextBlob polarity
- TextBlob subjectivity

### Linguistic

- Word count
- Character count
- Sentence count
- Average word length
- Average sentence length
- Unique word ratio
- Digit count
- Uppercase count
- Punctuation count

Total:

100 + 4 + 2 + 9 = 115 features

---

## Day 5 — Feature Union & Baseline Models

Combined:

5,000 TF-IDF + 115 auxiliary = 5,115 features

Baseline models:

| Model | CV F1 |
|---|---:|
| Logistic Regression | 99.39% |
| Naive Bayes | 95.71% |

---

## Day 6 — Advanced Models & Hyperparameter Tuning

Models evaluated:

- Logistic Regression
- Naive Bayes
- Linear SVM
- Random Forest
- Gradient Boosting
- XGBoost

Best model:

XGBoost

Best parameters:

- n_estimators = 120
- max_depth = 3
- learning_rate = 0.1

Cross-validation:

- F1: 99.77%
- ROC-AUC: 99.995%

---

## Day 7 — Final Model Evaluation

Final XGBoost on held-out test set:

| Metric | Score |
|---|---:|
| Accuracy | 99.7187% |
| Precision | 99.5769% |
| Recall | 99.9057% |
| F1 | 99.7410% |
| ROC-AUC | 99.9780% |

Required project target:

F1 > 0.88

Achieved:

F1 = 0.997410

---

## Day 8 — SHAP Explainability

Implemented:

- SHAP TreeExplainer
- Global feature importance
- SHAP bar plot
- SHAP beeswarm plot
- Individual force plots

Final feature space:

615 features

- 500 selected TF-IDF features
- 115 auxiliary features

Important features included:

- read more
- reuters
- featured image
- century wire
- washington reuters
- getty
- said

SHAP analysis also revealed strong source/template-related signals in the dataset.

---

## Day 9 — Flask API & Gradio UI

Implemented Flask REST API:

POST /predict

The API provides:

- Prediction
- Fake probability
- Real probability
- Credibility score
- SHAP-based explanation

Input validation:

- Invalid JSON → HTTP 400
- Missing text → HTTP 400
- Valid request → HTTP 200

### Gradio

Interactive article-classification interface created with:

- Article text input
- Fake/Real prediction
- Probability information
- Credibility score
- Explainability output

---

# External Validation — WELFake

The final model was additionally evaluated on a balanced sample of 2,000 WELFake articles.

Results:

| Metric | WELFake |
|---|---:|
| Accuracy | 79.15% |
| Precision | 98.34% |
| Recall | 59.30% |
| F1 | 73.99% |
| ROC-AUC | 92.83% |

Internal test F1:

99.74%

External WELFake F1:

73.99%

This demonstrates significant domain shift.

Therefore, NewsGuard should not be presented as a universal fake-news detector. It should be used as an experimental credibility-support system and not as the sole source for factual verification.

---

# Day 10 — Finalization

Completed:

### Model Card

Documents:

- Training data
- Intended use
- Performance
- Limitations
- Explainability
- External validation
- Responsible use
- Reproducibility

### MLflow

Experiment:

NewsGuard_Final_Model

Final XGBoost run tracked with:

- Model parameters
- Feature configuration
- Internal test metrics
- Cross-validation metrics
- WELFake validation metrics
- SHAP artifacts
- Model Card

### Automated Testing

PyTest result:

12 passed

All project integrity and performance checks passed.

---

# Final Architecture

Article
↓
Text Preprocessing
↓
TF-IDF + Word2Vec + Readability + Sentiment + Linguistic Features
↓
Feature Selection
↓
615 Final Features
↓
XGBoost
↓
Fake / Real Prediction
↓
Probability
↓
Credibility Score (0–100)
↓
SHAP Explanation
↓
Flask API / Gradio UI

---

# Repository Structure

NewsGuard_Fake_News_Detection/

├── notebooks/
│   ├── NewsGuard_Day_01_Dataset_Setup_EDA.ipynb
│   ├── NewsGuard_Day_02_Text_Preprocessing.ipynb
│   ├── NewsGuard_Day_03_TFIDF_Feature_Engineering.ipynb
│   ├── NewsGuard_Day_04_Auxiliary_NLP_Features.ipynb
│   ├── NewsGuard_Day_05_Feature_Union_Baseline_Models.ipynb
│   ├── NewsGuard_Day_06_Advanced_Models_and_Hyperparameter_Tuning.ipynb
│   ├── NewsGuard_Day_07_Final_Model_Evaluation.ipynb
│   ├── NewsGuard_Day_08_SHAP_Explainability.ipynb
│   ├── NewsGuard_Day_09_Flask_API_Gradio_UI.ipynb
│   └── NewsGuard_Day_10_Finalization.ipynb
│
├── results/
│   ├── day_05/
│   ├── day_06/
│   ├── day_07/
│   ├── day_08/
│   ├── day_09/
│   └── day_10/
│
├── tests/
│   └── test_newsguard.py
│
├── requirements.txt
├── LICENSE
└── README.md

---

# Final Performance

### Internal Test Set

F1 = 99.74%

### External WELFake

F1 = 73.99%

### Automated Tests

12/12 passed

### Model

XGBoost

### Final Feature Space

615 features

---

# Limitations

The model shows very high performance on the original dataset but lower performance on WELFake.

Potential reasons include:

- Dataset/domain shift
- Source-specific language
- Publication/template patterns
- Dataset construction differences
- Distribution differences between datasets

Therefore, the system should be treated as a decision-support and educational NLP system, not an authoritative fact-checking system.

---

# Future Deployment

Final deployment plan:

- Permanent Hugging Face Space
- Public Gradio interface
- Flask API deployment
- Visual credibility gauge
- SHAP feature explanations
- MLflow experiment tracking
- Final project documentation

---

# Project Status

Core ML/NLP pipeline: COMPLETE

Model evaluation: COMPLETE

SHAP explainability: COMPLETE

External validation: COMPLETE

Flask API: COMPLETE

Gradio UI: COMPLETE

Model Card: COMPLETE

MLflow tracking: COMPLETE

Automated tests: COMPLETE

Final deployment & presentation: IN PROGRESS

---

## Author

Sagar Sharma

B.Tech CSE (AI & ML)

Panipat Institute of Engineering and Technology (PIET)
'''
