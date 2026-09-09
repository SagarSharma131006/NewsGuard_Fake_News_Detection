# 🛡️ NewsGuard — Fake News Detection & Credibility Scoring System

> An end-to-end NLP-based fake news detection and credibility scoring system using **TF-IDF**, **Word2Vec**, **NLP preprocessing**, **machine learning classifiers**, **SHAP explainability**, **Flask REST API**, **Gradio**, and **MLflow**.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-orange.svg)](https://scikit-learn.org/)
[![NLP](https://img.shields.io/badge/NLP-Text%20Processing-green.svg)](https://www.nltk.org/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-purple.svg)](https://shap.readthedocs.io/)
[![Flask](https://img.shields.io/badge/Flask-REST%20API-black.svg)](https://flask.palletsprojects.com/)
[![Gradio](https://img.shields.io/badge/Gradio-Interactive%20UI-yellow.svg)](https://www.gradio.app/)
[![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-blue.svg)](https://mlflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

# 📌 Project Status

```text
╔══════════════════════════════════════════╗
║         NEWSGUARD PROJECT STATUS         ║
╠══════════════════════════════════════════╣
║ Day 01  ✅ Dataset Setup & EDA           ║
║ Day 02  ✅ Text Preprocessing            ║
║ Day 03  ⏳ TF-IDF Feature Engineering    ║
║ Day 04  ⏳ Auxiliary Features            ║
║ Day 05  ⏳ FeatureUnion + Baselines      ║
║ Day 06  ⏳ Advanced Models + Tuning      ║
║ Day 07  ⏳ Model Evaluation              ║
║ Day 08  ⏳ SHAP Explainability           ║
║ Day 09  ⏳ Flask API + Gradio UI         ║
║ Day 10  ⏳ MLflow + Testing + Docs       ║
╠══════════════════════════════════════════╣
║        🚧 PROJECT IN PROGRESS 🚧         ║
╚══════════════════════════════════════════╝
```

**Current Progress: 20% — Day 2/10 Completed**

---

# 📌 Project Overview

**NewsGuard** is an end-to-end Natural Language Processing and Machine Learning project designed to detect whether a textual claim is **Fake** or **Real** and provide a model-based **credibility score**.

The system combines:

- NLP text preprocessing
- TF-IDF features
- Word2Vec sentence embeddings
- Readability features
- Sentiment polarity
- Article/statement length
- Multiple machine learning classifiers
- Stratified cross-validation
- Hyperparameter tuning
- SHAP explainability
- Credibility scoring
- Flask REST API
- Gradio interface
- MLflow experiment tracking

The final system is designed to return:

```text
Prediction
Credibility Score
Important Words / Features
```

---

# 🎯 Objectives

The main objectives of NewsGuard are:

1. Build a complete NLP preprocessing pipeline.
2. Convert textual information into meaningful numerical features.
3. Use TF-IDF with unigram and bigram features.
4. Generate Word2Vec-based sentence representations.
5. Extract readability and sentiment-based auxiliary features.
6. Combine multiple feature types into a unified feature representation.
7. Train and compare at least five classification algorithms.
8. Evaluate models using stratified 5-fold cross-validation.
9. Select the best-performing model.
10. Achieve the project target of **F1 > 0.88** on the held-out test set.
11. Explain predictions using SHAP.
12. Generate a credibility score from model probability.
13. Deploy the final model through a Flask REST API.
14. Provide an interactive Gradio explanation interface.
15. Track experiments using MLflow.
16. Create reproducible documentation, tests, and a model card.

---

# 🗂️ Dataset

The project uses the **LIAR dataset** for fake/real classification.

### Dataset Source

The dataset was obtained from the official LIAR dataset distribution provided by the University of California, Santa Barbara.

The original dataset contains three files:

```text
train.tsv
valid.tsv
test.tsv
```

---

# 📊 Original Dataset

The original LIAR dataset contains **12,791 statements** across six truthfulness labels.

| Split | Samples |
|---|---:|
| Training | 10,240 |
| Validation | 1,284 |
| Testing | 1,267 |
| **Total** | **12,791** |

---

# 🏷️ Original Labels

The LIAR dataset contains six original labels:

```text
pants-fire
false
barely-true
half-true
mostly-true
true
```

These labels represent different levels of truthfulness.

---

# 🔄 Binary Class Mapping

The NewsGuard project converts the original six-class LIAR labels into two classes.

### Fake

```text
pants-fire
false
barely-true
half-true
```

### Real

```text
mostly-true
true
```

Therefore:

```text
Fake → 0
Real → 1
```

> ⚠️ This binary mapping is a project-defined transformation. It is not the native binary labeling of the LIAR dataset.

---

# 📊 Dataset Distribution

After combining the official dataset splits and removing duplicate statements:

| Class | Samples | Percentage |
|---|---:|---:|
| Fake | **8,263** | **64.73%** |
| Real | **4,502** | **35.27%** |
| **Total** | **12,765** | **100%** |

The dataset is therefore moderately imbalanced toward the Fake class.

---

# 🧹 Duplicate Removal

Duplicate statements were checked before creating the final dataset.

```text
Duplicate statements before removal : 26
Duplicate statements after removal  : 0
Final dataset size                  : 12,765
```

Removing duplicates helps reduce the possibility of repeated samples appearing across different splits.

---

# 🔀 Final Dataset Split

A new **stratified 80/10/10 split** was created after duplicate removal.

| Split | Samples | Fake | Real |
|---|---:|---:|---:|
| Training | **10,212** | 6,610 | 3,602 |
| Validation | **1,276** | 826 | 450 |
| Testing | **1,277** | 827 | 450 |
| **Total** | **12,765** | **8,263** | **4,502** |

Stratification was used to preserve the class distribution across all splits.

---

# 🔐 Data Leakage Verification

The final dataset splits were checked for overlapping statements.

```text
Train ∩ Validation : 0
Train ∩ Test       : 0
Validation ∩ Test  : 0
```

Therefore, no duplicate statement overlap was found between the final training, validation, and testing sets.

---

# 📏 Text Statistics

Basic text statistics were calculated during EDA.

```text
Average original words : 18.06
Minimum words          : 2
Maximum words          : 467
```

The dataset contains short textual claims rather than long-form news articles.

---

# 🛠️ Tech Stack

## Programming Language

- Python 3.11+

## NLP

- NLTK
- SpaCy
- Word Tokenization
- Stopword Removal
- Lemmatization

## Feature Engineering

- TF-IDF
- Word2Vec
- Readability Features
- Sentiment Polarity
- Article/Statement Length

## Machine Learning

- Scikit-learn
- Logistic Regression
- Support Vector Machine
- Random Forest
- Gradient Boosting
- XGBoost

## Explainable AI

- SHAP

## Deployment

- Flask
- Flask-CORS
- Gradio

## Experiment Tracking

- MLflow

## Data & Visualization

- NumPy
- Pandas
- Matplotlib
- Seaborn

## Development Environment

- Google Colab
- GitHub

---

# 🏗️ Project Architecture

```text
                    Input Text
                        │
                        ▼
              ┌───────────────────┐
              │ Text Preprocessing│
              └───────────────────┘
                        │
        ┌───────────────┼────────────────┐
        │               │                │
        ▼               ▼                ▼
     TF-IDF          Word2Vec       Auxiliary Features
   Unigram+Bigram   Sentence Vector  ┌───────────────┐
                                      │ Readability   │
                                      │ Sentiment     │
                                      │ Text Length   │
                                      └───────────────┘
        │               │                │
        └───────────────┼────────────────┘
                        ▼
              Feature Combination
                        │
                        ▼
             Multiple Classifiers
                        │
                        ▼
               Model Comparison
                        │
                        ▼
                 Best Model
                        │
             ┌──────────┴──────────┐
             │                     │
             ▼                     ▼
        Prediction              SHAP
             │                     │
             ▼                     ▼
      Fake / Real Label      Important Features
             │
             ▼
      Credibility Score
             │
             ▼
        Flask REST API
             │
             ▼
          Gradio UI
```

---

# 🧹 Day 1 — Dataset Setup & EDA ✅

The first day focused on dataset preparation, exploration, duplicate detection, binary label creation, and leakage-free dataset splitting.

## Day 1 Work Completed

- Downloaded the official LIAR dataset.
- Loaded:
  - `train.tsv`
  - `valid.tsv`
  - `test.tsv`
- Verified the dataset structure.
- Inspected all 13 available columns.
- Analyzed original six-class label distribution.
- Combined the official dataset splits.
- Converted six labels into Fake/Real.
- Checked missing values.
- Detected duplicate statements.
- Removed duplicate statements.
- Calculated text statistics.
- Created a fresh stratified 80/10/10 split.
- Verified no overlap between train, validation, and test data.

### Dataset Columns

```text
label
statement
subject
speaker
speaker_job_title
state_info
party_affiliation
barely_true_counts
false_counts
half_true_counts
mostly_true_counts
pants_on_fire_counts
context
```

---

# 📊 Day 1 Results

### Original Dataset

```text
Total samples : 12,791
```

### After Duplicate Removal

```text
Total samples : 12,765
Duplicates    : 0
```

### Final Class Distribution

```text
Fake : 8,263
Real : 4,502
```

### Final Stratified Split

```text
Train      : 10,212
Validation : 1,276
Test       : 1,277
```

### Leakage Check

```text
Train ∩ Validation : 0
Train ∩ Test       : 0
Validation ∩ Test  : 0
```

---

# 🧹 Day 2 — Text Preprocessing ✅

Day 2 focused on building the NLP text preprocessing pipeline.

The preprocessing pipeline was designed to convert raw textual statements into clean normalized text suitable for feature engineering.

---

# 🔄 Text Preprocessing Pipeline

```text
Raw Statement
      │
      ▼
Lowercase Conversion
      │
      ▼
URL Removal
      │
      ▼
Punctuation Removal
      │
      ▼
Tokenization
      │
      ▼
Stopword Removal
      │
      ▼
SpaCy Lemmatization
      │
      ▼
Cleaned Statement
```

---

# 🧠 Preprocessing Steps

## 1. Lowercasing

All text is converted to lowercase.

Example:

```text
"The Economy Is Growing"
```

becomes:

```text
"the economy is growing"
```

---

## 2. URL Removal

URLs are removed from the input text.

```text
http://example.com
www.example.com
https://example.com
```

---

## 3. Punctuation Removal

Punctuation characters are removed to normalize the text.

---

## 4. Tokenization

The text is converted into individual tokens using NLTK.

Example:

```text
"the economy is growing"
```

becomes:

```text
["the", "economy", "is", "growing"]
```

---

## 5. Stopword Removal

Common English stopwords are removed using the NLTK stopword corpus.

Examples:

```text
the
is
a
an
of
to
```

---

## 6. Lemmatization

SpaCy's English language model is used for lemmatization.

Example:

```text
running → run
cars    → car
studies → study
```

---

# 📊 Day 2 Results

The preprocessing pipeline was successfully applied to all **12,765 statements**.

```text
Total processed statements : 12,765
Empty cleaned statements   : 0
Missing cleaned statements : 0
```

### Text Statistics

```text
Average original words : 18.06
Average cleaned words  : 11.25
Minimum cleaned words  : 1
Maximum cleaned words  : 344
```

---

# 💾 Preprocessed Dataset

The processed dataset was saved as:

```text
processed/liar_preprocessed.csv
```

Final shape:

```text
(12,765, 15)
```

The final stratified preprocessed datasets were also saved:

```text
processed/train_preprocessed.csv
processed/validation_preprocessed.csv
processed/test_preprocessed.csv
```

---

# 🔐 Day 2 Leakage & Quality Check

Final preprocessing and split validation produced:

```text
Train ∩ Validation : 0
Train ∩ Test       : 0
Validation ∩ Test  : 0

Missing cleaned text:
Train      : 0
Validation : 0
Test       : 0

Empty cleaned statements:
Train      : 0
Validation : 0
Test       : 0
```

---

# 📓 Completed Notebooks

The completed notebooks are available inside the `notebooks/` directory.

### Day 1

```text
NewsGuard_Day1_Data_Setup_EDA.ipynb
```

### Day 2

```text
NewsGuard_Day2_Text_Preprocessing.ipynb
```

---

# 🔬 Feature Engineering

## Day 3 — TF-IDF ⏳

The next stage will convert cleaned text into numerical features using **TF-IDF**.

Planned configuration:

```text
TF-IDF
├── Unigrams
└── Bigrams
```

The vectorizer will be fitted **only on the training data** to prevent data leakage.

Validation and test data will only be transformed using the fitted training vectorizer.

---

# 🧠 Word2Vec Features

Word2Vec will be used to create dense semantic representations.

The planned pipeline is:

```text
Cleaned Text
     │
     ▼
Word Tokens
     │
     ▼
Word2Vec Embeddings
     │
     ▼
Sentence-Level Vector
```

---

# 📐 Auxiliary Features

Additional numerical features will include:

### Readability

Using TextStat:

- Flesch Reading Ease
- Flesch-Kincaid Grade
- Other relevant readability measures

### Sentiment

Using TextBlob:

```text
Sentiment Polarity
```

### Text Statistics

```text
Statement Length
Word Count
Character Count
```

---

# 🔗 Feature Combination

The final feature representation will combine:

```text
TF-IDF
   +
Word2Vec
   +
Readability
   +
Sentiment
   +
Text Length
```

The implementation will use a sparse-compatible feature combination strategy while following the assignment's FeatureUnion-based architecture.

---

# 🤖 Model Development

The project will compare at least five classification algorithms.

Planned models:

```text
1. Logistic Regression
2. Support Vector Machine
3. Random Forest
4. Gradient Boosting
5. XGBoost
```

The goal is to determine which model provides the best balance of:

- Accuracy
- Precision
- Recall
- F1 Score
- Generalization

---

# 🔁 Cross-Validation

Model evaluation will use **Stratified 5-Fold Cross-Validation**.

```text
Dataset
   │
   ├── Fold 1
   ├── Fold 2
   ├── Fold 3
   ├── Fold 4
   └── Fold 5
          │
          ▼
    Model Evaluation
          │
          ▼
     Mean CV Score
```

Stratification ensures that the Fake/Real class ratio remains approximately consistent across folds.

---

# 🎯 Model Selection

The primary model-selection metric will be:

```text
F1 Score
```

This is important because the dataset is moderately imbalanced.

The project target is:

```text
Held-out Test F1 > 0.88
```

> ⚠️ This is the project target. Final performance will be reported after model training and evaluation.

---

# 🔧 Hyperparameter Tuning

The best-performing models will be tuned using validation/CV-based approaches.

Potential parameters include:

```text
Learning Rate
Regularization
Tree Depth
Number of Estimators
Kernel
C
Gamma
```

Only training data will be used during model fitting and hyperparameter selection to avoid test-set leakage.

---

# 📊 Model Evaluation

Final evaluation will include:

```text
Accuracy
Precision
Recall
F1 Score
Confusion Matrix
Classification Report
ROC-AUC
```

The held-out test set will be used only for final evaluation.

---

# 🔥 SHAP Explainability

NewsGuard will use **SHAP (SHapley Additive exPlanations)** to explain model predictions.

The explainability pipeline will provide:

```text
Input Text
    │
    ▼
Best Model
    │
    ▼
SHAP
    │
    ├── Important Features
    ├── Positive Contributions
    └── Negative Contributions
```

Planned deliverables:

- SHAP summary plot
- 3 individual force/waterfall-style explanations
- Top contributing words/features

---

# 📈 Credibility Score

The system will generate a credibility score from the model's predicted probability for the **Real** class.

Conceptually:

```text
Real Class Probability
          │
          ▼
      × 100
          │
          ▼
Credibility Score
```

Example:

```text
Real Probability : 0.87
Credibility Score: 87
```

The score will be derived from the trained model's probability output and will **not be hardcoded**.

---

# 📡 Flask REST API

The final trained model will be exposed through a Flask REST API.

### Endpoint

```text
POST /predict
```

### Example Request

```json
{
  "text": "The government announced a new economic policy."
}
```

### Planned Response

```json
{
  "label": "Real",
  "credibility_score": 87.4,
  "top_features": [
    {
      "feature": "economic",
      "importance": 0.42
    },
    {
      "feature": "policy",
      "importance": 0.31
    }
  ]
}
```

The API will return valid JSON responses.

Malformed requests will return an appropriate HTTP `400` response.

---

# 🎨 Gradio Interface

A Gradio interface will be created to provide an interactive user experience.

Planned interface:

```text
┌─────────────────────────────────────┐
│         🛡️ NewsGuard                │
├─────────────────────────────────────┤
│ Enter Text                          │
│                                     │
│ [................................]  │
│ [................................]  │
│                                     │
│           [ Analyze ]               │
├─────────────────────────────────────┤
│ Prediction : Real                   │
│ Credibility: 87.4                   │
│                                     │
│ Important Features                  │
│ economic ██████████                 │
│ policy   ███████                    │
└─────────────────────────────────────┘
```

---

# 📊 MLflow Experiment Tracking

MLflow will be used to track:

- Model parameters
- Cross-validation scores
- Validation metrics
- Test metrics
- Feature configurations
- Model artifacts

Planned structure:

```text
Experiment
    │
    ├── Parameters
    ├── Metrics
    ├── Artifacts
    └── Model
```

---

# 🧪 Testing

Automated tests will be created using `pytest`.

Tests will cover:

```text
✓ Text preprocessing
✓ Feature transformation
✓ Model prediction
✓ API response
✓ Invalid API input
✓ Credibility score generation
```

The API will also be tested for malformed input and valid JSON responses.

---

# 📋 Model Card

A final model card will document:

- Model architecture
- Dataset
- Preprocessing
- Feature engineering
- Training procedure
- Evaluation results
- Intended use
- Limitations
- Ethical considerations
- Explainability
- Deployment details

Planned location:

```text
reports/NewsGuard_Model_Card.md
```

---

# 🗓️ 10-Day Development Roadmap

## Day 1 — Dataset Setup & EDA ✅

Completed:

- Dataset download
- Dataset loading
- Dataset inspection
- Six-class label analysis
- Binary Fake/Real mapping
- Duplicate removal
- Text statistics
- Stratified 80/10/10 split
- Leakage verification

---

## Day 2 — Text Preprocessing ✅

Completed:

- NLTK setup
- SpaCy setup
- Lowercasing
- URL removal
- Punctuation removal
- Tokenization
- Stopword removal
- Lemmatization
- Cleaned dataset generation
- Quality checks
- Final preprocessed splits

---

## Day 3 — TF-IDF Features ⏳

Planned:

- Load preprocessed datasets
- Create TF-IDF vectorizer
- Use unigram + bigram features
- Fit only on training data
- Transform validation/test
- Analyze vocabulary
- Save vectorizer
- Save feature artifacts

---

## Day 4 — Auxiliary Features ⏳

Planned:

- Word2Vec
- Sentence vectors
- Readability scores
- Sentiment polarity
- Statement length
- Auxiliary feature matrix
- Feature validation

---

## Day 5 — FeatureUnion + Baseline Models ⏳

Planned:

- Combine TF-IDF and auxiliary features
- Build baseline pipeline
- Train initial classifiers
- Establish baseline metrics

---

## Day 6 — Advanced Models + Tuning ⏳

Planned:

- Train five classification algorithms
- Hyperparameter tuning
- Stratified 5-fold CV
- Compare model performance
- Select promising candidates

---

## Day 7 — Final Evaluation ⏳

Planned:

- Final model selection
- Validation analysis
- Held-out test evaluation
- Accuracy
- Precision
- Recall
- F1
- Confusion matrix
- ROC-AUC

Target:

```text
F1 > 0.88
```

---

## Day 8 — SHAP Explainability ⏳

Planned:

- SHAP integration
- Global feature importance
- SHAP summary plot
- Individual explanations
- 3 explanation visualizations

---

## Day 9 — Flask API + Gradio UI ⏳

Planned:

- Save best model
- Create Flask application
- Implement `/predict`
- JSON request/response
- Error handling
- Credibility score
- SHAP feature importance
- Gradio interface

---

## Day 10 — MLflow + Testing + Documentation ⏳

Planned:

- MLflow tracking
- Pytest test suite
- Model card
- README finalization
- API documentation
- Reproducibility documentation
- Final project demonstration

---

# 📁 Project Structure

```text
NewsGuard_Fake_News_Detection/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   │   └── README.md
│   │
│   └── processed/
│       ├── README.md
│       └── .gitkeep
│
├── notebooks/
│   ├── NewsGuard_Day1_Data_Setup_EDA.ipynb
│   ├── NewsGuard_Day2_Text_Preprocessing.ipynb
│   ├── NewsGuard_Day3_TF_IDF_Features.ipynb
│   ├── NewsGuard_Day4_Auxiliary_Features.ipynb
│   ├── NewsGuard_Day5_FeatureUnion_Baseline_Models.ipynb
│   ├── NewsGuard_Day6_Advanced_Models_Tuning.ipynb
│   ├── NewsGuard_Day7_Final_Evaluation.ipynb
│   ├── NewsGuard_Day8_SHAP_Explainability.ipynb
│   ├── NewsGuard_Day9_Flask_Gradio.ipynb
│   └── NewsGuard_Day10_MLflow_Testing_Documentation.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── features.py
│   ├── models.py
│   └── utils.py
│
├── models/
│   └── README.md
│
├── results/
│   ├── figures/
│   ├── metrics/
│   └── predictions/
│
├── app/
│   ├── app.py
│   └── gradio_app.py
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_features.py
│   ├── test_model.py
│   └── test_api.py
│
├── reports/
│   └── NewsGuard_Model_Card.md
│
└── mlruns/
    └── .gitkeep
```

> Future source files will be added as each development stage is completed.

---

# 📦 Expected Final Deliverables

The completed project is expected to contain:

```text
1. Preprocessed LIAR dataset
2. Stratified train/validation/test split
3. TF-IDF feature pipeline
4. Word2Vec sentence representations
5. Readability features
6. Sentiment features
7. Combined feature representation
8. Five-model comparison
9. Stratified 5-fold CV results
10. Best-performing model
11. Saved model (.joblib)
12. F1 > 0.88 target evaluation
13. SHAP summary visualization
14. Three individual SHAP explanations
15. Flask POST /predict API
16. Credibility scoring system
17. Gradio interface
18. MLflow experiment tracking
19. Automated pytest tests
20. Model Card
21. API documentation
22. Final README
```

---

# 💾 Data & Model Management

Large datasets and trained model artifacts will not be committed directly to GitHub.

The repository will use `.gitignore` to exclude files such as:

```text
*.joblib
*.pkl
*.pickle
*.npy
*.npz
*.zip

data/raw/
data/processed/*.csv

.env
*api_key*
*secret*
*token*
```

This helps prevent:

- Large files from bloating the repository
- Dataset redistribution issues
- Accidental credential exposure
- Generated artifacts being committed unnecessarily

---

# 🔁 Reproducibility

The project is designed to be reproducible.

The workflow will maintain:

```text
Raw Dataset
     │
     ▼
Preprocessing
     │
     ▼
Feature Engineering
     │
     ▼
Model Training
     │
     ▼
Evaluation
     │
     ▼
Explainability
     │
     ▼
Deployment
```

Important reproducibility practices include:

- Fixed random seeds where applicable
- Stratified dataset splitting
- Training-only fitting of learned preprocessing components
- No test-set leakage
- Saved vectorizers
- Saved trained model
- Documented dependencies
- MLflow experiment tracking

---

# ⚠️ Important Data Leakage Prevention

The project follows strict leakage prevention.

### Training Data

Used for:

```text
Vectorizer fitting
Feature learning
Model training
Cross-validation
Hyperparameter selection
```

### Validation Data

Used for:

```text
Model comparison
Validation analysis
Tuning decisions
```

### Test Data

Used only for:

```text
Final performance evaluation
```

The TF-IDF vectorizer and other learned preprocessing components will never be fitted using validation or test data.

---

# ⚠️ Limitations

NewsGuard has several important limitations.

### 1. Dataset Limitation

The LIAR dataset contains short political statements rather than full-length news articles.

Therefore, performance on LIAR should not automatically be interpreted as performance on real-world news articles.

### 2. Binary Mapping Limitation

The Fake/Real labels are created through a project-defined mapping of the original six LIAR labels.

Different mappings could produce different results.

### 3. Domain Limitation

The dataset primarily represents political claims.

The model may not generalize well to:

- Medical misinformation
- Financial misinformation
- Scientific misinformation
- Social media misinformation
- Long-form journalism

### 4. Model Limitation

Machine learning predictions represent statistical patterns learned from training data.

A high credibility score does not guarantee that a statement is factually true.

### 5. Explainability Limitation

SHAP highlights features contributing to the model prediction.

It does not prove that those features represent factual evidence.

---

# ⚕️ Disclaimer

**NewsGuard is an educational and research project.**

It is **not a professional fact-checking system** and should not be used as the sole basis for determining whether real-world information is true or false.

A prediction represents the output of a machine learning model and should be independently verified using reliable sources.

The credibility score is a **model-derived confidence indicator**, not an objective measurement of truth.

---

# 📚 Documentation

The project documentation will include:

### Model Card

```text
reports/NewsGuard_Model_Card.md
```

The model card will contain:

- Intended use
- Dataset
- Preprocessing
- Feature engineering
- Model architecture
- Training
- Evaluation
- Explainability
- Limitations
- Ethical considerations
- Disclaimer

---

# 🔗 Important Links

### 💻 GitHub Repository

https://github.com/SagarSharma131006/NewsGuard_Fake_News_Detection

### 📚 Dataset

LIAR Dataset — University of California, Santa Barbara

### 🧪 Project Notebooks

```text
notebooks/
```

### 📊 Experiment Tracking

```text
MLflow
```

---

# 👨‍💻 Author

**Sagar Sharma**

B.Tech CSE — Artificial Intelligence & Machine Learning

GitHub:

https://github.com/SagarSharma131006

---

# 📋 Current Project Results

| Component | Status |
|---|---|
| Dataset Setup | ✅ Completed |
| Dataset EDA | ✅ Completed |
| Binary Label Mapping | ✅ Completed |
| Duplicate Removal | ✅ Completed |
| Stratified Split | ✅ Completed |
| Leakage Verification | ✅ Completed |
| Text Preprocessing | ✅ Completed |
| TF-IDF | ⏳ Upcoming |
| Word2Vec | ⏳ Upcoming |
| Auxiliary Features | ⏳ Upcoming |
| Feature Combination | ⏳ Upcoming |
| 5-Model Comparison | ⏳ Upcoming |
| 5-Fold CV | ⏳ Upcoming |
| Hyperparameter Tuning | ⏳ Upcoming |
| Final Evaluation | ⏳ Upcoming |
| SHAP | ⏳ Upcoming |
| Credibility Score | ⏳ Upcoming |
| Flask API | ⏳ Upcoming |
| Gradio UI | ⏳ Upcoming |
| MLflow | ⏳ Upcoming |
| Pytest | ⏳ Upcoming |
| Model Card | ⏳ Upcoming |

---

# ⭐ Project Philosophy

NewsGuard follows a simple principle:

```text
Clean the Data
      ↓
Engineer Better Features
      ↓
Compare Multiple Models
      ↓
Evaluate Honestly
      ↓
Explain Predictions
      ↓
Deploy Responsibly
```

The goal is not just to build a classifier, but to build a **reproducible, explainable, and deployable NLP machine learning system**.

---

# 🛡️ NewsGuard

**From Raw Text → NLP → Feature Engineering → Machine Learning → Explainable AI → API → Interactive UI**

🚀 **Built with Python | Powered by NLP | Explained with SHAP | Deployed with Flask & Gradio**
