# 🛡️ NewsGuard — Fake News Detection & Credibility Scoring

## 📌 Project Status
**Day 1 / 10 — Dataset Setup & Exploratory Data Analysis**

## 🎯 Objective
NewsGuard is an end-to-end NLP system for detecting fake news and generating a credibility score.

The project will:
- Preprocess news text
- Extract TF-IDF features
- Generate Word2Vec embeddings
- Calculate readability and sentiment features
- Compare multiple machine learning classifiers
- Explain predictions using SHAP
- Deploy the best model through a Flask REST API
- Provide an interactive Gradio interface

## 📊 Dataset

### LIAR Dataset
The project uses the official LIAR dataset.

Original dataset:
- Train: 10,240
- Validation: 1,284
- Test: 1,267
- Total: 12,791

After removing duplicate statements:
- Total samples: 12,765
- Fake: 8,263
- Real: 4,502

### Binary Label Mapping

For this project, the original six LIAR labels are converted into binary classes:

**Fake**
- pants-fire
- false
- barely-true
- half-true

**Real**
- mostly-true
- true

This binary mapping is a project-level design choice.

## 🏗️ Planned Pipeline

```text
Raw LIAR Dataset
       ↓
Data Cleaning
       ↓
Text Preprocessing
       ↓
Feature Engineering
       ├── TF-IDF
       ├── Word2Vec
       ├── Readability
       ├── Sentiment
       └── Article Length
       ↓
FeatureUnion
       ↓
Multiple ML Models
       ↓
5-Fold Cross Validation
       ↓
Best Model
       ↓
SHAP Explainability
       ↓
Credibility Score
       ↓
Flask REST API
       ↓
Gradio UI
