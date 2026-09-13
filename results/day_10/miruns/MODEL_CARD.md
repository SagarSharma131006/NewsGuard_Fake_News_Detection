# NewsGuard — Model Card

## 1. Model Overview

**Project:** NewsGuard — Fake News Detection & Credibility Scoring System

**Task:** Binary fake-news classification with credibility scoring and SHAP-based explainability.

**Primary Model:** XGBoost Classifier

**Feature Space:** 615 features
- 500 selected TF-IDF features
- 15 auxiliary NLP features
- 100 Word2Vec features

**Training Data:** Kaggle Fake and Real News Dataset

---

## 2. Intended Use

NewsGuard is intended as an educational and research-oriented system for:

- Fake-news classification experiments
- NLP feature engineering
- Model comparison
- Explainable AI demonstrations
- Credibility-score visualization
- REST API and UI deployment demonstrations

The credibility score represents the model's estimated probability
of the article belonging to the Real class, mapped to a 0–100 scale.

**This score is not a factual truth score and should not be treated
as a definitive assessment of journalistic credibility.**

---

## 3. Training Data

After preprocessing and duplicate removal:

- Total articles: **39,103**
- Training: **31,282**
- Validation: **3,910**
- Test: **3,911**

Class distribution was preserved using stratified splitting.

The original `subject` and `date` columns were removed because
subject contained strong label-correlated source/category information.

---

## 4. Model Performance

### Held-out Test Set

| Metric | Score |
|---|---:|
| Accuracy | 0.9972 |
| Precision | 0.9958 |
| Recall | 0.9991 |
| F1 Score | 0.9974 |
| ROC-AUC | 0.9998 |

### Cross-Validation

Best XGBoost CV:

- F1: **0.997701**
- ROC-AUC: **0.999954**

The required target of F1 > 0.88 was achieved on the internal
held-out test set.

---

## 5. Model Comparison

| Model | CV F1 |
|---|---:|
| XGBoost | 0.997701 |
| Linear SVM | 0.996700 |
| Gradient Boosting | 0.996174 |
| Logistic Regression | 0.993899 |
| Random Forest | 0.991537 |
| Naive Bayes | 0.957115 |

---

## 6. Explainability

SHAP TreeExplainer is used to explain XGBoost predictions.

The explanation pipeline provides:

- Global feature importance
- Local prediction explanations
- Top contributing features
- SHAP force plots

Example influential features observed during analysis include:

- `read more`
- `reuters`
- `featured image`
- `century wire`
- `washington reuters`
- `said`
- `nov`
- `getty`
- Word2Vec dimensions

These features demonstrate that the model can learn publication,
formatting, and source-related patterns in addition to semantic
content.

---

## 7. External Validation

The model was additionally evaluated on a balanced 2,000-article
sample from the WELFake dataset.

| Metric | Internal Test | WELFake |
|---|---:|---:|
| Accuracy | 0.997187 | 0.791500 |
| Precision | 0.995769 | 0.983416 |
| Recall | 0.999057 | 0.593000 |
| F1 | 0.997410 | 0.739863 |
| ROC-AUC | 0.999780 | 0.928321 |

The significant performance drop demonstrates domain and dataset
shift.

Therefore, the model should **not** be considered a universal
fake-news detector.

---

## 8. Limitations

1. The training dataset contains source and formatting patterns that
   may be correlated with labels.

2. SHAP analysis identified source-related terms such as Reuters,
   Getty, and article-format tokens as influential features.

3. Performance on external WELFake data is substantially lower than
   internal test performance.

4. The credibility score represents model probability, not factual
   verification.

5. The model should not be used as the sole decision-maker for
   journalism, moderation, legal decisions, public policy, or
   high-impact decisions.

6. Real-world deployment requires continuous monitoring, new data,
   calibration, and domain-specific validation.

7. Cross-validation stages involving preprocessing/feature selection
   should be interpreted with care because some transformations were
   prepared before CV rather than fully nested inside each fold.

---

## 9. Ethical Considerations

News classification models can amplify biases present in their
training data.

Users should verify important claims using reliable primary sources
and independent fact-checking organizations.

The system should be presented as an AI-assisted screening tool,
not as an automated fact-checking authority.

---

## 10. Reproducibility

The project maintains:

- Dataset preprocessing notebooks
- Feature engineering notebooks
- Model training notebooks
- Evaluation results
- SHAP explanations
- API implementation
- Deployment metadata
- External validation results

Random seeds and saved model artifacts are used where applicable.

---

## 11. Current Status

**Internal benchmark:** PASS

**External validation:** COMPLETE

**Deployment demo:** READY

**Universal fake-news detection claim:** NOT RECOMMENDED

**Project:** NewsGuard
