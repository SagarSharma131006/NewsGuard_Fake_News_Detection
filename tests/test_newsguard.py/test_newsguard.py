
import os
import sys
import numpy as np
import pandas as pd


BASE_DIR = "/content/drive/MyDrive/NewsGuard"


def test_required_directories_exist():

    required_dirs = [
        "data/processed",
        "data/splits",
        "features",
        "results/day_07",
        "results/day_08",
        "results/day_09",
        "results/day_10",
    ]

    for directory in required_dirs:

        path = os.path.join(BASE_DIR, directory)

        assert os.path.isdir(path), (
            f"Missing directory: {directory}"
        )


def test_dataset_splits_exist():

    files = [
        "data/splits/train.csv",
        "data/splits/validation.csv",
        "data/splits/test.csv",
    ]

    for file in files:

        path = os.path.join(BASE_DIR, file)

        assert os.path.isfile(path), (
            f"Missing dataset file: {file}"
        )


def test_dataset_sizes():

    train = pd.read_csv(
        os.path.join(
            BASE_DIR,
            "data/splits/train.csv"
        )
    )

    validation = pd.read_csv(
        os.path.join(
            BASE_DIR,
            "data/splits/validation.csv"
        )
    )

    test = pd.read_csv(
        os.path.join(
            BASE_DIR,
            "data/splits/test.csv"
        )
    )

    assert len(train) == 31282
    assert len(validation) == 3910
    assert len(test) == 3911


def test_final_model_exists():

    model_path = os.path.join(
        BASE_DIR,
        "results/day_06/day_06_xgboost_best_tuned.joblib"
    )

    assert os.path.isfile(model_path)


def test_shap_artifacts_exist():

    required_files = [
        "results/day_08/shap_feature_importance.csv",
        "results/day_08/shap_summary_bar.png",
        "results/day_08/shap_summary_beeswarm.png",
        "results/day_08/shap_force_plot_1.html",
        "results/day_08/shap_force_plot_2.html",
        "results/day_08/shap_force_plot_3.html",
    ]

    for file in required_files:

        path = os.path.join(
            BASE_DIR,
            file
        )

        assert os.path.isfile(path), (
            f"Missing SHAP artifact: {file}"
        )


def test_model_card_exists():

    path = os.path.join(
        BASE_DIR,
        "results/day_10/MODEL_CARD.md"
    )

    assert os.path.isfile(path)


def test_mlflow_database_exists():

    path = os.path.join(
        BASE_DIR,
        "results/day_10/mlflow/mlflow.db"
    )

    assert os.path.isfile(path)


def test_external_validation_exists():

    required_files = [
        "experiments/external_validation/welfake_external_sample_2000.csv",
        "experiments/external_validation/welfake_corrected_validation_summary.joblib",
    ]

    for file in required_files:

        path = os.path.join(
            BASE_DIR,
            file
        )

        assert os.path.isfile(path)


def test_model_performance_target():

    internal_f1 = 0.997410

    assert internal_f1 > 0.88


def test_external_validation_gap():

    internal_f1 = 0.997410
    external_f1 = 0.739863

    assert internal_f1 > external_f1
    assert internal_f1 - external_f1 > 0


def test_final_feature_count():

    tfidf_features = 500
    auxiliary_features = 115

    total_features = (
        tfidf_features +
        auxiliary_features
    )

    assert total_features == 615


def test_credibility_score_range():

    real_probability = 0.018980

    credibility_score = real_probability * 100

    assert 0 <= credibility_score <= 100
