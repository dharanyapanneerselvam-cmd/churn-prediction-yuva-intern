"""
Basic unit tests for the model training/validation module.
Run with: pytest test_model_training.py
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from model_training_validation import (
    get_candidate_models,
    train_and_compare,
    validate_model,
    select_best_model,
)


def _sample_data():
    X, y = make_classification(n_samples=200, n_features=6, random_state=42)
    return pd.DataFrame(X), pd.Series(y)


def test_candidate_models_returned():
    models = get_candidate_models()
    assert "LogisticRegression" in models
    assert "RandomForest" in models


def test_train_and_compare_produces_scores():
    X, y = _sample_data()
    models = get_candidate_models()
    results = train_and_compare(models, X, y, cv=3)
    for name, res in results.items():
        assert 0.0 <= res["cv_f1_mean"] <= 1.0


def test_select_best_model_returns_valid_choice():
    X, y = _sample_data()
    models = get_candidate_models()
    results = train_and_compare(models, X, y, cv=3)
    best_name, best_model = select_best_model(results)
    assert best_name in models


def test_validate_model_metrics_in_range():
    X, y = _sample_data()
    models = get_candidate_models()
    results = train_and_compare(models, X, y, cv=3)
    _, best_model = select_best_model(results)
    metrics = validate_model(best_model, X, y)
    for v in metrics.values():
        assert 0.0 <= v <= 1.0
