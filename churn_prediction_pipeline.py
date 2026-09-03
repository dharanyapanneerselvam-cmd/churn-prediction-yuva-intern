"""
Customer Churn Prediction System
Week 1 Task - Yuva Intern (Science Project Coordinator)
Author: Dharanya

This script accompanies the Week 1 Project Plan & Strategy document.
It sets up the technical skeleton for the churn-prediction pipeline
described in Section 3 (Project Phases & Python-Based Methodology).
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Phase 2: Data Strategy - Data Loading & Cleaning
# ---------------------------------------------------------------------------
def load_data(csv_path: str) -> pd.DataFrame:
    """Load raw customer data from CSV into a DataFrame."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: handle missing values and duplicate rows."""
    df = df.drop_duplicates()
    df = df.fillna(df.median(numeric_only=True))
    return df


# ---------------------------------------------------------------------------
# Phase 3: Technical Planning - Feature Engineering
# ---------------------------------------------------------------------------
def encode_categoricals(df: pd.DataFrame, cat_cols: list) -> pd.DataFrame:
    """Label-encode categorical columns (e.g., contract type, gender)."""
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    return df


def build_pipeline(df: pd.DataFrame, target_col: str):
    """Split data and scale features for model training."""
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


# ---------------------------------------------------------------------------
# Phase 3: Model Training & Evaluation
# ---------------------------------------------------------------------------
def train_model(X_train, y_train):
    """Train a Random Forest classifier as the churn-prediction model."""
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model and print key metrics."""
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    return y_pred


def plot_feature_importance(model, feature_names):
    """Visualize feature importance for stakeholder reporting."""
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1]

    plt.figure(figsize=(8, 5))
    plt.title("Feature Importance - Churn Prediction Model")
    plt.bar(range(len(importances)), importances[idx], align="center")
    plt.xticks(range(len(importances)), np.array(feature_names)[idx], rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    print("Saved feature_importance.png")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Replace with the actual dataset path once real/sample data is available
    DATA_PATH = "customer_data.csv"
    TARGET_COLUMN = "Churn"
    CATEGORICAL_COLUMNS = ["Gender", "ContractType", "PaymentMethod"]

    df = load_data(DATA_PATH)
    df = clean_data(df)
    df = encode_categoricals(df, CATEGORICAL_COLUMNS)

    X_train, X_test, y_train, y_test = build_pipeline(df, TARGET_COLUMN)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    plot_feature_importance(model, df.drop(columns=[TARGET_COLUMN]).columns)
