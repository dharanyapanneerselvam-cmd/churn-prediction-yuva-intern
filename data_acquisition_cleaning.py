"""
Data Acquisition & Cleaning Module
Week 2 Task - Yuva Intern (Data Science Project Coordinator)
Author: Dharanya

Implements the data acquisition and preprocessing workflow described in the
Week 2 Data Acquisition & Management Plan.
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


# ---------------------------------------------------------------------------
# Step 0: Data Acquisition
# ---------------------------------------------------------------------------
def acquire_data(source_path: str) -> pd.DataFrame:
    """Load raw data from a CSV source (public dataset or internal export)."""
    df = pd.read_csv(source_path)
    print(f"Acquired {df.shape[0]} rows, {df.shape[1]} columns from {source_path}")
    return df


# ---------------------------------------------------------------------------
# Step 1: Missing Value Handling
# ---------------------------------------------------------------------------
def handle_missing_values(df: pd.DataFrame, threshold: float = 0.4) -> pd.DataFrame:
    """Drop columns with excessive missingness, impute the rest with median/mode."""
    df = df.loc[:, df.isnull().mean() < threshold]
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown")
    return df


# ---------------------------------------------------------------------------
# Step 2: Duplicate Removal
# ---------------------------------------------------------------------------
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = df.shape[0]
    df = df.drop_duplicates()
    print(f"Removed {before - df.shape[0]} duplicate rows")
    return df


# ---------------------------------------------------------------------------
# Step 3: Data Type Correction
# ---------------------------------------------------------------------------
def correct_dtypes(df: pd.DataFrame, date_cols=None, numeric_cols=None) -> pd.DataFrame:
    date_cols = date_cols or []
    numeric_cols = numeric_cols or []
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Step 4: Outlier Detection (Z-score based)
# ---------------------------------------------------------------------------
def remove_outliers(df: pd.DataFrame, numeric_cols: list, z_thresh: float = 3.0) -> pd.DataFrame:
    for col in numeric_cols:
        if col in df.columns:
            z_scores = np.abs(stats.zscore(df[col].fillna(df[col].median())))
            df = df[z_scores < z_thresh]
    return df


# ---------------------------------------------------------------------------
# Step 5: Categorical Encoding
# ---------------------------------------------------------------------------
def encode_categoricals(df: pd.DataFrame, cat_cols: list) -> pd.DataFrame:
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    return df


# ---------------------------------------------------------------------------
# Step 6: Feature Scaling
# ---------------------------------------------------------------------------
def scale_features(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def run_cleaning_pipeline(source_path: str, date_cols, numeric_cols, cat_cols) -> pd.DataFrame:
    df = acquire_data(source_path)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = correct_dtypes(df, date_cols=date_cols, numeric_cols=numeric_cols)
    df = remove_outliers(df, numeric_cols=numeric_cols)
    df = encode_categoricals(df, cat_cols=cat_cols)
    df = scale_features(df, numeric_cols=numeric_cols)
    print("Data cleaning pipeline complete. Final shape:", df.shape)
    return df


if __name__ == "__main__":
    cleaned = run_cleaning_pipeline(
        source_path="customer_data.csv",
        date_cols=["SignupDate"],
        numeric_cols=["MonthlyCharges", "TotalCharges", "Tenure"],
        cat_cols=["Gender", "ContractType", "PaymentMethod"],
    )
    cleaned.to_csv("cleaned_customer_data.csv", index=False)
    print("Saved cleaned_customer_data.csv")
