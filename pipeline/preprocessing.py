"""Preprocessing utilities for numerical ML features."""
import pandas as pd
from sklearn.preprocessing import StandardScaler


def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            median = df[col].median()
            df[col] = df[col].fillna(0 if pd.isna(median) else median)
        else:
            mode = df[col].mode(dropna=True)
            df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else "Unknown")
    return df


def prepare_features(df: pd.DataFrame, feature_cols: list):
    if not feature_cols:
        raise ValueError("Please select at least one numerical feature.")
    X = df[feature_cols].copy()
    for col in feature_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        median = X[col].median()
        X[col] = X[col].fillna(0 if pd.isna(median) else median)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler
