# train_local_pipeline.py

"""
Offline local training script for CRYPTO AI.

Run this file once before using Streamlit live mode:

    python train_local_pipeline.py

What it does:
1. Loads dataset from data/raw/dataset.csv or data/raw/dataset.xlsx
2. Expands supported encoded columns into cipher rows
3. Generates cipher numerical features
4. Runs DBSCAN
5. Runs FCM
6. Generates pseudo/risk/cipher labels
7. Trains SVM-RBF locally
8. Saves trained artifacts to artifacts/
9. Streamlit later loads these artifacts instead of retraining SVM every run

Important:
- This is OFFLINE TRAINING.
- Streamlit should use LIVE INFERENCE after artifacts are saved.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from pipeline.ingestion import load_data
from pipeline.preprocessing import clean_missing_values, prepare_features
from pipeline.cipher_feature_engineering import add_cipher_features, expand_encoded_dataset
from pipeline.dbscan_clustering import apply_dbscan
from pipeline.fcm_clustering import apply_fcm
from pipeline.label_generator import generate_risk_labels
from pipeline.svm_classifier import train_svm_classifier, predict_with_confidence
from pipeline.model_persistence import save_training_artifacts

try:
    from utils.config import SUPPORTED_CIPHER_TYPES
except Exception:
    SUPPORTED_CIPHER_TYPES = ["Base64", "Base32", "Morse", "Hex", "ASCII"]


# ==============================
# Optional Training Augmentation
# ==============================

def encode_hex_text(text: str) -> str:
    """Encode plain text as uppercase hexadecimal text for training augmentation."""

    return str(text or "").encode("utf-8", errors="ignore").hex().upper()


def encode_ascii_decimal_text(text: str) -> str:
    """Encode plain text as decimal ASCII codes separated by spaces."""

    return " ".join(str(ord(ch)) for ch in str(text or ""))


def add_missing_hex_ascii_training_columns(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Adds Hex and ASCII encoded columns from plain_text when the dataset does not
    already provide them.

    Why this is needed:
    - The decoder can decode Hex/ASCII once the type is known.
    - The SVM can only learn Hex/ASCII if the offline training data contains
      examples for these classes.
    - If the original dataset has plain_text but no hex/ascii columns, we create
      deterministic training examples from the same plain text.
    """

    df = raw_df.copy()
    generated_columns: list[str] = []

    if "plain_text" not in df.columns:
        return df, generated_columns

    plain_series = df["plain_text"].fillna("").astype(str)
    valid_mask = plain_series.str.strip().ne("")

    if not valid_mask.any():
        return df, generated_columns

    lower_cols = {str(col).lower() for col in df.columns}

    if "hex" in SUPPORTED_CIPHER_TYPES and not any(
        col in lower_cols for col in {"hex", "hex_encoded", "hexadecimal", "hexadecimal_encoded"}
    ):
        df["hex_encoded"] = ""
        df.loc[valid_mask, "hex_encoded"] = plain_series.loc[valid_mask].map(encode_hex_text)
        generated_columns.append("hex_encoded")

    if "ASCII" in SUPPORTED_CIPHER_TYPES and not any(
        col in lower_cols for col in {"ascii", "ascii_encoded", "ascii_decimal", "ascii_decimal_encoded"}
    ):
        df["ascii_encoded"] = ""
        df.loc[valid_mask, "ascii_encoded"] = plain_series.loc[valid_mask].map(encode_ascii_decimal_text)
        generated_columns.append("ascii_encoded")

    return df, generated_columns


# ==============================
# Feature Selection
# ==============================

PREFERRED_FEATURES = [
    # General shape/statistical features
    "cypher_length",
    "cypher_entropy",
    "cypher_digit_count",
    "cypher_alpha_count",
    "cypher_upper_count",
    "cypher_lower_count",
    "cypher_symbol_count",
    "cypher_space_count",
    "cypher_equal_count",
    "cypher_plus_slash_count",
    "cypher_dot_dash_count",

    # Original cipher-family ratios
    "cypher_digit_ratio",
    "cypher_alpha_ratio",
    "cypher_upper_ratio",
    "cypher_lower_ratio",
    "cypher_symbol_ratio",
    "cypher_space_ratio",
    "cypher_base64_ratio",
    "cypher_base32_ratio",
    "cypher_morse_ratio",
    "cypher_dot_dash_ratio",

    # Added families: Hex / ASCII decimal-code features
    "cypher_hex_char_count",
    "cypher_hex_ratio",
    "cypher_hex_separator_count",
    "cypher_ascii_token_count",
    "cypher_ascii_valid_token_count",
    "cypher_ascii_valid_token_ratio",
    "cypher_ascii_separator_count",
    "cypher_is_likely_hex_shape",
    "cypher_is_likely_ascii_shape",
]


def select_training_columns(cleaned_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Selects feature columns for DBSCAN, FCM, SVM, and SHAP.

    This mirrors the live pipeline selection logic.
    """

    numeric_cols = cleaned_df.select_dtypes(include=["number"]).columns.tolist()

    feature_cols = [
        col for col in PREFERRED_FEATURES
        if col in numeric_cols
    ]

    if not feature_cols:
        feature_cols = [
            col for col in numeric_cols
            if col.startswith("cypher_")
        ]

    if not feature_cols:
        feature_cols = numeric_cols[: min(10, len(numeric_cols))]

    if not feature_cols:
        raise ValueError(
            "No numeric feature columns were found after feature engineering."
        )

    risk_feature = (
        "cypher_entropy"
        if "cypher_entropy" in numeric_cols
        else feature_cols[0]
    )

    forecast_target = (
        "cypher_entropy"
        if "cypher_entropy" in numeric_cols
        else feature_cols[0]
    )

    date_col = "record_date" if "record_date" in cleaned_df.columns else None

    if date_col is None:
        for col in cleaned_df.columns:
            if any(token in str(col).lower() for token in ("date", "time", "timestamp")):
                date_col = col
                break

    if date_col is None:
        cleaned_df["record_date"] = pd.date_range(
            start="2024-01-01",
            periods=len(cleaned_df),
            freq="D",
        )
        date_col = "record_date"

    return {
        "numeric_cols": numeric_cols,
        "feature_cols": feature_cols,
        "risk_feature": risk_feature,
        "forecast_target": forecast_target,
        "date_col": date_col,
    }


def build_training_mask(labeled_df: pd.DataFrame) -> pd.Series:
    """
    Selects rows used to train the SVM.

    Preferred:
    - train on known source cipher rows: Base64, Base32, Morse

    Fallback:
    - train on all labeled rows if known rows are too few
    """

    if "source_cipher_type" in labeled_df.columns:
        train_mask = labeled_df["source_cipher_type"].isin(SUPPORTED_CIPHER_TYPES)
    else:
        train_mask = pd.Series(
            [True] * len(labeled_df),
            index=labeled_df.index,
        )

    if int(train_mask.sum()) < 10:
        train_mask = pd.Series(
            [True] * len(labeled_df),
            index=labeled_df.index,
        )

    return train_mask


def make_shap_background_df(
    labeled_df: pd.DataFrame,
    feature_cols: list[str],
    max_rows: int = 80,
) -> pd.DataFrame:
    """
    Creates a small SHAP background sample.

    SHAP can be slow, so we save a compact representative sample.
    """

    available_features = [
        col for col in feature_cols
        if col in labeled_df.columns
    ]

    if not available_features:
        raise ValueError("Cannot create SHAP background: no feature columns found.")

    if len(labeled_df) <= max_rows:
        return labeled_df[available_features].copy()

    # Balanced sample if labels exist.
    if "risk_label" in labeled_df.columns:
        try:
            sampled = (
                labeled_df
                .groupby("risk_label", group_keys=False)
                .apply(
                    lambda group: group.sample(
                        n=min(len(group), max(5, max_rows // max(1, labeled_df["risk_label"].nunique()))),
                        random_state=42,
                    )
                )
            )

            if len(sampled) > max_rows:
                sampled = sampled.sample(n=max_rows, random_state=42)

            return sampled[available_features].copy()

        except Exception:
            pass

    return labeled_df.sample(
        n=min(max_rows, len(labeled_df)),
        random_state=42,
    )[available_features].copy()


# ==============================
# Training Pipeline
# ==============================

def train_and_save_local_models(
    max_rows_per_type: int = 700,
    shap_background_rows: int = 80,
) -> Dict[str, Any]:
    """
    Runs the complete offline training pipeline and saves artifacts.
    """

    print("=" * 80)
    print("CRYPTO AI — OFFLINE LOCAL TRAINING")
    print("=" * 80)

    # ------------------------------
    # 1. Load dataset
    # ------------------------------
    print("\n[1/9] Loading dataset...")

    raw_df = load_data(None)

    print(f"Loaded dataset shape: {raw_df.shape}")

    # If the dataset has plain_text but no Hex/ASCII columns, create deterministic
    # Hex/ASCII training examples so the SVM can learn the new cipher classes.
    raw_df, generated_training_columns = add_missing_hex_ascii_training_columns(raw_df)

    if generated_training_columns:
        print("Generated additional training columns:")
        for col in generated_training_columns:
            print(f"  - {col}")

    # ------------------------------
    # 2. Expand dataset
    # ------------------------------
    print("\n[2/9] Expanding encoded dataset...")

    expanded_df = expand_encoded_dataset(
        raw_df,
        max_rows_per_type=max_rows_per_type,
    )

    print(f"Expanded dataset shape: {expanded_df.shape}")

    if "source_cipher_type" in expanded_df.columns:
        print("Source cipher counts:")
        print(expanded_df["source_cipher_type"].value_counts())

    # ------------------------------
    # 3. Clean + feature engineering
    # ------------------------------
    print("\n[3/9] Cleaning data and generating cipher features...")

    cleaned_df = clean_missing_values(expanded_df)
    cleaned_df = add_cipher_features(cleaned_df)

    print(f"Feature-engineered dataset shape: {cleaned_df.shape}")

    # ------------------------------
    # 4. Select features + scale
    # ------------------------------
    print("\n[4/9] Selecting and scaling features...")

    selection = select_training_columns(cleaned_df)

    feature_cols = selection["feature_cols"]
    risk_feature = selection["risk_feature"]
    forecast_target = selection["forecast_target"]
    date_col = selection["date_col"]

    X_scaled, scaler = prepare_features(cleaned_df, feature_cols)

    print(f"Selected {len(feature_cols)} features:")
    for feature in feature_cols:
        print(f"  - {feature}")

    print(f"Risk feature: {risk_feature}")
    print(f"Forecast target: {forecast_target}")
    print(f"Date column: {date_col}")
    print(f"X_scaled shape: {X_scaled.shape}")

    # ------------------------------
    # 5. DBSCAN
    # ------------------------------
    print("\n[5/9] Running DBSCAN clustering...")

    clustered_df, dbscan_model = apply_dbscan(cleaned_df, X_scaled)

    print("DBSCAN cluster counts:")
    print(clustered_df["dbscan_cluster"].value_counts())

    # ------------------------------
    # 6. FCM
    # ------------------------------
    print("\n[6/9] Running FCM clustering...")

    clustered_df, fcm_info = apply_fcm(clustered_df, X_scaled)

    fpc = round(float(fcm_info.get("fpc", 0.0)), 4)

    print("FCM cluster counts:")
    print(clustered_df["fcm_cluster"].value_counts())
    print(f"FCM FPC score: {fpc}")
    print(f"FCM engine: {fcm_info.get('engine', 'unknown')}")

    # ------------------------------
    # 7. Label generation
    # ------------------------------
    print("\n[7/9] Generating pseudo/risk labels...")

    labeled_df = generate_risk_labels(clustered_df, risk_feature)

    # Force known source cipher rows to keep their true cipher label.
    # This is important for added deterministic families such as Hex and ASCII,
    # especially if label_generator.py was trained originally around Base64/Base32/Morse.
    if "source_cipher_type" in labeled_df.columns:
        known_source_mask = labeled_df["source_cipher_type"].isin(SUPPORTED_CIPHER_TYPES)
        labeled_df.loc[known_source_mask, "cipher_type_label"] = labeled_df.loc[
            known_source_mask,
            "source_cipher_type",
        ]
        labeled_df.loc[known_source_mask, "risk_label"] = labeled_df.loc[
            known_source_mask,
            "source_cipher_type",
        ]

    print("Risk/cipher label counts:")
    print(labeled_df["risk_label"].value_counts())

    # ------------------------------
    # 8. Train SVM-RBF
    # ------------------------------
    print("\n[8/9] Training SVM-RBF classifier...")

    train_mask = build_training_mask(labeled_df)

    y_train = labeled_df.loc[train_mask, "risk_label"]

    if y_train.nunique() < 2:
        raise ValueError(
            "SVM training requires at least two classes. "
            f"Detected classes: {y_train.unique().tolist()}"
        )

    svm_model, svm_metrics = train_svm_classifier(
        X_scaled[train_mask.to_numpy()],
        y_train,
    )

    predictions, confidence_scores = predict_with_confidence(
        svm_model,
        X_scaled,
    )

    labeled_df["svm_predicted_cipher_type"] = predictions
    labeled_df["svm_prediction"] = predictions
    labeled_df["svm_confidence"] = confidence_scores

    print(f"SVM accuracy: {round(float(svm_metrics.get('accuracy', 0.0)), 4)}")
    print(f"SVM classes: {svm_metrics.get('classes')}")
    print("SVM prediction counts:")
    print(labeled_df["svm_predicted_cipher_type"].value_counts())

    # ------------------------------
    # 9. Save artifacts
    # ------------------------------
    print("\n[9/9] Saving artifacts...")

    shap_background_df = make_shap_background_df(
        labeled_df=labeled_df,
        feature_cols=feature_cols,
        max_rows=shap_background_rows,
    )

    metadata = {
        "training_script": "train_local_pipeline.py",
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "supported_cipher_types": SUPPORTED_CIPHER_TYPES,
        "generated_training_columns": generated_training_columns,
        "source_cipher_counts": (
            expanded_df["source_cipher_type"].value_counts().to_dict()
            if "source_cipher_type" in expanded_df.columns
            else {}
        ),
        "label_counts": labeled_df["risk_label"].value_counts().to_dict(),
        "feature_cols": feature_cols,
        "max_rows_per_type": max_rows_per_type,
        "shap_background_rows": int(len(shap_background_df)),
        "raw_dataset_shape": list(raw_df.shape),
        "expanded_dataset_shape": list(expanded_df.shape),
        "cleaned_dataset_shape": list(cleaned_df.shape),
        "training_rows_used_for_svm": int(train_mask.sum()),
        "svm_accuracy": float(svm_metrics.get("accuracy", 0.0)),
        "svm_classes": svm_metrics.get("classes"),
        "fcm_fpc": fpc,
        "fcm_engine": fcm_info.get("engine", "unknown"),
        "risk_feature": risk_feature,
        "forecast_target": forecast_target,
        "date_col": date_col,
    }

    save_result = save_training_artifacts(
        scaler=scaler,
        svm_model=svm_model,
        dbscan_model=dbscan_model,
        fcm_info=fcm_info,
        feature_cols=feature_cols,
        labeled_df=labeled_df,
        shap_background_df=shap_background_df,
        metadata=metadata,
    )

    print("\nArtifacts saved successfully:")
    for name, path in save_result["saved_files"].items():
        print(f"  - {name}: {path}")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print("You can now run Streamlit live inference:")
    print("    streamlit run app.py")
    print("=" * 80)

    return {
        "raw_df": raw_df,
        "expanded_df": expanded_df,
        "cleaned_df": cleaned_df,
        "labeled_df": labeled_df,
        "feature_cols": feature_cols,
        "selection": selection,
        "scaler": scaler,
        "dbscan_model": dbscan_model,
        "fcm_info": fcm_info,
        "svm_model": svm_model,
        "svm_metrics": svm_metrics,
        "shap_background_df": shap_background_df,
        "save_result": save_result,
        "metadata": metadata,
    }


# ==============================
# CLI
# ==============================

def main():
    parser = argparse.ArgumentParser(
        description="Train CRYPTO AI models locally and save artifacts."
    )

    parser.add_argument(
        "--max-rows-per-type",
        type=int,
        default=700,
        help="Maximum training rows sampled per cipher type during dataset expansion.",
    )

    parser.add_argument(
        "--shap-background-rows",
        type=int,
        default=80,
        help="Number of rows saved for SHAP background.",
    )

    args = parser.parse_args()

    train_and_save_local_models(
        max_rows_per_type=args.max_rows_per_type,
        shap_background_rows=args.shap_background_rows,
    )


if __name__ == "__main__":
    main()