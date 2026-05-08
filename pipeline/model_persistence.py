# pipeline/model_persistence.py

"""
model_persistence.py

This file handles saving and loading trained CRYPTO AI artifacts.

Goal:
- Train models locally once.
- Save trained artifacts to disk.
- Streamlit/live mode loads saved artifacts instead of retraining SVM every run.

Saved artifacts:
- scaler.joblib
- svm_model.joblib
- dbscan_model.joblib
- fcm_info.pkl
- feature_cols.json
- training_metadata.json
- shap_background.csv
- training_labeled_df.csv
"""

from __future__ import annotations

import json
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd


ARTIFACTS_DIR = Path("artifacts")

SCALER_PATH = ARTIFACTS_DIR / "scaler.joblib"
SVM_MODEL_PATH = ARTIFACTS_DIR / "svm_model.joblib"
DBSCAN_MODEL_PATH = ARTIFACTS_DIR / "dbscan_model.joblib"
FCM_INFO_PATH = ARTIFACTS_DIR / "fcm_info.pkl"
FEATURE_COLS_PATH = ARTIFACTS_DIR / "feature_cols.json"
TRAINING_METADATA_PATH = ARTIFACTS_DIR / "training_metadata.json"
SHAP_BACKGROUND_PATH = ARTIFACTS_DIR / "shap_background.csv"
TRAINING_LABELED_DF_PATH = ARTIFACTS_DIR / "training_labeled_df.csv"


def ensure_artifacts_dir() -> None:
    """
    Creates artifacts/ folder if it does not exist.
    """

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, data: Dict[str, Any]) -> None:
    """
    Saves a dictionary as JSON.
    """

    ensure_artifacts_dir()

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def load_json(path: Path) -> Dict[str, Any]:
    """
    Loads a JSON file as dictionary.
    """

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_pickle(path: Path, data: Any) -> None:
    """
    Saves any Python object as pickle.
    """

    ensure_artifacts_dir()

    with open(path, "wb") as file:
        pickle.dump(data, file)


def load_pickle(path: Path) -> Any:
    """
    Loads a pickle file.
    """

    with open(path, "rb") as file:
        return pickle.load(file)


def artifacts_exist() -> bool:
    """
    Checks whether the minimum required live-inference artifacts exist.

    Required for live mode:
    - scaler
    - svm model
    - feature columns
    - training metadata
    - training labeled dataframe
    """

    required_paths = [
        SCALER_PATH,
        SVM_MODEL_PATH,
        FEATURE_COLS_PATH,
        TRAINING_METADATA_PATH,
        TRAINING_LABELED_DF_PATH,
    ]

    return all(path.exists() for path in required_paths)


def missing_artifacts() -> list[str]:
    """
    Returns missing required artifact file names.
    """

    required_paths = [
        SCALER_PATH,
        SVM_MODEL_PATH,
        FEATURE_COLS_PATH,
        TRAINING_METADATA_PATH,
        TRAINING_LABELED_DF_PATH,
    ]

    return [
        str(path)
        for path in required_paths
        if not path.exists()
    ]


def save_training_artifacts(
    scaler: Any,
    svm_model: Any,
    dbscan_model: Any,
    fcm_info: Dict[str, Any],
    feature_cols: list[str],
    labeled_df: pd.DataFrame,
    shap_background_df: pd.DataFrame,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Saves all artifacts required for live inference.

    Parameters:
    - scaler: fitted StandardScaler
    - svm_model: trained SVM-RBF classifier
    - dbscan_model: fitted DBSCAN model
    - fcm_info: FCM metadata, centers, membership matrix, FPC
    - feature_cols: selected numerical feature columns
    - labeled_df: final training dataframe after labels and SVM columns
    - shap_background_df: sample dataframe used as SHAP background
    - metadata: optional training metadata
    """

    ensure_artifacts_dir()

    metadata = metadata or {}

    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(svm_model, SVM_MODEL_PATH)
    joblib.dump(dbscan_model, DBSCAN_MODEL_PATH)

    save_pickle(FCM_INFO_PATH, fcm_info)

    save_json(
        FEATURE_COLS_PATH,
        {
            "feature_cols": feature_cols,
        },
    )

    labeled_df.to_csv(
        TRAINING_LABELED_DF_PATH,
        index=False,
        encoding="utf-8",
    )

    shap_background_df.to_csv(
        SHAP_BACKGROUND_PATH,
        index=False,
        encoding="utf-8",
    )

    label_counts = {}
    if "risk_label" in labeled_df.columns:
        label_counts = labeled_df["risk_label"].value_counts().to_dict()

    prediction_counts = {}
    if "svm_predicted_cipher_type" in labeled_df.columns:
        prediction_counts = labeled_df["svm_predicted_cipher_type"].value_counts().to_dict()

    metadata_payload = {
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "artifact_version": "1.0",
        "mode": "offline_training_saved_for_live_inference",
        "feature_count": len(feature_cols),
        "feature_cols": feature_cols,
        "training_rows": int(len(labeled_df)),
        "label_counts": label_counts,
        "prediction_counts": prediction_counts,
        **metadata,
    }

    save_json(TRAINING_METADATA_PATH, metadata_payload)

    return {
        "success": True,
        "message": "Training artifacts saved successfully.",
        "artifacts_dir": str(ARTIFACTS_DIR),
        "saved_files": {
            "scaler": str(SCALER_PATH),
            "svm_model": str(SVM_MODEL_PATH),
            "dbscan_model": str(DBSCAN_MODEL_PATH),
            "fcm_info": str(FCM_INFO_PATH),
            "feature_cols": str(FEATURE_COLS_PATH),
            "training_metadata": str(TRAINING_METADATA_PATH),
            "shap_background": str(SHAP_BACKGROUND_PATH),
            "training_labeled_df": str(TRAINING_LABELED_DF_PATH),
        },
        "metadata": metadata_payload,
    }


def load_training_artifacts() -> Dict[str, Any]:
    """
    Loads saved artifacts for Streamlit live inference.

    Raises:
    - FileNotFoundError if required artifacts do not exist.
    """

    if not artifacts_exist():
        missing = missing_artifacts()

        raise FileNotFoundError(
            "Trained artifacts are missing. Please run:\n\n"
            "python train_local_pipeline.py\n\n"
            "Missing files:\n"
            + "\n".join(missing)
        )

    scaler = joblib.load(SCALER_PATH)
    svm_model = joblib.load(SVM_MODEL_PATH)

    dbscan_model = None
    if DBSCAN_MODEL_PATH.exists():
        dbscan_model = joblib.load(DBSCAN_MODEL_PATH)

    fcm_info = {}
    if FCM_INFO_PATH.exists():
        fcm_info = load_pickle(FCM_INFO_PATH)

    feature_payload = load_json(FEATURE_COLS_PATH)
    feature_cols = feature_payload.get("feature_cols", [])

    metadata = load_json(TRAINING_METADATA_PATH)

    labeled_df = pd.read_csv(TRAINING_LABELED_DF_PATH)

    if SHAP_BACKGROUND_PATH.exists():
        shap_background_df = pd.read_csv(SHAP_BACKGROUND_PATH)
    else:
        shap_background_df = labeled_df[feature_cols].head(50).copy()

    return {
        "scaler": scaler,
        "svm_model": svm_model,
        "dbscan_model": dbscan_model,
        "fcm_info": fcm_info,
        "feature_cols": feature_cols,
        "metadata": metadata,
        "labeled_df": labeled_df,
        "shap_background_df": shap_background_df,
        "paths": {
            "scaler": str(SCALER_PATH),
            "svm_model": str(SVM_MODEL_PATH),
            "dbscan_model": str(DBSCAN_MODEL_PATH),
            "fcm_info": str(FCM_INFO_PATH),
            "feature_cols": str(FEATURE_COLS_PATH),
            "training_metadata": str(TRAINING_METADATA_PATH),
            "shap_background": str(SHAP_BACKGROUND_PATH),
            "training_labeled_df": str(TRAINING_LABELED_DF_PATH),
        },
    }


def get_artifact_status() -> Dict[str, Any]:
    """
    Returns artifact status for Streamlit display.
    """

    status = {
        "artifacts_dir": str(ARTIFACTS_DIR),
        "artifacts_exist": artifacts_exist(),
        "missing_artifacts": missing_artifacts(),
        "files": {
            "scaler": {
                "path": str(SCALER_PATH),
                "exists": SCALER_PATH.exists(),
            },
            "svm_model": {
                "path": str(SVM_MODEL_PATH),
                "exists": SVM_MODEL_PATH.exists(),
            },
            "dbscan_model": {
                "path": str(DBSCAN_MODEL_PATH),
                "exists": DBSCAN_MODEL_PATH.exists(),
            },
            "fcm_info": {
                "path": str(FCM_INFO_PATH),
                "exists": FCM_INFO_PATH.exists(),
            },
            "feature_cols": {
                "path": str(FEATURE_COLS_PATH),
                "exists": FEATURE_COLS_PATH.exists(),
            },
            "training_metadata": {
                "path": str(TRAINING_METADATA_PATH),
                "exists": TRAINING_METADATA_PATH.exists(),
            },
            "shap_background": {
                "path": str(SHAP_BACKGROUND_PATH),
                "exists": SHAP_BACKGROUND_PATH.exists(),
            },
            "training_labeled_df": {
                "path": str(TRAINING_LABELED_DF_PATH),
                "exists": TRAINING_LABELED_DF_PATH.exists(),
            },
        },
    }

    if TRAINING_METADATA_PATH.exists():
        try:
            status["metadata"] = load_json(TRAINING_METADATA_PATH)
        except Exception as e:
            status["metadata_error"] = str(e)

    return status


def clear_artifacts() -> Dict[str, Any]:
    """
    Deletes saved artifacts.

    Use only if you want to retrain from scratch.
    """

    paths = [
        SCALER_PATH,
        SVM_MODEL_PATH,
        DBSCAN_MODEL_PATH,
        FCM_INFO_PATH,
        FEATURE_COLS_PATH,
        TRAINING_METADATA_PATH,
        SHAP_BACKGROUND_PATH,
        TRAINING_LABELED_DF_PATH,
    ]

    deleted = []

    for path in paths:
        if path.exists():
            path.unlink()
            deleted.append(str(path))

    return {
        "success": True,
        "deleted": deleted,
    }