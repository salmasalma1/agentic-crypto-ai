"""Generate pseudo-labels after DBSCAN and FCM."""
import pandas as pd
from utils.config import SUPPORTED_CIPHER_TYPES


def infer_cipher_type_from_features(row: pd.Series) -> str:
    length = row.get("cypher_length", 0)
    entropy = row.get("cypher_entropy", 0)
    base64_ratio = row.get("cypher_base64_ratio", 0)
    base32_ratio = row.get("cypher_base32_ratio", 0)
    morse_ratio = row.get("cypher_morse_ratio", 0)
    dot_dash_ratio = row.get("cypher_dot_dash_ratio", 0)
    digit_ratio = row.get("cypher_digit_ratio", 0)
    alpha_ratio = row.get("cypher_alpha_ratio", 0)
    upper_ratio = row.get("cypher_upper_ratio", 0)
    lower_ratio = row.get("cypher_lower_ratio", 0)
    symbol_ratio = row.get("cypher_symbol_ratio", 0)
    equal_count = row.get("cypher_equal_count", 0)
    plus_slash_count = row.get("cypher_plus_slash_count", 0)

    if length <= 0:
        return "Unknown"
    if morse_ratio >= 0.80 and dot_dash_ratio >= 0.25 and alpha_ratio < 0.10:
        return "Morse"
    # Base32 must be checked before Base64, because Base32 alphabet is a subset of Base64-like chars.
    if base32_ratio >= 0.90 and alpha_ratio >= 0.45 and upper_ratio >= lower_ratio and plus_slash_count == 0 and symbol_ratio <= 0.40 and (equal_count > 0 or digit_ratio <= 0.45):
        return "Base32"
    if base64_ratio >= 0.90 and alpha_ratio >= 0.40 and entropy >= 3.0 and (plus_slash_count > 0 or lower_ratio > 0.05 or equal_count > 0 or length % 4 == 0):
        return "Base64"
    return "Unknown"


def _cluster_based_risk(row: pd.Series) -> str:
    if row.get("dbscan_cluster", 0) == -1:
        return "Anomaly"
    membership = float(row.get("fcm_membership_score", 0.0))
    entropy = float(row.get("cypher_entropy", 0.0))
    if membership < 0.55 or entropy >= 5.0:
        return "High Risk"
    if membership < 0.75 or entropy >= 4.0:
        return "Medium Risk"
    return "Low Risk"


def generate_cipher_type_labels(df: pd.DataFrame, risk_feature: str = None):
    df = df.copy()
    required = ["dbscan_cluster", "fcm_cluster", "fcm_membership_score", "cypher_length", "cypher_entropy", "cypher_base64_ratio", "cypher_base32_ratio", "cypher_morse_ratio", "cypher_dot_dash_ratio"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError("Missing required columns for label generation: " + ", ".join(missing))

    labels = []
    for _, row in df.iterrows():
        source = str(row.get("source_cipher_type", "")).strip()
        if source in SUPPORTED_CIPHER_TYPES:
            labels.append(source)
        elif row.get("dbscan_cluster", 0) == -1:
            inferred = infer_cipher_type_from_features(row)
            labels.append(inferred if inferred != "Unknown" else "Anomaly")
        else:
            labels.append(infer_cipher_type_from_features(row))

    df["cipher_type_label"] = labels
    # risk_label is kept for compatibility, but SVM now predicts cipher type.
    df["risk_label"] = df["cipher_type_label"]
    df["business_risk_label"] = df.apply(_cluster_based_risk, axis=1)
    return df


def generate_risk_labels(df: pd.DataFrame, risk_feature: str = None):
    return generate_cipher_type_labels(df, risk_feature)
