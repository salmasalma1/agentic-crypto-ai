"""Dynamic reclustering triggered by drift detection."""
from pipeline.dbscan_clustering import apply_dbscan
from pipeline.fcm_clustering import apply_fcm
from pipeline.label_generator import generate_risk_labels
from pipeline.svm_classifier import train_svm_classifier, predict_with_confidence
from utils.config import SUPPORTED_CIPHER_TYPES
import pandas as pd


def run_dynamic_reclustering(cleaned_df, X_scaled, feature_cols, risk_feature):
    reclustered_df, dbscan_model = apply_dbscan(cleaned_df, X_scaled)
    reclustered_df, fcm_info = apply_fcm(reclustered_df, X_scaled)
    relabeled_df = generate_risk_labels(reclustered_df, risk_feature)

    if "source_cipher_type" in relabeled_df.columns:
        train_mask = relabeled_df["source_cipher_type"].isin(SUPPORTED_CIPHER_TYPES)
        if int(train_mask.sum()) < 10:
            train_mask = pd.Series([True] * len(relabeled_df), index=relabeled_df.index)
    else:
        train_mask = pd.Series([True] * len(relabeled_df), index=relabeled_df.index)

    svm_model, svm_metrics = train_svm_classifier(X_scaled[train_mask.to_numpy()], relabeled_df.loc[train_mask, "risk_label"])
    predictions, confidence = predict_with_confidence(svm_model, X_scaled)
    relabeled_df["svm_prediction_after_reclustering"] = predictions
    relabeled_df["svm_confidence_after_reclustering"] = confidence

    return {
        "reclustered_df": relabeled_df,
        "dbscan_model": dbscan_model,
        "fcm_info": fcm_info,
        "svm_model": svm_model,
        "svm_metrics": svm_metrics,
        "status": "Dynamic Reclustering completed successfully.",
    }
