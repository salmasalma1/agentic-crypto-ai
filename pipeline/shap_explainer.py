"""Step 4-A: SHAP explainability for the SVM-RBF classifier."""
import numpy as np
import pandas as pd


def _fallback_importance(model, X_array, feature_cols):
    """Fallback importance based on probability sensitivity to feature perturbation."""
    base = model.predict_proba(X_array).max(axis=1).mean()
    values = []
    for idx, feature in enumerate(feature_cols):
        X_perm = X_array.copy()
        X_perm[:, idx] = np.median(X_perm[:, idx])
        changed = model.predict_proba(X_perm).max(axis=1).mean()
        values.append(abs(base - changed))
    importance_df = pd.DataFrame({"feature": feature_cols, "shap_importance": values}).sort_values("shap_importance", ascending=False).reset_index(drop=True)
    total = float(importance_df["shap_importance"].sum())
    score = 0.0 if total == 0 else float(importance_df["shap_importance"].iloc[0] / total)
    return {"shap_score": round(max(0.0, min(1.0, score)), 4), "importance_df": importance_df, "engine": "fallback_sensitivity"}


def run_shap_explanation(model, X_scaled, feature_cols, max_background_samples=20, max_explain_samples=10):
    X_array = np.asarray(X_scaled, dtype=float)
    if X_array.shape[0] == 0:
        raise ValueError("X_scaled is empty. Cannot run SHAP.")
    if len(feature_cols) != X_array.shape[1]:
        raise ValueError("feature_cols length does not match X_scaled width.")

    X_explain = X_array[: min(max_explain_samples, X_array.shape[0])]
    background = X_array[: min(max_background_samples, X_array.shape[0])]

    try:
        import shap  # type: ignore
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X_explain, nsamples=min(80, 2 * X_array.shape[1] + 20))
        if isinstance(shap_values, list):
            mean_abs = np.mean([np.abs(v).mean(axis=0) for v in shap_values], axis=0)
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            mean_abs = np.abs(shap_values).mean(axis=(0, 2))
        else:
            mean_abs = np.abs(shap_values).mean(axis=0)
        importance_df = pd.DataFrame({"feature": feature_cols, "shap_importance": mean_abs}).sort_values("shap_importance", ascending=False).reset_index(drop=True)
        total = float(importance_df["shap_importance"].sum())
        score = 0.0 if total == 0 else float(importance_df["shap_importance"].iloc[0] / total)
        return {"shap_score": round(max(0.0, min(1.0, score)), 4), "importance_df": importance_df, "engine": "shap.KernelExplainer"}
    except Exception:
        return _fallback_importance(model, X_explain, feature_cols)
