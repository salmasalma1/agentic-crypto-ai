from utils.config import DRIFT_THRESHOLD


def detect_drift(sarima_result: dict, threshold: float = DRIFT_THRESHOLD):
    expected_change_pct = abs(float(sarima_result.get("expected_change_pct", 0.0)))
    drift_score = expected_change_pct / 100.0
    drift_detected = drift_score >= threshold
    return {
        "drift_detected": bool(drift_detected),
        "drift_score": round(float(drift_score), 4),
        "recommendation": "Dynamic Reclustering is recommended: re-run DBSCAN, FCM, labels, and SVM." if drift_detected else "No major drift detected.",
    }
