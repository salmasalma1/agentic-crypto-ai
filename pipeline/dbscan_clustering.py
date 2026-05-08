"""Step 2-A: DBSCAN clustering."""
import pandas as pd
from sklearn.cluster import DBSCAN
from utils.config import DBSCAN_EPS, DBSCAN_MIN_SAMPLES


def apply_dbscan(df: pd.DataFrame, X_scaled, eps: float = DBSCAN_EPS, min_samples: int = DBSCAN_MIN_SAMPLES):
    df = df.copy()
    if len(df) == 0:
        raise ValueError("Cannot run DBSCAN on an empty dataset.")
    model = DBSCAN(eps=eps, min_samples=min_samples)
    df["dbscan_cluster"] = model.fit_predict(X_scaled)
    return df, model
