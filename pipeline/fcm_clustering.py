"""Step 2-B: Fuzzy C-Means clustering.

Uses scikit-fuzzy when installed. Otherwise falls back to a compact NumPy
implementation of the FCM update equations, so the project remains runnable.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from utils.config import FCM_CLUSTERS, FCM_M, FCM_ERROR, FCM_MAX_ITER, RANDOM_STATE


def _numpy_fcm(X, c=3, m=2.0, error=0.005, maxiter=300):
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    c = max(2, min(int(c), n))
    # KMeans initialization provides stable centers for the first membership matrix.
    km = KMeans(n_clusters=c, random_state=RANDOM_STATE, n_init=1, max_iter=50).fit(X)
    centers = km.cluster_centers_
    u = np.zeros((c, n), dtype=float)
    labels = km.labels_
    for i, label in enumerate(labels):
        u[:, i] = 0.01 / max(c - 1, 1)
        u[label, i] = 0.99

    jm = []
    for iteration in range(min(maxiter, 80)):
        u_old = u.copy()
        um = u ** m
        centers = (um @ X) / np.maximum(um.sum(axis=1, keepdims=True), 1e-12)
        dist = np.linalg.norm(X[None, :, :] - centers[:, None, :], axis=2)
        dist = np.maximum(dist, 1e-8)
        power = 2.0 / (m - 1.0)
        for k in range(c):
            denom = np.sum((dist[k:k+1, :] / dist) ** power, axis=0)
            u[k, :] = 1.0 / np.maximum(denom, 1e-12)
        objective = np.sum((u ** m) * (dist ** 2))
        jm.append(objective)
        if np.linalg.norm(u - u_old) < error:
            break
    fpc = float(np.sum(u ** 2) / n)
    return centers, u, None, dist, jm, iteration + 1, fpc


def apply_fcm(df: pd.DataFrame, X_scaled):
    df = df.copy()
    X = np.asarray(X_scaled, dtype=float)
    if len(df) == 0:
        raise ValueError("Cannot run FCM on an empty dataset.")
    c = min(FCM_CLUSTERS, len(df))

    try:
        import skfuzzy as fuzz  # type: ignore
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            data=X.T, c=c, m=FCM_M, error=FCM_ERROR, maxiter=FCM_MAX_ITER, init=None
        )
        engine = "skfuzzy.cmeans"
    except Exception:
        cntr, u, u0, d, jm, p, fpc = _numpy_fcm(X, c=c, m=FCM_M, error=FCM_ERROR, maxiter=FCM_MAX_ITER)
        engine = "numpy_fcm_fallback"

    df["fcm_cluster"] = np.argmax(u, axis=0)
    df["fcm_membership_score"] = np.max(u, axis=0)
    return df, {"centers": cntr, "membership_matrix": u, "fpc": fpc, "engine": engine}
