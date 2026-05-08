"""Step 3: SVM-RBF classifier."""
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from utils.config import TEST_SIZE, RANDOM_STATE, SVM_KERNEL


def _can_stratify(y):
    _, counts = np.unique(y, return_counts=True)
    return len(counts) > 1 and counts.min() >= 2


def train_svm_classifier(X_scaled, y):
    y_arr = np.asarray(y)
    if len(y_arr) < 2:
        raise ValueError("SVM training requires at least two rows.")
    classes = np.unique(y_arr)
    if len(classes) < 2:
        raise ValueError("SVM training requires at least two classes.")

    stratify = y_arr if _can_stratify(y_arr) else None
    test_size = TEST_SIZE
    if len(y_arr) < 10:
        test_size = max(1 / len(y_arr), 0.25)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_arr, test_size=test_size, random_state=RANDOM_STATE, stratify=stratify
    )
    model = SVC(kernel=SVM_KERNEL, probability=True, random_state=RANDOM_STATE, class_weight="balanced")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        "classes": list(model.classes_),
        "kernel": SVM_KERNEL,
    }


def predict_with_confidence(model, X_scaled):
    predictions = model.predict(X_scaled)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_scaled)
        confidence = probabilities.max(axis=1)
    else:
        confidence = np.ones(len(predictions)) * 0.5
    return predictions, confidence
