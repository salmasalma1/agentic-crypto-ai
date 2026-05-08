"""Optional RAG-KNN support context for the final report."""
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from utils.config import DEFAULT_KNOWLEDGE_PATH


def load_knowledge_base(path: str = DEFAULT_KNOWLEDGE_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=["title", "cipher_type", "content", "rag_document"])
    df = pd.read_csv(path).fillna("")
    required = ["title", "cipher_type", "content"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Missing required knowledge base columns: " + ", ".join(missing))
    df["rag_document"] = df["title"].astype(str) + " | " + df["cipher_type"].astype(str) + " | " + df["content"].astype(str)
    return df


def retrieve_rag_context(query: str, top_k: int = 3, knowledge_path: str = DEFAULT_KNOWLEDGE_PATH):
    query = str(query or "").strip()
    if not query:
        return {"query": "", "retrieved_context": "", "retrieved_items": []}
    df = load_knowledge_base(knowledge_path)
    if df.empty:
        return {"query": query, "retrieved_context": "", "retrieved_items": []}
    vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2))
    vectors = vectorizer.fit_transform(df["rag_document"].tolist())
    model = NearestNeighbors(n_neighbors=min(top_k, len(df)), metric="cosine")
    model.fit(vectors)
    distances, indices = model.kneighbors(vectorizer.transform([query]), n_neighbors=min(top_k, len(df)))
    items = []
    for distance, index in zip(distances[0], indices[0]):
        row = df.iloc[index]
        items.append({"title": row["title"], "cipher_type": row["cipher_type"], "content": row["content"], "distance": round(float(distance), 4), "similarity": round(float(1 - distance), 4)})
    context = "\n\n".join(f"[{i['cipher_type']}] {i['title']}: {i['content']}" for i in items)
    return {"query": query, "retrieved_context": context, "retrieved_items": items}


def build_cipher_rag_query(predicted_cipher_type: str, cypher_text: str, decoded_plain_text: str, svm_confidence=None, shap_summary: str = "") -> str:
    parts = [
        f"Predicted cipher type: {predicted_cipher_type}",
        f"Cypher text: {cypher_text}",
        f"Decoded plain text: {decoded_plain_text}",
        "Explain the cipher pattern, SVM prediction, decoder result, DBSCAN, FCM, SHAP and SARIMA evidence.",
    ]
    if svm_confidence is not None:
        parts.append(f"SVM confidence: {svm_confidence}")
    if shap_summary:
        parts.append(f"SHAP summary: {shap_summary}")
    return " ".join(parts)
