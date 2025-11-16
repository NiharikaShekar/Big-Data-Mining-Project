from pathlib import Path

import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib  # sklearn uses this internally anyway


def build_tfidf_features(
    df: pd.DataFrame,
    text_col: str,
    out_vectors: Path,
    out_vectorizer: Path,
    max_features: int = 20000,
):
    """
    TF-IDF baseline for the project.

    Saves:
        - sparse matrix -> out_vectors (.npz)
        - fitted vectorizer -> out_vectorizer (.joblib)
    """
    if text_col not in df.columns:
        raise KeyError(f"'{text_col}' is not a column in df")

    print(f"[INFO] starting TF-IDF on '{text_col}' (max_features={max_features})")

    
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        stop_words="english",
    )

    # just in case we accidentally pass non-strings
    texts = df[text_col].astype(str).values

    try:
        X = tfidf.fit_transform(texts)
    except Exception as e:
        print("[ERROR] something went wrong during TF-IDF fit/transform:", e)
        raise

    out_vectors.parent.mkdir(parents=True, exist_ok=True)
    sp.save_npz(out_vectors, X)
    joblib.dump(tfidf, out_vectorizer)

    print("  -> TF-IDF shape:", X.shape)
    print(f"[OK] saved TF-IDF matrix to {out_vectors}")
    print(f"[OK] saved TF-IDF vectorizer to {out_vectorizer}")

    return X, tfidf
