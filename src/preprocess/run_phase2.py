"""
Phase 2 driver script.

Rough idea:
    1. load + merge FakeNewsNet CSVs
    2. clean titles/text
    3. save unified table
    4. compute TF-IDF baseline
    5. compute BERT embeddings (MiniLM)

I kept this as a single "main" so teammates can just run:
    python -m src.preprocess.run_phase2
"""

from pathlib import Path

from .parse_news import load_all_news, save_news_table
from .clean_text import add_clean_column
from .embeddings_tfidf import build_tfidf_features
from .embeddings_bert import make_bert_embeddings


# again, I like having ROOT to avoid hardcoding absolute paths everywhere
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "FakeNewsNet"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def run_phase_two():
    print("=== PHASE 2: preprocessing & embeddings ===")
    print(f"[INFO] project root: {PROJECT_ROOT}")
    print(f"[INFO] raw data dir: {RAW_DIR}")

    # 1) load CSVs and stack them
    news_df = load_all_news()

    # 2) clean text column (we treat titles as text for this dataset)
    news_df = add_clean_column(news_df, src_col="text", dest_col="clean_text")

    # 3) save the cleaned unified table
    clean_csv = PROCESSED_DIR / "news_clean.csv"
    save_news_table(news_df, clean_csv)

    # 4) TF-IDF baseline on clean text
    tfidf_matrix_path = PROCESSED_DIR / "tfidf_vectors.npz"
    tfidf_model_path = PROCESSED_DIR / "tfidf_vectorizer.joblib"
    X_tfidf, _ = build_tfidf_features(
        news_df,
        text_col="clean_text",
        out_vectors=tfidf_matrix_path,
        out_vectorizer=tfidf_model_path,
        max_features=20000,
    )

    print ("[INFO] finished TF-IDF step, matrix size:", X_tfidf.shape)

    # 5) BERT embeddings on the same cleaned text
    bert_out = PROCESSED_DIR / "bert_embeddings.npy"
    emb = make_bert_embeddings(
        news_df,
        text_col="clean_text",
        out_path=bert_out,
        model_name="all-MiniLM-L6-v2",
        batch_size=64,
    )

    print("[INFO] finished BERT step, embedding shape:", emb.shape)
    print("=== DONE: Phase 2 preprocessing complete. ===")


def main():
    # thin wrapper so we can change function name later if needed
    run_phase_two()


if __name__ == "__main__":
    main()
