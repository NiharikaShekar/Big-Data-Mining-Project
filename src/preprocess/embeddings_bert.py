from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def make_bert_embeddings(
    df: pd.DataFrame,
    text_col: str,
    out_path: Path,
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 64,
):
    """
    Uses a SentenceTransformer model to create dense embeddings.
    We just save them as a big .npy array (float32).
    """
    if text_col not in df.columns:
        raise KeyError(f"column '{text_col}' not found in dataframe")

    the_texts = df[text_col].astype(str).tolist()

    print(f"[INFO] loading SentenceTransformer model: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"[INFO] encoding {len(the_texts)} docs with batch_size={batch_size}")
    # progress bar is nice for seeing that it's actually doing something
    try:
        emb = model.encode(
            the_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
    except Exception as e:
        print("[ERROR] BERT encoding failed:", e)
        raise

    emb = emb.astype("float32")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, emb)

    print("  -> BERT embeddings shape:", emb.shape)
    print(f"[OK] saved BERT embeddings to {out_path}")

    return emb
