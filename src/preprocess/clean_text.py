import re
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# basic regex patterns – nothing fancy, just enough for this project
URL_PATTERN = re.compile(r"http\S+|www\.\S+")
HTML_PATTERN = re.compile(r"<[^>]+>")
NON_ALNUM_PATTERN = re.compile(r"[^a-z0-9\s]+")


def clean_one_text(t):
    """
    Simple cleaner for titles/text.
    Lowercases, strips URLs, removes weird chars + stopwords.
    Not super aggressive on purpose.
    """
    if not isinstance(t, str):
        return ""

    txt = t.lower()
    txt = URL_PATTERN.sub(" ", txt)
    txt = HTML_PATTERN.sub(" ", txt)
    txt = NON_ALNUM_PATTERN.sub(" ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()

    # remove stopwords at the end so we don't accidentally nuke punctuation first
    tokens = [w for w in txt.split() if w not in ENGLISH_STOP_WORDS]
    return " ".join(tokens)


def add_clean_column(df: pd.DataFrame, src_col="text", dest_col="clean_text"):
    """
    Adds df[dest_col] using the cleaner above.
    """
    if src_col not in df.columns:
        # just in case someone renames things later
        raise KeyError(f"expected column '{src_col}' in df, got {list(df.columns)}")

    print(f"[INFO] cleaning column '{src_col}' → '{dest_col}'")
    # a tiny bit slow but fine for this dataset size
    df[dest_col] = df[src_col].astype(str).apply(clean_one_text)
    return df
