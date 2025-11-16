# tiny helper to load + normalize the FakeNewsNet CSVs
# (mostly titles, since we don't have full article text)

from pathlib import Path
import pandas as pd

# I like having a global ROOT so paths don't break if we move things around
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "FakeNewsNet"


def _find_csv_file(file_name: str) -> Path:
    """
    Tries a couple of locations to find the CSV.
    If something goes wrong we just crash and see the error.
    """
    direct_path = RAW_DIR / file_name
    if direct_path.is_file():
        return direct_path

    # fallback: search under the dataset folder (in case structure changes)
    matches = list(RAW_DIR.rglob(file_name))
    if matches:
        return matches[0]

    
    raise FileNotFoundError(f"couldn't find {file_name} under {RAW_DIR}")


def _pick_text_column(df: pd.DataFrame, filename: str) -> str:
    """
    The dataset doesn't always use the same name for text.
    For the version we got, 'title' is the main text-ish field.
    """
    # I put 'title' first since we know gossipcop_fake has that
    text_candidates = [
        "title",           # <- what our CSV actually has
        "text",
        "content",
        "body",
        "article",
        "news_article",
        "tweet_text",
        "description",
        "clean_title",
    ]

    for col in text_candidates:
        if col in df.columns:
            return col

    # if we end up here something is wrong
    raise ValueError(
        f"Could not figure out a text column for {filename}. "
        f"Columns were: {list(df.columns)}"
    )


def load_all_news() -> pd.DataFrame:
    """
    Loads all four CSVs (gossipcop/politifact x fake/real),
    and merges them into a single dataframe.

    Output columns:
        - news_id
        - text      (raw text / title)
        - label     (0 = fake, 1 = real)
        - source    ('gossipcop' or 'politifact')
    """
    files_meta = {
        "gossipcop_fake": ("gossipcop_fake.csv", 0, "gossipcop"),
        "gossipcop_real": ("gossipcop_real.csv", 1, "gossipcop"),
        "politifact_fake": ("politifact_fake.csv", 0, "politifact"),
        "politifact_real": ("politifact_real.csv", 1, "politifact"),
    }

    all_parts = []

    for short_name, (fname, label_val, src_name) in files_meta.items():
        csv_path = _find_csv_file(fname)
        print(f"[INFO] loading {short_name} from {csv_path}")

        try:
            df_raw = pd.read_csv(csv_path)
        except Exception as e:
            # generic handler, just so the script doesn't explode silently
            print(f"[ERROR] something went wrong reading {csv_path}: {e}")
            raise

        # try to find an ID column; if not, we just make one
        id_candidates = ["news_id", "id", "newsid"]
        id_col = None
        for c in id_candidates:
            if c in df_raw.columns:
                id_col = c
                break

        if id_col is None:
            df_raw = df_raw.reset_index().rename(columns={"index": "news_id"})
            id_col = "news_id"

        text_col = _pick_text_column(df_raw, fname)

        # just keep what we need for the rest of the pipeline
        df_small = df_raw[[id_col, text_col]].copy()
        df_small = df_small.rename(columns={id_col: "news_id", text_col: "text"})
        df_small["label"] = label_val
        df_small["source"] = src_name

        all_parts.append(df_small)

    news_df = pd.concat(all_parts, ignore_index=True)
    print(f"[INFO] combined dataset shape = {news_df.shape}")
    return news_df


def save_news_table(df: pd.DataFrame, out_file: Path) -> None:
    """Just a tiny wrapper to save the unified table."""
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)
    print(f"[OK] saved news table to {out_file}")
