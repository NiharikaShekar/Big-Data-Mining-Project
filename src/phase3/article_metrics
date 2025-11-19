from __future__ import annotations
from pathlib import Path
from typing import List
import pandas as pd
from .utils import parse_tweet_ids, ensure_parent

_DOMAINS: List[str] = ["gossipcop", "politifact"]

def _load_raw(fp: Path) -> pd.DataFrame:
    df = pd.read_csv(fp, low_memory=False)
    if "id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "id"})
    if "tweet_ids" not in df.columns:
        df["tweet_ids"] = ""
    return df[["id", "tweet_ids"]].astype({"id": "string", "tweet_ids": "string"})

def _stack_raw(root: Path) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    for dom in _DOMAINS:
        d = root / dom
        if not d.is_dir():
            continue
        for name in ("fake", "real"):
            for fp in sorted(d.glob(f"*_{name}.csv")):
                parts.append(_load_raw(fp))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["id","tweet_ids"])

def build_article_metrics(news_clean_fp: Path, raw_root: Path, out_csv: Path) -> pd.DataFrame:
    """
    Produce per-article tweet_count from Phase-1 raw CSVs; join to Phase-2 table.
    Why: counts are the interaction signal we can compute without user edges.
    """
    news = pd.read_csv(news_clean_fp, low_memory=False)
    req = {"news_id", "label", "source"}
    miss = req - set(news.columns)
    if miss:
        raise ValueError(f"news_clean.csv missing {sorted(miss)}")
    news["news_id"] = news["news_id"].astype("string")

    raw = _stack_raw(raw_root)
    raw["id"] = raw["id"].astype("string")
    raw["tweet_list"] = raw["tweet_ids"].apply(parse_tweet_ids)
    raw["tweet_count"] = raw["tweet_list"].apply(len)

    merged = news.merge(raw[["id","tweet_list","tweet_count"]], how="left",
                        left_on="news_id", right_on="id")
    merged["tweet_count"] = merged["tweet_count"].fillna(0).astype(int)
    merged["has_tweets"] = merged["tweet_count"] > 0
    merged = merged.drop(columns=["id"]).copy()

    ensure_parent(out_csv)
    merged.to_csv(out_csv, index=False)
    return merged
