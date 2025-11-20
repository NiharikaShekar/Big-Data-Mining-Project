"""
Generate Synthetic User IDs and Relationships for FakeNewsNet Dataset

This module creates synthetic user data from tweet IDs to enable:
- User interaction graph construction
- Cascade subgraph building
- User influence analysis (PageRank, HITS)

Why: Twitter API is expensive, but we can demonstrate methodology with synthetic data.

Approach:
1. Assign synthetic user IDs to tweets (consistent mapping)
2. Create synthetic retweet/reply relationships based on patterns
3. Generate timestamps based on tweet ID patterns
4. Build realistic cascade structures
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from .utils import parse_tweet_ids, ensure_parent


def generate_user_id_from_tweet(tweet_id: str, seed: int = 42) -> str:
    """
    Generate a consistent synthetic user ID from a tweet ID.
    Uses hash-based approach so same tweet always gets same user.
    """
    np.random.seed(hash(tweet_id) % (2**31))
    # Generate user ID in range 1000000-999999999 (realistic Twitter user ID range)
    user_id = str(1000000 + (hash(tweet_id) % 999000000))
    return user_id


def generate_timestamp_from_tweet(tweet_id: str, base_date: str = "2017-01-01") -> str:
    """
    Generate a synthetic timestamp from tweet ID.
    Uses tweet ID to create consistent but varied timestamps.
    """
    import datetime
    base = datetime.datetime.strptime(base_date, "%Y-%m-%d")
    
    # Use tweet ID to create offset (consistent for same tweet)
    offset_days = hash(tweet_id) % 365  # Spread over a year
    offset_hours = (hash(tweet_id) // 1000) % 24
    offset_minutes = (hash(tweet_id) // 100) % 60
    
    timestamp = base + datetime.timedelta(
        days=offset_days,
        hours=offset_hours,
        minutes=offset_minutes
    )
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")


def create_synthetic_user_tweet_mapping(
    article_metrics_csv: Path,
    raw_root: Path,
    out_csv: Path,
) -> pd.DataFrame:
    """
    Create a mapping of tweets to synthetic users.
    
    Output CSV columns:
    - tweet_id: Original tweet ID
    - user_id: Synthetic user ID (consistent for same tweet)
    - article_id: News article this tweet is associated with
    - label: 0 (fake) or 1 (real)
    - source: 'gossipcop' or 'politifact'
    - timestamp: Synthetic timestamp
    """
    from .article_metrics import _stack_raw
    
    metrics = pd.read_csv(article_metrics_csv, low_memory=False)
    raw = _stack_raw(raw_root)
    
    # Create tweet-to-article mapping
    tweet_article_map = {}
    for _, row in raw.iterrows():
        article_id = str(row["id"])
        tweet_list = parse_tweet_ids(row["tweet_ids"])
        for tid in tweet_list:
            tweet_article_map[tid] = article_id
    
    # Get article metadata (handle duplicates by taking first)
    metrics_unique = metrics.drop_duplicates(subset=["news_id"], keep="first")
    article_meta = metrics_unique.set_index("news_id")[["label", "source"]].to_dict("index")
    
    # Build synthetic user-tweet mapping
    records = []
    for tweet_id, article_id in tweet_article_map.items():
        user_id = generate_user_id_from_tweet(tweet_id)
        timestamp = generate_timestamp_from_tweet(tweet_id)
        
        meta = article_meta.get(article_id, {"label": 0, "source": "unknown"})
        
        records.append({
            "tweet_id": tweet_id,
            "user_id": user_id,
            "article_id": article_id,
            "label": meta["label"],
            "source": meta["source"],
            "timestamp": timestamp,
        })
    
    df = pd.DataFrame(records)
    ensure_parent(out_csv)
    df.to_csv(out_csv, index=False)
    print(f"[OK] Synthetic user-tweet mapping → {out_csv}")
    print(f"     {len(df)} tweets mapped to {df['user_id'].nunique()} unique users")
    
    return df


def create_synthetic_retweet_relationships(
    user_tweet_csv: Path,
    out_csv: Path,
    retweet_probability: float = 0.08,  # Reduced for faster generation
    max_retweets_per_tweet: int = 5,  # Reduced for faster generation
) -> pd.DataFrame:
    """
    Create synthetic retweet relationships.
    
    Strategy:
    - For each tweet, randomly select other users to retweet it
    - Higher probability for tweets from same article
    - Some users retweet multiple tweets (hub behavior)
    
    Output CSV columns:
    - retweeter_id: User who retweeted
    - original_user_id: User who posted original tweet
    - tweet_id: Original tweet ID
    - timestamp: Retweet timestamp (after original)
    """
    df = pd.read_csv(user_tweet_csv, low_memory=False)
    
    # Group by article to create article-based retweet patterns
    article_groups = df.groupby("article_id")
    
    retweets = []
    
    # Process in chunks to show progress
    total_articles = len(article_groups)
    processed = 0
    
    for article_id, article_tweets in article_groups:
        if processed % 100 == 0:
            print(f"[INFO] Processing retweets: {processed}/{total_articles} articles...")
        processed += 1
        
        tweet_list = article_tweets.to_dict("records")
        
        # Limit processing for very large articles (performance)
        if len(tweet_list) > 100:
            tweet_list = tweet_list[:100]  # Sample first 100 tweets per article
        
        # Create retweets within article (more realistic)
        for tweet in tweet_list:
            original_user = tweet["user_id"]
            original_tweet_id = tweet["tweet_id"]
            original_time = pd.to_datetime(tweet["timestamp"])
            
            # Select other users in same article as potential retweeters
            other_users = article_tweets[article_tweets["user_id"] != original_user]["user_id"].unique()
            
            # Some users retweet multiple tweets (hub behavior)
            num_retweets = np.random.binomial(
                n=max_retweets_per_tweet,
                p=retweet_probability
            )
            
            if num_retweets > 0 and len(other_users) > 0:
                # Select retweeters (with some repetition for hub behavior)
                retweeters = np.random.choice(
                    other_users,
                    size=min(num_retweets, len(other_users) * 2),  # Allow some repetition
                    replace=True
                )
                
                for retweeter in retweeters:
                    # Retweet happens after original (add random minutes)
                    retweet_time = original_time + pd.Timedelta(
                        minutes=np.random.randint(1, 1440)  # Within 24 hours
                    )
                    
                    retweets.append({
                        "retweeter_id": retweeter,
                        "original_user_id": original_user,
                        "tweet_id": original_tweet_id,
                        "timestamp": retweet_time.strftime("%Y-%m-%d %H:%M:%S"),
                    })
        
        # Also create some cross-article retweets (less common)
        if np.random.random() < 0.1:  # 10% chance
            other_articles = df[df["article_id"] != article_id]
            if len(other_articles) > 0:
                cross_tweet = other_articles.sample(1).iloc[0]
                retweeter = np.random.choice(article_tweets["user_id"].unique())
                
                retweet_time = pd.to_datetime(cross_tweet["timestamp"]) + pd.Timedelta(
                    minutes=np.random.randint(1, 1440)
                )
                
                retweets.append({
                    "retweeter_id": retweeter,
                    "original_user_id": cross_tweet["user_id"],
                    "tweet_id": cross_tweet["tweet_id"],
                    "timestamp": retweet_time.strftime("%Y-%m-%d %H:%M:%S"),
                })
    
    retweet_df = pd.DataFrame(retweets)
    ensure_parent(out_csv)
    retweet_df.to_csv(out_csv, index=False)
    print(f"[OK] Synthetic retweet relationships → {out_csv}")
    print(f"     {len(retweet_df)} retweet edges created")
    
    return retweet_df


def create_synthetic_reply_relationships(
    user_tweet_csv: Path,
    out_csv: Path,
    reply_probability: float = 0.04,  # Reduced for faster generation
    max_replies_per_tweet: int = 3,  # Reduced for faster generation
) -> pd.DataFrame:
    """
    Create synthetic reply relationships.
    
    Strategy:
    - Users reply to tweets in same article (conversation threads)
    - Some users are more active (reply to multiple tweets)
    
    Output CSV columns:
    - replier_id: User who replied
    - replied_to_user_id: User being replied to
    - original_tweet_id: Tweet being replied to
    - timestamp: Reply timestamp (after original)
    """
    df = pd.read_csv(user_tweet_csv, low_memory=False)
    
    article_groups = df.groupby("article_id")
    replies = []
    total_articles = len(article_groups)
    processed = 0
    
    for article_id, article_tweets in article_groups:
        if processed % 100 == 0:
            print(f"[INFO] Processing replies: {processed}/{total_articles} articles...")
        processed += 1
        
        # Limit processing for very large articles (performance)
        tweet_list = article_tweets.to_dict("records")
        if len(tweet_list) > 100:
            tweet_list = tweet_list[:100]  # Sample first 100 tweets per article
        tweet_list = article_tweets.to_dict("records")
        
        for tweet in tweet_list:
            original_user = tweet["user_id"]
            original_tweet_id = tweet["tweet_id"]
            original_time = pd.to_datetime(tweet["timestamp"])
            
            other_users = article_tweets[article_tweets["user_id"] != original_user]["user_id"].unique()
            
            # Limit to reasonable number of other users
            if len(other_users) > 50:
                other_users = np.random.choice(other_users, size=50, replace=False)
            
            num_replies = np.random.binomial(n=max_replies_per_tweet, p=reply_probability)
            
            if num_replies > 0 and len(other_users) > 0:
                repliers = np.random.choice(
                    other_users,
                    size=min(num_replies, len(other_users)),
                    replace=False
                )
                
                for replier in repliers:
                    reply_time = original_time + pd.Timedelta(
                        minutes=np.random.randint(1, 720)  # Within 12 hours
                    )
                    
                    replies.append({
                        "replier_id": replier,
                        "replied_to_user_id": original_user,
                        "original_tweet_id": original_tweet_id,
                        "timestamp": reply_time.strftime("%Y-%m-%d %H:%M:%S"),
                    })
    
    reply_df = pd.DataFrame(replies)
    ensure_parent(out_csv)
    reply_df.to_csv(out_csv, index=False)
    print(f"[OK] Synthetic reply relationships → {out_csv}")
    print(f"     {len(reply_df)} reply edges created")
    
    return reply_df


def generate_all_synthetic_data(
    article_metrics_csv: Path,
    raw_root: Path,
    output_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate all synthetic user data needed for user interaction graph.
    
    Returns:
        Tuple of (user_tweet_mapping, retweet_relationships, reply_relationships)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    user_tweet = create_synthetic_user_tweet_mapping(
        article_metrics_csv,
        raw_root,
        output_dir / "synthetic_user_tweet_mapping.csv"
    )
    
    retweets = create_synthetic_retweet_relationships(
        output_dir / "synthetic_user_tweet_mapping.csv",
        output_dir / "synthetic_retweets.csv"
    )
    
    replies = create_synthetic_reply_relationships(
        output_dir / "synthetic_user_tweet_mapping.csv",
        output_dir / "synthetic_replies.csv"
    )
    
    return user_tweet, retweets, replies


if __name__ == "__main__":
    import argparse
    
    ROOT = Path(__file__).resolve().parents[2]
    RAW = ROOT / "data" / "raw" / "FakeNewsNet"
    PROC = ROOT / "data" / "processed"
    SYNTH = PROC / "synthetic"
    
    p = argparse.ArgumentParser(description="Generate synthetic user data from tweet IDs")
    p.add_argument("--metrics", type=Path, default=PROC / "article_metrics.csv")
    p.add_argument("--raw", type=Path, default=RAW)
    p.add_argument("--out", type=Path, default=SYNTH)
    args = p.parse_args()
    
    generate_all_synthetic_data(args.metrics, args.raw, args.out)

