"""
Cascade Subgraph Builder 

Builds per-article diffusion cascades showing how information spread through users.

NOTE: With current data (only tweet IDs), we can build a simplified cascade structure
based on tweet associations. For full temporal cascades, we need timestamps and user data.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import networkx as nx
from .utils import parse_tweet_ids, ensure_parent


@dataclass
class CascadeMetrics:
    """Metrics for a single article cascade."""
    news_id: str
    cascade_size: int  # number of tweets
    cascade_depth: int  # max depth (if we had user tree structure)
    cascade_width: int  # max width at any level
    has_timestamps: bool
    tweet_ids: List[str]


def build_article_cascade_subgraph(
    article_metrics_csv: Path,
    raw_root: Path,
    news_id: str,
    synthetic_user_data: Optional[Path] = None,
    out_graphml: Optional[Path] = None,
) -> nx.DiGraph:
    """
    Build a cascade subgraph for a single article.
    
    With synthetic user data, this creates a full cascade:
    - Root node: Article
    - User nodes: Users who posted/retweeted/replied
    - Edges: Article → User → User (retweet/reply chains)
    - Timestamps: Order the cascade temporally
    
    Args:
        article_metrics_csv: Path to article_metrics.csv
        raw_root: Root directory with raw CSV files
        news_id: News article ID to build cascade for
        synthetic_user_data: Directory with synthetic user data (if available)
        out_graphml: Optional output path
    
    Returns:
        Directed graph representing the cascade
    """
    # Load article metrics
    metrics = pd.read_csv(article_metrics_csv, low_memory=False)
    metrics = metrics[metrics["news_id"] == news_id]
    
    if len(metrics) == 0:
        raise ValueError(f"Article {news_id} not found in metrics")
    
    # Load raw data to get tweet IDs
    from .article_metrics import _stack_raw
    raw = _stack_raw(raw_root)
    raw = raw[raw["id"] == news_id]
    
    if len(raw) == 0:
        raise ValueError(f"Article {news_id} not found in raw data")
    
    tweet_list = parse_tweet_ids(raw.iloc[0]["tweet_ids"])
    
    # Build cascade graph
    G = nx.DiGraph()
    
    # Root node: article
    article_node = f"A_{news_id}"
    G.add_node(article_node, node_type="article", news_id=news_id)
    
    # Try to use synthetic user data for full cascade
    if synthetic_user_data:
        synth_dir = Path(synthetic_user_data)
        user_tweet_csv = synth_dir / "synthetic_user_tweet_mapping.csv"
        retweets_csv = synth_dir / "synthetic_retweets.csv"
        replies_csv = synth_dir / "synthetic_replies.csv"
        
        if user_tweet_csv.exists():
            # Load synthetic data
            user_tweet = pd.read_csv(user_tweet_csv, low_memory=False)
            article_users = user_tweet[user_tweet["article_id"] == news_id]
            
            # Add user nodes and article → user edges
            for _, row in article_users.iterrows():
                user_id = str(row["user_id"])
                G.add_node(user_id, node_type="user", user_id=user_id)
                G.add_edge(article_node, user_id, 
                          edge_type="posts", 
                          tweet_id=str(row["tweet_id"]),
                          timestamp=str(row["timestamp"]))
            
            # Add retweet edges (user → user)
            if retweets_csv.exists():
                retweets = pd.read_csv(retweets_csv, low_memory=False)
                article_retweets = retweets[retweets["tweet_id"].isin(article_users["tweet_id"])]
                for _, row in article_retweets.iterrows():
                    retweeter = str(row["retweeter_id"])
                    original = str(row["original_user_id"])
                    if retweeter in G and original in G:
                        G.add_edge(retweeter, original,
                                  edge_type="retweet",
                                  tweet_id=str(row["tweet_id"]),
                                  timestamp=str(row["timestamp"]))
            
            # Add reply edges (user → user)
            if replies_csv.exists():
                replies = pd.read_csv(replies_csv, low_memory=False)
                article_replies = replies[replies["original_tweet_id"].isin(article_users["tweet_id"])]
                for _, row in article_replies.iterrows():
                    replier = str(row["replier_id"])
                    replied_to = str(row["replied_to_user_id"])
                    if replier in G and replied_to in G:
                        G.add_edge(replier, replied_to,
                                  edge_type="reply",
                                  tweet_id=str(row["original_tweet_id"]),
                                  timestamp=str(row["timestamp"]))
        else:
            # Fallback: simplified structure (article → tweets)
            for tid in tweet_list:
                tweet_node = f"T_{tid}"
                G.add_node(tweet_node, node_type="tweet", tweet_id=tid)
                G.add_edge(article_node, tweet_node, edge_type="association")
    else:
        # Fallback: simplified structure (article → tweets)
        for tid in tweet_list:
            tweet_node = f"T_{tid}"
            G.add_node(tweet_node, node_type="tweet", tweet_id=tid)
            G.add_edge(article_node, tweet_node, edge_type="association")
    
    if out_graphml:
        ensure_parent(out_graphml)
        nx.write_graphml(G, out_graphml)
        print(f"[OK] Cascade for {news_id} → {out_graphml}")
        print(f"     |V|={G.number_of_nodes()}, |E|={G.number_of_edges()}")
    
    return G


def build_all_cascades(
    article_metrics_csv: Path,
    raw_root: Path,
    out_dir: Path,
    max_articles: Optional[int] = None,
    synthetic_user_data: Optional[Path] = None,
) -> Dict[str, CascadeMetrics]:
    """
    Build cascade subgraphs for all articles (or a subset).
    
    Args:
        article_metrics_csv: Path to article_metrics.csv
        raw_root: Root directory with raw CSV files
        out_dir: Directory to save individual cascade graphs
        max_articles: Limit number of articles (for testing)
    
    Returns:
        Dictionary mapping news_id -> CascadeMetrics
    """
    metrics = pd.read_csv(article_metrics_csv, low_memory=False)
    
    if max_articles:
        metrics = metrics.head(max_articles)
    
    cascade_metrics = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    
    from .article_metrics import _stack_raw
    raw = _stack_raw(raw_root)
    raw_dict = dict(zip(raw["id"].tolist(), raw["tweet_ids"].tolist()))
    
    for _, row in metrics.iterrows():
        news_id = str(row["news_id"])
        tweet_list = parse_tweet_ids(raw_dict.get(news_id, ""))
        
        cascade_metrics[news_id] = CascadeMetrics(
            news_id=news_id,
            cascade_size=len(tweet_list),
            cascade_depth=1,  # Simplified: depth=1 (article -> tweets)
            cascade_width=len(tweet_list),  # All tweets at same level
            has_timestamps=False,  # No timestamp data available
            tweet_ids=tweet_list,
        )
        
        # Save individual cascade graph
        cascade_path = out_dir / f"cascade_{news_id}.graphml"
        build_article_cascade_subgraph(
            article_metrics_csv, raw_root, news_id, synthetic_user_data, cascade_path
        )
    
    # Save summary CSV
    summary_df = pd.DataFrame([
        {
            "news_id": m.news_id,
            "cascade_size": m.cascade_size,
            "cascade_depth": m.cascade_depth,
            "cascade_width": m.cascade_width,
            "has_timestamps": m.has_timestamps,
        }
        for m in cascade_metrics.values()
    ])
    summary_path = out_dir / "cascade_metrics_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"[OK] Cascade summary → {summary_path}")
    print(f"     Processed {len(cascade_metrics)} articles")
    
    return cascade_metrics


if __name__ == "__main__":
    import argparse
    
    ROOT = Path(__file__).resolve().parents[2]
    RAW = ROOT / "data" / "raw" / "FakeNewsNet"
    PROC = ROOT / "data" / "processed"
    CASCADES = PROC / "cascades"
    
    p = argparse.ArgumentParser(description="Build cascade subgraphs")
    p.add_argument("--metrics", type=Path, default=PROC / "article_metrics.csv")
    p.add_argument("--raw", type=Path, default=RAW)
    p.add_argument("--out", type=Path, default=CASCADES)
    p.add_argument("--max", type=int, help="Limit number of articles")
    p.add_argument("--article-id", type=str, help="Build cascade for single article")
    args = p.parse_args()
    
    synthetic_dir = PROC / "synthetic"
    
    if args.article_id:
        G = build_article_cascade_subgraph(
            args.metrics, args.raw, args.article_id, synthetic_dir,
            args.out / f"cascade_{args.article_id}.graphml"
        )
    else:
        build_all_cascades(args.metrics, args.raw, args.out, args.max, synthetic_dir)

