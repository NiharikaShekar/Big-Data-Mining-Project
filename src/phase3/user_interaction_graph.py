"""
User Interaction Graph Builder

Builds user interaction graph from FakeNewsNet JSON dataset structure.

Expected dataset structure (from FakeNewsNet full dataset):
├── gossipcop/
│   ├── fake/
│   │   ├── gossipcop-1/
│   │   │   ├── tweets/
│   │   │   │   ├── <tweet_id>.json
│   │   │   └── retweets/
│   │   │       ├── <tweet_id>.json
│   └── real/...
├── politifact/...
├── user_profiles/
│   ├── <user_id>.json
└── user_followers/
    ├── <user_id>.json

Tweet JSON format (from Twitter API):
- user.id: User ID who posted
- created_at: Timestamp
- retweeted_status: Original tweet (if this is a retweet)
- in_reply_to_user_id: User being replied to
- entities.user_mentions: Users mentioned

Retweet JSON: Array of retweet objects with user.id and created_at
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import json
import pandas as pd
import networkx as nx
from .utils import ensure_parent


def _parse_tweet_json(tweet_path: Path) -> Optional[dict]:
    """Parse a single tweet JSON file and extract user interaction data."""
    try:
        with open(tweet_path, 'r', encoding='utf-8') as f:
            tweet = json.load(f)
        
        user_id = tweet.get('user', {}).get('id_str') or str(tweet.get('user', {}).get('id', ''))
        if not user_id:
            return None
        
        return {
            'tweet_id': str(tweet.get('id_str') or tweet.get('id', '')),
            'user_id': user_id,
            'created_at': tweet.get('created_at', ''),
            'is_retweet': 'retweeted_status' in tweet,
            'original_user_id': tweet.get('retweeted_status', {}).get('user', {}).get('id_str') if 'retweeted_status' in tweet else None,
            'in_reply_to_user_id': tweet.get('in_reply_to_user_id_str') or str(tweet.get('in_reply_to_user_id', '')) if tweet.get('in_reply_to_user_id') else None,
            'mentioned_user_ids': [m.get('id_str') or str(m.get('id', '')) for m in tweet.get('entities', {}).get('user_mentions', [])],
        }
    except Exception as e:
        return None


def _parse_retweets_json(retweets_path: Path) -> list[dict]:
    """Parse retweets JSON file and extract retweet relationships."""
    retweets = []
    try:
        with open(retweets_path, 'r', encoding='utf-8') as f:
            retweet_list = json.load(f)
        
        if not isinstance(retweet_list, list):
            return retweets
        
        for rt in retweet_list:
            retweeter_id = rt.get('user', {}).get('id_str') or str(rt.get('user', {}).get('id', ''))
            original_user_id = rt.get('retweeted_status', {}).get('user', {}).get('id_str') if rt.get('retweeted_status') else None
            
            if retweeter_id and original_user_id:
                retweets.append({
                    'retweeter_id': retweeter_id,
                    'original_user_id': original_user_id,
                    'tweet_id': str(rt.get('retweeted_status', {}).get('id_str') or rt.get('retweeted_status', {}).get('id', '')),
                    'created_at': rt.get('created_at', ''),
                })
    except Exception:
        pass
    
    return retweets


def build_user_interaction_graph_from_synthetic(
    synthetic_dir: Path,
    out_graphml: Optional[Path] = None,
) -> nx.DiGraph:
    """
    Build user interaction graph from synthetic user data.
    
    Args:
        synthetic_dir: Directory containing synthetic CSV files:
            - synthetic_user_tweet_mapping.csv
            - synthetic_retweets.csv
            - synthetic_replies.csv
        out_graphml: Output path for GraphML file
    
    Returns:
        Directed graph with user nodes and interaction edges
    """
    G = nx.DiGraph()
    
    user_tweet_path = synthetic_dir / "synthetic_user_tweet_mapping.csv"
    retweets_path = synthetic_dir / "synthetic_retweets.csv"
    replies_path = synthetic_dir / "synthetic_replies.csv"
    
    if not user_tweet_path.exists():
        print(f"[WARNING] Synthetic data not found in {synthetic_dir}")
        print("[INFO] Run: python -m src.phase3.generate_synthetic_users")
        return G
    
    # Load user-tweet mapping
    user_tweet = pd.read_csv(user_tweet_path, low_memory=False)
    
    # Add all users as nodes
    for user_id in user_tweet["user_id"].unique():
        G.add_node(str(user_id), node_type="user")
    
    # Add retweet edges
    if retweets_path.exists():
        retweets = pd.read_csv(retweets_path, low_memory=False)
        for _, row in retweets.iterrows():
            retweeter = str(row["retweeter_id"])
            original = str(row["original_user_id"])
            G.add_edge(
                retweeter, original,
                interaction_type="retweet",
                tweet_id=str(row["tweet_id"]),
                timestamp=str(row["timestamp"])
            )
        print(f"[INFO] Added {len(retweets)} retweet edges")
    
    # Add reply edges
    if replies_path.exists():
        replies = pd.read_csv(replies_path, low_memory=False)
        for _, row in replies.iterrows():
            replier = str(row["replier_id"])
            replied_to = str(row["replied_to_user_id"])
            G.add_edge(
                replier, replied_to,
                interaction_type="reply",
                tweet_id=str(row["original_tweet_id"]),
                timestamp=str(row["timestamp"])
            )
        print(f"[INFO] Added {len(replies)} reply edges")
    
    print(f"[INFO] Graph: |V|={G.number_of_nodes()}, |E|={G.number_of_edges()}")
    
    if out_graphml:
        ensure_parent(out_graphml)
        nx.write_graphml(G, out_graphml)
        print(f"[OK] User interaction graph → {out_graphml}")
    
    return G


def build_user_interaction_graph(
    fakenewsnet_root: Path,
    out_graphml: Optional[Path] = None,
    synthetic_dir: Optional[Path] = None,
) -> nx.DiGraph:
    """
    Build a directed user interaction graph from FakeNewsNet JSON dataset or synthetic data.
    
    Args:
        fakenewsnet_root: Root directory of full FakeNewsNet dataset (with JSON structure)
        out_graphml: Output path for GraphML file
        synthetic_dir: If provided, use synthetic data instead of JSON dataset
    
    Returns:
        Directed graph with user nodes and interaction edges
    """
    # Try synthetic data first (for course project)
    if synthetic_dir:
        synthetic_path = Path(synthetic_dir)
        if synthetic_path.exists():
            return build_user_interaction_graph_from_synthetic(synthetic_path, out_graphml)
    
    G = nx.DiGraph()
    
    if not fakenewsnet_root.exists():
        print(f"[WARNING] FakeNewsNet root not found: {fakenewsnet_root}")
        print("[INFO] Trying synthetic data generation...")
        
        # Try to generate synthetic data
        from .generate_synthetic_users import generate_all_synthetic_data
        from pathlib import Path as P
        
        proc_dir = fakenewsnet_root.parent.parent / "processed"
        synth_dir = proc_dir / "synthetic"
        metrics_csv = proc_dir / "article_metrics.csv"
        raw_root = fakenewsnet_root.parent / "FakeNewsNet"
        
        if metrics_csv.exists() and raw_root.exists():
            print("[INFO] Generating synthetic user data...")
            generate_all_synthetic_data(metrics_csv, raw_root, synth_dir)
            return build_user_interaction_graph_from_synthetic(synth_dir, out_graphml)
        
        print("[INFO] To build user interaction graph, you need:")
        print("  1. Full JSON dataset, OR")
        print("  2. Run: python -m src.phase3.generate_synthetic_users")
        return G
    
    domains = ["gossipcop", "politifact"]
    labels = ["fake", "real"]
    
    tweet_count = 0
    retweet_count = 0
    
    # Process tweets and retweets from each article folder
    for domain in domains:
        for label in labels:
            domain_path = fakenewsnet_root / domain / label
            if not domain_path.exists():
                continue
            
            # Find all article folders (e.g., gossipcop-1, politifact-15014)
            for article_dir in domain_path.iterdir():
                if not article_dir.is_dir():
                    continue
                
                # Process tweets
                tweets_dir = article_dir / "tweets"
                if tweets_dir.exists():
                    for tweet_file in tweets_dir.glob("*.json"):
                        tweet_data = _parse_tweet_json(tweet_file)
                        if not tweet_data:
                            continue
                        
                        user_id = tweet_data['user_id']
                        G.add_node(user_id, node_type='user')
                        
                        # Add retweet edge (if this tweet is a retweet)
                        if tweet_data['is_retweet'] and tweet_data['original_user_id']:
                            orig_user = tweet_data['original_user_id']
                            G.add_node(orig_user, node_type='user')
                            G.add_edge(user_id, orig_user,
                                      interaction_type='retweet',
                                      tweet_id=tweet_data['tweet_id'],
                                      timestamp=tweet_data['created_at'])
                            retweet_count += 1
                        
                        # Add reply edge
                        if tweet_data['in_reply_to_user_id']:
                            reply_to = tweet_data['in_reply_to_user_id']
                            G.add_node(reply_to, node_type='user')
                            G.add_edge(user_id, reply_to,
                                      interaction_type='reply',
                                      tweet_id=tweet_data['tweet_id'],
                                      timestamp=tweet_data['created_at'])
                        
                        # Add mention edges
                        for mentioned_id in tweet_data['mentioned_user_ids']:
                            if mentioned_id:
                                G.add_node(mentioned_id, node_type='user')
                                G.add_edge(user_id, mentioned_id,
                                          interaction_type='mention',
                                          tweet_id=tweet_data['tweet_id'],
                                          timestamp=tweet_data['created_at'])
                        
                        tweet_count += 1
                
                # Process retweets folder
                retweets_dir = article_dir / "retweets"
                if retweets_dir.exists():
                    for retweet_file in retweets_dir.glob("*.json"):
                        retweets = _parse_retweets_json(retweet_file)
                        for rt in retweets:
                            retweeter = rt['retweeter_id']
                            original = rt['original_user_id']
                            G.add_node(retweeter, node_type='user')
                            G.add_node(original, node_type='user')
                            G.add_edge(retweeter, original,
                                      interaction_type='retweet',
                                      tweet_id=rt['tweet_id'],
                                      timestamp=rt['created_at'])
                            retweet_count += 1
    
    print(f"[INFO] Processed {tweet_count} tweets, {retweet_count} retweets")
    print(f"[INFO] Graph: |V|={G.number_of_nodes()}, |E|={G.number_of_edges()}")
    
    if out_graphml:
        ensure_parent(out_graphml)
        nx.write_graphml(G, out_graphml)
        print(f"[OK] User interaction graph → {out_graphml}")
    
    return G


def build_fake_vs_real_subgraphs(
    user_graph: nx.DiGraph,
    article_metrics_csv: Path,
    out_fake_graphml: Path,
    out_real_graphml: Path,
) -> tuple[nx.DiGraph, nx.DiGraph]:
    """
    Split the user interaction graph into fake and real news subgraphs.
    
    Args:
        user_graph: Full user interaction graph
        article_metrics_csv: CSV with news_id and label columns
        out_fake_graphml: Output path for fake news subgraph
        out_real_graphml: Output path for real news subgraph
    
    Returns:
        Tuple of (fake_subgraph, real_subgraph)
    """
    # TODO: Implement when user graph is available
    # This would filter edges based on which articles (fake/real) they're associated with
    fake_G = nx.DiGraph()
    real_G = nx.DiGraph()
    
    if out_fake_graphml:
        ensure_parent(out_fake_graphml)
        nx.write_graphml(fake_G, out_fake_graphml)
    
    if out_real_graphml:
        ensure_parent(out_real_graphml)
        nx.write_graphml(real_G, out_real_graphml)
    
    return fake_G, real_G


if __name__ == "__main__":
    import argparse
    
    ROOT = Path(__file__).resolve().parents[2]
    RAW = ROOT / "data" / "raw" / "FakeNewsNet_full"
    PROC = ROOT / "data" / "processed"
    GRAPHS = PROC / "graphs"
    
    p = argparse.ArgumentParser(description="Build user interaction graph from FakeNewsNet JSON")
    p.add_argument("--fakenewsnet-root", type=Path, default=RAW,
                   help="Root directory of full FakeNewsNet dataset (with JSON structure)")
    p.add_argument("--synthetic-dir", type=Path, default=PROC / "synthetic",
                   help="Directory with synthetic user data CSV files")
    p.add_argument("--out", type=Path, default=GRAPHS / "user_interaction.graphml")
    args = p.parse_args()
    
    G = build_user_interaction_graph(args.fakenewsnet_root, args.out, args.synthetic_dir)

