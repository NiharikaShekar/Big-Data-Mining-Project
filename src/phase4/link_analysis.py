"""
Link Analysis: PageRank and HITS Algorithms

This module implements link analysis algorithms to identify influential users
in the misinformation cascade network.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, Optional
import pandas as pd
import networkx as nx
from tqdm import tqdm

from src.phase3.utils import ensure_parent


def compute_pagerank(
    graph: nx.DiGraph,
    damping_factor: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
    weight: Optional[str] = None
) -> Dict[str, float]:
    """
    Compute PageRank scores for nodes in a directed graph.
    
    PageRank measures the importance of nodes based on the structure
    of incoming links. Higher scores indicate more influential nodes.
    
    Args:
        graph: Directed NetworkX graph
        damping_factor: Damping parameter (default: 0.85)
        max_iter: Maximum iterations for convergence
        tol: Tolerance for convergence
        weight: Edge weight attribute (optional)
    
    Returns:
        Dictionary mapping node IDs to PageRank scores
    """
    print(f"[INFO] Computing PageRank on graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    # Use NetworkX's built-in PageRank
    pagerank_scores = nx.pagerank(
        graph,
        alpha=damping_factor,
        max_iter=max_iter,
        tol=tol,
        weight=weight
    )
    
    print(f"[OK] PageRank computed for {len(pagerank_scores)} nodes")
    return pagerank_scores


def compute_hits(
    graph: nx.DiGraph,
    max_iter: int = 100,
    tol: float = 1e-8,
    normalized: bool = True
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Compute HITS (Hyperlink-Induced Topic Search) scores.
    
    HITS assigns two scores to each node:
    - Authority: How often this node's content is referenced
    - Hub: How often this node references other authoritative content
    
    Args:
        graph: Directed NetworkX graph
        max_iter: Maximum iterations for convergence
        tol: Tolerance for convergence
        normalized: Whether to normalize scores
    
    Returns:
        Tuple of (authority_scores, hub_scores) dictionaries
    """
    print(f"[INFO] Computing HITS on graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    # Use NetworkX's built-in HITS
    authority_scores, hub_scores = nx.hits(
        graph,
        max_iter=max_iter,
        tol=tol,
        normalized=normalized
    )
    
    print(f"[OK] HITS computed for {len(authority_scores)} nodes")
    return authority_scores, hub_scores


def get_top_influencers(
    scores: Dict[str, float],
    top_k: int = 20,
    label: str = "score"
) -> pd.DataFrame:
    """
    Get top-k nodes ranked by their scores.
    
    Args:
        scores: Dictionary mapping node IDs to scores
        top_k: Number of top nodes to return
        label: Name for the score column
    
    Returns:
        DataFrame with columns: node_id, score, rank
    """
    # Convert to DataFrame and sort
    df = pd.DataFrame([
        {"node_id": node, label: score}
        for node, score in scores.items()
    ])
    
    df = df.sort_values(label, ascending=False).head(top_k)
    df["rank"] = range(1, len(df) + 1)
    
    return df[["rank", "node_id", label]]


def analyze_user_interaction_graph(
    graph_path: Path,
    output_dir: Path,
    damping_factor: float = 0.85,
    top_k: int = 50
) -> Dict[str, pd.DataFrame]:
    """
    Run PageRank and HITS analysis on user interaction graph.
    
    Args:
        graph_path: Path to user interaction graph GraphML file
        output_dir: Directory to save results
        damping_factor: PageRank damping factor
        top_k: Number of top influencers to extract
    
    Returns:
        Dictionary with results DataFrames
    """
    print(f"\n{'='*60}")
    print("LINK ANALYSIS: User Interaction Graph")
    print(f"{'='*60}\n")
    
    # Load graph
    print(f"[INFO] Loading graph from {graph_path}")
    graph = nx.read_graphml(graph_path)
    print(f"[OK] Loaded graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    # Convert to directed if needed
    if not graph.is_directed():
        print("[INFO] Converting to directed graph...")
        graph = graph.to_directed()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # 1. PageRank
    print("\n[1/2] Computing PageRank...")
    pagerank_scores = compute_pagerank(graph, damping_factor=damping_factor)
    
    pagerank_df = get_top_influencers(pagerank_scores, top_k=top_k, label="pagerank_score")
    pagerank_output = output_dir / "pagerank_top_influencers.csv"
    pagerank_df.to_csv(pagerank_output, index=False)
    print(f"[OK] Top {top_k} PageRank influencers saved to {pagerank_output}")
    results["pagerank"] = pagerank_df
    
    # 2. HITS
    print("\n[2/2] Computing HITS (Authority & Hub scores)...")
    authority_scores, hub_scores = compute_hits(graph)
    
    authority_df = get_top_influencers(authority_scores, top_k=top_k, label="authority_score")
    authority_output = output_dir / "hits_top_authorities.csv"
    authority_df.to_csv(authority_output, index=False)
    print(f"[OK] Top {top_k} authorities saved to {authority_output}")
    results["authorities"] = authority_df
    
    hub_df = get_top_influencers(hub_scores, top_k=top_k, label="hub_score")
    hub_output = output_dir / "hits_top_hubs.csv"
    hub_df.to_csv(hub_output, index=False)
    print(f"[OK] Top {top_k} hubs saved to {hub_output}")
    results["hubs"] = hub_df
    
    # Save all scores
    all_scores_df = pd.DataFrame({
        "node_id": list(pagerank_scores.keys()),
        "pagerank": [pagerank_scores.get(n, 0) for n in pagerank_scores.keys()],
        "authority": [authority_scores.get(n, 0) for n in pagerank_scores.keys()],
        "hub": [hub_scores.get(n, 0) for n in pagerank_scores.keys()]
    })
    
    all_scores_output = output_dir / "link_analysis_all_scores.csv"
    all_scores_df.to_csv(all_scores_output, index=False)
    print(f"[OK] All scores saved to {all_scores_output}")
    results["all_scores"] = all_scores_df
    
    return results


def analyze_fake_vs_real_subgraphs(
    graph_path: Path,
    article_metrics_path: Path,
    synthetic_data_dir: Path,
    output_dir: Path,
    damping_factor: float = 0.85,
    top_k: int = 50
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Compare PageRank and HITS between fake and real news subgraphs.
    
    This creates separate subgraphs for fake and real news articles
    and runs link analysis on each to identify differences.
    
    Args:
        graph_path: Path to user interaction graph
        article_metrics_path: Path to article_metrics.csv
        synthetic_data_dir: Directory containing synthetic user-tweet mappings
        output_dir: Directory to save results
        damping_factor: PageRank damping factor
        top_k: Number of top influencers to extract
    
    Returns:
        Nested dictionary: {"fake": {...}, "real": {...}}
    """
    print(f"\n{'='*60}")
    print("LINK ANALYSIS: Fake vs Real News Comparison")
    print(f"{'='*60}\n")
    
    # Load article labels
    print(f"[INFO] Loading article metrics from {article_metrics_path}")
    metrics = pd.read_csv(article_metrics_path)
    fake_articles = set(metrics[metrics["label"] == 0]["news_id"].astype(str))
    real_articles = set(metrics[metrics["label"] == 1]["news_id"].astype(str))
    
    print(f"[INFO] Fake articles: {len(fake_articles)}, Real articles: {len(real_articles)}")
    
    # Load user-tweet mapping to identify which users engaged with which articles
    mapping_path = synthetic_data_dir / "synthetic_user_tweet_mapping.csv"
    if not mapping_path.exists():
        raise FileNotFoundError(f"User-tweet mapping not found: {mapping_path}")
    
    print(f"[INFO] Loading user-tweet mapping from {mapping_path}")
    user_tweet_map = pd.read_csv(mapping_path)
    
    # Load graph
    print(f"[INFO] Loading user interaction graph from {graph_path}")
    graph = nx.read_graphml(graph_path)
    if not graph.is_directed():
        graph = graph.to_directed()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {"fake": {}, "real": {}}
    
    for label_name, article_set in [("fake", fake_articles), ("real", real_articles)]:
        print(f"\n[{label_name.upper()}] Analyzing subgraph...")
        
        # Find users who engaged with articles of this label
        # Note: synthetic_user_tweet_mapping.csv uses 'article_id' not 'news_id'
        label_tweets = user_tweet_map[
            user_tweet_map["article_id"].astype(str).isin(article_set)
        ]["tweet_id"].astype(str).tolist()
        
        label_users = set(user_tweet_map[
            user_tweet_map["article_id"].astype(str).isin(article_set)
        ]["user_id"].astype(str).tolist())
        
        print(f"[INFO] Found {len(label_users)} users engaged with {label_name} news")
        
        # Create subgraph with these users
        subgraph = graph.subgraph(label_users).copy()
        print(f"[INFO] Subgraph: {subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges")
        
        if subgraph.number_of_nodes() == 0:
            print(f"[WARNING] Empty subgraph for {label_name} news, skipping...")
            continue
        
        # PageRank
        print(f"\n[{label_name}] Computing PageRank...")
        pagerank_scores = compute_pagerank(subgraph, damping_factor=damping_factor)
        pagerank_df = get_top_influencers(pagerank_scores, top_k=top_k, label="pagerank_score")
        
        pagerank_output = output_dir / f"{label_name}_pagerank_top_influencers.csv"
        pagerank_df.to_csv(pagerank_output, index=False)
        results[label_name]["pagerank"] = pagerank_df
        
        # HITS
        print(f"\n[{label_name}] Computing HITS...")
        authority_scores, hub_scores = compute_hits(subgraph)
        
        authority_df = get_top_influencers(authority_scores, top_k=top_k, label="authority_score")
        authority_output = output_dir / f"{label_name}_hits_top_authorities.csv"
        authority_df.to_csv(authority_output, index=False)
        results[label_name]["authorities"] = authority_df
        
        hub_df = get_top_influencers(hub_scores, top_k=top_k, label="hub_score")
        hub_output = output_dir / f"{label_name}_hits_top_hubs.csv"
        hub_df.to_csv(hub_output, index=False)
        results[label_name]["hubs"] = hub_df
    
    return results

