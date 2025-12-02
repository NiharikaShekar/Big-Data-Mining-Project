"""
Community Detection: Louvain Algorithm

This module implements community detection to identify clusters of users
who frequently interact and share similar information (echo chambers).
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import networkx as nx
from tqdm import tqdm

try:
    import igraph as ig
    HAS_IGRAPH = True
except ImportError:
    HAS_IGRAPH = False
    print("[WARNING] python-igraph not available. Falling back to NetworkX Louvain implementation.")

from src.phase3.utils import ensure_parent


def detect_communities_louvain(
    graph: nx.Graph,
    resolution: float = 1.0,
    random_state: Optional[int] = None
) -> Dict[str, int]:
    """
    Detect communities using the Louvain algorithm.
    
    The Louvain algorithm maximizes modularity to identify communities
    where nodes are more densely connected within communities than between.
    
    Args:
        graph: NetworkX graph (will be converted to undirected if needed)
        resolution: Resolution parameter for modularity (higher = more communities)
        random_state: Random seed for reproducibility
    
    Returns:
        Dictionary mapping node IDs to community IDs
    """
    print(f"[INFO] Detecting communities using Louvain algorithm...")
    print(f"[INFO] Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    # Try igraph first (much faster for large graphs)
    if HAS_IGRAPH:
        try:
            print("[INFO] Using igraph for faster community detection...")
            # Convert NetworkX to igraph
            if graph.is_directed():
                print("[INFO] Converting to undirected (igraph handles this efficiently)")
                graph_undir = graph.to_undirected()
            else:
                graph_undir = graph
            
            # Convert to igraph
            edges = [(str(u), str(v)) for u, v in graph_undir.edges()]
            g_ig = ig.Graph.TupleList(edges, directed=False)
            
            # Run Louvain
            communities_ig = g_ig.community_multilevel(resolution_parameter=resolution)
            
            # Convert back to dict
            communities = {}
            for comm_id, comm_nodes in enumerate(communities_ig):
                for node in comm_nodes:
                    communities[str(node)] = comm_id
            
            num_communities = len(set(communities.values()))
            print(f"[OK] Found {num_communities} communities using igraph")
            return communities
        except Exception as e:
            print(f"[WARNING] igraph failed: {e}, falling back to python-louvain")
    
    # Convert to undirected if needed (for python-louvain)
    if graph.is_directed():
        print("[INFO] Converting directed graph to undirected (this may take a moment for large graphs)...")
        graph = graph.to_undirected()
        print("[OK] Conversion complete")
    
    # Use python-louvain if available, otherwise NetworkX
    try:
        import community.community_louvain as community_louvain
        print("[INFO] Running Louvain algorithm (this may take several minutes for large graphs)...")
        communities = community_louvain.best_partition(
            graph,
            resolution=resolution,
            random_state=random_state
        )
        print(f"[OK] Louvain communities detected using python-louvain")
    except ImportError:
        # Fallback to NetworkX greedy modularity communities
        print("[INFO] python-louvain not found, using NetworkX greedy modularity")
        print("[WARNING] This will be very slow for large graphs!")
        communities_dict = nx.community.greedy_modularity_communities(
            graph,
            weight=None,
            resolution=resolution,
            cutoff=1,
            best_n=None
        )
        # Convert to node->community mapping
        communities = {}
        for comm_id, comm_nodes in enumerate(communities_dict):
            for node in comm_nodes:
                communities[node] = comm_id
        print(f"[OK] Communities detected using NetworkX greedy modularity")
    
    num_communities = len(set(communities.values()))
    print(f"[OK] Found {num_communities} communities")
    
    return communities


def compute_modularity(
    graph: nx.Graph,
    communities: Dict[str, int],
    weight: Optional[str] = None
) -> float:
    """
    Compute modularity score for given communities.
    
    Modularity measures the quality of community structure.
    Values range from -1 to 1, where:
    - > 0: Community structure exists
    - Higher values: Better community structure
    
    Args:
        graph: NetworkX graph
        communities: Dictionary mapping nodes to community IDs
        weight: Edge weight attribute (optional)
    
    Returns:
        Modularity score
    """
    if graph.is_directed():
        graph = graph.to_undirected()
    
    # Convert communities dict to list of sets
    community_sets = {}
    for node, comm_id in communities.items():
        if comm_id not in community_sets:
            community_sets[comm_id] = set()
        community_sets[comm_id].add(node)
    
    community_list = list(community_sets.values())
    
    modularity = nx.community.modularity(
        graph,
        community_list,
        weight=weight
    )
    
    return modularity


def analyze_communities(
    graph: nx.Graph,
    communities: Dict[str, int]
) -> pd.DataFrame:
    """
    Analyze community structure and compute statistics.
    
    Args:
        graph: NetworkX graph
        communities: Dictionary mapping nodes to community IDs
    
    Returns:
        DataFrame with community statistics
    """
    if graph.is_directed():
        graph = graph.to_undirected()
    
    # Convert to sets for easier analysis
    community_sets = {}
    for node, comm_id in communities.items():
        if comm_id not in community_sets:
            community_sets[comm_id] = set()
        community_sets[comm_id].add(node)
    
    # Compute statistics for each community
    stats = []
    for comm_id, nodes in tqdm(community_sets.items(), desc="Analyzing communities"):
        subgraph = graph.subgraph(nodes)
        
        stats.append({
            "community_id": comm_id,
            "size": len(nodes),
            "num_edges": subgraph.number_of_edges(),
            "density": nx.density(subgraph),
            "avg_degree": sum(dict(subgraph.degree()).values()) / len(nodes) if len(nodes) > 0 else 0
        })
    
    df = pd.DataFrame(stats).sort_values("size", ascending=False)
    return df


def detect_user_communities(
    graph_path: Path,
    output_dir: Path,
    resolution: float = 1.0,
    random_state: Optional[int] = None,
    sample_size: Optional[int] = None
) -> Tuple[Dict[str, int], pd.DataFrame, float]:
    """
    Detect communities in user interaction graph using Louvain algorithm.
    
    Args:
        graph_path: Path to user interaction graph GraphML file
        output_dir: Directory to save results
        resolution: Resolution parameter for modularity
        random_state: Random seed for reproducibility
        sample_size: If provided, sample N nodes for faster processing
    
    Returns:
        Tuple of (communities_dict, community_stats_df, modularity_score)
    """
    print(f"\n{'='*60}")
    print("COMMUNITY DETECTION: Louvain Algorithm")
    print(f"{'='*60}\n")
    
    # Load graph
    print(f"[INFO] Loading graph from {graph_path}")
    graph = nx.read_graphml(graph_path)
    print(f"[OK] Loaded graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    # Sample graph if requested (for faster processing)
    if sample_size is not None and sample_size < graph.number_of_nodes():
        print(f"[INFO] Sampling {sample_size:,} nodes from {graph.number_of_nodes():,} for faster processing...")
        # Get largest connected component first, then sample
        if graph.is_directed():
            graph_undir = graph.to_undirected()
        else:
            graph_undir = graph
        
        # Get largest connected component
        largest_cc = max(nx.connected_components(graph_undir), key=len)
        subgraph_lcc = graph.subgraph(largest_cc)
        
        # Sample nodes
        import random
        if random_state is not None:
            random.seed(random_state)
        sampled_nodes = random.sample(list(subgraph_lcc.nodes()), min(sample_size, len(subgraph_lcc)))
        graph = graph.subgraph(sampled_nodes).copy()
        print(f"[OK] Sampled graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Detect communities
    communities = detect_communities_louvain(
        graph,
        resolution=resolution,
        random_state=random_state
    )
    
    # Compute modularity
    print("\n[INFO] Computing modularity...")
    modularity = compute_modularity(graph, communities)
    print(f"[OK] Modularity score: {modularity:.4f}")
    
    # Analyze communities
    print("\n[INFO] Analyzing community structure...")
    community_stats = analyze_communities(graph, communities)
    
    # Save results
    communities_df = pd.DataFrame([
        {"node_id": node, "community_id": comm_id}
        for node, comm_id in communities.items()
    ])
    
    communities_output = output_dir / "communities.csv"
    communities_df.to_csv(communities_output, index=False)
    print(f"[OK] Community assignments saved to {communities_output}")
    
    stats_output = output_dir / "community_statistics.csv"
    community_stats.to_csv(stats_output, index=False)
    print(f"[OK] Community statistics saved to {stats_output}")
    
    # Save modularity
    modularity_output = output_dir / "modularity_score.txt"
    with open(modularity_output, 'w') as f:
        f.write(f"Modularity: {modularity:.6f}\n")
        f.write(f"Number of communities: {len(set(communities.values()))}\n")
    
    print(f"[OK] Modularity score saved to {modularity_output}")
    
    return communities, community_stats, modularity


def compare_fake_vs_real_communities(
    graph_path: Path,
    article_metrics_path: Path,
    synthetic_data_dir: Path,
    output_dir: Path,
    resolution: float = 1.0,
    random_state: Optional[int] = None
) -> Dict[str, Tuple[Dict[str, int], pd.DataFrame, float]]:
    """
    Compare community structure between fake and real news subgraphs.
    
    Args:
        graph_path: Path to user interaction graph
        article_metrics_path: Path to article_metrics.csv
        synthetic_data_dir: Directory containing synthetic user-tweet mappings
        output_dir: Directory to save results
        resolution: Resolution parameter for modularity
        random_state: Random seed for reproducibility
    
    Returns:
        Dictionary with results for fake and real subgraphs
    """
    print(f"\n{'='*60}")
    print("COMMUNITY DETECTION: Fake vs Real Comparison")
    print(f"{'='*60}\n")
    
    # Load article labels
    print(f"[INFO] Loading article metrics from {article_metrics_path}")
    metrics = pd.read_csv(article_metrics_path)
    fake_articles = set(metrics[metrics["label"] == 0]["news_id"].astype(str))
    real_articles = set(metrics[metrics["label"] == 1]["news_id"].astype(str))
    
    print(f"[INFO] Fake articles: {len(fake_articles)}, Real articles: {len(real_articles)}")
    
    # Load user-tweet mapping
    mapping_path = synthetic_data_dir / "synthetic_user_tweet_mapping.csv"
    if not mapping_path.exists():
        raise FileNotFoundError(f"User-tweet mapping not found: {mapping_path}")
    
    print(f"[INFO] Loading user-tweet mapping from {mapping_path}")
    user_tweet_map = pd.read_csv(mapping_path)
    
    # Load graph
    print(f"[INFO] Loading user interaction graph from {graph_path}")
    graph = nx.read_graphml(graph_path)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for label_name, article_set in [("fake", fake_articles), ("real", real_articles)]:
        print(f"\n[{label_name.upper()}] Analyzing communities...")
        
        # Find users who engaged with articles of this label
        # Note: synthetic_user_tweet_mapping.csv uses 'article_id' not 'news_id'
        label_users = set(user_tweet_map[
            user_tweet_map["article_id"].astype(str).isin(article_set)
        ]["user_id"].astype(str).tolist())
        
        print(f"[INFO] Found {len(label_users)} users engaged with {label_name} news")
        
        # Create subgraph
        subgraph = graph.subgraph(label_users).copy()
        print(f"[INFO] Subgraph: {subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges")
        
        if subgraph.number_of_nodes() == 0:
            print(f"[WARNING] Empty subgraph for {label_name} news, skipping...")
            continue
        
        # Detect communities
        communities = detect_communities_louvain(
            subgraph,
            resolution=resolution,
            random_state=random_state
        )
        
        # Compute modularity
        modularity = compute_modularity(subgraph, communities)
        print(f"[OK] Modularity: {modularity:.4f}")
        
        # Analyze communities
        community_stats = analyze_communities(subgraph, communities)
        
        # Save results
        communities_df = pd.DataFrame([
            {"node_id": node, "community_id": comm_id}
            for node, comm_id in communities.items()
        ])
        
        communities_output = output_dir / f"{label_name}_communities.csv"
        communities_df.to_csv(communities_output, index=False)
        
        stats_output = output_dir / f"{label_name}_community_statistics.csv"
        community_stats.to_csv(stats_output, index=False)
        
        results[label_name] = (communities, community_stats, modularity)
    
    return results

