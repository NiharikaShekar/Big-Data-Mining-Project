"""
Independent Cascade Model (ICM) for Temporal Diffusion Simulation

This module implements the Independent Cascade Model to simulate how information
spreads through the user interaction network. It computes cascade metrics including
depth, width, spread rate, and compares fake vs real news diffusion patterns.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd
import networkx as nx
import numpy as np
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class CascadeSimulationResult:
    """Results from a single ICM simulation."""
    article_id: str
    label: str  # 'fake' or 'real'
    activated_nodes: Set[str]
    activation_times: Dict[str, int]  # node -> time step
    cascade_depth: int
    cascade_width: int
    cascade_size: int
    spread_rate: float  # nodes activated per time step (average)
    max_width_level: int  # time step with maximum width


@dataclass
class CascadeMetrics:
    """Aggregated metrics for cascade analysis."""
    article_id: str
    label: str
    cascade_size: int
    cascade_depth: int
    max_width: int
    avg_spread_rate: float
    time_to_max_width: int
    total_time_steps: int


def simulate_icm(
    graph: nx.DiGraph,
    seed_nodes: Set[str],
    activation_prob: float = 0.1,
    max_iterations: int = 50,
    random_state: Optional[int] = None
) -> Tuple[Set[str], Dict[str, int]]:
    """
    Simulate Independent Cascade Model (ICM) on a directed graph.
    
    In ICM:
    - Each activated node has one chance to activate its neighbors
    - Activation succeeds with probability `activation_prob`
    - Process continues until no new activations occur
    
    Args:
        graph: Directed graph representing user interactions
        seed_nodes: Initial set of activated nodes
        activation_prob: Probability of successful activation (default: 0.1)
        max_iterations: Maximum number of time steps
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (activated_nodes, activation_times)
        - activated_nodes: Set of all nodes that became activated
        - activation_times: Dict mapping node -> time step when activated
    """
    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)
    
    activated = set(seed_nodes)
    activation_times = {node: 0 for node in seed_nodes}
    newly_activated = set(seed_nodes)
    
    for t in range(1, max_iterations + 1):
        next_activated = set()
        
        # Each newly activated node tries to activate its neighbors
        for node in newly_activated:
            if node not in graph:
                continue
            
            # Get out-neighbors (nodes this user influences)
            neighbors = list(graph.successors(node))
            
            for neighbor in neighbors:
                # Skip if already activated
                if neighbor in activated:
                    continue
                
                # Try to activate with probability activation_prob
                if random.random() < activation_prob:
                    next_activated.add(neighbor)
        
        # Update activated set
        if not next_activated:
            break  # No new activations, cascade stopped
        
        activated.update(next_activated)
        for node in next_activated:
            activation_times[node] = t
        
        newly_activated = next_activated
    
    return activated, activation_times


def simulate_lt(
    graph: nx.DiGraph,
    seed_nodes: Set[str],
    threshold_distribution: str = "uniform",
    max_iterations: int = 50,
    random_state: Optional[int] = None
) -> Tuple[Set[str], Dict[str, int]]:
    """
    Simulate Linear Threshold (LT) Model on a directed graph.
    
    In LT:
    - Each node has a threshold (randomly assigned)
    - A node activates when sum of influences from activated neighbors >= threshold
    - Influence weights are uniform (1/degree) or based on edge weights
    
    Args:
        graph: Directed graph representing user interactions
        seed_nodes: Initial set of activated nodes
        threshold_distribution: How to assign thresholds ("uniform" or "normal")
        max_iterations: Maximum number of time steps
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (activated_nodes, activation_times)
        - activated_nodes: Set of all nodes that became activated
        - activation_times: Dict mapping node -> time step when activated
    """
    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)
    
    # Assign thresholds to all nodes
    thresholds = {}
    for node in graph.nodes():
        if threshold_distribution == "uniform":
            thresholds[node] = random.uniform(0.1, 0.5)  # Threshold between 0.1-0.5
        else:  # normal
            thresholds[node] = max(0.1, min(0.9, np.random.normal(0.3, 0.1)))
    
    # Compute influence weights (uniform: 1/indegree for each incoming edge)
    influence_weights = {}
    for node in graph.nodes():
        in_neighbors = list(graph.predecessors(node))
        if in_neighbors:
            weight = 1.0 / len(in_neighbors)  # Uniform weight distribution
            influence_weights[node] = {pred: weight for pred in in_neighbors}
        else:
            influence_weights[node] = {}
    
    activated = set(seed_nodes)
    activation_times = {node: 0 for node in seed_nodes}
    newly_activated = set(seed_nodes)
    
    for t in range(1, max_iterations + 1):
        next_activated = set()
        
        # Check all inactive nodes
        for node in graph.nodes():
            if node in activated:
                continue
            
            # Calculate total influence from activated neighbors
            total_influence = 0.0
            for neighbor in graph.predecessors(node):
                if neighbor in activated:
                    if node in influence_weights and neighbor in influence_weights[node]:
                        total_influence += influence_weights[node][neighbor]
                    else:
                        # Fallback: uniform weight
                        in_degree = graph.in_degree(node)
                        if in_degree > 0:
                            total_influence += 1.0 / in_degree
            
            # Activate if influence >= threshold
            if total_influence >= thresholds.get(node, 0.5):
                next_activated.add(node)
        
        # Update activated set
        if not next_activated:
            break  # No new activations, cascade stopped
        
        activated.update(next_activated)
        for node in next_activated:
            activation_times[node] = t
        
        newly_activated = next_activated
    
    return activated, activation_times


def compute_cascade_metrics(
    activated_nodes: Set[str],
    activation_times: Dict[str, int]
) -> Tuple[int, int, float, int]:
    """
    Compute cascade metrics from activation results.
    
    Args:
        activated_nodes: Set of activated nodes
        activation_times: Dict mapping node -> activation time
    
    Returns:
        Tuple of (depth, max_width, spread_rate, max_width_level)
        - depth: Maximum time step reached
        - max_width: Maximum number of nodes activated in a single time step
        - spread_rate: Average nodes activated per time step
        - max_width_level: Time step with maximum width
    """
    if not activation_times:
        return 0, 0, 0.0, 0
    
    # Compute depth (max time step)
    depth = max(activation_times.values()) if activation_times else 0
    
    # Compute width at each time step
    width_by_time = {}
    for node, time in activation_times.items():
        width_by_time[time] = width_by_time.get(time, 0) + 1
    
    # Find max width and its time step
    if width_by_time:
        max_width = max(width_by_time.values())
        max_width_level = max(width_by_time.items(), key=lambda x: x[1])[0]
    else:
        max_width = len(activated_nodes)
        max_width_level = 0
    
    # Compute average spread rate
    if depth > 0:
        spread_rate = len(activated_nodes) / depth
    else:
        spread_rate = len(activated_nodes)
    
    return depth, max_width, spread_rate, max_width_level


def simulate_article_cascade_icm(
    graph: nx.DiGraph,
    article_id: str,
    user_tweet_map: pd.DataFrame,
    activation_prob: float = 0.1,
    max_iterations: int = 50,
    random_state: Optional[int] = None
) -> Optional[CascadeSimulationResult]:
    """
    Simulate cascade for a single article.
    
    Args:
        graph: User interaction graph
        article_id: Article ID
        user_tweet_map: DataFrame with columns [user_id, article_id, tweet_id]
        activation_prob: ICM activation probability
        max_iterations: Maximum time steps
        random_state: Random seed
    
    Returns:
        CascadeSimulationResult or None if no users found for article
    """
    # Find seed nodes (users who posted tweets about this article)
    article_users = user_tweet_map[user_tweet_map["article_id"] == article_id]["user_id"].unique()
    seed_nodes = set(str(uid) for uid in article_users if str(uid) in graph)
    
    if not seed_nodes:
        return None
    
    # Run ICM simulation
    activated, times = simulate_icm(
        graph=graph,
        seed_nodes=seed_nodes,
        activation_prob=activation_prob,
        max_iterations=max_iterations,
        random_state=random_state
    )
    
    # Compute metrics
    depth, max_width, spread_rate, max_width_level = compute_cascade_metrics(activated, times)
    
    return CascadeSimulationResult(
        article_id=article_id,
        label="",  # Will be filled later
        activated_nodes=activated,
        activation_times=times,
        cascade_depth=depth,
        cascade_width=max_width,
        cascade_size=len(activated),
        spread_rate=spread_rate,
        max_width_level=max_width_level
    )


def simulate_article_cascade_lt(
    graph: nx.DiGraph,
    article_id: str,
    user_tweet_map: pd.DataFrame,
    threshold_distribution: str = "uniform",
    max_iterations: int = 50,
    random_state: Optional[int] = None
) -> Optional[CascadeSimulationResult]:
    """
    Simulate cascade for a single article using Linear Threshold model.
    
    Args:
        graph: User interaction graph
        article_id: Article ID
        user_tweet_map: DataFrame with columns [user_id, article_id, tweet_id]
        threshold_distribution: How to assign thresholds
        max_iterations: Maximum time steps
        random_state: Random seed
    
    Returns:
        CascadeSimulationResult or None if no users found for article
    """
    # Find seed nodes (users who posted tweets about this article)
    article_users = user_tweet_map[user_tweet_map["article_id"] == article_id]["user_id"].unique()
    seed_nodes = set(str(uid) for uid in article_users if str(uid) in graph)
    
    if not seed_nodes:
        return None
    
    # Run LT simulation
    activated, times = simulate_lt(
        graph=graph,
        seed_nodes=seed_nodes,
        threshold_distribution=threshold_distribution,
        max_iterations=max_iterations,
        random_state=random_state
    )
    
    # Compute metrics
    depth, max_width, spread_rate, max_width_level = compute_cascade_metrics(activated, times)
    
    return CascadeSimulationResult(
        article_id=article_id,
        label="",  # Will be filled later
        activated_nodes=activated,
        activation_times=times,
        cascade_depth=depth,
        cascade_width=max_width,
        cascade_size=len(activated),
        spread_rate=spread_rate,
        max_width_level=max_width_level
    )


def analyze_cascades(
    graph_path: Path,
    user_tweet_map_path: Path,
    article_metrics_path: Path,
    output_dir: Path,
    activation_prob: float = 0.1,
    max_iterations: int = 50,
    max_articles: Optional[int] = None,
    random_state: Optional[int] = None
) -> pd.DataFrame:
    """
    Analyze cascades for all articles using ICM.
    
    Args:
        graph_path: Path to user interaction graph
        user_tweet_map_path: Path to synthetic_user_tweet_mapping.csv
        article_metrics_path: Path to article_metrics.csv
        output_dir: Directory to save results
        activation_prob: ICM activation probability
        max_iterations: Maximum time steps per simulation
        max_articles: Limit number of articles (for testing)
        random_state: Random seed
    
    Returns:
        DataFrame with cascade metrics for all articles
    """
    print(f"\n{'='*60}")
    print("CASCADE MODELING: Independent Cascade Model (ICM)")
    print(f"{'='*60}\n")
    
    # Load data
    print(f"[INFO] Loading graph from {graph_path}")
    graph = nx.read_graphml(graph_path)
    print(f"[OK] Loaded graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    print(f"[INFO] Loading user-tweet mapping...")
    user_tweet_map = pd.read_csv(user_tweet_map_path, low_memory=False)
    print(f"[OK] Loaded {len(user_tweet_map):,} user-tweet mappings")
    
    print(f"[INFO] Loading article metrics...")
    article_metrics = pd.read_csv(article_metrics_path, low_memory=False)
    print(f"[OK] Loaded {len(article_metrics):,} articles")
    
    # Filter articles if needed (shuffle first to get mix of fake/real)
    if max_articles:
        if random_state is not None:
            np.random.seed(random_state)
        article_metrics = article_metrics.sample(n=min(max_articles, len(article_metrics)), random_state=random_state).reset_index(drop=True)
        print(f"[INFO] Shuffled and limited to {len(article_metrics)} articles for testing")
        # Show label distribution
        if "label" in article_metrics.columns:
            label_counts = article_metrics["label"].value_counts()
            print(f"[INFO] Sample label distribution: {dict(label_counts)}")
    
    # Create label mapping (convert 0/1 to "fake"/"real")
    # Based on FakeNewsNet: 0 = fake, 1 = real
    news_data = pd.read_csv(Path("data/processed/news_clean.csv"), low_memory=False)
    label_map = {}
    for news_id, label in zip(news_data["news_id"].astype(str), news_data["label"]):
        if label == 0:
            label_map[news_id] = "fake"
        elif label == 1:
            label_map[news_id] = "real"
        else:
            label_map[news_id] = "unknown"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Simulate cascades
    print(f"\n[INFO] Simulating cascades with activation_prob={activation_prob}...")
    results = []
    
    for _, row in tqdm(article_metrics.iterrows(), total=len(article_metrics), desc="Simulating cascades"):
        article_id = str(row["news_id"])
        label = label_map.get(article_id, "unknown")
        
        result = simulate_article_cascade_icm(
            graph=graph,
            article_id=article_id,
            user_tweet_map=user_tweet_map,
            activation_prob=activation_prob,
            max_iterations=max_iterations,
            random_state=random_state
        )
        
        if result is None:
            continue
        
        result.label = label
        
        # Create metrics record
        metrics = CascadeMetrics(
            article_id=article_id,
            label=label,
            cascade_size=result.cascade_size,
            cascade_depth=result.cascade_depth,
            max_width=result.cascade_width,
            avg_spread_rate=result.spread_rate,
            time_to_max_width=result.max_width_level,
            total_time_steps=result.cascade_depth
        )
        
        results.append(metrics)
    
    # Convert to DataFrame
    results_df = pd.DataFrame([
        {
            "article_id": m.article_id,
            "label": m.label,
            "cascade_size": m.cascade_size,
            "cascade_depth": m.cascade_depth,
            "max_width": m.max_width,
            "avg_spread_rate": m.avg_spread_rate,
            "time_to_max_width": m.time_to_max_width,
            "total_time_steps": m.total_time_steps
        }
        for m in results
    ])
    
    # Save results
    output_path = output_dir / "cascade_metrics.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\n[OK] Saved cascade metrics to {output_path}")
    print(f"[INFO] Simulated {len(results_df)} cascades")
    
    return results_df


def compare_fake_vs_real_cascades(
    cascade_metrics_path: Path,
    output_dir: Path
) -> pd.DataFrame:
    """
    Compare cascade metrics between fake and real news.
    
    Args:
        cascade_metrics_path: Path to cascade_metrics.csv
        output_dir: Directory to save comparison results
    
    Returns:
        DataFrame with aggregated statistics
    """
    print(f"\n{'='*60}")
    print("COMPARING FAKE VS REAL CASCADES")
    print(f"{'='*60}\n")
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(cascade_metrics_path)
    
    # Filter valid labels
    df = df[df["label"].isin(["fake", "real"])]
    
    # Compute statistics by label
    stats = []
    for label in ["fake", "real"]:
        subset = df[df["label"] == label]
        
        stats.append({
            "label": label,
            "count": len(subset),
            "avg_cascade_size": subset["cascade_size"].mean(),
            "std_cascade_size": subset["cascade_size"].std(),
            "avg_cascade_depth": subset["cascade_depth"].mean(),
            "std_cascade_depth": subset["cascade_depth"].std(),
            "avg_max_width": subset["max_width"].mean(),
            "std_max_width": subset["max_width"].std(),
            "avg_spread_rate": subset["avg_spread_rate"].mean(),
            "std_spread_rate": subset["avg_spread_rate"].std(),
            "median_cascade_size": subset["cascade_size"].median(),
            "median_cascade_depth": subset["cascade_depth"].median(),
            "max_cascade_size": subset["cascade_size"].max(),
            "max_cascade_depth": subset["cascade_depth"].max(),
        })
    
    comparison_df = pd.DataFrame(stats)
    
    # Save comparison
    output_path = output_dir / "cascade_comparison_fake_vs_real.csv"
    comparison_df.to_csv(output_path, index=False)
    print(f"[OK] Saved comparison to {output_path}")
    
    # Print summary
    print("\n[SUMMARY] Cascade Metrics Comparison:")
    print(comparison_df.to_string(index=False))
    
    # Generate visualizations
    visualize_cascade_comparison(df, output_dir)
    
    return comparison_df


def analyze_cascades_lt(
    graph_path: Path,
    user_tweet_map_path: Path,
    article_metrics_path: Path,
    output_dir: Path,
    threshold_distribution: str = "uniform",
    max_iterations: int = 50,
    max_articles: Optional[int] = None,
    random_state: Optional[int] = None
) -> pd.DataFrame:
    """
    Analyze cascades for all articles using Linear Threshold (LT) model.
    
    Args:
        graph_path: Path to user interaction graph
        user_tweet_map_path: Path to synthetic_user_tweet_mapping.csv
        article_metrics_path: Path to article_metrics.csv
        output_dir: Directory to save results
        threshold_distribution: How to assign thresholds
        max_iterations: Maximum time steps per simulation
        max_articles: Limit number of articles (for testing)
        random_state: Random seed
    
    Returns:
        DataFrame with cascade metrics for all articles
    """
    print(f"\n{'='*60}")
    print("CASCADE MODELING: Linear Threshold Model (LT)")
    print(f"{'='*60}\n")
    
    # Load data (same as ICM)
    print(f"[INFO] Loading graph from {graph_path}")
    graph = nx.read_graphml(graph_path)
    print(f"[OK] Loaded graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    print(f"[INFO] Loading user-tweet mapping...")
    user_tweet_map = pd.read_csv(user_tweet_map_path, low_memory=False)
    print(f"[OK] Loaded {len(user_tweet_map):,} user-tweet mappings")
    
    print(f"[INFO] Loading article metrics...")
    article_metrics = pd.read_csv(article_metrics_path, low_memory=False)
    print(f"[OK] Loaded {len(article_metrics):,} articles")
    
    # Filter articles if needed
    if max_articles:
        if random_state is not None:
            np.random.seed(random_state)
        article_metrics = article_metrics.sample(n=min(max_articles, len(article_metrics)), random_state=random_state).reset_index(drop=True)
        print(f"[INFO] Shuffled and limited to {len(article_metrics)} articles for testing")
        if "label" in article_metrics.columns:
            label_counts = article_metrics["label"].value_counts()
            print(f"[INFO] Sample label distribution: {dict(label_counts)}")
    
    # Create label mapping
    news_data = pd.read_csv(Path("data/processed/news_clean.csv"), low_memory=False)
    label_map = {}
    for news_id, label in zip(news_data["news_id"].astype(str), news_data["label"]):
        if label == 0:
            label_map[news_id] = "fake"
        elif label == 1:
            label_map[news_id] = "real"
        else:
            label_map[news_id] = "unknown"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Simulate cascades with LT
    print(f"\n[INFO] Simulating cascades with LT model (threshold_distribution={threshold_distribution})...")
    results = []
    
    for _, row in tqdm(article_metrics.iterrows(), total=len(article_metrics), desc="Simulating LT cascades"):
        article_id = str(row["news_id"])
        label = label_map.get(article_id, "unknown")
        
        result = simulate_article_cascade_lt(
            graph=graph,
            article_id=article_id,
            user_tweet_map=user_tweet_map,
            threshold_distribution=threshold_distribution,
            max_iterations=max_iterations,
            random_state=random_state
        )
        
        if result is None:
            continue
        
        # Create metrics record
        metrics = CascadeMetrics(
            article_id=article_id,
            label=label,
            cascade_size=result.cascade_size,
            cascade_depth=result.cascade_depth,
            max_width=result.cascade_width,
            avg_spread_rate=result.spread_rate,
            time_to_max_width=result.max_width_level,
            total_time_steps=result.cascade_depth
        )
        
        results.append(metrics)
    
    # Convert to DataFrame
    results_df = pd.DataFrame([
        {
            "article_id": m.article_id,
            "label": m.label,
            "cascade_size": m.cascade_size,
            "cascade_depth": m.cascade_depth,
            "max_width": m.max_width,
            "avg_spread_rate": m.avg_spread_rate,
            "time_to_max_width": m.time_to_max_width,
            "total_time_steps": m.total_time_steps
        }
        for m in results
    ])
    
    # Save results
    output_path = output_dir / "cascade_metrics_lt.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\n[OK] Saved LT cascade metrics to {output_path}")
    print(f"[INFO] Simulated {len(results_df)} cascades with LT model")
    
    return results_df


def compare_cascade_models(
    icm_metrics_path: Path,
    lt_metrics_path: Path,
    output_dir: Path
) -> pd.DataFrame:
    """
    Compare results between ICM and LT cascade models.
    
    Args:
        icm_metrics_path: Path to cascade_metrics.csv (ICM results)
        lt_metrics_path: Path to cascade_metrics_lt.csv (LT results)
        output_dir: Directory to save comparison results
    
    Returns:
        DataFrame with model comparison statistics
    """
    print(f"\n{'='*60}")
    print("COMPARING CASCADE MODELS: ICM vs LT")
    print(f"{'='*60}\n")
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load both datasets
    icm_df = pd.read_csv(icm_metrics_path)
    lt_df = pd.read_csv(lt_metrics_path)
    
    # Add model column
    icm_df["model"] = "ICM"
    lt_df["model"] = "LT"
    
    # Combine
    combined_df = pd.concat([icm_df, lt_df], ignore_index=True)
    
    # Compare by model and label
    stats = []
    for model in ["ICM", "LT"]:
        for label in ["fake", "real"]:
            subset = combined_df[(combined_df["model"] == model) & (combined_df["label"] == label)]
            
            if len(subset) == 0:
                continue
            
            stats.append({
                "model": model,
                "label": label,
                "count": len(subset),
                "avg_cascade_size": subset["cascade_size"].mean(),
                "avg_cascade_depth": subset["cascade_depth"].mean(),
                "avg_max_width": subset["max_width"].mean(),
                "avg_spread_rate": subset["avg_spread_rate"].mean(),
                "median_cascade_size": subset["cascade_size"].median(),
            })
    
    comparison_df = pd.DataFrame(stats)
    
    # Save comparison
    output_path = output_dir / "cascade_model_comparison.csv"
    comparison_df.to_csv(output_path, index=False)
    print(f"[OK] Saved model comparison to {output_path}")
    
    # Print summary
    print("\n[SUMMARY] ICM vs LT Model Comparison:")
    print(comparison_df.to_string(index=False))
    
    # Create comparison visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Cascade Size Comparison
    for model in ["ICM", "LT"]:
        for label in ["fake", "real"]:
            subset = combined_df[(combined_df["model"] == model) & (combined_df["label"] == label)]
            if len(subset) > 0:
                axes[0, 0].hist(subset["cascade_size"], alpha=0.5, label=f"{model} {label}", bins=30)
    axes[0, 0].set_xlabel("Cascade Size")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title("Cascade Size Distribution: ICM vs LT")
    axes[0, 0].legend()
    
    # Spread Rate Comparison
    for model in ["ICM", "LT"]:
        for label in ["fake", "real"]:
            subset = combined_df[(combined_df["model"] == model) & (combined_df["label"] == label)]
            if len(subset) > 0:
                axes[0, 1].hist(subset["avg_spread_rate"], alpha=0.5, label=f"{model} {label}", bins=30)
    axes[0, 1].set_xlabel("Spread Rate")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title("Spread Rate Distribution: ICM vs LT")
    axes[0, 1].legend()
    
    # Bar chart comparison
    metrics = ["avg_cascade_size", "avg_spread_rate"]
    for idx, metric in enumerate(metrics):
        ax = axes[1, idx]
        model_data = comparison_df.pivot(index="label", columns="model", values=metric)
        model_data.plot(kind="bar", ax=ax, rot=0)
        ax.set_title(f"{metric.replace('_', ' ').title()}: ICM vs LT")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.legend(title="Model")
    
    plt.tight_layout()
    fig_path = output_dir / "figures" / "cascade_model_comparison.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved comparison visualization to {fig_path}")
    plt.close()
    
    return comparison_df


def visualize_cascade_comparison(
    cascade_metrics_df: pd.DataFrame,
    output_dir: Path
) -> None:
    """
    Generate visualizations comparing fake vs real cascade metrics.
    
    Args:
        cascade_metrics_df: DataFrame with cascade metrics
        output_dir: Directory to save figures
    """
    print(f"\n[INFO] Generating visualizations...")
    
    # Filter valid labels
    df = cascade_metrics_df[cascade_metrics_df["label"].isin(["fake", "real"])].copy()
    
    if len(df) == 0:
        print("[WARNING] No valid labels found for visualization")
        return
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # Create figures directory
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Cascade Size Distribution (Box Plot)
    fig, ax = plt.subplots(figsize=(10, 6))
    df_box = df[df["cascade_size"] > 0]  # Filter out zero-size cascades
    if len(df_box) > 0:
        sns.boxplot(data=df_box, x="label", y="cascade_size", ax=ax)
        ax.set_title("Cascade Size Distribution: Fake vs Real News", fontsize=14, fontweight='bold')
        ax.set_xlabel("News Type", fontsize=12)
        ax.set_ylabel("Cascade Size (Number of Activated Nodes)", fontsize=12)
        ax.set_yscale('log')  # Log scale for better visualization
        plt.tight_layout()
        plt.savefig(figures_dir / "cascade_size_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] Saved: cascade_size_distribution.png")
    
    # 2. Cascade Depth Distribution (Box Plot)
    fig, ax = plt.subplots(figsize=(10, 6))
    df_depth = df[df["cascade_depth"] > 0]
    if len(df_depth) > 0:
        sns.boxplot(data=df_depth, x="label", y="cascade_depth", ax=ax)
        ax.set_title("Cascade Depth Distribution: Fake vs Real News", fontsize=14, fontweight='bold')
        ax.set_xlabel("News Type", fontsize=12)
        ax.set_ylabel("Cascade Depth (Maximum Time Steps)", fontsize=12)
        plt.tight_layout()
        plt.savefig(figures_dir / "cascade_depth_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] Saved: cascade_depth_distribution.png")
    
    # 3. Spread Rate Comparison (Box Plot)
    fig, ax = plt.subplots(figsize=(10, 6))
    df_rate = df[df["avg_spread_rate"] > 0]
    if len(df_rate) > 0:
        sns.boxplot(data=df_rate, x="label", y="avg_spread_rate", ax=ax)
        ax.set_title("Average Spread Rate: Fake vs Real News", fontsize=14, fontweight='bold')
        ax.set_xlabel("News Type", fontsize=12)
        ax.set_ylabel("Average Spread Rate (Nodes per Time Step)", fontsize=12)
        ax.set_yscale('log')
        plt.tight_layout()
        plt.savefig(figures_dir / "spread_rate_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] Saved: spread_rate_comparison.png")
    
    # 4. Max Width Comparison (Box Plot)
    fig, ax = plt.subplots(figsize=(10, 6))
    df_width = df[df["max_width"] > 0]
    if len(df_width) > 0:
        sns.boxplot(data=df_width, x="label", y="max_width", ax=ax)
        ax.set_title("Maximum Cascade Width: Fake vs Real News", fontsize=14, fontweight='bold')
        ax.set_xlabel("News Type", fontsize=12)
        ax.set_ylabel("Maximum Width (Nodes Activated in Single Step)", fontsize=12)
        ax.set_yscale('log')
        plt.tight_layout()
        plt.savefig(figures_dir / "max_width_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] Saved: max_width_comparison.png")
    
    # 5. Comparison Bar Chart (Average Metrics)
    comparison_data = []
    for label in ["fake", "real"]:
        subset = df[df["label"] == label]
        if len(subset) > 0:
            comparison_data.append({
                "label": label,
                "avg_size": subset["cascade_size"].mean(),
                "avg_depth": subset["cascade_depth"].mean(),
                "avg_spread_rate": subset["avg_spread_rate"].mean(),
                "avg_max_width": subset["max_width"].mean()
            })
    
    if comparison_data:
        comp_df = pd.DataFrame(comparison_data)
        comp_df_melted = comp_df.melt(
            id_vars=["label"],
            value_vars=["avg_size", "avg_depth", "avg_spread_rate", "avg_max_width"],
            var_name="metric",
            value_name="value"
        )
        
        # Normalize values for better comparison (0-1 scale)
        for metric in comp_df_melted["metric"].unique():
            subset = comp_df_melted[comp_df_melted["metric"] == metric]
            max_val = subset["value"].max()
            if max_val > 0:
                comp_df_melted.loc[comp_df_melted["metric"] == metric, "value"] = (
                    comp_df_melted[comp_df_melted["metric"] == metric]["value"] / max_val
                )
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(data=comp_df_melted, x="metric", y="value", hue="label", ax=ax)
        ax.set_title("Normalized Cascade Metrics Comparison: Fake vs Real", fontsize=14, fontweight='bold')
        ax.set_xlabel("Metric", fontsize=12)
        ax.set_ylabel("Normalized Value", fontsize=12)
        ax.set_xticklabels(["Size", "Depth", "Spread Rate", "Max Width"], rotation=45, ha='right')
        ax.legend(title="News Type")
        plt.tight_layout()
        plt.savefig(figures_dir / "cascade_metrics_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] Saved: cascade_metrics_comparison.png")
    
    # 6. Histogram: Cascade Size Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for idx, label in enumerate(["fake", "real"]):
        subset = df[df["label"] == label]
        if len(subset) > 0:
            axes[idx].hist(subset["cascade_size"], bins=50, alpha=0.7, edgecolor='black')
            axes[idx].set_title(f"{label.capitalize()} News: Cascade Size Distribution", fontsize=12, fontweight='bold')
            axes[idx].set_xlabel("Cascade Size", fontsize=10)
            axes[idx].set_ylabel("Frequency", fontsize=10)
            axes[idx].set_yscale('log')
            axes[idx].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "cascade_size_histogram.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: cascade_size_histogram.png")
    
    print(f"\n[OK] All visualizations saved to {figures_dir}/")

