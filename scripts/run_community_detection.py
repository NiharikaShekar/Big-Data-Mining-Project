"""
Driver script for community detection using Louvain algorithm

This script detects communities in the user interaction graph to identify
echo chambers and user clusters.
"""

from __future__ import annotations
from pathlib import Path
import argparse

from src.phase4.community_detection import (
    detect_user_communities,
    compare_fake_vs_real_communities
)

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
GRAPHS = PROC / "graphs"
SYNTHETIC = PROC / "synthetic"
OUTPUT = PROC / "community_detection"


def main():
    parser = argparse.ArgumentParser(
        description="Run community detection (Louvain) on user interaction graph"
    )
    parser.add_argument(
        "--graph",
        type=Path,
        default=GRAPHS / "user_interaction.graphml",
        help="Path to user interaction graph GraphML file"
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=PROC / "article_metrics.csv",
        help="Path to article_metrics.csv"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT,
        help="Output directory for community detection results"
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=1.0,
        help="Resolution parameter for modularity (higher = more communities, default: 1.0)"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Sample N nodes for faster processing (e.g., 100000 for testing)"
    )
    parser.add_argument(
        "--no-comparison",
        action="store_true",
        help="Skip fake vs real comparison (faster)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("COMMUNITY DETECTION: Louvain Algorithm")
    print("="*60)
    
    # 1. Detect communities in global graph
    print("\n[STEP 1] Detecting communities in global user interaction graph...")
    communities, community_stats, modularity = detect_user_communities(
        graph_path=args.graph,
        output_dir=args.output,
        resolution=args.resolution,
        random_state=args.random_state,
        sample_size=args.sample_size
    )
    
    print(f"\n[SUMMARY]")
    print(f"  Number of communities: {len(set(communities.values()))}")
    print(f"  Modularity score: {modularity:.4f}")
    print(f"  Largest community: {community_stats.iloc[0]['size']} users")
    
    # 2. Compare fake vs real subgraphs (optional)
    if not args.no_comparison:
        print("\n[STEP 2] Comparing communities in fake vs real news subgraphs...")
        comparison_results = compare_fake_vs_real_communities(
            graph_path=args.graph,
            article_metrics_path=args.metrics,
            synthetic_data_dir=SYNTHETIC,
            output_dir=args.output,
            resolution=args.resolution,
            random_state=args.random_state
        )
        
        print(f"\n[COMPARISON SUMMARY]")
        for label in ["fake", "real"]:
            if label in comparison_results:
                _, stats, mod = comparison_results[label]
                print(f"\n  {label.upper()} news:")
                print(f"    Number of communities: {len(set(comparison_results[label][0].values()))}")
                print(f"    Modularity score: {mod:.4f}")
                print(f"    Largest community: {stats.iloc[0]['size']} users")
    
    print("\n" + "="*60)
    print("COMMUNITY DETECTION COMPLETE!")
    print("="*60)
    print(f"\nResults saved to: {args.output}")
    print("\nGenerated files:")
    print("  - communities.csv")
    print("  - community_statistics.csv")
    print("  - modularity_score.txt")
    if not args.no_comparison:
        print("  - fake_communities.csv")
        print("  - fake_community_statistics.csv")
        print("  - real_communities.csv")
        print("  - real_community_statistics.csv")


if __name__ == "__main__":
    main()

