"""
Driver script for link analysis (PageRank and HITS)

This script runs link analysis on the user interaction graph and compares
fake vs real news subgraphs to identify influential users.
"""

from __future__ import annotations
from pathlib import Path
import argparse

from src.phase4.link_analysis import (
    analyze_user_interaction_graph,
    analyze_fake_vs_real_subgraphs
)

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
GRAPHS = PROC / "graphs"
SYNTHETIC = PROC / "synthetic"
OUTPUT = PROC / "link_analysis"


def main():
    parser = argparse.ArgumentParser(
        description="Run link analysis (PageRank and HITS) on user interaction graph"
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
        help="Output directory for link analysis results"
    )
    parser.add_argument(
        "--damping",
        type=float,
        default=0.85,
        help="PageRank damping factor (default: 0.85)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Number of top influencers to extract (default: 50)"
    )
    parser.add_argument(
        "--no-comparison",
        action="store_true",
        help="Skip fake vs real comparison (faster)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("LINK ANALYSIS: PageRank and HITS")
    print("="*60)
    
    # 1. Analyze global user interaction graph
    print("\n[STEP 1] Analyzing global user interaction graph...")
    global_results = analyze_user_interaction_graph(
        graph_path=args.graph,
        output_dir=args.output,
        damping_factor=args.damping,
        top_k=args.top_k
    )
    
    # 2. Compare fake vs real subgraphs (optional)
    if not args.no_comparison:
        print("\n[STEP 2] Comparing fake vs real news subgraphs...")
        comparison_results = analyze_fake_vs_real_subgraphs(
            graph_path=args.graph,
            article_metrics_path=args.metrics,
            synthetic_data_dir=SYNTHETIC,
            output_dir=args.output,
            damping_factor=args.damping,
            top_k=args.top_k
        )
    
    print("\n" + "="*60)
    print("LINK ANALYSIS COMPLETE!")
    print("="*60)
    print(f"\nResults saved to: {args.output}")
    print("\nGenerated files:")
    print("  - pagerank_top_influencers.csv")
    print("  - hits_top_authorities.csv")
    print("  - hits_top_hubs.csv")
    print("  - link_analysis_all_scores.csv")
    if not args.no_comparison:
        print("  - fake_pagerank_top_influencers.csv")
        print("  - fake_hits_top_authorities.csv")
        print("  - fake_hits_top_hubs.csv")
        print("  - real_pagerank_top_influencers.csv")
        print("  - real_hits_top_authorities.csv")
        print("  - real_hits_top_hubs.csv")


if __name__ == "__main__":
    main()

