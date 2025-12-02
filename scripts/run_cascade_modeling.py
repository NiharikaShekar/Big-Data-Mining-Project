#!/usr/bin/env python3
"""
Driver script for Independent Cascade Model (ICM) simulation and analysis.

This script:
1. Simulates information diffusion using ICM on the user interaction graph
2. Computes cascade metrics (depth, width, spread rate)
3. Compares fake vs real news cascade patterns
"""

import argparse
from pathlib import Path
import sys

# Add src to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.phase4.cascade_modeling import analyze_cascades, compare_fake_vs_real_cascades


def main():
    parser = argparse.ArgumentParser(
        description="Run Independent Cascade Model (ICM) simulation"
    )
    
    # Paths
    PROC = ROOT / "data" / "processed"
    GRAPHS = PROC / "graphs"
    OUTPUT = PROC / "cascade_modeling"
    
    parser.add_argument(
        "--graph",
        type=Path,
        default=GRAPHS / "user_interaction.graphml",
        help="Path to user interaction graph GraphML file"
    )
    parser.add_argument(
        "--user-tweet-map",
        type=Path,
        default=PROC / "synthetic" / "synthetic_user_tweet_mapping.csv",
        help="Path to synthetic_user_tweet_mapping.csv"
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
        help="Output directory for cascade modeling results"
    )
    
    # ICM parameters
    parser.add_argument(
        "--activation-prob",
        type=float,
        default=0.1,
        help="ICM activation probability (default: 0.1)"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=50,
        help="Maximum time steps per simulation (default: 50)"
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=None,
        help="Limit number of articles for testing (default: all)"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--no-comparison",
        action="store_true",
        help="Skip fake vs real comparison (faster)"
    )
    
    args = parser.parse_args()
    
    # Run cascade analysis
    results_df = analyze_cascades(
        graph_path=args.graph,
        user_tweet_map_path=args.user_tweet_map,
        article_metrics_path=args.metrics,
        output_dir=args.output,
        activation_prob=args.activation_prob,
        max_iterations=args.max_iterations,
        max_articles=args.max_articles,
        random_state=args.random_state
    )
    
    # Compare fake vs real
    if not args.no_comparison and len(results_df) > 0:
        cascade_metrics_path = args.output / "cascade_metrics.csv"
        compare_fake_vs_real_cascades(
            cascade_metrics_path=cascade_metrics_path,
            output_dir=args.output
        )
    
    print("\n[OK] Cascade modeling complete!")


if __name__ == "__main__":
    main()

