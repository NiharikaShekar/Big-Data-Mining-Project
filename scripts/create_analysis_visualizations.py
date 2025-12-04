#!/usr/bin/env python3
"""
Create visualizations for Link Analysis and Community Detection results.

Generates:
1. PageRank top influencers bar chart
2. HITS authorities and hubs bar charts
3. Community size distribution
4. Top communities statistics
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Add src to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Paths
PROC = ROOT / "data" / "processed"
LINK_ANALYSIS = PROC / "link_analysis"
COMMUNITY = PROC / "community_detection"
OUTPUT = ROOT / "reports" / "figures"

OUTPUT.mkdir(parents=True, exist_ok=True)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def create_pagerank_visualization():
    """Create bar chart for top PageRank influencers"""
    print("[INFO] Creating PageRank visualization...")
    
    df = pd.read_csv(LINK_ANALYSIS / "pagerank_top_influencers.csv")
    
    # Take top 20
    top_20 = df.head(20)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    bars = ax.barh(range(len(top_20)), top_20["pagerank_score"], color='steelblue')
    ax.set_yticks(range(len(top_20)))
    ax.set_yticklabels([f"User {i+1}" for i in range(len(top_20))], fontsize=10)
    ax.set_xlabel("PageRank Score", fontsize=12, fontweight='bold')
    ax.set_title("Top 20 PageRank Influencers", fontsize=14, fontweight='bold')
    ax.invert_yaxis()  # Highest at top
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(top_20.iterrows()):
        ax.text(row["pagerank_score"] * 0.5, i, f"{row['pagerank_score']:.2e}", 
                va='center', fontsize=9, color='white', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT / "pagerank_top_influencers.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {OUTPUT / 'pagerank_top_influencers.png'}")


def create_hits_visualizations():
    """Create bar charts for HITS authorities and hubs"""
    print("[INFO] Creating HITS visualizations...")
    
    # Authorities
    auth_df = pd.read_csv(LINK_ANALYSIS / "hits_top_authorities.csv")
    top_auth = auth_df.head(20)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    bars = ax.barh(range(len(top_auth)), top_auth["authority_score"], color='coral')
    ax.set_yticks(range(len(top_auth)))
    ax.set_yticklabels([f"User {i+1}" for i in range(len(top_auth))], fontsize=10)
    ax.set_xlabel("Authority Score", fontsize=12, fontweight='bold')
    ax.set_title("Top 20 HITS Authorities", fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    for i, (idx, row) in enumerate(top_auth.iterrows()):
        ax.text(row["authority_score"] * 0.5, i, f"{row['authority_score']:.2e}", 
                va='center', fontsize=9, color='white', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT / "hits_top_authorities.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {OUTPUT / 'hits_top_authorities.png'}")
    
    # Hubs
    hub_df = pd.read_csv(LINK_ANALYSIS / "hits_top_hubs.csv")
    top_hub = hub_df.head(20)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    bars = ax.barh(range(len(top_hub)), top_hub["hub_score"], color='mediumseagreen')
    ax.set_yticks(range(len(top_hub)))
    ax.set_yticklabels([f"User {i+1}" for i in range(len(top_hub))], fontsize=10)
    ax.set_xlabel("Hub Score", fontsize=12, fontweight='bold')
    ax.set_title("Top 20 HITS Hubs", fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    for i, (idx, row) in enumerate(top_hub.iterrows()):
        ax.text(row["hub_score"] * 0.5, i, f"{row['hub_score']:.2e}", 
                va='center', fontsize=9, color='white', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT / "hits_top_hubs.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {OUTPUT / 'hits_top_hubs.png'}")


def create_community_visualizations():
    """Create visualizations for community detection results"""
    print("[INFO] Creating community visualizations...")
    
    df = pd.read_csv(COMMUNITY / "community_statistics.csv")
    
    # 1. Community size distribution
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(df["size"], bins=50, edgecolor='black', alpha=0.7, color='purple')
    ax.set_xlabel("Community Size (Number of Users)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Frequency", fontsize=12, fontweight='bold')
    ax.set_title("Community Size Distribution", fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT / "community_size_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {OUTPUT / 'community_size_distribution.png'}")
    
    # 2. Top 20 largest communities
    top_communities = df.nlargest(20, "size")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    bars = ax.barh(range(len(top_communities)), top_communities["size"], color='mediumpurple')
    ax.set_yticks(range(len(top_communities)))
    ax.set_yticklabels([f"Community {int(row['community_id'])}" for _, row in top_communities.iterrows()], 
                       fontsize=10)
    ax.set_xlabel("Community Size (Number of Users)", fontsize=12, fontweight='bold')
    ax.set_title("Top 20 Largest Communities", fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    for i, (idx, row) in enumerate(top_communities.iterrows()):
        ax.text(row["size"] * 0.5, i, f"{int(row['size'])}", 
                va='center', fontsize=10, color='white', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT / "top_communities.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {OUTPUT / 'top_communities.png'}")
    
    # 3. Community density distribution
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(df["density"], bins=50, edgecolor='black', alpha=0.7, color='teal')
    ax.set_xlabel("Community Density", fontsize=12, fontweight='bold')
    ax.set_ylabel("Frequency", fontsize=12, fontweight='bold')
    ax.set_title("Community Density Distribution", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT / "community_density_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {OUTPUT / 'community_density_distribution.png'}")


def create_comparison_visualizations():
    """Create fake vs real comparison visualizations"""
    print("[INFO] Creating fake vs real comparison visualizations...")
    
    # PageRank comparison
    fake_pr = pd.read_csv(LINK_ANALYSIS / "fake_pagerank_top_influencers.csv")
    real_pr = pd.read_csv(LINK_ANALYSIS / "real_pagerank_top_influencers.csv")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Fake PageRank
    top_fake = fake_pr.head(15)
    axes[0].barh(range(len(top_fake)), top_fake["pagerank_score"], color='crimson')
    axes[0].set_yticks(range(len(top_fake)))
    axes[0].set_yticklabels([f"User {i+1}" for i in range(len(top_fake))], fontsize=9)
    axes[0].set_xlabel("PageRank Score", fontsize=11, fontweight='bold')
    axes[0].set_title("Top 15 PageRank: Fake News Network", fontsize=12, fontweight='bold')
    axes[0].invert_yaxis()
    
    # Real PageRank
    top_real = real_pr.head(15)
    axes[1].barh(range(len(top_real)), top_real["pagerank_score"], color='steelblue')
    axes[1].set_yticks(range(len(top_real)))
    axes[1].set_yticklabels([f"User {i+1}" for i in range(len(top_real))], fontsize=9)
    axes[1].set_xlabel("PageRank Score", fontsize=11, fontweight='bold')
    axes[1].set_title("Top 15 PageRank: Real News Network", fontsize=12, fontweight='bold')
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(OUTPUT / "pagerank_fake_vs_real.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {OUTPUT / 'pagerank_fake_vs_real.png'}")


def main():
    print("\n" + "="*60)
    print("CREATING ANALYSIS VISUALIZATIONS")
    print("="*60 + "\n")
    
    try:
        create_pagerank_visualization()
        create_hits_visualizations()
        create_community_visualizations()
        create_comparison_visualizations()
        
        print("\n" + "="*60)
        print("✅ All visualizations created successfully!")
        print(f"📁 Output directory: {OUTPUT}")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

