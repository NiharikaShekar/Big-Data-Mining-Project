from __future__ import annotations
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import scipy.sparse as sp
import numpy as np
from sklearn.decomposition import PCA

from src.phase3.article_metrics import build_article_metrics
from src.phase3.bipartite_graph import build_article_tweet_bipartite
from src.phase3.similarity_graph import build_similarity_graph
from src.phase3.user_interaction_graph import build_user_interaction_graph
from src.phase3.cascade_subgraphs import build_all_cascades
from src.phase3.utils import ensure_parent

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "FakeNewsNet"
PROC = ROOT / "data" / "processed"
GRAPHS = PROC / "graphs"
FIGS = ROOT / "reports" / "figures"

def _fig(name: str) -> Path:
    p = FIGS / name
    ensure_parent(p)
    return p

def _plot_hist_by_label(df: pd.DataFrame) -> None:
    fig = plt.figure()
    for lbl, sub in df.groupby("label"):
        sub["tweet_count"].plot(kind="hist", bins=30, alpha=0.5, label=f"label={lbl}")
    plt.xlabel("tweet_count"); plt.ylabel("frequency"); plt.title("Tweet count by label"); plt.legend()
    fig.savefig(_fig("tweet_count_hist.png")); plt.close(fig)

def _plot_topk(df: pd.DataFrame, k: int = 20) -> None:
    fig = plt.figure()
    top = df.sort_values("tweet_count", ascending=False).head(k)
    plt.bar(top["news_id"].astype(str), top["tweet_count"].astype(int))
    plt.xticks(rotation=90); plt.ylabel("tweet_count"); plt.title(f"Top {k} articles by tweet_count")
    fig.tight_layout(); fig.savefig(_fig("top_articles_by_tweet_count.png")); plt.close(fig)

def _plot_pca() -> None:
    news = pd.read_csv(PROC / "news_clean.csv", low_memory=False)
    Xp = PROC / "bert_embeddings.npy"
    if Xp.exists():
        X = np.load(Xp)
    else:
        X = sp.load_npz(PROC / "tfidf_vectors.npz").toarray()
    X2 = PCA(n_components=2, random_state=0).fit_transform(X)
    fig = plt.figure()
    plt.scatter(X2[:,0], X2[:,1], s=6, c=news["label"].values)
    plt.title("Article embeddings (PCA)")
    fig.savefig(_fig("embeddings_pca.png")); plt.close(fig)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["tfidf","bert"], default="tfidf")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--thr", type=float, default=0.25)
    args = ap.parse_args()

    # 1) metrics
    metrics_fp = PROC / "article_metrics.csv"
    metrics = build_article_metrics(PROC / "news_clean.csv", RAW, metrics_fp)
    print(f"[OK] metrics → {metrics_fp} rows={len(metrics)}")

    # 2) bipartite
    bipath = GRAPHS / "article_tweet_bipartite.graphml"
    G_bip = build_article_tweet_bipartite(metrics_fp, RAW, bipath)
    print(f"[OK] bipartite → {bipath} |V|={G_bip.number_of_nodes()} |E|={G_bip.number_of_edges()}")

    # 3) similarity
    simpath = GRAPHS / "article_similarity.graphml"
    G_sim = build_similarity_graph(PROC, simpath, mode=args.mode, k=args.k, threshold=args.thr)
    print(f"[OK] similarity → {simpath} |V|={G_sim.number_of_nodes()} |E|={G_sim.number_of_edges()}")

    # 4) user interaction graph (from JSON dataset or synthetic data)
    fakenewsnet_full = ROOT / "data" / "raw" / "FakeNewsNet_full"
    synthetic_dir = PROC / "synthetic"
    user_graph_path = GRAPHS / "user_interaction.graphml"
    user_G = build_user_interaction_graph(
        fakenewsnet_full, 
        user_graph_path,
        synthetic_dir=synthetic_dir
    )
    if user_G.number_of_nodes() > 0:
        print(f"[OK] user interaction graph → {user_graph_path}")
    else:
        print(f"[INFO] user interaction graph → {user_graph_path} (empty)")
    
    # 5) cascade subgraphs (with synthetic user data if available)
    # NOTE: Building all cascades takes hours. For testing, use max_articles=1000
    # For full run, set max_articles=None (but expect 2-4 hours)
    cascades_dir = PROC / "cascades"
    synthetic_dir = PROC / "synthetic"
    
    # Limit to 1000 articles for faster completion (remove this limit for full dataset)
    MAX_CASCADE_ARTICLES = 1000  # Set to None for all articles
    
    cascade_metrics = build_all_cascades(
        metrics_fp, RAW, cascades_dir, 
        max_articles=MAX_CASCADE_ARTICLES,
        synthetic_user_data=synthetic_dir if (synthetic_dir / "synthetic_user_tweet_mapping.csv").exists() else None
    )
    print(f"[OK] cascades → {cascades_dir} ({len(cascade_metrics)} articles)")
    
    # 6) figures
    _plot_hist_by_label(metrics)
    _plot_topk(metrics)
    _plot_pca()
    print(f"[OK] figures → {FIGS}")

if __name__ == "__main__":
    main()
