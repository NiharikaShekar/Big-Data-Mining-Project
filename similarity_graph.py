from __future__ import annotations
from pathlib import Path
from typing import Literal, Tuple
import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

def _load_vectors(proc_dir: Path, mode: Literal["tfidf","bert"]) -> Tuple[pd.DataFrame, object]:
    news = pd.read_csv(proc_dir / "news_clean.csv", low_memory=False)
    news["news_id"] = news["news_id"].astype("string")
    if mode == "tfidf":
        X = sp.load_npz(proc_dir / "tfidf_vectors.npz")
    elif mode == "bert":
        X = np.load(proc_dir / "bert_embeddings.npy")
    else:
        raise ValueError("mode must be 'tfidf' or 'bert'")
    return news, X

def build_similarity_graph(proc_dir: Path, out_graphml: Path,
                           mode: Literal["tfidf","bert"]="tfidf",
                           k: int = 10, threshold: float = 0.25) -> nx.Graph:
    """
    Build article↔article similarity graph via cosine nearest neighbors.
    Why: supports community analysis without user interactions.
    """
    news, X = _load_vectors(proc_dir, mode)
    n = len(news)
    if n == 0:
        raise ValueError("news_clean.csv is empty")

    nn = NearestNeighbors(n_neighbors=min(k+1, n), metric="cosine")
    nn.fit(X)
    dists, idx = nn.kneighbors(X, return_distance=True)
    sims = 1.0 - dists

    G = nx.Graph()
    for _, r in news.iterrows():
        G.add_node(str(r["news_id"]), label=int(r["label"]), source=str(r["source"]))

    for i in range(n):
        u = str(news.iloc[i]["news_id"])
        for j, s in zip(idx[i][1:], sims[i][1:]):  # skip self
            if s < threshold:
                continue
            v = str(news.iloc[j]["news_id"])
            w = float(s)
            if G.has_edge(u, v):
                if G[u][v]["weight"] < w:
                    G[u][v]["weight"] = w
            else:
                G.add_edge(u, v, weight=w)

    ensure_parent(out_graphml)
    nx.write_graphml(G, out_graphml)
    return G