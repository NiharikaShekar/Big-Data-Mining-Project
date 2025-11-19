from __future__ import annotations
from pathlib import Path
import pandas as pd
import networkx as nx
from .utils import ensure_parent, parse_tweet_ids

def build_article_tweet_bipartite(metrics_fp: Path, raw_root: Path, out_graphml: Path) -> nx.Graph:
    """
    Build Article↔Tweet bipartite graph (association only).
    Why: visualize coverage when user interaction graph isn't available.
    """
    df = pd.read_csv(metrics_fp, low_memory=False).astype({"news_id":"string"})
    # Reload raw for exact tweet lists
    from .article_metrics import _stack_raw
    raw = _stack_raw(raw_root).astype({"id":"string"})
    raw["tweet_list"] = raw["tweet_ids"].apply(parse_tweet_ids)
    lookup = dict(zip(raw["id"].tolist(), raw["tweet_list"].tolist()))

    G = nx.Graph()
    for _, r in df.iterrows():
        a = f"A_{r['news_id']}"
        G.add_node(a, bipartite="article", news_id=str(r["news_id"]),
                   label=int(r["label"]), source=str(r["source"]),
                   tweet_count=int(r["tweet_count"]))
        for tid in lookup.get(str(r["news_id"]), []):
            t = f"T_{tid}"
            if t not in G:
                G.add_node(t, bipartite="tweet", tweet_id=str(tid))
            G.add_edge(t, a)

    ensure_parent(out_graphml)
    nx.write_graphml(G, out_graphml)
    return G