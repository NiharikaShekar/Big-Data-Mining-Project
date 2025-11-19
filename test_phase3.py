import pandas as pd
from pathlib import Path
import networkx as nx

from src.phase3.article_metrics import build_article_metrics
from src.phase3.bipartite_graph import build_article_tweet_bipartite
from src.phase3.similarity_graph import build_similarity_graph

def test_toy(tmp_path: Path):
    proc = tmp_path / "data" / "processed"; proc.mkdir(parents=True)
    raw = tmp_path / "data" / "raw" / "FakeNewsNet" / "gossipcop"; raw.mkdir(parents=True)

    # tiny news_clean
    pd.DataFrame([
        {"news_id":"1","text":"a","clean_text":"a","label":0,"source":"gossipcop"},
        {"news_id":"2","text":"b","clean_text":"b","label":1,"source":"gossipcop"},
    ]).to_csv(proc/"news_clean.csv", index=False)

    # tiny raw with tweet_ids
    pd.DataFrame([
        {"id":"1","news_url":"","title":"t1","tweet_ids":"111\t222 333"},
        {"id":"2","news_url":"","title":"t2","tweet_ids":""},
    ]).to_csv(raw/"gossipcop_fake.csv", index=False)

    # fake tfidf vectors (sparse 2x2 identity) for similarity graph
    import numpy as np, scipy.sparse as sp
    X = sp.csr_matrix([[1,0],[0,1]])
    sp.save_npz(proc/"tfidf_vectors.npz", X)

    metrics = build_article_metrics(proc/"news_clean.csv", tmp_path/"data"/"raw"/"FakeNewsNet", proc/"article_metrics.csv")
    assert metrics.set_index("news_id").loc["1","tweet_count"] == 3
    assert metrics.set_index("news_id").loc["2","tweet_count"] == 0

    G_bip = build_article_tweet_bipartite(proc/"article_metrics.csv", tmp_path/"data"/"raw"/"FakeNewsNet", proc/"graphs.graphml")
    assert isinstance(G_bip, nx.Graph)
    assert G_bip.number_of_edges() == 3

    G_sim = build_similarity_graph(proc, proc/"sim.graphml", mode="tfidf", k=1, threshold=0.01)
    assert G_sim.number_of_nodes() == 2
    assert G_sim.number_of_edges() >= 1