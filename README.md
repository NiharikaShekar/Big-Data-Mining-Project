# Misinformation Cascade Analysis on X (Twitter)

This repository hosts the Fact Checkers team project for modeling how misinformation emerges, spreads, and evolves across X (Twitter). We combine graph-based mining, cascade modeling, and comparative analysis between fake and factual narratives using sources such as FakeNewsNet.

## Project Objectives

- Reconstruct the interaction graph that underpins misinformation cascades.
- Identify high-impact sources and amplifiers via link analysis (PageRank, HITS).
- Detect community structure and compare echo chambers between fake and real narratives.
- Model temporal diffusion dynamics with cascade metrics and Independent Cascade simulations.
- Produce quantitative summaries and visuals that contrast misinformation with factual spread patterns.

## Repository Layout

```
├── config/              # Configuration files, API keys templates, experiment settings
├── data/
│   ├── raw/             # Immutable original datasets (FakeNewsNet dumps, etc.)
│   └── processed/       # Cleaned/derived datasets ready for analysis
├── notebooks/           # Exploratory analysis and prototyping notebooks
├── reports/
│   └── figures/         # Generated plots for reports/presentations
├── scripts/             # CLI utilities for data pulls, preprocessing, batch jobs
├── src/                 # Core Python package for reusable pipeline modules
└── README.md
```

## Environment Setup

1. **Create and activate a virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   ```

2. **Install project dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   - Copy `config/.env.example` to `config/.env` and populate Twitter API credentials if live data collection is required.

## Dependency Stack

The `requirements.txt` file seeds the environment with the following capabilities:
- Data wrangling: `pandas`, `numpy`, `pyarrow`
- Graph analysis: `networkx`, `python-igraph`, `graphviz`
- Machine learning & embeddings: `scikit-learn`, `sentence-transformers`
- Visualization: `matplotlib`, `seaborn`, `plotly`
- Experiment support: `tqdm`, `python-dotenv`, `jupyter`

## Completed Work

### Data Acquisition and Preparation

We have successfully acquired and processed the FakeNewsNet dataset from Arizona State University:
- **Dataset Size:** 23,196 news articles
- **Label Distribution:** 5,755 fake news (24.8%) and 17,441 real news (75.2%)
- **Sources:** GossipCop (22,140 articles) and PolitiFact (1,056 articles)
- **Data Cleaning:** Implemented comprehensive text cleaning pipeline removing URLs, HTML tags, special characters, and stopwords
- **Output:** Cleaned dataset saved as `data/processed/news_clean.csv`

### Exploratory Data Analysis

We conducted comprehensive EDA revealing key insights:
- **Tweet Count Statistics:** Mean of 89.2 tweets per article, with significant variance (std: 489.3)
- **Fake vs Real Comparison:** Fake news shows higher average engagement (132.8 vs 74.8 tweets per article)
- **Statistical Test:** Mann-Whitney U test confirms significant difference (p < 0.05)
- **Text Analysis:** Average text length is 68 characters, with 20.3% reduction after cleaning

Run EDA analysis:
```bash
source .venv/bin/activate
PYTHONPATH=. python scripts/run_eda.py
```

### Text Embeddings and Feature Generation

We generated numerical representations of articles for similarity analysis:

**TF-IDF Embeddings:**
- Method: Scikit-learn's TfidfVectorizer
- Output: Sparse matrix (`data/processed/tfidf_vectors.npz`)
- Vectorizer saved for consistency (`data/processed/tfidf_vectorizer.joblib`)

**BERT Embeddings:**
- Model: SentenceTransformer with BERT base
- Output: Dense numpy array (`data/processed/bert_embeddings.npy`)
- Dimension: 768 dimensions per article
- Purpose: Captures semantic similarity between articles

### Graph Construction

We have constructed multiple graph representations to model information flow:

**Article Metrics:**
- Computed tweet counts for all 23,200 articles
- Output: `data/processed/article_metrics.csv`

**Bipartite Graph (Article-Tweet):**
- Graph Type: Bipartite, undirected
- Nodes: Articles and tweets as distinct sets
- Edges: Connections between articles and their associated tweets
- Output: `data/processed/graphs/article_tweet_bipartite.graphml` (389 MB)
- Purpose: Track which tweets are associated with which articles

**Article Similarity Graph:**
- Similarity Metric: Cosine similarity from TF-IDF or BERT embeddings
- Parameters: Top-k nearest neighbors (default: 10), similarity threshold (default: 0.25)
- Output: `data/processed/graphs/article_similarity.graphml` (19 MB)
- Purpose: Identify clusters of similar articles and misinformation echo chambers

**Synthetic User Data Generation:**
- User-Tweet Mapping: 1,949,028 synthetic user IDs
- Retweet Relationships: 2,923,350 synthetic edges
- Reply Relationships: 778,172 synthetic edges
- Generation Strategy: Probabilistic assignment based on article tweet counts
- Output Files: `data/processed/synthetic/synthetic_user_tweet_mapping.csv`, `synthetic_retweets.csv`, `synthetic_replies.csv`

**User Interaction Graph:**
- Graph Type: Directed, time-stamped
- Nodes: 1,900,000+ unique synthetic users
- Edges: 3,600,000+ interactions (retweets and replies)
- Output: `data/processed/graphs/user_interaction.graphml` (802 MB)
- Purpose: Foundation for link analysis and community detection

**Cascade Subgraphs:**
- Method: Subgraph extraction from global user interaction graph
- Scope: 1,000 articles (representative sample for analysis)
- Cascade Metrics: Size, depth, width, timestamp availability
- Output: Individual cascade files (`data/processed/cascades/cascade_{news_id}.graphml`) and summary metrics (`cascade_metrics_summary.csv`)
- Purpose: Model how information spreads through user networks for each article

## Usage

### Running Data Preprocessing
```bash
source .venv/bin/activate
PYTHONPATH=. python src/preprocess/run_phase2.py
```

### Running Graph Construction
```bash
source .venv/bin/activate
PYTHONPATH=. python scripts/build_phase3.py --mode tfidf --k 10 --thr 0.25
```

### Running EDA
```bash
source .venv/bin/activate
PYTHONPATH=. python scripts/run_eda.py
```

## Results Summary

### Dataset Statistics
- Processed 23,196 articles with cleaned text
- Generated embeddings for all articles (TF-IDF and BERT)
- Computed metrics for 23,200 articles
- Built cascades for 1,000 articles (representative sample)

### Graph Statistics
- **User Interaction Graph:** 1.9M nodes, 3.6M edges
- **Bipartite Graph:** Article-tweet connections (389 MB)
- **Similarity Graph:** Article-article similarity (19 MB)
- **Cascades:** 1,000 individual cascade subgraphs (64 MB total)

### Key Insights
1. Fake news articles show higher average engagement (132.8 vs. 74.8 tweets)
2. Text cleaning reduces article length by approximately 20%
3. Dataset is imbalanced (75% real, 25% fake), accounted for in analysis
4. GossipCop dominates the dataset (95% of articles)

## Note on Large Files

Due to the large size of generated files (processed CSV files and graph files can exceed 1GB), these files are excluded from version control via `.gitignore`. However, all code, scripts, and intermediate outputs can be accessed through this repository. The scripts provided can regenerate all processed files from the raw data.

## Next Steps

The following tasks remain to be completed:

- **Link Analysis:** Implement PageRank and HITS algorithms on user interaction graph
- **Community Detection:** Apply Louvain algorithm to identify communities
- **Temporal Cascade Modeling:** Implement Independent Cascade Model (ICM) simulations
- **Comparative Analysis:** Compare fake vs. real news network structures
- **Final Report:** Compile comprehensive results and visualizations

## Contributing Workflow

1. Create a feature branch for each major task.
2. Run lint/tests locally before committing (tooling TBD).
3. Submit PRs for peer review; include figures/metrics as appropriate.

## Team

- **Group Leader:** Niharika Belavadi Shekar
- **Group Name:** Fact Checkers
- **Team Members:**
  - Ahmad Suleiman
  - Celine Taki
  - Pradyun Shrestha
  - Niharika Belavadi Shekar
  - Andy Mai
  - Jeremy Albios



