# Misinformation Cascade Analysis on X (Twitter)

This repository hosts the Fact Checkers team project for modeling how misinformation emerges, spreads, and evolves across X (Twitter). We will combine graph-based mining, cascade modeling, and comparative analysis between fake and factual narratives using sources such as FakeNewsNet and (optionally) live Twitter API collections.

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

## Initial Dependency Stack
The `requirements.txt` file seeds the environment with the following
capabilities:
- Data wrangling: `pandas`, `numpy`, `pyarrow`
- Graph analysis: `networkx`, `python-igraph`, `graphviz`
- Machine learning & embeddings: `scikit-learn`, `sentence-transformers`
- Visualization: `matplotlib`, `seaborn`, `plotly`
- Experiment support: `tqdm`, `python-dotenv`, `jupyter`

We will extend the list as the implementation matures (e.g., PyTorch for custom embedding pipelines or Snap for large-scale cascade simulation).

## Getting Started Checklist
- [ ] Download FakeNewsNet (see forthcoming data acquisition scripts).
- [ ] Parse social context JSON and construct the global interaction graph.
- [ ] Run baseline PageRank/HITS and visualize top influencers.
- [ ] Evaluate community structure differences between fake and real subgraphs.
- [ ] Simulate diffusion dynamics and capture cascade metrics.
- [ ] Compile findings into the final report and presentation deck.

## Contributing Workflow
1. Create a feature branch for each major task.
2. Run lint/tests locally before committing (tooling TBD).
3. Submit PRs for peer review; include figures/metrics as appropriate.

## License
TBD — clarify with course requirements before public release.
