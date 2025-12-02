#!/usr/bin/env python3
"""
Interactive Demo: Misinformation Detection & Cascade Analysis

A simple Streamlit web app that:
1. Takes user input (tweet/article text)
2. Compares against FakeNewsNet dataset using similarity matching
3. Predicts fake/real using nearest neighbor approach
4. Shows cascade trends and misinformation flow statistics
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# Add src to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocess.clean_text import clean_one_text

# Paths
PROC = ROOT / "data" / "processed"
TFIDF_VECT = PROC / "tfidf_vectors.npz"
TFIDF_VECTO = PROC / "tfidf_vectorizer.joblib"
NEWS_CLEAN = PROC / "news_clean.csv"
CASCADE_METRICS = PROC / "cascade_modeling" / "cascade_metrics.csv"
COMPARISON = PROC / "cascade_modeling" / "cascade_comparison_fake_vs_real.csv"


@st.cache_data
def load_data():
    """Load all necessary data (cached for performance)"""
    print("[INFO] Loading data...")
    
    # Load news data
    news_df = pd.read_csv(NEWS_CLEAN, low_memory=False)
    
    # Load TF-IDF vectors and vectorizer
    from scipy.sparse import load_npz
    tfidf_vectors = load_npz(TFIDF_VECT)
    vectorizer = joblib.load(TFIDF_VECTO)
    
    # Load cascade metrics if available
    cascade_df = None
    comparison_df = None
    if CASCADE_METRICS.exists():
        cascade_df = pd.read_csv(CASCADE_METRICS)
    if COMPARISON.exists():
        comparison_df = pd.read_csv(COMPARISON)
    
    return news_df, tfidf_vectors, vectorizer, cascade_df, comparison_df


def find_similar_articles(text, news_df, tfidf_vectors, vectorizer, top_k=5):
    """Find most similar articles in dataset"""
    # Clean input text
    cleaned_text = clean_one_text(text)
    
    # Vectorize input
    input_vector = vectorizer.transform([cleaned_text])
    
    # Compute cosine similarity
    similarities = cosine_similarity(input_vector, tfidf_vectors).flatten()
    
    # Get top-k most similar
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        article = news_df.iloc[idx]
        results.append({
            "news_id": article["news_id"],
            "text": article["text"][:200] + "..." if len(article["text"]) > 200 else article["text"],
            "label": "Fake" if article["label"] == 0 else "Real",
            "similarity": float(similarities[idx]),
            "source": article.get("source", "unknown")
        })
    
    return results


def predict_fake_real(similar_articles):
    """Predict fake/real based on similar articles"""
    if not similar_articles:
        return "Unknown", 0.5
    
    # Weighted voting based on similarity
    fake_score = 0.0
    real_score = 0.0
    total_weight = 0.0
    
    for article in similar_articles:
        weight = article["similarity"]
        total_weight += weight
        
        if article["label"] == "Fake":
            fake_score += weight
        else:
            real_score += weight
    
    if total_weight == 0:
        return "Unknown", 0.5
    
    fake_prob = fake_score / total_weight
    real_prob = real_score / total_weight
    
    if fake_prob > real_prob:
        return "Fake", fake_prob
    else:
        return "Real", real_prob


def main():
    st.set_page_config(
        page_title="Misinformation Cascade Analyzer",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Misinformation Cascade Analyzer")
    st.markdown("**Analyze tweets/articles and detect misinformation using our FakeNewsNet dataset**")
    st.divider()
    
    # Load data
    try:
        with st.spinner("Loading dataset and models..."):
            news_df, tfidf_vectors, vectorizer, cascade_df, comparison_df = load_data()
        st.success(f"✅ Loaded {len(news_df):,} articles from FakeNewsNet dataset")
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.stop()
    
    # Sidebar for trends/stats
    with st.sidebar:
        st.header("📊 Dataset Statistics")
        st.metric("Total Articles", f"{len(news_df):,}")
        
        fake_count = len(news_df[news_df["label"] == 0])
        real_count = len(news_df[news_df["label"] == 1])
        st.metric("Fake News", f"{fake_count:,} ({fake_count/len(news_df)*100:.1f}%)")
        st.metric("Real News", f"{real_count:,} ({real_count/len(news_df)*100:.1f}%)")
        
        if comparison_df is not None and len(comparison_df) > 0:
            st.divider()
            st.header("📈 Cascade Trends")
            fake_row = comparison_df[comparison_df["label"] == "fake"]
            real_row = comparison_df[comparison_df["label"] == "real"]
            
            if len(fake_row) > 0:
                st.subheader("Fake News Cascades")
                st.metric("Avg Cascade Size", f"{fake_row['avg_cascade_size'].values[0]:.1f}")
                st.metric("Avg Spread Rate", f"{fake_row['avg_spread_rate'].values[0]:.1f}")
            
            if len(real_row) > 0:
                st.subheader("Real News Cascades")
                st.metric("Avg Cascade Size", f"{real_row['avg_cascade_size'].values[0]:.1f}")
                st.metric("Avg Spread Rate", f"{real_row['avg_spread_rate'].values[0]:.1f}")
    
    # Main input area
    st.header("🔎 Analyze Article/Tweet")
    
    input_text = st.text_area(
        "Paste the article text or tweet here:",
        height=150,
        placeholder="Enter the text you want to analyze..."
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        analyze_button = st.button("🔍 Analyze", type="primary", use_container_width=True)
    
    if analyze_button and input_text.strip():
        with st.spinner("Analyzing text and comparing with dataset..."):
            # Find similar articles
            similar_articles = find_similar_articles(input_text, news_df, tfidf_vectors, vectorizer, top_k=5)
            
            # Predict fake/real
            prediction, confidence = predict_fake_real(similar_articles)
            
            # Display results
            st.divider()
            st.header("🎯 Prediction Result")
            
            # Prediction with color coding
            if prediction == "Fake":
                st.error(f"⚠️ **Predicted: {prediction}** (Confidence: {confidence*100:.1f}%)")
            else:
                st.success(f"✅ **Predicted: {prediction}** (Confidence: {confidence*100:.1f}%)")
            
            # Show similar articles
            st.subheader("📚 Most Similar Articles in Dataset")
            
            for i, article in enumerate(similar_articles, 1):
                with st.expander(f"Article {i}: {article['label']} (Similarity: {article['similarity']*100:.1f}%)"):
                    st.write(f"**ID:** {article['news_id']}")
                    st.write(f"**Source:** {article['source']}")
                    st.write(f"**Text:** {article['text']}")
                    if article['label'] == "Fake":
                        st.error(f"Label: {article['label']}")
                    else:
                        st.success(f"Label: {article['label']}")
            
            # Show insights
            st.divider()
            st.subheader("💡 Insights")
            
            fake_similar = sum(1 for a in similar_articles if a["label"] == "Fake")
            real_similar = sum(1 for a in similar_articles if a["label"] == "Real")
            
            st.write(f"- Found **{fake_similar} fake** and **{real_similar} real** similar articles")
            st.write(f"- Average similarity: **{np.mean([a['similarity'] for a in similar_articles])*100:.1f}%**")
            
            if comparison_df is not None and len(comparison_df) > 0:
                fake_row = comparison_df[comparison_df['label'] == 'fake']
                if len(fake_row) > 0:
                    spread_rate = fake_row['avg_spread_rate'].values[0]
                    st.write(f"- Based on cascade analysis: Fake news spreads **{spread_rate:.1f}** nodes per time step on average")
    
    elif analyze_button:
        st.warning("⚠️ Please enter some text to analyze!")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Built with FakeNewsNet Dataset | Misinformation Cascade Analysis Project</p>
        <p>Group: Fact Checkers | Big Data Mining Course</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

