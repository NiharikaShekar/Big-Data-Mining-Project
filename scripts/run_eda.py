"""
Exploratory Data Analysis (EDA) for Raw and Processed Data

This script performs EDA on:
- Raw FakeNewsNet data (CSV files)
- Processed news data (cleaned text, labels)
- Article metrics (tweet counts)
- Basic statistics and distributions

NOTE: This focuses on data exploration, NOT graph analysis.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "FakeNewsNet"
PROC = ROOT / "data" / "processed"
FIGS = ROOT / "reports" / "figures"

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def save_fig(name: str) -> Path:
    p = FIGS / name
    ensure_dir(p)
    return p

def analyze_raw_data():
    """Analyze raw FakeNewsNet CSV files"""
    print("\n" + "="*60)
    print("RAW DATA ANALYSIS")
    print("="*60)
    
    # Find all CSV files in raw data
    csv_files = list(RAW.glob("*.csv")) if RAW.exists() else []
    
    if not csv_files:
        print(f"\n⚠ No CSV files found in {RAW}")
        return None
    
    print(f"\nFound {len(csv_files)} CSV file(s):")
    for f in csv_files:
        print(f"  - {f.name}")
    
    # Try to load and analyze each CSV
    all_data = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, low_memory=False)
            print(f"\n{csv_file.name}:")
            print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
            print(f"  Columns: {', '.join(df.columns.tolist()[:10])}")
            if len(df.columns) > 10:
                print(f"  ... and {len(df.columns) - 10} more columns")
            
            # Check for label column
            if 'label' in df.columns:
                print(f"  Label distribution:")
                print(df['label'].value_counts().sort_index())
            
            all_data.append((csv_file.name, df))
        except Exception as e:
            print(f"  ⚠ Error reading {csv_file.name}: {e}")
    
    return all_data

def analyze_processed_news():
    """Analyze processed news data"""
    print("\n" + "="*60)
    print("PROCESSED NEWS DATA ANALYSIS")
    print("="*60)
    
    news_file = PROC / "news_clean.csv"
    if not news_file.exists():
        print(f"\n⚠ Processed news file not found: {news_file}")
        return None
    
    news = pd.read_csv(news_file, low_memory=False)
    
    print(f"\nDataset shape: {news.shape[0]:,} rows × {news.shape[1]} columns")
    print(f"\nColumns: {', '.join(news.columns.tolist())}")
    
    print(f"\nLabel distribution:")
    label_counts = news['label'].value_counts().sort_index()
    print(label_counts)
    print(f"\nLabel percentages:")
    print((label_counts / len(news) * 100).round(2))
    
    print(f"\nMissing values:")
    missing = news.isnull().sum()
    print(missing[missing > 0])
    
    # Text length analysis
    if 'text' in news.columns:
        news['text_length'] = news['text'].astype(str).str.len()
        print(f"\nText length statistics (characters):")
        print(news['text_length'].describe())
        
        if 'clean_text' in news.columns:
            news['clean_text_length'] = news['clean_text'].astype(str).str.len()
            print(f"\nClean text length statistics (characters):")
            print(news['clean_text_length'].describe())
            print(f"\nAverage reduction: {((news['text_length'] - news['clean_text_length']) / news['text_length'] * 100).mean():.1f}%")
    
    # Source analysis
    if 'source' in news.columns:
        print(f"\nSource distribution:")
        print(news['source'].value_counts())
    
    return news

def analyze_article_metrics():
    """Analyze article-level metrics"""
    print("\n" + "="*60)
    print("ARTICLE METRICS ANALYSIS")
    print("="*60)
    
    metrics_file = PROC / "article_metrics.csv"
    if not metrics_file.exists():
        print(f"\n⚠ Article metrics file not found: {metrics_file}")
        return None
    
    metrics = pd.read_csv(metrics_file)
    
    print(f"\nTotal articles: {len(metrics):,}")
    print(f"\nLabel distribution:")
    label_counts = metrics['label'].value_counts().sort_index()
    print(label_counts)
    print(f"\nLabel percentages:")
    print((label_counts / len(metrics) * 100).round(2))
    
    print(f"\nTweet count statistics:")
    print(metrics['tweet_count'].describe())
    
    print(f"\nTweet count by label:")
    print(metrics.groupby('label')['tweet_count'].describe())
    
    # Statistical test: fake vs real tweet counts
    fake_counts = metrics[metrics['label'] == 0]['tweet_count']
    real_counts = metrics[metrics['label'] == 1]['tweet_count']
    
    if len(fake_counts) > 0 and len(real_counts) > 0:
        stat, pvalue = stats.mannwhitneyu(fake_counts, real_counts, alternative='two-sided')
        print(f"\nMann-Whitney U test (fake vs real tweet counts):")
        print(f"  Statistic: {stat:.2f}, p-value: {pvalue:.4f}")
        print(f"  {'Significant difference' if pvalue < 0.05 else 'No significant difference'} (α=0.05)")
    
    return metrics

def create_visualizations(news, metrics):
    """Create EDA visualizations"""
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    if news is None or metrics is None:
        print("\n⚠ Skipping visualizations - missing data")
        return
    
    # Prepare news columns for merge (include computed columns if they exist)
    # Don't include 'label' from news since metrics already has it
    news_cols = ['news_id', 'text', 'clean_text', 'source']
    if 'text_length' in news.columns:
        news_cols.append('text_length')
    if 'clean_text_length' in news.columns:
        news_cols.append('clean_text_length')
    
    # Merge news and metrics for comprehensive analysis
    combined = metrics.merge(
        news[news_cols], 
        on='news_id', 
        how='left'
    )
    
    # 1. Article Metrics Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1.1 Distribution of tweet counts by label
    ax = axes[0, 0]
    for label in sorted(metrics['label'].unique()):
        subset = metrics[metrics['label'] == label]
        ax.hist(subset['tweet_count'], bins=50, alpha=0.6, 
                label=f"{'Fake' if label == 0 else 'Real'}", density=True)
    ax.set_xlabel('Tweet Count')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Tweet Counts by Label')
    ax.legend()
    # Remove outliers for clarity
    q95 = metrics['tweet_count'].quantile(0.95)
    ax.set_xlim(0, q95)
    
    # 1.2 Box plot: tweet counts by label
    ax = axes[0, 1]
    metrics.boxplot(column='tweet_count', by='label', ax=ax)
    ax.set_xlabel('Label (0=Fake, 1=Real)')
    ax.set_ylabel('Tweet Count')
    ax.set_title('Tweet Count Distribution by Label')
    plt.suptitle('')  # Remove default title
    
    # 1.3 Log scale histogram
    ax = axes[1, 0]
    for label in sorted(metrics['label'].unique()):
        subset = metrics[metrics['label'] == label]
        ax.hist(np.log1p(subset['tweet_count']), bins=50, alpha=0.6,
                label=f"{'Fake' if label == 0 else 'Real'}", density=True)
    ax.set_xlabel('Log(Tweet Count + 1)')
    ax.set_ylabel('Density')
    ax.set_title('Log-Transformed Tweet Count Distribution')
    ax.legend()
    
    # 1.4 Cumulative distribution
    ax = axes[1, 1]
    for label in sorted(metrics['label'].unique()):
        subset = metrics[metrics['label'] == label]
        sorted_counts = np.sort(subset['tweet_count'])
        cumulative = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)
        ax.plot(sorted_counts, cumulative, label=f"{'Fake' if label == 0 else 'Real'}", linewidth=2)
    ax.set_xlabel('Tweet Count')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Cumulative Distribution of Tweet Counts')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_fig('eda_article_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: eda_article_metrics.png")
    
    # 2. Text Analysis Visualizations
    if 'text_length' in combined.columns:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 2.1 Text length distribution
        ax = axes[0, 0]
        for label in sorted(combined['label'].dropna().unique()):
            subset = combined[combined['label'] == label]
            if len(subset) > 0:
                ax.hist(subset['text_length'].dropna(), bins=50, alpha=0.6,
                        label=f"{'Fake' if label == 0 else 'Real'}", density=True)
        ax.set_xlabel('Text Length (characters)')
        ax.set_ylabel('Density')
        ax.set_title('Text Length Distribution by Label')
        ax.legend()
        
        # 2.2 Clean text length distribution
        if 'clean_text_length' in combined.columns:
            ax = axes[0, 1]
            for label in sorted(combined['label'].dropna().unique()):
                subset = combined[combined['label'] == label]
                if len(subset) > 0:
                    ax.hist(subset['clean_text_length'].dropna(), bins=50, alpha=0.6,
                            label=f"{'Fake' if label == 0 else 'Real'}", density=True)
            ax.set_xlabel('Clean Text Length (characters)')
            ax.set_ylabel('Density')
            ax.set_title('Clean Text Length Distribution by Label')
            ax.legend()
        
        # 2.3 Text length vs tweet count
        ax = axes[1, 0]
        valid_data = combined[['text_length', 'tweet_count']].dropna()
        if len(valid_data) > 0:
            ax.scatter(valid_data['text_length'], valid_data['tweet_count'], 
                       alpha=0.3, s=10)
        ax.set_xlabel('Text Length (characters)')
        ax.set_ylabel('Tweet Count')
        ax.set_title('Text Length vs Tweet Count')
        ax.set_yscale('log')
        
        # 2.4 Source distribution
        if 'source' in combined.columns:
            ax = axes[1, 1]
            source_counts = combined['source'].value_counts()
            ax.bar(range(len(source_counts)), source_counts.values)
            ax.set_xticks(range(len(source_counts)))
            ax.set_xticklabels(source_counts.index, rotation=45, ha='right')
            ax.set_ylabel('Number of Articles')
            ax.set_title('Articles by Source')
        
        plt.tight_layout()
        plt.savefig(save_fig('eda_text_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: eda_text_analysis.png")
    
    # 3. Label Distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 3.1 Label counts
    ax = axes[0]
    label_counts = metrics['label'].value_counts().sort_index()
    ax.bar(['Fake (0)', 'Real (1)'], label_counts.values, color=['red', 'green'], alpha=0.7)
    ax.set_ylabel('Number of Articles')
    ax.set_title('Label Distribution')
    for i, v in enumerate(label_counts.values):
        ax.text(i, v, f'{v:,}', ha='center', va='bottom')
    
    # 3.2 Label percentages
    ax = axes[1]
    label_pct = (label_counts / len(metrics) * 100).round(1)
    ax.pie(label_counts.values, labels=['Fake (0)', 'Real (1)'], 
           autopct='%1.1f%%', colors=['red', 'green'], startangle=90)
    ax.set_title('Label Distribution (Percentage)')
    
    plt.tight_layout()
    plt.savefig(save_fig('eda_label_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: eda_label_distribution.png")

def generate_summary_report(news, metrics):
    """Generate a summary EDA report"""
    print("\n" + "="*60)
    print("GENERATING SUMMARY REPORT")
    print("="*60)
    
    report_lines = [
        "# Exploratory Data Analysis (EDA) Summary\n",
        f"**Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        "\n---\n",
        "\n## Dataset Overview\n"
    ]
    
    if news is not None:
        report_lines.extend([
            f"- **Total Articles:** {len(news):,}\n",
            f"- **Fake News (label=0):** {(news['label'] == 0).sum():,} ({(news['label'] == 0).mean()*100:.1f}%)\n",
            f"- **Real News (label=1):** {(news['label'] == 1).sum():,} ({(news['label'] == 1).mean()*100:.1f}%)\n"
        ])
        
        if 'source' in news.columns:
            report_lines.append(f"- **Sources:** {', '.join(news['source'].unique())}\n")
    
    if metrics is not None:
        report_lines.extend([
            "\n## Article Metrics\n",
            f"- **Mean Tweet Count:** {metrics['tweet_count'].mean():.1f}\n",
            f"- **Median Tweet Count:** {metrics['tweet_count'].median():.1f}\n",
            f"- **Min Tweet Count:** {metrics['tweet_count'].min():,}\n",
            f"- **Max Tweet Count:** {metrics['tweet_count'].max():,}\n",
            f"- **Std Dev:** {metrics['tweet_count'].std():.1f}\n"
        ])
        
        # Fake vs Real comparison
        fake_mean = metrics[metrics['label'] == 0]['tweet_count'].mean()
        real_mean = metrics[metrics['label'] == 1]['tweet_count'].mean()
        report_lines.extend([
            "\n## Fake vs Real Comparison\n",
            f"- **Fake News Mean Tweet Count:** {fake_mean:.1f}\n",
            f"- **Real News Mean Tweet Count:** {real_mean:.1f}\n",
            f"- **Difference:** {abs(fake_mean - real_mean):.1f} ({'Fake' if fake_mean > real_mean else 'Real'} has more)\n"
        ])
    
    if news is not None and 'text_length' in news.columns:
        report_lines.extend([
            "\n## Text Statistics\n",
            f"- **Mean Text Length:** {news['text_length'].mean():.0f} characters\n",
            f"- **Median Text Length:** {news['text_length'].median():.0f} characters\n"
        ])
        if 'clean_text_length' in news.columns:
            reduction = ((news['text_length'] - news['clean_text_length']) / news['text_length'] * 100).mean()
            report_lines.append(f"- **Average Text Reduction (cleaning):** {reduction:.1f}%\n")
    
    report_lines.extend([
        "\n---\n",
        "\n## Generated Visualizations\n",
        "- `eda_article_metrics.png` - Article metrics and tweet count analysis\n",
        "- `eda_text_analysis.png` - Text length and source analysis\n",
        "- `eda_label_distribution.png` - Label distribution charts\n"
    ])
    
    report_text = "\n".join(report_lines)
    
    report_path = FIGS / "eda_summary_report.md"
    ensure_dir(report_path)
    with open(report_path, 'w') as f:
        f.write(report_text)
    
    print(f"\n✓ Saved summary report: {report_path}")
    print("\n" + report_text)

def main():
    """Run all EDA analyses"""
    print("\n" + "="*60)
    print("EXPLORATORY DATA ANALYSIS (EDA)")
    print("Raw and Processed Data Only")
    print("="*60)
    
    # Analyze raw data
    raw_data = analyze_raw_data()
    
    # Analyze processed news
    news = analyze_processed_news()
    
    # Analyze article metrics
    metrics = analyze_article_metrics()
    
    # Create visualizations
    create_visualizations(news, metrics)
    
    # Generate summary report
    generate_summary_report(news, metrics)
    
    print("\n" + "="*60)
    print("EDA COMPLETE!")
    print("="*60)
    print(f"\nAll figures saved to: {FIGS}")
    print(f"Summary report: {FIGS / 'eda_summary_report.md'}")

if __name__ == "__main__":
    main()
