"""
Helper script to collect full FakeNewsNet dataset with user interaction data.

This script helps set up and run the official FakeNewsNet data collection tool
to get the complete dataset including:
- Tweet JSON objects (with user IDs, timestamps)
- Retweet JSON objects (retweet relationships)
- User profiles
- User followers/following

Prerequisites:
1. Twitter API keys (get from https://developer.twitter.com/)
2. Clone FakeNewsNet repository
3. Configure API keys in tweet_keys_file.json

Usage:
    python -m src.get_data.collect_full_fakenewsnet --help
"""

from __future__ import annotations
from pathlib import Path
import json
import subprocess
import argparse


def create_config_template(config_path: Path) -> None:
    """Create a template config.json for FakeNewsNet data collection."""
    config = {
        "num_process": 4,
        "tweet_keys_file": "code/resources/tweet_keys_file.json",
        "data_collection_choice": [
            {"news_source": "politifact", "label": "fake"},
            {"news_source": "politifact", "label": "real"},
            {"news_source": "gossipcop", "label": "fake"},
            {"news_source": "gossipcop", "label": "real"}
        ],
        "data_features_to_collect": [
            "news_articles",
            "tweets",
            "retweets",
            "user_profile",
            "user_timeline_tweets",
            "user_followers",
            "user_following"
        ],
        "dataset_dir": "data/raw/FakeNewsNet_full"
    }
    
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"[OK] Created config template → {config_path}")


def create_keys_template(keys_path: Path) -> None:
    """Create a template tweet_keys_file.json for Twitter API keys."""
    template = [
        {
            "app_key": "YOUR_APP_KEY",
            "app_secret": "YOUR_APP_SECRET",
            "oauth_token": "YOUR_OAUTH_TOKEN",
            "oauth_token_secret": "YOUR_OAUTH_TOKEN_SECRET"
        }
    ]
    
    keys_path.parent.mkdir(parents=True, exist_ok=True)
    with open(keys_path, 'w') as f:
        json.dump(template, f, indent=2)
    print(f"[OK] Created API keys template → {keys_path}")
    print("[INFO] Please fill in your Twitter API credentials!")


def check_fakenewsnet_repo(repo_path: Path) -> bool:
    """Check if FakeNewsNet repository is cloned."""
    if not repo_path.exists():
        return False
    if not (repo_path / "code" / "main.py").exists():
        return False
    return True


def main():
    ROOT = Path(__file__).resolve().parents[2]
    FAKENEWSNET_REPO = ROOT / "external" / "FakeNewsNet"
    
    p = argparse.ArgumentParser(
        description="Helper for collecting full FakeNewsNet dataset"
    )
    p.add_argument("--setup", action="store_true",
                   help="Create config and keys templates")
    p.add_argument("--check", action="store_true",
                   help="Check if FakeNewsNet repo is available")
    p.add_argument("--repo-path", type=Path, default=FAKENEWSNET_REPO,
                   help="Path to FakeNewsNet repository")
    args = p.parse_args()
    
    if args.setup:
        config_path = ROOT / "config" / "fakenewsnet_config.json"
        keys_path = ROOT / "config" / "tweet_keys_file.json"
        create_config_template(config_path)
        create_keys_template(keys_path)
        print("\n[INFO] Next steps:")
        print("1. Fill in your Twitter API keys in config/tweet_keys_file.json")
        print("2. Clone FakeNewsNet: git clone https://github.com/KaiDMML/FakeNewsNet.git external/FakeNewsNet")
        print("3. Copy config files to FakeNewsNet/code/ directory")
        print("4. Run: cd external/FakeNewsNet/code && python main.py")
    
    if args.check:
        if check_fakenewsnet_repo(args.repo_path):
            print(f"[OK] FakeNewsNet repository found at {args.repo_path}")
        else:
            print(f"[WARN] FakeNewsNet repository not found at {args.repo_path}")
            print("[INFO] Clone it with: git clone https://github.com/KaiDMML/FakeNewsNet.git external/FakeNewsNet")


if __name__ == "__main__":
    main()

