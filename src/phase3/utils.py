from __future__ import annotations
from pathlib import Path
from typing import List, Optional
import re

"""
Utilities for Phase 3-lite.

Why:
- FakeNewsNet CSVs store tweet IDs as messy strings (tabs/commas/pipes/spaces).
- Downstream steps need a consistent list of tweet IDs and safe output dirs.
"""

# Robust splitter: commas, tabs, semicolons, pipes, any whitespace
_SEP = re.compile(r"[,\t;|\s]+")

def parse_tweet_ids(raw: Optional[str]) -> List[str]:
    """
    Return a cleaned list of tweet IDs from a raw CSV field.
    Keeps only tokens that contain at least one digit (defensive for odd artifacts).

    Examples:
        "111\t222 333"  -> ["111", "222", "333"]
        " 444,555|666 " -> ["444", "555", "666"]
        "" / None / "nan" -> []
    """
    if raw is None:
        return []
    s = str(raw).strip()
    if not s or s.lower() in {"nan", "none", "[]"}:
        return []
    parts = [p for p in _SEP.split(s) if p]
    return [p for p in parts if any(ch.isdigit() for ch in p)]

def ensure_parent(path: Path) -> None:
    """
    Ensure the parent directory for 'path' exists (idempotent).
    Why: avoids scattered mkdir logic before writing CSVs/graphs/figures.
    """
    path.parent.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # Quick self-check you can run from VS Code terminal:
    samples = ["111\t222 333", "444,555|666", "", None, "nan", "[]", "x-777 y"]
    for s in samples:
        print(s, "→", parse_tweet_ids(s))
