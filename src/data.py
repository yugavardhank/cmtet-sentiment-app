"""
data.py - Parse, dedupe, clean, compute CMI, and split the raw dataset.
"""
import re
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from config import (
    DATA_RAW, CLEAN_CSV, SPLITS_JSON,
    LABELS, SPLIT_SEED, SMALL_DATASET_THRESHOLD,
    N_FOLDS, TRAIN_RATIO, DEV_RATIO
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Regex for label line: "NEG: text", "POS: text", "NTL: text"
_LABEL_RE = re.compile(r"^(NEG|POS|NTL):\s*(.+)$", re.IGNORECASE)

# ── Clean helpers ──────────────────────────────────────────────────────────────
_URL_RE    = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_REPEAT_RE = re.compile(r"(.)\1{2,}")   # 3+ repeated chars -> 2


def clean(text: str) -> str:
    """Normalise a single text string."""
    text = _URL_RE.sub("<url>", text)
    text = _MENTION_RE.sub("@user", text)
    text = _REPEAT_RE.sub(r"\1\1", text)
    # lowercase only ASCII (Latin) characters, preserve Telugu/emojis/hashtags
    result = []
    for ch in text:
        if ch.isascii() and ch.isalpha():
            result.append(ch.lower())
        else:
            result.append(ch)
    return "".join(result)


# ── CMI ────────────────────────────────────────────────────────────────────────

def cmi(tags: str) -> float:
    """
    Code-Mixing Index (Gamback & Sikdar 2016).
    CMI = 100 * (1 - max(n_en, n_te) / (n_en + n_te))
    Returns 0 if no en/te tags.
    """
    parts = tags.strip().split()
    n_en = parts.count("en")
    n_te = parts.count("te")
    total = n_en + n_te
    if total == 0:
        return 0.0
    return 100.0 * (1.0 - max(n_en, n_te) / total)


# ── Parser ─────────────────────────────────────────────────────────────────────

def parse_file(path: Path = DATA_RAW) -> pd.DataFrame:
    """
    Parse the raw data file into a DataFrame.

    Expected format (groups separated by one or more blank lines):
        LABEL: text
        tag1 tag2 tag3 ...

    Only rows where word count == tag count are kept.
    """
    records = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        lines = [ln.rstrip("\r\n") for ln in fh]

    i = 0
    total = len(lines)
    while i < total:
        # Skip blank lines
        if not lines[i].strip():
            i += 1
            continue

        m = _LABEL_RE.match(lines[i].strip())
        if not m:
            i += 1
            continue

        label = m.group(1).upper()
        text  = m.group(2).strip()
        i += 1

        # Next non-blank line should be the tag line
        while i < total and not lines[i].strip():
            i += 1

        if i >= total:
            break

        tag_line = lines[i].strip()
        i += 1

        # Alignment check
        words = text.split()
        tags  = tag_line.split()
        if len(words) != len(tags):
            continue   # drop misaligned row

        records.append({"text": text, "label": label, "tags": tag_line})

    df = pd.DataFrame(records)
    log.info("Parsed %d aligned rows from %s", len(df), path)
    return df


# ── Deduplication ──────────────────────────────────────────────────────────────

def _make_key(text: str) -> str:
    return " ".join(text.lower().split())


def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group rows by normalised text key.
    - If a key has conflicting labels -> drop all copies.
    - Otherwise keep one copy (the first).
    """
    df = df.copy()
    df["_key"] = df["text"].apply(_make_key)

    groups = df.groupby("_key")["label"].nunique()
    conflict_keys = set(groups[groups > 1].index)

    before = len(df)
    df = df[~df["_key"].isin(conflict_keys)]
    df = df.drop_duplicates(subset="_key")
    df = df.drop(columns=["_key"])

    log.info(
        "Deduplication: %d -> %d rows (%d conflicting keys dropped)",
        before, len(df), len(conflict_keys)
    )
    return df.reset_index(drop=True)


# ── Split ──────────────────────────────────────────────────────────────────────

def split(df: pd.DataFrame) -> dict:
    """
    Stratified split or K-fold CV.
    Returns a dict with key 'folds' (list of fold dicts with train/dev/test indices)
    or a single fold dict for the hold-out case.
    Saves indices to splits.json.
    """
    n = len(df)
    labels = df["label"].tolist()

    if n < SMALL_DATASET_THRESHOLD:
        log.info("Small dataset (%d rows) -> 5-fold CV", n)
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SPLIT_SEED)
        folds = []
        for fold_i, (traindev_idx, test_idx) in enumerate(skf.split(df, labels)):
            # Further split traindev into train / dev (90 / 10 of traindev)
            traindev_labels = [labels[i] for i in traindev_idx]
            train_idx, dev_idx = train_test_split(
                traindev_idx,
                test_size=0.10,
                stratify=traindev_labels,
                random_state=SPLIT_SEED,
            )
            folds.append({
                "fold": fold_i,
                "train": train_idx.tolist(),
                "dev":   dev_idx.tolist(),
                "test":  test_idx.tolist(),
            })
        result = {"mode": "cv", "folds": folds}
    else:
        log.info("Large dataset (%d rows) -> 80/10/10 split", n)
        idx = list(range(n))
        train_idx, testdev_idx, train_labels, testdev_labels = train_test_split(
            idx, labels, test_size=(DEV_RATIO + DEV_RATIO),
            stratify=labels, random_state=SPLIT_SEED
        )
        dev_idx, test_idx = train_test_split(
            testdev_idx,
            test_size=0.5,
            stratify=testdev_labels,
            random_state=SPLIT_SEED,
        )
        result = {
            "mode": "holdout",
            "folds": [{
                "fold": 0,
                "train": train_idx,
                "dev":   dev_idx,
                "test":  test_idx,
            }]
        }

    SPLITS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(SPLITS_JSON, "w") as f:
        json.dump(result, f, indent=2)

    log.info("Splits saved to %s", SPLITS_JSON)
    return result


# ── Main pipeline ──────────────────────────────────────────────────────────────

def build_dataset(raw_path: Path = DATA_RAW) -> pd.DataFrame:
    """Full pipeline: parse -> dedupe -> clean -> CMI -> split -> save."""
    df = parse_file(raw_path)
    df = dedupe(df)

    df["clean_text"] = df["text"].apply(clean)
    df["cmi"] = df["tags"].apply(cmi)

    # Reorder columns
    df = df[["text", "clean_text", "label", "cmi", "tags"]]

    # Print label distribution
    counts = df["label"].value_counts()
    log.info("Label distribution:\n%s", counts.to_string())
    print("\nLabel distribution:")
    for lbl, cnt in counts.items():
        print(f"  {lbl}: {cnt:>6,}  ({100*cnt/len(df):.1f}%)")
    print(f"  TOTAL: {len(df):>6,}\n")

    # Save
    CLEAN_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_CSV, index=False)
    log.info("Saved clean.csv (%d rows) -> %s", len(df), CLEAN_CSV)

    # Split
    split(df)

    return df


if __name__ == "__main__":
    build_dataset()
