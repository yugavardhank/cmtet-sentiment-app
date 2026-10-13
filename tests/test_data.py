"""
test_data.py - Tests for the data pipeline (parse, dedupe, split).
"""
import json
import sys
import pytest
import pandas as pd
from pathlib import Path

# Allow src imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data import parse_file, dedupe, split, clean, cmi, _make_key
from config import DATA_RAW, CLEAN_CSV, SPLITS_JSON


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def raw_df():
    """Parse the real raw data file once for the whole module."""
    return parse_file(DATA_RAW)


@pytest.fixture(scope="module")
def clean_df():
    """Return deduplicated dataframe."""
    df = parse_file(DATA_RAW)
    return dedupe(df)


# ── Test 1: Parser keeps only aligned rows ─────────────────────────────────────

def test_parser_alignment(raw_df):
    """Every row in raw_df must have word count == tag count."""
    for _, row in raw_df.iterrows():
        words = row["text"].split()
        tags  = row["tags"].split()
        assert len(words) == len(tags), (
            f"Misaligned row: '{row['text'][:60]}' "
            f"(words={len(words)}, tags={len(tags)})"
        )


def test_parser_labels_valid(raw_df):
    """All labels are one of NEG, NTL, POS."""
    assert set(raw_df["label"].unique()).issubset({"NEG", "NTL", "POS"})


def test_parser_nonempty(raw_df):
    """Parser should produce at least some rows."""
    assert len(raw_df) > 0


# ── Test 2: No duplicate keys remain after dedupe ──────────────────────────────

def test_no_duplicate_keys(clean_df):
    """After deduplication, every normalised text key is unique."""
    keys = clean_df["text"].apply(_make_key)
    assert keys.nunique() == len(clean_df), "Duplicate keys found after deduplication"


def test_no_conflicting_labels(clean_df):
    """No two rows with the same text key have different labels."""
    clean_df = clean_df.copy()
    clean_df["_key"] = clean_df["text"].apply(_make_key)
    per_key = clean_df.groupby("_key")["label"].nunique()
    conflicts = per_key[per_key > 1]
    assert len(conflicts) == 0, f"Conflicting labels for keys: {list(conflicts.index[:5])}"


# ── Test 3: Splits don't overlap ───────────────────────────────────────────────

def test_splits_no_overlap():
    """Train, dev, and test index sets are disjoint for every fold."""
    if not SPLITS_JSON.exists():
        pytest.skip("splits.json not yet generated; run data.py first")

    with open(SPLITS_JSON) as f:
        splits = json.load(f)

    for fold in splits["folds"]:
        train_set = set(fold["train"])
        dev_set   = set(fold["dev"])
        test_set  = set(fold["test"])

        assert len(train_set & dev_set)  == 0, "Train/dev overlap in fold"
        assert len(train_set & test_set) == 0, "Train/test overlap in fold"
        assert len(dev_set   & test_set) == 0, "Dev/test overlap in fold"


def test_splits_cover_all_rows():
    """Union of all fold indices covers all rows."""
    if not SPLITS_JSON.exists():
        pytest.skip("splits.json not yet generated")
    if not CLEAN_CSV.exists():
        pytest.skip("clean.csv not yet generated")

    df = pd.read_csv(CLEAN_CSV)
    with open(SPLITS_JSON) as f:
        splits = json.load(f)

    if splits["mode"] == "holdout":
        fold = splits["folds"][0]
        all_idx = set(fold["train"]) | set(fold["dev"]) | set(fold["test"])
        assert all_idx == set(range(len(df))), "Some rows are missing from the split"


# ── Test 4: clean() function ───────────────────────────────────────────────────

def test_clean_url():
    assert "<url>" in clean("Check https://example.com now")


def test_clean_mention():
    assert "@user" in clean("@SomeUser hello")


def test_clean_repeated_chars():
    result = clean("sooooooo good")
    assert "soo" in result and "sooo" not in result


def test_clean_lowercase_latin():
    result = clean("Super Movie")
    assert result == "super movie"


def test_clean_preserves_emoji():
    result = clean("Nice 😊 work")
    assert "😊" in result


# ── Test 5: cmi() function ─────────────────────────────────────────────────────

def test_cmi_pure_english():
    assert cmi("en en en en") == 0.0


def test_cmi_pure_telugu():
    assert cmi("te te te te") == 0.0


def test_cmi_mixed():
    val = cmi("en en te te")
    assert abs(val - 50.0) < 1e-6


def test_cmi_no_relevant_tags():
    assert cmi("univ univ univ") == 0.0
