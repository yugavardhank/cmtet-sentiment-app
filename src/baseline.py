"""
baseline.py - TF-IDF + LinearSVC baseline (M1).
"""
import json
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, f1_score, accuracy_score
)
from sklearn.model_selection import cross_val_score

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    CLEAN_CSV, SPLITS_JSON, RESULTS_DIR, LABELS,
    TFIDF_PARAMS, SVM_C_GRID, LABEL2ID
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def load_splits():
    df = pd.read_csv(CLEAN_CSV)
    with open(SPLITS_JSON) as f:
        splits = json.load(f)
    return df, splits


def build_pipeline(C: float = 1.0) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
        ("svc",   LinearSVC(C=C, class_weight="balanced", max_iter=5000)),
    ])


def evaluate_fold(pipe, X_train, y_train, X_dev, y_dev, X_test, y_test):
    pipe.fit(X_train, y_train)
    dev_preds  = pipe.predict(X_dev)
    test_preds = pipe.predict(X_test)
    dev_f1  = f1_score(y_dev,  dev_preds,  average="macro", zero_division=0)
    test_f1 = f1_score(y_test, test_preds, average="macro", zero_division=0)
    test_acc = accuracy_score(y_test, test_preds)
    report = classification_report(
        y_test, test_preds,
        target_names=LABELS, output_dict=True, zero_division=0
    )
    return {
        "dev_f1": dev_f1,
        "test_f1": test_f1,
        "test_acc": test_acc,
        "report": report,
        "test_preds": test_preds.tolist(),
    }


def run_baseline():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df, splits = load_splits()

    mode  = splits["mode"]
    folds = splits["folds"]

    all_test_preds  = []
    all_test_labels = []
    fold_metrics    = []
    best_metrics    = None

    for fold_data in folds:
        train_idx = fold_data["train"]
        dev_idx   = fold_data["dev"]
        test_idx  = fold_data["test"]

        X_train = df.iloc[train_idx]["clean_text"].tolist()
        y_train = df.iloc[train_idx]["label"].tolist()
        X_dev   = df.iloc[dev_idx]["clean_text"].tolist()
        y_dev   = df.iloc[dev_idx]["label"].tolist()
        X_test  = df.iloc[test_idx]["clean_text"].tolist()
        y_test  = df.iloc[test_idx]["label"].tolist()

        # Tune C on dev set
        best_c, best_dev_f1 = SVM_C_GRID[0], -1.0
        for c in SVM_C_GRID:
            pipe = build_pipeline(C=c)
            pipe.fit(X_train, y_train)
            preds  = pipe.predict(X_dev)
            dev_f1 = f1_score(y_dev, preds, average="macro", zero_division=0)
            if dev_f1 > best_dev_f1:
                best_dev_f1 = dev_f1
                best_c = c

        log.info("Fold %d  best C=%.1f  dev macro-F1=%.4f",
                 fold_data["fold"], best_c, best_dev_f1)

        # Final evaluation on test
        pipe = build_pipeline(C=best_c)
        res  = evaluate_fold(pipe, X_train, y_train, X_dev, y_dev, X_test, y_test)
        fold_metrics.append(res)

        all_test_preds.extend(res["test_preds"])
        all_test_labels.extend(y_test)

        if best_metrics is None or res["test_f1"] > best_metrics["test_f1"]:
            best_metrics = res

    # Aggregate
    macro_f1s = [m["test_f1"] for m in fold_metrics]
    accs      = [m["test_acc"] for m in fold_metrics]
    mean_f1   = float(np.mean(macro_f1s))
    std_f1    = float(np.std(macro_f1s))
    mean_acc  = float(np.mean(accs))

    # Per-class F1 from best fold (or overall if single fold)
    per_class = {}
    for lbl in LABELS:
        per_class[lbl] = best_metrics["report"][lbl]["f1-score"]

    metrics_out = {
        "model": "TF-IDF + LinearSVC",
        "macro_f1_mean": mean_f1,
        "macro_f1_std":  std_f1,
        "accuracy":      mean_acc,
        "per_class_f1":  per_class,
        "num_folds":     len(fold_metrics),
    }

    # Save metrics
    with open(RESULTS_DIR / "m1_metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    # Save predictions (only test indices from first/only fold for single-fold)
    first_fold = folds[0]
    test_idx   = first_fold["test"]
    preds_df = pd.DataFrame({
        "index":  test_idx,
        "text":   df.iloc[test_idx]["text"].tolist(),
        "true":   df.iloc[test_idx]["label"].tolist(),
        "pred":   fold_metrics[0]["test_preds"],
    })
    preds_df.to_csv(RESULTS_DIR / "m1_preds.csv", index=False)

    print("\n=== Baseline (M1) Results ===")
    print(f"  Macro-F1 : {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"  Accuracy : {mean_acc:.4f}")
    print("  Per-class F1:")
    for lbl, f1 in per_class.items():
        print(f"    {lbl}: {f1:.4f}")

    log.info("Results saved to %s", RESULTS_DIR)
    return metrics_out


if __name__ == "__main__":
    run_baseline()
