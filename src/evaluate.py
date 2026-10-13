"""
evaluate.py - Build results table, CMI breakdown, confusion matrix, error analysis.
"""
import json
import sys
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    RESULTS_DIR, CLEAN_CSV, SPLITS_JSON, LABELS, MODELS_DIR
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", palette="muted")


# ── Load all results ───────────────────────────────────────────────────────────

def load_all_metrics() -> pd.DataFrame:
    """Load all *_metrics.json files from results/."""
    rows = []
    for p in sorted(RESULTS_DIR.glob("*_metrics.json")):
        if p.stem == "best_model_info":
            continue
        with open(p) as f:
            d = json.load(f)
        row = {
            "model": d.get("model", p.stem),
            "seed":  d.get("seed", "all"),
            "macro_f1": d.get("macro_f1") or d.get("macro_f1_mean"),
            "macro_f1_std": d.get("macro_f1_std", 0.0),
            "accuracy": d.get("accuracy"),
        }
        for lbl in LABELS:
            row[f"f1_{lbl}"] = d.get("per_class_f1", {}).get(lbl, 0.0)
        rows.append(row)
    return pd.DataFrame(rows)


# ── Main results table ─────────────────────────────────────────────────────────

def build_results_table(df_metrics: pd.DataFrame) -> pd.DataFrame:
    # Group by model, aggregate seeds
    agg = df_metrics.groupby("model").agg(
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        accuracy=("accuracy", "mean"),
        **{f"f1_{l}": (f"f1_{l}", "mean") for l in LABELS}
    ).reset_index()
    agg["macro_f1_std"] = agg["macro_f1_std"].fillna(0.0)
    return agg


# ── CMI breakdown ──────────────────────────────────────────────────────────────

def cmi_breakdown(preds_files: list[Path], df_clean: pd.DataFrame,
                  splits: dict) -> pd.DataFrame:
    """
    Bin test rows by CMI tercile and compute macro-F1 per bin per model.
    """
    test_idx = splits["folds"][0]["test"]
    test_cmi = df_clean.iloc[test_idx]["cmi"].values
    bins = np.percentile(test_cmi, [33.3, 66.7])

    def get_bin(v):
        if v <= bins[0]:
            return "low"
        elif v <= bins[1]:
            return "medium"
        return "high"

    cmi_labels = [get_bin(v) for v in test_cmi]
    true_labels = df_clean.iloc[test_idx]["label"].tolist()

    rows = []
    for pf in preds_files:
        pf = Path(pf)
        if not pf.exists():
            continue
        preds_df = pd.read_csv(pf)
        pred_labels = preds_df["pred"].tolist()

        model_name = pf.stem.replace("_preds", "").replace("_seed42", "")
        for bin_name in ["low", "medium", "high"]:
            mask = [b == bin_name for b in cmi_labels]
            if not any(mask):
                continue
            y_true = [t for t, m in zip(true_labels, mask) if m]
            y_pred = [p for p, m in zip(pred_labels, mask) if m]
            macro_f1 = f1_score(y_true, y_pred, average="macro",
                                labels=LABELS, zero_division=0)
            rows.append({
                "model": model_name,
                "cmi_bin": bin_name,
                "macro_f1": macro_f1,
                "n_samples": sum(mask),
            })

    return pd.DataFrame(rows)


def plot_cmi_breakdown(cmi_df: pd.DataFrame):
    if cmi_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    pivot = cmi_df.pivot(index="cmi_bin", columns="model", values="macro_f1")
    pivot = pivot.reindex(["low", "medium", "high"])
    pivot.plot(kind="bar", ax=ax, rot=0)
    ax.set_title("Macro-F1 by CMI Bin", fontsize=14)
    ax.set_xlabel("CMI Bin")
    ax.set_ylabel("Macro-F1")
    ax.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    out = RESULTS_DIR / "cmi_breakdown.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("CMI breakdown plot saved to %s", out)


# ── Confusion matrix ───────────────────────────────────────────────────────────

def plot_confusion_matrix(true_labels: list, pred_labels: list,
                          model_name: str = "best_model"):
    cm = confusion_matrix(true_labels, pred_labels, labels=LABELS)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=LABELS, yticklabels=LABELS, ax=ax
    )
    ax.set_title(f"Confusion Matrix: {model_name}", fontsize=13)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    safe = model_name.replace("/", "_").replace(" ", "_")
    out = RESULTS_DIR / f"confusion_matrix_{safe}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("Confusion matrix saved to %s", out)


# ── Error analysis ─────────────────────────────────────────────────────────────

def error_analysis(preds_file: Path, df_clean: pd.DataFrame,
                   n_samples: int = 40) -> pd.DataFrame:
    preds_df = pd.read_csv(preds_file)
    errors = preds_df[preds_df["true"] != preds_df["pred"]].copy()

    if len(errors) > n_samples:
        errors = errors.sample(n=n_samples, random_state=42)

    # Heuristic error categories
    def categorise(row):
        text = str(row["text"]).lower()
        cats = []
        if any(w in text for w in ["not", "no", "never", "n't", "ledu", "kadhu", "kaadu"]):
            cats.append("negation")
        if any(w in text for w in ["lol", "haha", "😂", "😆", "😅", "kidding"]):
            cats.append("sarcasm")
        if row.get("cmi", 50) < 15:
            cats.append("non_telugu_text")
        if not cats:
            cats.append("transliteration_variant")
        return "; ".join(cats)

    if "cmi" in preds_df.columns:
        idx_map = {r["index"]: r for _, r in preds_df.iterrows()}
        errors["cmi"] = errors["index"].apply(
            lambda i: df_clean.iloc[i]["cmi"] if i < len(df_clean) else 0.0
        )
    errors["error_category"] = errors.apply(categorise, axis=1)

    out = RESULTS_DIR / "error_samples.csv"
    errors.to_csv(out, index=False)
    log.info("Error samples saved to %s", out)

    # Print category counts
    cats_flat = []
    for cats in errors["error_category"].str.split("; "):
        cats_flat.extend(cats)
    from collections import Counter
    counts = Counter(cats_flat)
    print("\nError category counts:")
    for cat, cnt in counts.most_common():
        print(f"  {cat}: {cnt}")

    return errors


# ── Run all ────────────────────────────────────────────────────────────────────

def run_evaluation():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df_clean = pd.read_csv(CLEAN_CSV)
    with open(SPLITS_JSON) as f:
        splits = json.load(f)

    # 1. Results table
    metrics_df = load_all_metrics()
    if not metrics_df.empty:
        table = build_results_table(metrics_df)
        table.to_csv(RESULTS_DIR / "results_table.csv", index=False)
        print("\n=== Results Table ===")
        print(table.to_string(index=False))

    # 2. CMI breakdown - find all preds files
    preds_files = list(RESULTS_DIR.glob("*_preds.csv"))
    if preds_files:
        cmi_df = cmi_breakdown(preds_files, df_clean, splits)
        if not cmi_df.empty:
            cmi_df.to_csv(RESULTS_DIR / "cmi_breakdown.csv", index=False)
            plot_cmi_breakdown(cmi_df)

    # 3. Confusion matrix for the best model
    best_info_file = RESULTS_DIR / "best_model_info.json"
    best_preds_file = None
    if best_info_file.exists():
        with open(best_info_file) as f:
            best_info = json.load(f)
        model_name = best_info.get("model", "best_model")
        seed = best_info.get("seed", 42)
        safe = model_name.replace("/", "_")
        candidate = RESULTS_DIR / f"{safe}_seed{seed}_preds.csv"
        if candidate.exists():
            best_preds_file = candidate

    if best_preds_file is None and preds_files:
        best_preds_file = sorted(preds_files)[-1]

    if best_preds_file and best_preds_file.exists():
        bp_df = pd.read_csv(best_preds_file)
        true_labels = bp_df["true"].tolist()
        pred_labels = bp_df["pred"].tolist()
        model_label = best_info_file.exists() and best_info.get("model", best_preds_file.stem)
        plot_confusion_matrix(true_labels, pred_labels,
                              model_name=str(model_label) if model_label else best_preds_file.stem)
        # 4. Error analysis
        error_analysis(best_preds_file, df_clean)

    log.info("Evaluation complete.")


if __name__ == "__main__":
    run_evaluation()
