"""
use_pretrained.py - Adapt a pre-trained multilingual sentiment model to CMTet labels.

Uses cardiffnlp/twitter-xlm-roberta-base-sentiment (positive/neutral/negative)
and remaps to our NEG/NTL/POS labels. Evaluates on our test split and saves
results to results/ for use in Module 6.

Run:
  python src/use_pretrained.py
"""
import json
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score, accuracy_score, classification_report

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    CLEAN_CSV, SPLITS_JSON, MODELS_DIR, RESULTS_DIR, LABELS, LABEL2ID
)
from data import clean

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Pretrained model
MODEL_ID = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

# Label mapping: Cardiff labels -> our labels
CARDIFF_MAP = {
    "positive": "POS",
    "neutral":  "NTL",
    "negative": "NEG",
}


def run_pretrained_eval():
    df = pd.read_csv(CLEAN_CSV)
    with open(SPLITS_JSON) as f:
        splits = json.load(f)

    test_idx = splits["folds"][0]["test"]
    test_df  = df.iloc[test_idx].copy().reset_index(drop=True)

    log.info("Loading model: %s", MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Cardiff model id2label
    cardiff_id2label = model.config.id2label     # e.g. {0: "negative", 1: "neutral", 2: "positive"}

    log.info("Running inference on %d test samples ...", len(test_df))
    preds = []
    probs_all = []
    batch_size = 64

    for i in range(0, len(test_df), batch_size):
        batch_texts = test_df["clean_text"].iloc[i:i+batch_size].tolist()
        enc = tokenizer(
            batch_texts,
            truncation=True,
            max_length=64,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        batch_probs = F.softmax(logits, dim=-1).cpu().numpy()
        probs_all.append(batch_probs)

        for row_probs in batch_probs:
            cardiff_id = int(np.argmax(row_probs))
            cardiff_lbl = cardiff_id2label[cardiff_id].lower()
            preds.append(CARDIFF_MAP.get(cardiff_lbl, "NTL"))

        if (i // batch_size + 1) % 5 == 0:
            log.info("  Processed %d/%d", min(i + batch_size, len(test_df)), len(test_df))

    true_labels = test_df["label"].tolist()
    macro_f1 = f1_score(true_labels, preds, average="macro",
                        labels=LABELS, zero_division=0)
    acc = accuracy_score(true_labels, preds)
    report = classification_report(
        true_labels, preds, target_names=LABELS,
        output_dict=True, zero_division=0
    )
    per_class = {lbl: report[lbl]["f1-score"] for lbl in LABELS}

    metrics_out = {
        "model":        MODEL_ID,
        "seed":         0,
        "mode":         "zero_shot_pretrained",
        "macro_f1":     float(macro_f1),
        "accuracy":     float(acc),
        "per_class_f1": per_class,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "xlmr_sentiment_seed0_metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    preds_df = pd.DataFrame({
        "index": test_idx,
        "text":  test_df["text"].tolist(),
        "true":  true_labels,
        "pred":  preds,
    })
    preds_df.to_csv(RESULTS_DIR / "xlmr_sentiment_seed0_preds.csv", index=False)

    log.info("Saving best checkpoint to models/best/ ...")
    best_dir = MODELS_DIR / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))

    # Save label_map so predict.py knows the remapping
    label_map = {
        v.lower(): k for k, v in CARDIFF_MAP.items()
    }  # unused but saved for reference
    with open(best_dir / "cardiff_label_map.json", "w") as f:
        json.dump({"id2label": {k: v for k, v in enumerate(LABELS)},
                   "cardiff_id2label": {str(k): v for k, v in cardiff_id2label.items()},
                   "cardiff_map": CARDIFF_MAP}, f, indent=2)

    with open(RESULTS_DIR / "best_model_info.json", "w") as f:
        json.dump({**metrics_out, "checkpoint_dir": str(best_dir)}, f, indent=2)

    print(f"\n=== Zero-shot XLM-R Sentiment (cardiffnlp) ===")
    print(f"  Macro-F1 : {macro_f1:.4f}")
    print(f"  Accuracy : {acc:.4f}")
    for lbl, v in per_class.items():
        print(f"    {lbl}: {v:.4f}")

    return metrics_out


if __name__ == "__main__":
    run_pretrained_eval()
