"""
train_transformer.py - Fine-tune XLM-RoBERTa, MuRIL, or any HF model for code-mixed sentiment.

Usage:
  python src/train_transformer.py --model_name distilbert-base-multilingual-cased --seed 42
  python src/train_transformer.py --model_name xlm-roberta-base --seed 42
  python src/train_transformer.py --model_name google/muril-base-cased --seed 42

Quick CPU demo options:
  --epochs 1             Use 1 training epoch
  --subsample 0.2        Train on only 20% of the training data
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
    set_seed,
)
from sklearn.metrics import f1_score, accuracy_score, classification_report

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    CLEAN_CSV, SPLITS_JSON, MODELS_DIR, RESULTS_DIR,
    LABELS, LABEL2ID, ID2LABEL,
    MAX_LENGTH, LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS,
    WEIGHT_DECAY, WARMUP_RATIO, FP16,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ── Custom Trainer with weighted CE ───────────────────────────────────────────

class WeightedTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device)
        )
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    acc = accuracy_score(labels, preds)
    return {"macro_f1": macro_f1, "accuracy": acc}


# ── Main ───────────────────────────────────────────────────────────────────────

def train(model_name: str, seed: int, epochs: int = NUM_EPOCHS,
          subsample: float = 1.0, batch_size: int = BATCH_SIZE):
    set_seed(seed)

    df = pd.read_csv(CLEAN_CSV)
    with open(SPLITS_JSON) as f:
        splits = json.load(f)

    fold_data = splits["folds"][0]
    train_idx = fold_data["train"]
    dev_idx   = fold_data["dev"]
    test_idx  = fold_data["test"]

    df["label_id"] = df["label"].map(LABEL2ID)

    train_df = df.iloc[train_idx][["clean_text", "label_id"]].rename(
        columns={"clean_text": "text", "label_id": "label"}
    ).reset_index(drop=True)

    # Subsample training data for quick CPU runs
    if subsample < 1.0:
        from sklearn.model_selection import train_test_split as tts
        n_sub = max(100, int(len(train_df) * subsample))
        _, train_df = tts(
            train_df, test_size=n_sub,
            stratify=train_df["label"], random_state=seed
        )
        train_df = train_df.reset_index(drop=True)
        log.info("Subsampled training set: %d rows (%.0f%%)", len(train_df), 100 * subsample)

    dev_df = df.iloc[dev_idx][["clean_text", "label_id"]].rename(
        columns={"clean_text": "text", "label_id": "label"}
    ).reset_index(drop=True)
    test_df = df.iloc[test_idx][["clean_text", "label_id"]].rename(
        columns={"clean_text": "text", "label_id": "label"}
    ).reset_index(drop=True)

    log.info("Split sizes: train=%d  dev=%d  test=%d",
             len(train_df), len(dev_df), len(test_df))

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            max_length=MAX_LENGTH,
            truncation=True,
            padding=False,
        )

    train_ds = Dataset.from_pandas(train_df).map(tokenize, batched=True,
                                                 remove_columns=["text"])
    dev_ds   = Dataset.from_pandas(dev_df).map(tokenize, batched=True,
                                               remove_columns=["text"])
    test_ds  = Dataset.from_pandas(test_df).map(tokenize, batched=True,
                                                remove_columns=["text"])

    # Class weights (inverse frequency, balanced)
    label_counts = train_df["label"].value_counts().sort_index()
    weights_list = []
    for i in range(len(LABELS)):
        cnt = label_counts.get(i, 1)
        weights_list.append(1.0 / max(cnt, 1))
    weights = torch.tensor(weights_list, dtype=torch.float)
    weights = weights / weights.sum() * len(LABELS)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(LABELS),
        id2label={i: l for i, l in enumerate(LABELS)},
        label2id=LABEL2ID,
    )

    safe_model_name = model_name.replace("/", "_")
    output_dir = MODELS_DIR / f"{safe_model_name}_seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    use_fp16 = FP16 and torch.cuda.is_available()

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=use_fp16,
        seed=seed,
        logging_steps=50,
        report_to="none",
        no_cuda=not torch.cuda.is_available(),
    )

    data_collator = DataCollatorWithPadding(tokenizer)

    trainer = WeightedTrainer(
        class_weights=weights,
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    log.info("Training %s (seed=%d, epochs=%d, subsample=%.0f%%) ...",
             model_name, seed, epochs, 100 * subsample)
    trainer.train()

    # Evaluate on test set
    test_preds_out = trainer.predict(test_ds)
    preds = np.argmax(test_preds_out.predictions, axis=-1)
    test_labels = test_df["label"].tolist()

    macro_f1 = f1_score(test_labels, preds, average="macro", zero_division=0)
    acc = accuracy_score(test_labels, preds)
    report = classification_report(
        test_labels, preds,
        target_names=LABELS, output_dict=True, zero_division=0
    )
    per_class = {lbl: report[lbl]["f1-score"] for lbl in LABELS}

    metrics_out = {
        "model":      model_name,
        "seed":       seed,
        "epochs":     epochs,
        "subsample":  subsample,
        "macro_f1":   float(macro_f1),
        "accuracy":   float(acc),
        "per_class_f1": per_class,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = model_name.replace("/", "_")
    metrics_file = RESULTS_DIR / f"{safe_name}_seed{seed}_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics_out, f, indent=2)

    preds_df = pd.DataFrame({
        "index": test_idx,
        "text":  df.iloc[test_idx]["text"].tolist(),
        "true":  [LABELS[l] for l in test_labels],
        "pred":  [LABELS[int(p)] for p in preds],
    })
    preds_df.to_csv(RESULTS_DIR / f"{safe_name}_seed{seed}_preds.csv", index=False)

    log.info("Done: macro_F1=%.4f  acc=%.4f  -> %s", macro_f1, acc, metrics_file)

    # Track best checkpoint
    best_metric_file = RESULTS_DIR / "best_model_info.json"
    is_best = False
    if best_metric_file.exists():
        with open(best_metric_file) as fh:
            best_info = json.load(fh)
        if macro_f1 > best_info.get("macro_f1", 0):
            is_best = True
    else:
        is_best = True

    if is_best:
        import shutil
        best_dir = MODELS_DIR / "best"
        best_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(best_dir))
        tokenizer.save_pretrained(str(best_dir))
        with open(best_metric_file, "w") as fh:
            json.dump({**metrics_out, "checkpoint_dir": str(best_dir)}, fh, indent=2)
        log.info("New best model saved to %s", best_dir)

    print(f"\n=== {model_name} (seed={seed}) ===")
    print(f"  Macro-F1 : {macro_f1:.4f}")
    print(f"  Accuracy : {acc:.4f}")
    for lbl, f1v in per_class.items():
        print(f"    {lbl}: {f1v:.4f}")

    return metrics_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS,
                        help="Override number of training epochs")
    parser.add_argument("--subsample", type=float, default=1.0,
                        help="Fraction of training data to use (0.0-1.0)")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    train(args.model_name, args.seed, args.epochs, args.subsample, args.batch_size)
