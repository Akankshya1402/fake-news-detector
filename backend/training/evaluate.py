"""
Evaluation Script
=================
Runs the fine-tuned model checkpoint against the LIAR test set
and prints a full classification report + confusion matrix.

Usage:
    python training/evaluate.py --model_path ./checkpoints/fake-news-bert
"""

import argparse
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FAKE_LIAR_IDS = {0, 1, 2, 3}
MAX_LENGTH = 512


def binarise(example):
    example["binary_label"] = 0 if example["label"] in FAKE_LIAR_IDS else 1
    return example


@torch.no_grad()
def evaluate(model_path: str, batch_size: int = 32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()

    # Load test data
    raw = load_dataset("liar", split="test")
    raw = raw.map(binarise)
    texts = [f"{row['speaker']} [SEP] {row['statement']}" for row in raw]
    true_labels = [row["binary_label"] for row in raw]

    # Batch inference
    all_preds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(batch, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits
        preds = torch.argmax(logits, dim=-1).cpu().tolist()
        all_preds.extend(preds)
        print(f"  Processed {min(i + batch_size, len(texts))}/{len(texts)}", end="\r")

    print()
    # Metrics
    acc = accuracy_score(true_labels, all_preds)
    f1 = f1_score(true_labels, all_preds, average="weighted")
    print(f"\n{'='*60}")
    print(f"Accuracy : {acc:.4f} ({acc*100:.2f}%)")
    print(f"F1 Score : {f1:.4f}")
    print(f"{'='*60}")
    print(classification_report(true_labels, all_preds, target_names=["FAKE", "REAL"]))

    # Confusion matrix
    cm = confusion_matrix(true_labels, all_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["FAKE", "REAL"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix — Fake News BERT")
    out_path = Path(model_path) / "confusion_matrix.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nConfusion matrix saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="./checkpoints/fake-news-bert")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    evaluate(args.model_path, args.batch_size)
