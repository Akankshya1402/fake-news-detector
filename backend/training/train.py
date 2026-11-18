"""
BERT Fine-Tuning Script
=======================
Fine-tunes DistilBERT on the LIAR dataset (12,836 labelled statements)
for binary fake-news classification (REAL vs FAKE).

Usage:
    python training/train.py

Environment variables:
    OUTPUT_DIR   — where to save the checkpoint (default: ./checkpoints/fake-news-bert)
    EPOCHS       — number of training epochs (default: 3)
    BATCH_SIZE   — per-device train batch size (default: 16)
    LR           — learning rate (default: 2e-5)
"""

import os
import logging
from pathlib import Path

import numpy as np
from datasets import load_dataset, ClassLabel
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, f1_score, classification_report
import torch

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_CHECKPOINT = "distilbert-base-uncased"
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./checkpoints/fake-news-bert")
EPOCHS = int(os.getenv("EPOCHS", 3))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 16))
LR = float(os.getenv("LR", 2e-5))
MAX_LENGTH = 512

# LIAR dataset label mapping:
# 0=pants-fire, 1=false, 2=barely-true  → FAKE (0)
# 3=half-true                            → FAKE (0)  [conservative threshold]
# 4=mostly-true, 5=true                 → REAL (1)
FAKE_LIAR_IDS = {0, 1, 2, 3}
REAL_LIAR_IDS = {4, 5}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_and_binarise_liar():
    """
    Downloads the LIAR dataset via HuggingFace datasets and converts the
    6-class fine-grained labels to binary FAKE (0) / REAL (1).
    """
    logger.info("Downloading LIAR dataset …")
    raw = load_dataset("liar")

    def binarise(example):
        label_id = example["label"]
        example["binary_label"] = 0 if label_id in FAKE_LIAR_IDS else 1
        return example

    dataset = raw.map(binarise)
    logger.info(f"Train samples: {len(dataset['train'])}")
    logger.info(f"Validation samples: {len(dataset['validation'])}")
    return dataset


def tokenize_dataset(dataset, tokenizer):
    """Tokenise the 'statement' column, combining with speaker for context."""

    def tokenize_fn(batch):
        texts = [
            f"{spk} [SEP] {stmt}"
            for spk, stmt in zip(batch.get("speaker", [""] * len(batch["statement"])), batch["statement"])
        ]
        return tokenizer(texts, truncation=True, max_length=MAX_LENGTH)

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset["train"].column_names)
    tokenized = tokenized.rename_column("binary_label", "labels")
    return tokenized


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    logger.info(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    logger.info(f"Output dir: {OUTPUT_DIR}")

    # 1. Load data
    dataset = load_and_binarise_liar()

    # 2. Tokenise
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_CHECKPOINT)
    tokenized = tokenize_dataset(dataset, tokenizer)
    tokenized.set_format("torch")

    # 3. Model
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=2,
        id2label={0: "FAKE", 1: "REAL"},
        label2id={"FAKE": 0, "REAL": 1},
    )

    # 4. Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        learning_rate=LR,
        warmup_steps=500,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        report_to="none",
    )

    # 5. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # 6. Train
    logger.info("Starting fine-tuning …")
    trainer.train()

    # 7. Evaluate on test set
    logger.info("Evaluating on test set …")
    test_results = trainer.evaluate(tokenized["test"])
    logger.info(f"Test results: {test_results}")

    # 8. Save final checkpoint + tokenizer
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    logger.info(f"Model saved to {OUTPUT_DIR}")

    # 9. Print classification report
    preds_output = trainer.predict(tokenized["test"])
    preds = np.argmax(preds_output.predictions, axis=-1)
    labels = preds_output.label_ids
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(labels, preds, target_names=["FAKE", "REAL"]))


if __name__ == "__main__":
    main()
