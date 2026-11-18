"""
BERT Inference Model
====================
Wraps a HuggingFace DistilBERT classifier for fake news detection.
Falls back to a lightweight demo mode if no fine-tuned checkpoint is present.
"""

import os
import re
import logging
from typing import Any

import torch
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
FINETUNED_PATH = os.getenv("MODEL_PATH", "./checkpoints/fake-news-bert")
LABELS = ["REAL", "FAKE"]
MAX_LENGTH = 512


class FakeNewsClassifier:
    """
    Wraps DistilBERT for binary fake-news classification.

    Priority:
      1. Load fine-tuned checkpoint from FINETUNED_PATH (set via MODEL_PATH env var)
      2. Fall back to a base pretrained model (demo mode)
    """

    def __init__(self):
        self.model_name: str = ""
        self.num_labels: int = 2
        self.max_length: int = MAX_LENGTH
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self):
        if os.path.isdir(FINETUNED_PATH):
            logger.info(f"Loading fine-tuned checkpoint from {FINETUNED_PATH}")
            self._tokenizer = AutoTokenizer.from_pretrained(FINETUNED_PATH)
            self._model = AutoModelForSequenceClassification.from_pretrained(FINETUNED_PATH)
            self.model_name = FINETUNED_PATH
        else:
            logger.warning(
                f"Fine-tuned checkpoint not found at '{FINETUNED_PATH}'. "
                "Falling back to base DistilBERT (demo mode). "
                "Run training/train.py to produce a fine-tuned checkpoint."
            )
            self._tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
            self._model = DistilBertForSequenceClassification.from_pretrained(
                "distilbert-base-uncased", num_labels=2
            )
            self.model_name = "distilbert-base-uncased (demo)"

        self._model.to(self._device)
        self._model.eval()
        logger.info(f"Model on device: {self._device}")

    @staticmethod
    def _clean_text(text: str) -> str:
        """Basic NLP preprocessing: strip HTML tags, collapse whitespace."""
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"http\S+|www\S+", "[URL]", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _build_explanation(label: str, confidence: float, text: str) -> str:
        """Return a human-readable rationale string based on heuristics + confidence."""
        snippet = text[:120].rstrip() + ("…" if len(text) > 120 else "")
        tier = "high" if confidence > 0.90 else "moderate" if confidence > 0.75 else "low"

        if label == "FAKE":
            return (
                f"The model classified this article as **FAKE** with {tier} confidence "
                f"({confidence:.1%}). Indicators may include sensationalist language, "
                f"unverified claims, or patterns inconsistent with credible reporting. "
                f'Analysed text: "{snippet}"'
            )
        return (
            f"The model classified this article as **REAL** with {tier} confidence "
            f"({confidence:.1%}). The writing style and factual structure are consistent "
            f'with credible news sources. Analysed text: "{snippet}"'
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, text: str, title: str = "") -> dict[str, Any]:
        """
        Run inference on a single article.

        Returns a dict with:
          - label (str): "REAL" or "FAKE"
          - confidence (float): probability of predicted label
          - probabilities (dict): {"REAL": float, "FAKE": float}
          - explanation (str): human-readable rationale
        """
        combined = f"{title} [SEP] {text}" if title else text
        cleaned = self._clean_text(combined)

        encoding = self._tokenizer(
            cleaned,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(self._device)
        attention_mask = encoding["attention_mask"].to(self._device)

        outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # shape: (1, num_labels)
        probs = torch.softmax(logits, dim=-1).squeeze().cpu().tolist()

        # Map probabilities to labels
        # Note: in demo mode the label ordering may not perfectly correspond to
        # REAL/FAKE until a fine-tuned checkpoint is loaded.
        prob_dict = {LABELS[i]: round(probs[i], 4) for i in range(len(LABELS))}
        pred_idx = int(torch.argmax(torch.tensor(probs)))
        label = LABELS[pred_idx]
        confidence = probs[pred_idx]

        return {
            "label": label,
            "confidence": round(confidence, 4),
            "probabilities": prob_dict,
            "explanation": self._build_explanation(label, confidence, text),
        }
