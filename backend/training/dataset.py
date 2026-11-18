"""
Dataset Utilities
=================
Helpers for loading and preprocessing the LIAR dataset.
"""

from datasets import load_dataset


FAKE_LIAR_IDS = {0, 1, 2, 3}  # pants-fire, false, barely-true, half-true
REAL_LIAR_IDS = {4, 5}         # mostly-true, true


def load_liar_binary(split: str = "train"):
    """
    Load a split of the LIAR dataset and binarise labels.

    Args:
        split: "train", "validation", or "test"

    Returns:
        HuggingFace Dataset with added 'binary_label' column:
        0 → FAKE, 1 → REAL
    """
    dataset = load_dataset("liar", split=split)

    def _binarise(example):
        example["binary_label"] = 0 if example["label"] in FAKE_LIAR_IDS else 1
        return example

    return dataset.map(_binarise)


def build_text(example) -> str:
    """Combine speaker + statement into a single input string."""
    speaker = example.get("speaker", "")
    statement = example.get("statement", "")
    return f"{speaker} [SEP] {statement}" if speaker else statement
