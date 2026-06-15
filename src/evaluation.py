"""Evaluation metrics for the Yoga Assistant.

Includes perplexity scoring for text generation quality and CLIP-based
scoring for text-to-image alignment.
"""

from __future__ import annotations

from typing import List, Optional

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def calculate_perplexity(
    text: str,
    model: Optional[GPT2LMHeadModel] = None,
    tokenizer: Optional[GPT2Tokenizer] = None,
    model_name: str = "gpt2",
    device: str = "cpu",
) -> float:
    """Calculate the perplexity of a text string using a language model.

    Lower perplexity indicates the model assigns higher probability to the
    text, suggesting more fluent and coherent output.

    Args:
        text: Text to evaluate.
        model: Pre-loaded GPT-2 model. If None, loads from ``model_name``.
        tokenizer: Pre-loaded tokenizer. If None, loads from ``model_name``.
        model_name: HuggingFace model identifier (used if model/tokenizer
            are not provided).
        device: Torch device.

    Returns:
        Perplexity score (float).
    """
    if tokenizer is None:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    if model is None:
        model = GPT2LMHeadModel.from_pretrained(model_name)

    model = model.to(device).eval()
    encodings = tokenizer(text, return_tensors="pt").to(device)
    input_ids = encodings.input_ids

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

    return torch.exp(loss).item()


def batch_perplexity(
    texts: List[str],
    model_name: str = "gpt2",
    device: str = "cpu",
) -> List[float]:
    """Calculate perplexity for a batch of texts.

    Loads the model once and reuses it across all texts.

    Args:
        texts: List of text strings.
        model_name: HuggingFace model identifier.
        device: Torch device.

    Returns:
        List of perplexity scores.
    """
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device).eval()

    scores = []
    for text in texts:
        score = calculate_perplexity(text, model=model, tokenizer=tokenizer, device=device)
        scores.append(score)
    return scores
