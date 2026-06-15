#!/usr/bin/env python3
"""Evaluate Yoga Assistant response quality.

Usage:
    python scripts/evaluate.py \
        --llm-model ft:gpt-3.5-turbo-0125:personal::XXXXX \
        --queries "What are the benefits of Surya Namaskar?" "How to do Warrior pose?"

Computes perplexity scores for generated text responses.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain.chat_models import ChatOpenAI

from src.evaluation import batch_perplexity
from src.inference import generate_response_direct


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Yoga Assistant responses")
    parser.add_argument(
        "--llm-model", type=str, required=True,
        help="Fine-tuned GPT-3.5 model identifier",
    )
    parser.add_argument(
        "--queries", type=str, nargs="+", required=True,
        help="One or more test queries",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.9,
        help="LLM temperature (default: 0.9)",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device for perplexity evaluation (default: cpu)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    llm = ChatOpenAI(temperature=args.temperature, model=args.llm_model)

    print("Generating responses...")
    responses = []
    for query in args.queries:
        response = generate_response_direct(query, llm)
        responses.append(response)
        print(f"\n  Q: {query}")
        print(f"  A: {response[:200]}{'...' if len(response) > 200 else ''}")

    print("\nCalculating perplexity scores...")
    scores = batch_perplexity(responses, device=args.device)

    print("\nResults:")
    print(f"{'Query':<50} {'Perplexity':>10}")
    print("-" * 62)
    for query, score in zip(args.queries, scores):
        label = query[:47] + "..." if len(query) > 50 else query
        print(f"{label:<50} {score:>10.2f}")

    avg = sum(scores) / len(scores)
    print("-" * 62)
    print(f"{'Average':<50} {avg:>10.2f}")


if __name__ == "__main__":
    main()
