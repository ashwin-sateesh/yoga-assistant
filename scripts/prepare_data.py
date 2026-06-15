#!/usr/bin/env python3
"""Prepare fine-tuning data from raw PDF documents.

Usage:
    python scripts/prepare_data.py --pdf-dir ./data/pdfs --output ./data/yoga_prompts_completions.jsonl

This script:
    1. Extracts text from all PDFs in the source directory
    2. Chunks the text and generates question-completion pairs via GPT-3.5
    3. Formats the pairs into OpenAI chat fine-tuning JSONL format
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing import (
    extract_texts_from_directory,
    format_pairs_to_messages,
    generate_pairs_from_texts,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare GPT-3.5 fine-tuning data")
    parser.add_argument(
        "--pdf-dir", type=str, required=True,
        help="Directory containing source PDF files",
    )
    parser.add_argument(
        "--output", type=str, default="./data/yoga_prompts_completions.jsonl",
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=500,
        help="Characters per text chunk (default: 500)",
    )
    parser.add_argument(
        "--delay", type=float, default=1.0,
        help="Seconds between API calls to avoid rate limiting (default: 1.0)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Extracting text from PDFs in: {args.pdf_dir}")
    texts = extract_texts_from_directory(args.pdf_dir)
    print(f"  Extracted text from {len(texts)} PDF files")

    print(f"Generating prompt-completion pairs (chunk_size={args.chunk_size})...")
    pairs = generate_pairs_from_texts(
        texts, chunk_size=args.chunk_size, delay=args.delay
    )
    print(f"  Generated {len(pairs)} pairs")

    print("Formatting for OpenAI fine-tuning...")
    records = format_pairs_to_messages(pairs)

    output_path = write_jsonl(records, args.output)
    print(f"  Wrote {len(records)} records to {output_path}")
    print("Done.")


if __name__ == "__main__":
    main()
