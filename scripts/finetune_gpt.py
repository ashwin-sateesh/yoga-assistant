#!/usr/bin/env python3
"""Fine-tune GPT-3.5 Turbo on yoga Q&A data via the OpenAI API.

Usage:
    python scripts/finetune_gpt.py --data ./data/yoga_prompts_completions.jsonl

This script:
    1. Uploads the JSONL training file to OpenAI
    2. Creates a fine-tuning job
    3. Polls until the job completes
    4. Prints the resulting fine-tuned model name
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.finetuning import (
    create_fine_tuning_job,
    get_job_status,
    get_model_name,
    upload_training_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune GPT-3.5 on yoga data")
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to the JSONL training file",
    )
    parser.add_argument(
        "--model", type=str, default="gpt-3.5-turbo-0125",
        help="Base model to fine-tune (default: gpt-3.5-turbo-0125)",
    )
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Number of training epochs (default: 10)",
    )
    parser.add_argument(
        "--lr-multiplier", type=float, default=0.1,
        help="Learning rate multiplier (default: 0.1)",
    )
    parser.add_argument(
        "--poll-interval", type=int, default=60,
        help="Seconds between status checks (default: 60)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = OpenAI()

    print(f"Uploading training file: {args.data}")
    file_id = upload_training_file(client, args.data)
    print(f"  File ID: {file_id}")

    print(f"Creating fine-tuning job (model={args.model}, epochs={args.epochs})...")
    job_id = create_fine_tuning_job(
        client,
        training_file_id=file_id,
        model=args.model,
        n_epochs=args.epochs,
        learning_rate_multiplier=args.lr_multiplier,
    )
    print(f"  Job ID: {job_id}")

    print("Polling for completion...")
    while True:
        status = get_job_status(client, job_id)
        if status == "completed":
            break
        print(f"  Status: {status} — checking again in {args.poll_interval}s...")
        time.sleep(args.poll_interval)

    model_name = get_model_name(client, job_id)
    print(f"\nFine-tuning complete!")
    print(f"  Model: {model_name}")
    print(f"\nUse this model name in your config or pass it to scripts/chat.py.")


if __name__ == "__main__":
    main()
