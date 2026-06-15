#!/usr/bin/env python3
"""Prepare a HuggingFace ImageFolder dataset for Stable Diffusion fine-tuning.

Usage:
    python scripts/prepare_sd_data.py \
        --image-dir ./data/yoga_poses \
        --output-dir ./data/yoga_img_dataset

This script:
    1. Scans pose subdirectories for images
    2. Generates text prompts for each image
    3. Creates a flat ImageFolder dataset with metadata.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sd_data import build_text_image_pairs, create_hf_image_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Stable Diffusion training dataset"
    )
    parser.add_argument(
        "--image-dir", type=str, required=True,
        help="Root directory with pose subdirectories containing images",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./data/yoga_img_dataset",
        help="Output directory for the HuggingFace ImageFolder dataset",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Scanning image directory: {args.image_dir}")
    pairs = build_text_image_pairs(args.image_dir)
    print(f"  Found {len(pairs)} text-image pairs")

    print(f"Creating HuggingFace dataset at: {args.output_dir}")
    metadata_path = create_hf_image_dataset(pairs, args.output_dir)
    print(f"  Wrote metadata to {metadata_path}")
    print("Done. Use this dataset with the HuggingFace diffusers training script.")


if __name__ == "__main__":
    main()
