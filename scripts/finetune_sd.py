#!/usr/bin/env python3
"""Launch Stable Diffusion fine-tuning using HuggingFace Accelerate.

Usage:
    python scripts/finetune_sd.py \
        --dataset-dir ./data/yoga_img_dataset \
        --output-dir ./artifacts/yoga-stable-diffusion-v1-4

This is a wrapper around the HuggingFace diffusers text_to_image training
script. It requires the diffusers repo to be cloned locally:

    git clone https://github.com/huggingface/diffusers
    pip install -U -r diffusers/examples/text_to_image/requirements.txt
    accelerate config default --mixed_precision fp16
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs import StableDiffusionConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune Stable Diffusion on yoga images"
    )
    parser.add_argument(
        "--dataset-dir", type=str, required=True,
        help="Path to the HuggingFace ImageFolder dataset",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory to save the fine-tuned model",
    )
    parser.add_argument(
        "--diffusers-dir", type=str, default="./diffusers",
        help="Path to the cloned HuggingFace diffusers repo",
    )
    parser.add_argument(
        "--max-train-steps", type=int, default=None,
        help="Override max training steps (default: from config)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = StableDiffusionConfig()

    if args.max_train_steps:
        cfg.max_train_steps = args.max_train_steps

    train_script = Path(args.diffusers_dir) / "examples/text_to_image/train_text_to_image.py"
    if not train_script.exists():
        print(f"Error: Training script not found at {train_script}")
        print("Clone the diffusers repo first:")
        print("  git clone https://github.com/huggingface/diffusers")
        sys.exit(1)

    cmd = [
        "accelerate", "launch",
        f"--mixed_precision={cfg.mixed_precision}",
        str(train_script),
        f"--pretrained_model_name_or_path={cfg.pretrained_model}",
        f"--train_data_dir={args.dataset_dir}",
        f"--resolution={cfg.resolution}",
        f"--train_batch_size={cfg.train_batch_size}",
        f"--gradient_accumulation_steps={cfg.gradient_accumulation_steps}",
        "--gradient_checkpointing",
        f"--max_train_steps={cfg.max_train_steps}",
        f"--learning_rate={cfg.learning_rate}",
        f"--max_grad_norm={cfg.max_grad_norm}",
        f"--lr_scheduler={cfg.lr_scheduler}",
        f"--lr_warmup_steps={cfg.lr_warmup_steps}",
        f"--output_dir={args.output_dir}",
    ]

    if cfg.use_ema:
        cmd.append("--use_ema")
    if cfg.center_crop:
        cmd.append("--center_crop")
    if cfg.random_flip:
        cmd.append("--random_flip")

    print("Launching Stable Diffusion fine-tuning:")
    print(f"  Model: {cfg.pretrained_model}")
    print(f"  Dataset: {args.dataset_dir}")
    print(f"  Steps: {cfg.max_train_steps}")
    print(f"  Output: {args.output_dir}")
    print()

    subprocess.run(cmd, check=True)
    print("\nFine-tuning complete!")


if __name__ == "__main__":
    main()
