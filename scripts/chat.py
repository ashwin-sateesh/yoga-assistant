#!/usr/bin/env python3
"""Run the Yoga Assistant interactive chatbot.

Usage:
    python scripts/chat.py \
        --llm-model ft:gpt-3.5-turbo-0125:personal::XXXXX \
        --sd-model-path ./artifacts/yoga-stable-diffusion-v1-4

Requires:
    - OPENAI_API_KEY environment variable set
    - Fine-tuned GPT-3.5 model name (from finetune_gpt.py output)
    - Fine-tuned Stable Diffusion checkpoint on disk
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference import YogaAssistant


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Yoga Assistant interactive chat")
    parser.add_argument(
        "--llm-model", type=str, required=True,
        help="Fine-tuned GPT-3.5 model identifier (e.g. ft:gpt-3.5-turbo-0125:...)",
    )
    parser.add_argument(
        "--sd-model-path", type=str, required=True,
        help="Path to the fine-tuned Stable Diffusion checkpoint",
    )
    parser.add_argument(
        "--sd-unet-path", type=str, default=None,
        help="Optional path to a specific UNet checkpoint",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.9,
        help="LLM sampling temperature (default: 0.9)",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device for Stable Diffusion: 'cuda', 'cpu', or 'auto'",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Loading Yoga Assistant...")
    assistant = YogaAssistant(
        llm_model=args.llm_model,
        sd_model_path=args.sd_model_path,
        sd_unet_path=args.sd_unet_path,
        temperature=args.temperature,
        device=args.device,
    )

    print("\nYoga Assistant")
    print("-" * 40)
    print("Hello! I'm your Yoga Assistant.")
    print("I can answer yoga questions and generate yoga pose images.")
    print("Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Yoga Assistant: Goodbye! Namaste.")
            break

        response = assistant.respond(user_input)

        if response.endswith((".png", ".jpg", ".jpeg")):
            print(f"Yoga Assistant: Image saved to {response}")
            try:
                img = Image.open(response)
                img.show()
            except Exception:
                pass
        else:
            print(f"Yoga Assistant: {response}\n")


if __name__ == "__main__":
    main()
