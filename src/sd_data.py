"""Stable Diffusion dataset preparation.

Creates text-image pairs and metadata files for fine-tuning
Stable Diffusion using the HuggingFace diffusers training scripts.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


def create_pose_prompts(pose_name: str) -> List[str]:
    """Generate a set of visualization-focused text prompts for a yoga pose.

    Args:
        pose_name: Human-readable name of the yoga pose.

    Returns:
        List of 10 descriptive prompt strings.
    """
    return [
        f"Describe the {pose_name} pose.",
        f"How does the {pose_name} pose look?",
        f"Imagine the {pose_name} pose. What does it look like?",
        f"Visualize the {pose_name} pose and describe its details.",
        f"Picture someone in the {pose_name} pose. How do they look?",
        f"What is the {pose_name} pose?",
        f"Describe the key features of the {pose_name} pose.",
        f"How would you visualize the {pose_name} pose?",
        f"Think about the {pose_name} pose. What do you see?",
        f"Provide a visual description of the {pose_name} pose.",
    ]


def build_text_image_pairs(
    image_dir: str | Path,
) -> List[Tuple[str, str]]:
    """Build (prompt, image_path) pairs from a directory of yoga pose images.

    Expects ``image_dir`` to contain subdirectories named after poses,
    each containing image files.

    Args:
        image_dir: Root directory with pose subdirectories.

    Returns:
        List of (prompt_text, image_path) tuples.
    """
    image_dir = Path(image_dir)
    pairs: List[Tuple[str, str]] = []

    for pose_dir in sorted(image_dir.iterdir()):
        if not pose_dir.is_dir():
            continue
        prompts = create_pose_prompts(pose_dir.name)
        images = sorted(
            f for f in pose_dir.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )
        for i, img_path in enumerate(images):
            prompt = prompts[i % len(prompts)]
            pairs.append((prompt, str(img_path)))

    return pairs


def create_hf_image_dataset(
    pairs: List[Tuple[str, str]],
    output_dir: str | Path,
) -> Path:
    """Create a HuggingFace ImageFolder dataset with metadata.

    Copies images into a flat directory and writes a ``metadata.jsonl``
    file mapping filenames to text prompts, following the format
    documented at https://huggingface.co/docs/datasets/image_load.

    Args:
        pairs: List of (prompt_text, source_image_path) tuples.
        output_dir: Destination directory for the dataset.

    Returns:
        Path to the created metadata.jsonl file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = []
    for prompt, img_path in pairs:
        img_path = Path(img_path)
        pose_name = img_path.parent.name
        new_name = f"{pose_name}_{img_path.name}"
        shutil.copy(img_path, output_dir / new_name)
        metadata.append({"file_name": new_name, "text": prompt})

    metadata_path = output_dir / "metadata.jsonl"
    with open(metadata_path, "w", encoding="utf-8") as f:
        for entry in metadata:
            f.write(json.dumps(entry) + "\n")

    return metadata_path
