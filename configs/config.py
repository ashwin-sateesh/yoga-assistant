"""Centralized configuration for the Yoga Assistant project."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class PathConfig:
    """File and directory paths. Override via CLI or environment variables."""

    data_dir: Path = Path("./data")
    artifacts_dir: Path = Path("./artifacts")
    output_images_dir: Path = Path("./outputs/images")

    # Stable Diffusion artifacts
    sd_model_dir: str = "yoga-stable-diffusion-v1-4"
    sd_checkpoint_subdir: str = "checkpoint-2500/unet"

    # Fine-tuning data
    tuning_data_file: str = "yoga_prompts_completions.jsonl"
    raw_data_subdir: str = "pdfs"
    image_dataset_subdir: str = "yoga_img_dataset"

    @property
    def sd_model_path(self) -> Path:
        return self.artifacts_dir / self.sd_model_dir

    @property
    def sd_unet_path(self) -> Path:
        return self.sd_model_path / self.sd_checkpoint_subdir

    @property
    def tuning_data_path(self) -> Path:
        return self.data_dir / self.tuning_data_file

    @property
    def raw_pdf_dir(self) -> Path:
        return self.data_dir / self.raw_data_subdir

    @property
    def image_dataset_path(self) -> Path:
        return self.data_dir / self.image_dataset_subdir


@dataclass
class GPTFineTuneConfig:
    """Hyperparameters for GPT-3.5 fine-tuning via the OpenAI API."""

    base_model: str = "gpt-3.5-turbo-0125"
    n_epochs: int = 10
    learning_rate_multiplier: float = 0.1
    temperature: float = 0.9
    system_prompt: str = (
        "You are a knowledgeable yoga assistant. "
        "Answer questions based on the provided yoga related text."
    )
    chunk_size: int = 500
    num_chunks_per_doc: int = 10
    update_epochs: int = 2


@dataclass
class StableDiffusionConfig:
    """Hyperparameters for Stable Diffusion fine-tuning and inference."""

    pretrained_model: str = "CompVis/stable-diffusion-v1-4"
    resolution: int = 512
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_train_steps: int = 3000
    learning_rate: float = 1e-5
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 0
    max_grad_norm: float = 1.0
    mixed_precision: str = "fp16"
    use_ema: bool = True
    center_crop: bool = True
    random_flip: bool = True
    # Inference
    torch_dtype: str = "float16"
    num_inference_steps: int = 50
    guidance_scale: float = 7.5


@dataclass
class ScrapingConfig:
    """Settings for web scraping and URL extraction."""

    request_timeout: int = 10
    user_agent: str = (
        "Mozilla/5.0 (compatible; YogaAssistant/1.0; "
        "+https://github.com/ashwin-sateesh/yoga-assistant)"
    )

    visualization_keywords: List[str] = field(default_factory=lambda: [
        "show", "display", "visualize", "image", "picture",
        "illustrate", "depict", "render", "sketch", "draw",
        "demonstrate", "exhibit", "present", "graph", "diagram",
        "chart", "photograph", "snapshot", "view", "portrait",
        "photo", "visual", "figure", "scene", "design",
    ])
