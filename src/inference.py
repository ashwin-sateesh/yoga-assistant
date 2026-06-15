"""Inference pipelines for the Yoga Assistant.

Provides text generation (via fine-tuned GPT-3.5) and image generation
(via fine-tuned Stable Diffusion), plus the unified ``respond()`` entry
point that routes queries to the appropriate pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from .classifier import classify_query
from .scraping import extract_url_and_text, scrape_content


# ---------------------------------------------------------------------------
# Text generation
# ---------------------------------------------------------------------------

def generate_response_with_context(
    query: str,
    scraped_content: str,
    llm: ChatOpenAI,
) -> str:
    """Generate a response using scraped web content as context.

    Args:
        query: The user's question (URL stripped).
        scraped_content: Text scraped from the provided URL.
        llm: Configured LangChain ChatOpenAI instance.

    Returns:
        Generated text response.
    """
    prompt = ChatPromptTemplate.from_template(
        "Using the content below, give an appropriate response to the "
        "following query:\n\n"
        "Content:\n{content}\n\n"
        "Query:\n{query}\n\n"
        "Response:"
    )
    chain = LLMChain(llm=llm, prompt=prompt, verbose=False)
    return chain.run(content=scraped_content, query=query)


def generate_response_direct(
    query: str,
    llm: ChatOpenAI,
) -> str:
    """Generate a response directly from the fine-tuned model.

    Args:
        query: The user's yoga-related question.
        llm: Configured LangChain ChatOpenAI instance.

    Returns:
        Generated text response.
    """
    prompt = ChatPromptTemplate.from_template(
        "Give an appropriate yoga related response to the following query:\n\n"
        "Query:\n{query}\n\n"
        "Response:"
    )
    chain = LLMChain(llm=llm, prompt=prompt, verbose=False)
    return chain.run(query)


# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------

def generate_image(
    query: str,
    model_path: str | Path,
    unet_path: Optional[str | Path] = None,
    output_path: Optional[str | Path] = None,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.float16,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
) -> Path:
    """Generate an image using the fine-tuned Stable Diffusion model.

    Args:
        query: Text prompt describing the desired image.
        model_path: Path to the fine-tuned Stable Diffusion checkpoint.
        unet_path: Optional path to a specific UNet checkpoint. If None,
            uses the UNet from ``model_path``.
        output_path: Where to save the generated image. Defaults to
            ``./outputs/images/generated_image.png``.
        device: Torch device string.
        torch_dtype: Data type for inference (float16 recommended for GPU).
        num_inference_steps: Number of denoising steps.
        guidance_scale: Classifier-free guidance scale.

    Returns:
        Path to the saved image.
    """
    model_path = Path(model_path)

    if unet_path:
        unet = UNet2DConditionModel.from_pretrained(
            str(unet_path), torch_dtype=torch_dtype
        )
        pipe = StableDiffusionPipeline.from_pretrained(
            str(model_path), unet=unet, torch_dtype=torch_dtype
        )
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            str(model_path), torch_dtype=torch_dtype
        )

    pipe = pipe.to(device)
    image = pipe(
        prompt=query,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    ).images[0]

    if output_path is None:
        output_path = Path("./outputs/images/generated_image.png")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)

    return output_path


# ---------------------------------------------------------------------------
# Unified response router
# ---------------------------------------------------------------------------

class YogaAssistant:
    """End-to-end inference pipeline for the Yoga Assistant.

    Routes queries to either the text (GPT-3.5) or image (Stable Diffusion)
    pipeline based on keyword classification.

    Args:
        llm_model: Fine-tuned GPT-3.5 model identifier.
        sd_model_path: Path to fine-tuned Stable Diffusion checkpoint.
        sd_unet_path: Optional path to a specific UNet checkpoint.
        temperature: LLM sampling temperature.
        device: Torch device for Stable Diffusion inference.
    """

    def __init__(
        self,
        llm_model: str,
        sd_model_path: str | Path,
        sd_unet_path: Optional[str | Path] = None,
        temperature: float = 0.9,
        device: str = "auto",
    ) -> None:
        self.llm = ChatOpenAI(temperature=temperature, model=llm_model)
        self.sd_model_path = Path(sd_model_path)
        self.sd_unet_path = Path(sd_unet_path) if sd_unet_path else None
        self.device = (
            "cuda" if device == "auto" and torch.cuda.is_available() else device
        )

    def respond(self, input_query: str) -> str:
        """Process a user query and return a text or image response.

        Args:
            input_query: Raw user input (may contain a URL).

        Returns:
            Generated text response, or the path to a generated image.
        """
        url, text = extract_url_and_text(input_query)
        query_type = classify_query(input_query)

        if query_type == "image":
            image_path = generate_image(
                query=input_query,
                model_path=self.sd_model_path,
                unet_path=self.sd_unet_path,
                device=self.device,
            )
            return str(image_path)

        if url is not None:
            scraped = scrape_content(url)
            return generate_response_with_context(text, scraped, self.llm)

        return generate_response_direct(input_query, self.llm)
