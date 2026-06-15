# Yoga Assistant: Multimodal Yoga Q&A with Fine-Tuned LLMs

A multimodal assistant that answers yoga-related queries with both text and image responses, powered by fine-tuned GPT-3.5 for text generation and a distilled Stable Diffusion model for yoga pose visualization.

![Architecture](/assets/yoga_assistant_architecture.png)

## Architecture Overview

The system operates through a three-stage pipeline:

**Stage 1 — Query Classification:** User input is analyzed to determine whether it requires a text response or an image response using keyword-based classification.

**Stage 2 — Text Processing (GPT-3.5 Pipeline):**
- If the query contains a URL, the linked page is scraped and the content is injected into the prompt as context (RAG-style retrieval augmentation).
- If no URL is present, the query is sent directly to the fine-tuned GPT-3.5 model.
- A queue-based fine-tuning system enables incremental model updates: scraped content is formatted into training data and queued for background fine-tuning, so the model continuously improves without blocking inference.

**Stage 3 — Image Generation (Stable Diffusion Pipeline):**
- Image queries are routed to a fine-tuned Stable Diffusion model (CompVis/stable-diffusion-v1-4).
- The original 1B+ parameter model was distilled to a smaller Latent Diffusion Model by pruning the UNet while preserving the Text Encoder and VAE, achieving 2x faster inference.

## Project Structure

```
yoga-assistant/
├── configs/
│   ├── __init__.py
│   └── config.py              # PathConfig, GPTFineTuneConfig, StableDiffusionConfig, ScrapingConfig
├── src/
│   ├── __init__.py
│   ├── preprocessing.py       # PDF extraction, text chunking, JSONL formatting
│   ├── scraping.py            # URL extraction and web scraping
│   ├── classifier.py          # Query type classification (text vs image)
│   ├── finetuning.py          # OpenAI fine-tuning API, queue-based model updates
│   ├── sd_data.py             # Stable Diffusion dataset preparation
│   ├── inference.py           # YogaAssistant class, text and image generation
│   └── evaluation.py          # Perplexity scoring
├── scripts/
│   ├── prepare_data.py        # PDF → JSONL data pipeline
│   ├── prepare_sd_data.py     # Image directory → HuggingFace ImageFolder dataset
│   ├── finetune_gpt.py        # GPT-3.5 fine-tuning via OpenAI API
│   ├── finetune_sd.py         # Stable Diffusion fine-tuning via Accelerate
│   ├── evaluate.py            # Response quality evaluation
│   └── chat.py                # Interactive chatbot
├── data/                      # Training data (not tracked in git)
├── artifacts/                 # Model checkpoints (not tracked in git)
├── assets/                    # Architecture diagrams
├── requirements.txt
├── .gitignore
└── README.md
```

## Models

| Component | Model | Parameters | Details |
|---|---|---|---|
| Text Generation | GPT-3.5 Turbo (fine-tuned) | — | Fine-tuned via OpenAI API, 10 epochs, LR multiplier 0.1 |
| Image Generation | Stable Diffusion v1.4 (fine-tuned) | 1.07B total | UNet: 860M, Text Encoder: 123M, VAE: 84M |
| Distilled Image Gen | Latent Diffusion (pruned UNet) | ~500M | 2x inference speedup via UNet distillation |
| Evaluation | GPT-2 | 124M | Perplexity scoring for text quality |
| Evaluation | CLIP | — | Text-to-image alignment scoring |

## Setup

```bash
git clone https://github.com/ashwin-sateesh/yoga-assistant.git
cd yoga-assistant

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Data Preparation

**For GPT-3.5 fine-tuning** — convert yoga PDFs to training data:

```bash
python scripts/prepare_data.py \
    --pdf-dir ./data/pdfs \
    --output ./data/yoga_prompts_completions.jsonl
```

**For Stable Diffusion fine-tuning** — prepare an ImageFolder dataset:

```bash
python scripts/prepare_sd_data.py \
    --image-dir ./data/yoga_poses \
    --output-dir ./data/yoga_img_dataset
```

The image directory should contain subdirectories named after poses, each with image files:

```
data/yoga_poses/
├── warrior_pose/
│   ├── img1.jpg
│   └── img2.jpg
├── tree_pose/
│   ├── img1.jpg
│   └── img2.jpg
└── ...
```

## Training

**Fine-tune GPT-3.5:**

```bash
python scripts/finetune_gpt.py \
    --data ./data/yoga_prompts_completions.jsonl \
    --model gpt-3.5-turbo-0125 \
    --epochs 10
```

**Fine-tune Stable Diffusion:**

```bash
# Clone the diffusers repo (one-time setup)
git clone https://github.com/huggingface/diffusers
pip install -U -r diffusers/examples/text_to_image/requirements.txt
accelerate config default --mixed_precision fp16

# Launch training
python scripts/finetune_sd.py \
    --dataset-dir ./data/yoga_img_dataset \
    --output-dir ./artifacts/yoga-stable-diffusion-v1-4
```

## Evaluation

```bash
python scripts/evaluate.py \
    --llm-model ft:gpt-3.5-turbo-0125:personal::XXXXX \
    --queries "What are the benefits of Surya Namaskar?" "How to do Warrior pose?"
```

## Interactive Chat

```bash
python scripts/chat.py \
    --llm-model ft:gpt-3.5-turbo-0125:personal::XXXXX \
    --sd-model-path ./artifacts/yoga-stable-diffusion-v1-4 \
    --sd-unet-path ./artifacts/yoga-stable-diffusion-v1-4/checkpoint-2500/unet
```

Example interaction:

```
Yoga Assistant
----------------------------------------
Hello! I'm your Yoga Assistant.
I can answer yoga questions and generate yoga pose images.
Type 'quit' to exit.

You: What are the benefits of practicing yoga daily?
Yoga Assistant: Practicing yoga daily offers numerous benefits including improved
flexibility, strength, and balance. Regular practice also reduces stress...

You: Show me the warrior pose
Yoga Assistant: Image saved to ./outputs/images/generated_image.png

You: quit
Yoga Assistant: Goodbye! Namaste.
```

## Programmatic Usage

```python
from src.inference import YogaAssistant

assistant = YogaAssistant(
    llm_model="ft:gpt-3.5-turbo-0125:personal::XXXXX",
    sd_model_path="./artifacts/yoga-stable-diffusion-v1-4",
    sd_unet_path="./artifacts/yoga-stable-diffusion-v1-4/checkpoint-2500/unet",
)

# Text response
answer = assistant.respond("What are the benefits of Surya Namaskar?")

# Image response (returns path to saved image)
image_path = assistant.respond("Show me the tree pose")

# RAG-style response with URL context
answer = assistant.respond(
    "Summarize the yoga benefits from https://example.com/yoga-guide"
)
```

## Key Design Decisions

- **Queue-based incremental fine-tuning**: When users provide URLs, the scraped content is not only used for immediate RAG-style responses but also queued as fine-tuning data. The model improves continuously in the background without blocking inference.
- **UNet distillation for Stable Diffusion**: Rather than serving the full 1B parameter model, the UNet was pruned and distilled while keeping the Text Encoder and VAE intact, cutting inference time in half with minimal quality loss.
- **Keyword-based query classification**: A lightweight regex approach routes queries to text or image pipelines without requiring an additional classification model, keeping latency low.
- **CLIP evaluation for image quality**: Text-to-image alignment is measured via CLIP scores rather than pixel-level metrics, capturing semantic relevance over visual fidelity.

## References

- Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models" ([arXiv:2112.10752](https://arxiv.org/abs/2112.10752))
- [HuggingFace Diffusers Text-to-Image Training](https://github.com/huggingface/diffusers/tree/main/examples/text_to_image)
- [OpenAI Fine-Tuning Documentation](https://platform.openai.com/docs/guides/fine-tuning)

