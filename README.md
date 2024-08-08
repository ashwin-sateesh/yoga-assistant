# Yoga Assistant

## Objective

The Yoga Assistant is designed to provide accurate and real-time responses to yoga-related queries, offering both text and image outputs. The assistant leverages fine-tuned GPT-3.5 for text generation and a distilled Stable Diffusion model for image generation, ensuring high performance and quick response times.

## Architecture Overview

The architecture of the Yoga Assistant is built to efficiently handle both text and image queries through a scalable pipeline:
![Yoga Assistant Architecture](https://github.com/ashwin-sateesh/yoga-assistant/blob/main/assets/Yoga%20Assistant%20Workflow.png)

1. **Input Query & Query Type Classification:**
   - User queries are classified into text or image categories.

2. **Text Query Processing:**
   - Text queries are processed through a queue-based fine-tuning system.
   - If a relevant link is found, the content is scraped and queued for response generation.
   - A fine-tuned GPT-3.5 model generates the final text response.

3. **Image Query Processing:**
   - Image queries utilize a fine-tuned Stable Diffusion model.
   - The original model, containing over 1 billion parameters, was distilled to a Latent Diffusion Model to reduce its size and inference time.
   - The distilled model generates and returns the image response.

## Model Training & Evaluation

1. **Stable Diffusion Fine-tuning:**
   - Fine-tuned on yoga posture images, the Stable Diffusion model was distilled to improve performance.
   - For training details, refer to the [official Hugging Face documentation](https://github.com/huggingface/diffusers/tree/main/examples/text_to_image).
   - CLIP scores were used to evaluate the text-to-image response quality.

2. **GPT-3.5 Fine-tuning:**
   - The GPT-3.5 Turbo model was fine-tuned using yoga-related datasets.
   - Evaluation metrics included perplexity scores and semantic similarity to ensure high-quality responses.
   

## Acknowledgments

This project leverages resources from the [Hugging Face diffusers repository](https://github.com/huggingface/diffusers) and [OpenAI Finetuning Documentation](https://platform.openai.com/docs/guides/fine-tuning). Special thanks to the developers and contributors to these tools.
