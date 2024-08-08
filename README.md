# Yoga Assistant

## Objective

The Yoga Assistant is a multimodal tool designed to provide accurate and real-time responses to yoga-related queries, offering both text and image outputs. The assistant leverages fine-tuned GPT-3.5 for text generation and a distilled Stable Diffusion model for image generation, ensuring high performance and quick response times.

## Architecture Overview

The architecture of the Yoga Assistant is built to efficiently handle both text and image queries through a scalable pipeline:

<img src="https://github.com/ashwin-sateesh/yoga-assistant/blob/main/assets/Yoga%20Assistant%20Workflow.png" alt="Yoga Assistant Architecture" width="600" height="400">

1. **Input Query & Query Type Classification:**
   - User queries are classified into to know whether it is a text-based or image-based query.

2. **Text Query Processing:**
   - The text query processing is handled through a sophisticated, queue-based fine-tuning system:
     - **Question and Link Check:** 
       - Initially, the query is analyzed to determine if it includes a link that can be enhanced by web content.
       - If the query is just a question without a link, it is directly passed to the fine-tuned GPT-3.5 model to generate a response.
       - If the query contains a link, the question and the link are separated.
     - **Web Scraping and Queue Integration:**
       - The link is scraped, and the retrieved content is added to a processing queue.
       - This scraped content enriches the original question, helping to form a new, more informed prompt.
     - **Parallel Processing with RAG (Retrieval-Augmented Generation):**
       - The newly generated prompt, combining the question and relevant content, is sent to the fine-tuned GPT-3.5 model.
       - The fine-tuned model generates a response faster because the model is continuously updated with the latest data as it becomes available. 
       - The RAG approach allows the model to provide real-time, accurate responses without waiting for comprehensive fine-tuning each time, as the model is automatically updated in the background.


3. **Image Query Processing:**
   - Image queries utilize a fine-tuned Stable Diffusion model.
   - The original model, containing over 1 billion parameters, was distilled to a simple Latent Diffusion Model half its size to reduce inference time.
   - The distilled model generates and returns the image response.

## Model Training & Evaluation

1. **Stable Diffusion Fine-tuning:**
   - The Stable Diffusion model (CompVis/stable-diffusion-v1-4) was fine-tuned on a set of yoga images and corresponding text prompts. The fine-tuning was carried out at a resolution of 512x512 pixels for 3,000 steps, using the quantized `float16` datatype and A100 GPUs to optimize memory usage.
   - The architecture of the Stable Diffusion model includes the following components:
     - **UNet Model:** 859,520,964 parameters.
     - **Text Encoder:** 123,060,480 parameters.
     - **VAE Model:** 83,653,863 parameters.
     - **Total Parameters:** 1,066,235,307 parameters.
   - **Distillation Process:** For distillation, the UNet model was pruned while the Text Encoder and VAE components were kept intact. The model was then distilled using a distillation loss function, reducing the overall model size and improving efficiency.
.
   - For training details, refer to the [official Hugging Face documentation](https://github.com/huggingface/diffusers/tree/main/examples/text_to_image).
   - CLIP scores were used to evaluate the text-to-image response quality.

2. **GPT-3.5 Fine-tuning:**
   - The GPT-3.5 Turbo model was fine-tuned using yoga-related datasets.
   - For training details, refer to the [official OpenAI documentation](https://platform.openai.com/docs/guides/finetuing)
   - Evaluation metrics included perplexity scores and semantic similarity to ensure high-quality responses.
   

## Acknowledgments

This project leverages resources from the [Hugging Face diffusers repository](https://github.com/huggingface/diffusers) and [OpenAI Documentation](https://platform.openai.com/docs/guides). Special thanks to the developers and contributors to these tools.
