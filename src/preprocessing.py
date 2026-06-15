"""Data preprocessing: PDF extraction, prompt generation, and fine-tuning data formatting.

This module handles the full data pipeline from raw PDF documents to
OpenAI-compatible JSONL fine-tuning files.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import fitz  # PyMuPDF
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate


# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """Extract all text content from a PDF file.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Concatenated text from all pages.
    """
    document = fitz.open(str(pdf_path))
    pages = [document.load_page(i).get_text() for i in range(len(document))]
    document.close()
    return "\n".join(pages)


def extract_texts_from_directory(pdf_dir: str | Path) -> List[str]:
    """Extract text from all PDF files in a directory.

    Args:
        pdf_dir: Directory containing PDF files.

    Returns:
        List of extracted text strings, one per PDF.
    """
    pdf_dir = Path(pdf_dir)
    texts = []
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        texts.append(extract_text_from_pdf(pdf_path))
    return texts


# ---------------------------------------------------------------------------
# Prompt-completion pair generation
# ---------------------------------------------------------------------------

def generate_prompt_completion(
    chunk: str,
    llm_model: str = "gpt-3.5-turbo",
    temperature: float = 0.9,
) -> Dict[str, str]:
    """Use GPT-3.5 to generate a question from a text chunk.

    Args:
        chunk: Raw text chunk to generate a question for.
        llm_model: OpenAI model identifier.
        temperature: Sampling temperature.

    Returns:
        Dict with 'prompt' (generated question) and 'completion' (original chunk).
    """
    prompt_template = ChatPromptTemplate.from_template(
        "Create an appropriate question or prompt from the following text:\n\n"
        "{chunk}\n\n"
        "Question:"
    )
    llm = ChatOpenAI(temperature=temperature, model=llm_model)
    chain = LLMChain(llm=llm, prompt=prompt_template, verbose=False)
    question = chain.run(chunk)
    return {"prompt": question, "completion": chunk}


def chunk_text_fixed_size(text: str, chunk_size: int = 500) -> List[str]:
    """Split text into fixed-size character chunks.

    Args:
        text: Input text.
        chunk_size: Maximum characters per chunk.

    Returns:
        List of text chunks.
    """
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def chunk_text_fixed_count(text: str, num_chunks: int = 10) -> List[str]:
    """Split text into a fixed number of roughly equal chunks.

    Args:
        text: Input text.
        num_chunks: Desired number of chunks.

    Returns:
        List of text chunks.
    """
    chunk_size = max(1, len(text) // num_chunks)
    chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
    if len(chunks) > num_chunks:
        chunks = chunks[: num_chunks - 1] + ["".join(chunks[num_chunks - 1 :])]
    return chunks


def generate_pairs_from_texts(
    texts: List[str],
    chunk_size: int = 500,
    llm_model: str = "gpt-3.5-turbo",
    delay: float = 1.0,
) -> List[Dict[str, str]]:
    """Generate prompt-completion pairs from a list of documents.

    Args:
        texts: Raw document texts.
        chunk_size: Characters per chunk.
        llm_model: OpenAI model for question generation.
        delay: Seconds between API calls to avoid rate limiting.

    Returns:
        List of {'prompt': ..., 'completion': ...} dicts.
    """
    pairs: List[Dict[str, str]] = []
    for text in texts:
        chunks = chunk_text_fixed_size(text, chunk_size)
        for chunk in chunks:
            pair = generate_prompt_completion(chunk, llm_model=llm_model)
            pairs.append(pair)
            time.sleep(delay)
    return pairs


# ---------------------------------------------------------------------------
# JSONL formatting for OpenAI fine-tuning
# ---------------------------------------------------------------------------

def format_pairs_to_messages(
    pairs: List[Dict[str, str]],
    system_prompt: str = (
        "You are a knowledgeable yoga assistant. "
        "Answer questions based on the provided yoga related text."
    ),
) -> List[Dict]:
    """Convert prompt-completion pairs to OpenAI chat fine-tuning format.

    Args:
        pairs: List of {'prompt': ..., 'completion': ...}.
        system_prompt: System message for the assistant.

    Returns:
        List of {'messages': [...]} dicts ready for JSONL serialization.
    """
    records = []
    for pair in pairs:
        records.append(
            {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": pair["prompt"]},
                    {"role": "assistant", "content": pair["completion"]},
                ]
            }
        )
    return records


def write_jsonl(records: List[Dict], output_path: str | Path) -> Path:
    """Write a list of dicts to a JSONL file.

    Args:
        records: Data records.
        output_path: Destination file path.

    Returns:
        The resolved output path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    return output_path
