"""GPT-3.5 fine-tuning and queue-based model management.

Handles initial fine-tuning via the OpenAI API, queue-based incremental
updates with scraped web content, and model status tracking.
"""

from __future__ import annotations

from pathlib import Path
from queue import Queue
from typing import Dict, List, Optional, Tuple

from openai import OpenAI

from .preprocessing import (
    chunk_text_fixed_count,
    format_pairs_to_messages,
    generate_prompt_completion,
    write_jsonl,
)


# ---------------------------------------------------------------------------
# Fine-tuning job management
# ---------------------------------------------------------------------------

def create_fine_tuning_job(
    client: OpenAI,
    training_file_id: str,
    model: str = "gpt-3.5-turbo-0125",
    n_epochs: int = 10,
    learning_rate_multiplier: float = 0.1,
) -> str:
    """Submit a fine-tuning job to the OpenAI API.

    Args:
        client: Authenticated OpenAI client.
        training_file_id: ID of the uploaded training file.
        model: Base model to fine-tune.
        n_epochs: Number of training epochs.
        learning_rate_multiplier: Learning rate scaling factor.

    Returns:
        The fine-tuning job ID.
    """
    response = client.fine_tuning.jobs.create(
        training_file=training_file_id,
        model=model,
        hyperparameters={
            "n_epochs": n_epochs,
            "learning_rate_multiplier": learning_rate_multiplier,
        },
    )
    return response.id


def upload_training_file(client: OpenAI, file_path: str | Path) -> str:
    """Upload a JSONL training file to OpenAI.

    Args:
        client: Authenticated OpenAI client.
        file_path: Path to the JSONL file.

    Returns:
        The uploaded file ID.
    """
    with open(file_path, "rb") as f:
        file_details = client.files.create(file=f, purpose="fine-tune")
    return file_details.id


def get_job_status(client: OpenAI, job_id: str) -> str:
    """Check the status of a fine-tuning job.

    Args:
        client: Authenticated OpenAI client.
        job_id: Fine-tuning job ID.

    Returns:
        ``"completed"`` if the job finished successfully,
        ``"not completed"`` otherwise.
    """
    events = client.fine_tuning.jobs.list_events(
        fine_tuning_job_id=job_id, limit=5
    )
    for event in events.data:
        if "The job has successfully completed" in event.message:
            return "completed"
    return "not completed"


def get_model_name(
    client: OpenAI,
    job_id: str,
    fallback_model: str = "gpt-3.5-turbo",
) -> str:
    """Retrieve the fine-tuned model name from a completed job.

    Args:
        client: Authenticated OpenAI client.
        job_id: Fine-tuning job ID.
        fallback_model: Model name to return if no new model is found.

    Returns:
        The fine-tuned model identifier.
    """
    events = client.fine_tuning.jobs.list_events(
        fine_tuning_job_id=job_id, limit=5
    )
    for event in events.data:
        if "New fine-tuned model created" in event.message:
            return event.message.split(": ")[1]
    return fallback_model


# ---------------------------------------------------------------------------
# Queue-based incremental model updates
# ---------------------------------------------------------------------------

class ModelUpdateManager:
    """Manages queue-based fine-tuning updates with scraped content.

    This class tracks the current fine-tuned model state and queues new
    training data from web scrapes for incremental updates.

    Args:
        client: Authenticated OpenAI client.
        initial_model_name: Starting fine-tuned model identifier.
        initial_job_id: Most recent fine-tuning job ID.
        initial_file_id: Most recent training file ID.
        system_prompt: System message for training data.
        output_dir: Directory for intermediate JSONL files.
    """

    def __init__(
        self,
        client: OpenAI,
        initial_model_name: str,
        initial_job_id: str,
        initial_file_id: str,
        system_prompt: str = (
            "You are a knowledgeable yoga assistant. "
            "Answer questions based on the provided yoga related text."
        ),
        output_dir: str | Path = "./data",
    ) -> None:
        self.client = client
        self.model_name = initial_model_name
        self.job_id = initial_job_id
        self.file_id = initial_file_id
        self.system_prompt = system_prompt
        self.output_dir = Path(output_dir)
        self.queue: Queue = Queue()

    @property
    def current_model(self) -> str:
        """Return the latest fine-tuned model identifier."""
        return self.model_name

    def update_with_content(
        self,
        content: str,
        num_chunks: int = 10,
        update_epochs: int = 2,
        learning_rate_multiplier: float = 0.1,
    ) -> str:
        """Process scraped content and trigger an incremental fine-tuning update.

        Args:
            content: Scraped web page text.
            num_chunks: Number of chunks to split content into.
            update_epochs: Training epochs for the incremental update.
            learning_rate_multiplier: LR multiplier for the update.

        Returns:
            The current model name (may be updated if job completed).
        """
        # Generate training pairs from the scraped content
        chunks = chunk_text_fixed_count(content, num_chunks)
        pairs = []
        for chunk in chunks:
            pair = generate_prompt_completion(chunk)
            pairs.append(pair)

        # Format and write JSONL
        records = format_pairs_to_messages(pairs, self.system_prompt)
        jsonl_path = write_jsonl(
            records, self.output_dir / "yoga_bot_data_update.jsonl"
        )

        # Upload and queue
        file_id = upload_training_file(self.client, jsonl_path)
        self.queue.put(file_id)

        # If previous job is complete, start a new one
        if get_job_status(self.client, self.job_id) == "completed":
            queued_file_id = self.queue.get()
            self.file_id = queued_file_id

            job_id = create_fine_tuning_job(
                self.client,
                training_file_id=queued_file_id,
                model=self.model_name,
                n_epochs=update_epochs,
                learning_rate_multiplier=learning_rate_multiplier,
            )
            self.job_id = job_id

        # Refresh model name
        self.model_name = get_model_name(
            self.client, self.job_id, fallback_model=self.model_name
        )
        return self.model_name

    def refresh_model_name(self) -> str:
        """Check for a newly completed model without submitting new data.

        Returns:
            The current model name.
        """
        self.model_name = get_model_name(
            self.client, self.job_id, fallback_model=self.model_name
        )
        return self.model_name
