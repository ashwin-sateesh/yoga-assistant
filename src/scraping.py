"""Web scraping and URL extraction utilities.

Handles extracting URLs from user queries and scraping webpage content
for RAG-style prompt enrichment.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

import requests
from bs4 import BeautifulSoup


def extract_url_and_text(prompt: str) -> Tuple[Optional[str], str]:
    """Extract a URL and the remaining text from a user prompt.

    If the prompt contains a URL, returns (url, remaining_text).
    If no URL is found, returns (None, original_prompt).

    Args:
        prompt: Raw user input that may contain a URL.

    Returns:
        Tuple of (url_or_None, text_without_url).
    """
    url_match = re.search(r"(?P<url>https?://[^\s]+)", prompt)
    if url_match:
        url = url_match.group("url")
        before = prompt[: url_match.start()].strip()
        after = prompt[url_match.end() :].strip()
        combined = f"{before} {after}".strip()
        return url, combined
    return None, prompt


def scrape_content(
    url: str,
    timeout: int = 10,
    user_agent: Optional[str] = None,
) -> str:
    """Scrape and return the text content of a webpage.

    Args:
        url: Target URL.
        timeout: Request timeout in seconds.
        user_agent: Optional custom User-Agent header.

    Returns:
        Extracted text content from the page.

    Raises:
        requests.RequestException: If the request fails.
    """
    headers = {}
    if user_agent:
        headers["User-Agent"] = user_agent

    response = requests.get(url, timeout=timeout, headers=headers)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, "html.parser")

    # Remove script and style elements
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text
