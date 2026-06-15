"""Query classification: route user queries to text or image pipelines.

Uses keyword matching to determine whether a query expects a visual
(image) response or a textual answer.
"""

from __future__ import annotations

import re
from typing import List, Literal

# Default visualization keywords
_DEFAULT_KEYWORDS: List[str] = [
    "show", "display", "visualize", "image", "picture",
    "illustrate", "depict", "render", "sketch", "draw",
    "demonstrate", "exhibit", "present", "graph", "diagram",
    "chart", "photograph", "snapshot", "view", "portrait",
    "photo", "visual", "figure", "scene", "design",
]


def classify_query(
    query: str,
    visualization_keywords: List[str] | None = None,
) -> Literal["text", "image"]:
    """Classify a user query as requesting text or image output.

    Args:
        query: The user's input query.
        visualization_keywords: Optional custom keyword list. Falls back
            to the built-in default if not provided.

    Returns:
        ``"image"`` if the query matches a visualization keyword,
        ``"text"`` otherwise.
    """
    keywords = visualization_keywords or _DEFAULT_KEYWORDS
    for keyword in keywords:
        if re.search(rf"\b{re.escape(keyword)}\b", query, re.IGNORECASE):
            return "image"
    return "text"
