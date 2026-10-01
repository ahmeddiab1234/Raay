"""Queue payload encoding/decoding for the micro-batch consumer.

Queue elements are either a plain review string or a JSON object
``{"id": ..., "text": ...}``. Results are always JSON
``{"id"?, "text", "label", "score"}``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from loguru import logger


@dataclass(frozen=True)
class ReviewMessage:
    """One queue element: the review text plus an optional correlation id."""

    text: str
    id: str | None = None


def encode_message(message: ReviewMessage) -> str:
    """Encode a message for the wire: JSON when it carries an id, else raw text."""
    if message.id is None:
        return message.text
    return json.dumps({"id": message.id, "text": message.text}, ensure_ascii=False)


def decode_message(payload: str) -> ReviewMessage | None:
    """Decode a queue payload; ``None`` for elements we must skip (bad JSON)."""
    payload = payload if isinstance(payload, str) else str(payload)
    try:
        data = json.loads(payload)
    except ValueError:
        return ReviewMessage(text=payload)
    if isinstance(data, dict) and isinstance(data.get("text"), str):
        msg_id = data.get("id")
        return ReviewMessage(text=data["text"], id=str(msg_id) if msg_id else None)
    logger.warning(f"Dropping undecodable queue element: {payload[:120]!r}")
    return None


def encode_result(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False)
