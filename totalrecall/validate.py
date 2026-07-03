"""JSON output validation and enforcement for LLM responses."""

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def validate_compression_output(data: Any) -> tuple[bool, str]:
    """Validate that parsed JSON matches the expected compression schema.

    Returns (is_valid, error_message). Empty error_message means valid.
    """
    if not isinstance(data, dict):
        return False, "Root element is not a JSON object"

    subjects = data.get("subjects")
    if not isinstance(subjects, list) or not all(isinstance(s, str) for s in subjects):
        return False, "'subjects' must be a list of strings"

    memories = data.get("memories")
    if not isinstance(memories, list):
        return False, "'memories' must be a list"

    for i, mem in enumerate(memories):
        if not isinstance(mem, dict):
            return False, f"memory[{i}] is not an object"
        tags = mem.get("tags")
        if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
            return False, f"memory[{i}].tags must be a list of strings"
        info = mem.get("information")
        if not isinstance(info, str):
            return False, f"memory[{i}].information must be a string"

    return True, ""


def enforce_json(response_text: str, max_retries: int = 3) -> dict | None:
    """Parse and validate LLM JSON output with retry logic.

    Returns validated dict or None if all retries exhausted.
    """
    for attempt in range(1, max_retries + 1):
        # Strip markdown code fences if present
        cleaned = response_text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove first and last fence lines
            cleaned = "\n".join(lines[1:-1]).strip()
            # Also strip optional language tag like 'json'
            if cleaned.startswith("{") or not cleaned.replace("-", "").replace("_", "").isalpha():
                pass  # keep as is after removing fences

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            if attempt == max_retries:
                logger.warning("JSON parse failed after %d attempts: %s", max_retries, e)
                return _best_effort_parse(response_text)
            continue

        valid, error = validate_compression_output(data)
        if valid:
            return data

        if attempt == max_retries:
            logger.warning("Validation failed after %d attempts: %s", max_retries, error)
            return _best_effort_parse(response_text)

    return None


def _best_effort_parse(text: str) -> dict | None:
    """Attempt to extract partial data from malformed LLM output."""
    # Try finding JSON-like substring
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            data = json.loads(text[start:end])
            valid, _ = validate_compression_output(data)
            if valid:
                return data
        except json.JSONDecodeError:
            pass

    # Fallback: create a minimal valid structure from raw text
    logger.warning("Falling back to storing raw output as single memory")
    return {
        "subjects": ["unknown"],
        "memories": [{"tags": ["raw-fallback"], "information": text[:4000]}]
    }
