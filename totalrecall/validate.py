"""JSON output validation and enforcement for LLM responses."""

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def strip_thinking(text: str) -> str:
    """Remove LLM thinking process content from response text.

    Models often emit a thinking preamble (numbered analysis, reasoning steps,
    <thinking> tags, etc.) before the actual structured output.  This function
    strips that preamble so only the real payload remains.
    """
    if not text:
        return text

    # Strip <thinking>...</thinking> blocks (some models use these explicitly)
    text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL | re.IGNORECASE)

    # Strip thinking preamble: "Here's a thinking process:" through the end of
    # the numbered/lettered analysis list, stopping at the first line that looks
    # like actual content (JSON brace, markdown header, or non-list paragraph).
    preamble_re = re.compile(
        r'(?:'
        r"Here's?\s+a\s+thinking\s+process"
        r'|Thinking\s+process'
        r'|Let\s+me\s+think'
        r'|Step\s+\d+'
        r'|Analysis\s*:?'
        r')',
        re.IGNORECASE,
    )
    m = preamble_re.search(text)
    if m:
        candidate = text[m.end():].lstrip()
        lines = candidate.split('\n')
        skip = True
        kept: list[str] = []
        for line in lines:
            stripped = line.strip()
            if skip:
                # Skip blank lines and pure punctuation
                if not stripped or stripped in (':', '::'):
                    continue
                # Skip numbered analysis items (1., 2., 3., etc.)
                if re.match(r'^\d+[.)]\s', stripped):
                    continue
                # Skip bullet points that are meta-commentary
                if re.match(r'^[-*]\s*\*\*', stripped):
                    continue
                # Skip meta-commentary keywords (Draft, Refine, Polish, etc.)
                if re.match(
                    r'^(?:Draft|Refine|Polish|Target|Let\'s\s+draft|Attempt\s+\d+|'
                    r'Refining|Polishing|Drafting)',
                    stripped, re.IGNORECASE,
                ):
                    continue
                # Reached actual content — stop skipping
                skip = False
            kept.append(line)
        if kept:
            text = '\n'.join(kept).strip()

    return text


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
    # Strip thinking preamble before any parsing
    cleaned = strip_thinking(response_text).strip()

    for attempt in range(1, max_retries + 1):
        # Strip markdown code fences if present
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1]).strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            if attempt == max_retries:
                logger.warning("JSON parse failed after %d attempts: %s", max_retries, e)
                return _best_effort_parse(cleaned)
            continue

        valid, error = validate_compression_output(data)
        if valid:
            return data

        if attempt == max_retries:
            logger.warning("Validation failed after %d attempts: %s", max_retries, error)
            return _best_effort_parse(cleaned)

    return None


def _best_effort_parse(text: str) -> dict | None:
    """Attempt to extract partial data from malformed LLM output."""
    cleaned = strip_thinking(text).strip()

    # Try finding JSON-like substring
    start = cleaned.find("{")
    end = cleaned.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            data = json.loads(cleaned[start:end])
            valid, _ = validate_compression_output(data)
            if valid:
                return data
        except json.JSONDecodeError:
            pass

    # Fallback: create a minimal valid structure from cleaned text
    logger.warning("Falling back to storing raw output as single memory")
    return {
        "subjects": ["unknown"],
        "memories": [{"tags": ["raw-fallback"], "information": cleaned[:4000]}]
    }
