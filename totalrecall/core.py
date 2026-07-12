"""TotalRecall — recursive memory compression with LLM distillation."""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable

from openai import OpenAI

from .schema import init_db
from .templates import L0_TO_L1_TEMPLATE, L1_TO_L2_TEMPLATE, CJK_INSTRUCTION
from .validate import enforce_json

logger = logging.getLogger(__name__)


class CompressionError(Exception):
    """Raised when the LLM backend fails during compression.

    Distinguishes "LLM is broken" from "LLM returned nothing useful".
    Callers should catch this to avoid silently dropping turns.
    """
    pass


class TotalRecall:
    """Memory compression system with recursive distillation."""

    def __init__(self, db_dir: str = "~/.totalrecall", model: str | None = None,
                 base_url: str | None = None, api_key: str | None = None,
                 cjk_opt: str = "YES", llm_backend: Callable | None = None):
        self.db_dir = Path(db_dir).expanduser()
        self.model = model or os.environ.get("LYCUS_MODEL", os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
        self.cjk_enabled = cjk_opt == "YES"

        client_kwargs: dict[str, Any] = {}
        if base_url:
            client_kwargs["base_url"] = base_url
        if api_key:
            client_kwargs["api_key"] = api_key
        else:
            key = os.environ.get("OPENAI_API_KEY")
            if key:
                client_kwargs["api_key"] = key

        # Only create the client if we have credentials; defer until first LLM call otherwise
        self._client_kwargs = client_kwargs
        self._client: OpenAI | None = None
        if client_kwargs.get("api_key"):
            self._client = OpenAI(**client_kwargs)

        # External LLM backend (callable) — takes priority over OpenAI client
        self._llm_backend = llm_backend

        # Failure tracking — surfaced via status() for visibility
        self._llm_failure_count = 0
        self._last_llm_error: str = ""

        # Single database connection for everything
        self.conn = init_db(self.db_dir)
        logger.info("TotalRecall initialized at %s", self.db_dir)

    # ── Ingestion ────────────────────────────────────────────────

    def ingest(self, session_id: str, input_text: str, output_text: str,
               error_log: str = "") -> int:
        """Append a command to the database. Returns command id."""
        cur = self.conn.execute(
            "INSERT INTO commands (session_id, chunk_number, layer, input, output, error_log) "
            "VALUES (?, -1, 0, ?, ?, ?)",
            (session_id, input_text, output_text, error_log),
        )
        self.conn.commit()
        return cur.lastrowid if cur.lastrowid is not None else 0

    # ── Chunking ─────────────────────────────────────────────────

    def assign_chunk(self) -> int | None:
        """Assign unassigned commands to next chunk number. Returns chunk_number or None."""
        rows = self.conn.execute(
            "SELECT id FROM commands WHERE chunk_number = -1 ORDER BY id"
        ).fetchall()
        if not rows:
            return None

        ids = [r["id"] for r in rows]
        cur = self.conn.execute("SELECT COALESCE(MAX(chunk_number), 0) + 1 FROM chunks")
        chunk_num = cur.fetchone()[0]

        self.conn.executemany(
            "UPDATE commands SET chunk_number = ? WHERE id = ?",
            [(chunk_num, iid) for iid in ids],
        )
        self.conn.execute("INSERT INTO chunks (chunk_number) VALUES (?)", (chunk_num,))
        self.conn.commit()
        logger.info("Assigned %d commands to chunk %d", len(ids), chunk_num)
        return chunk_num

    # ── Compression L0→L1 ────────────────────────────────────────

    def compress_chunk(self, chunk_number: int) -> list[int]:
        """Compress a single L0 chunk into L1 memories. Returns list of new memory ids."""
        rows = self.conn.execute(
            "SELECT id, input, output, error_log FROM commands WHERE chunk_number = ? ORDER BY id",
            (chunk_number,),
        ).fetchall()

        if not rows:
            logger.warning("Chunk %d has no commands", chunk_number)
            return []

        parts = []
        for r in rows:
            part = f"[CMD {r['id']}]\nInput: {r['input']}\nOutput: {r['output']}"
            if r["error_log"]:
                part += f"\nError: {r['error_log']}"
            parts.append(part)

        commands_text = "\n\n---\n\n".join(parts)
        cjk_instr = CJK_INSTRUCTION if self.cjk_enabled else ""
        prompt = L0_TO_L1_TEMPLATE.format(commands_text=commands_text, cjk_instruction=cjk_instr)

        result = self._llm_call(prompt)
        return self._store_memories(result, level=1, source_ids=[r["id"] for r in rows])

    # ── Compression L1+→L2+ ──────────────────────────────────────

    def compress_memories(self, memory_ids: list[int]) -> list[int]:
        """Compress existing memories to next level. Returns list of new memory ids."""
        if not memory_ids:
            return []

        rows = self.conn.execute(
            "SELECT id, level, tags, information FROM memories WHERE id IN (" +
            ",".join("?" * len(memory_ids)) + ")",
            memory_ids,
        ).fetchall()

        if not rows:
            logger.warning("No memories found for ids %s", memory_ids)
            return []

        # Determine output level
        levels = {r["level"] for r in rows}
        if len(levels) == 1:
            out_level = rows[0]["level"] + 1
        else:
            out_level = 1

        source_level = min(r["level"] for r in rows)
        parts = []
        for r in rows:
            tags_str = ", ".join(json.loads(r["tags"])) if r["tags"] else "none"
            parts.append(f"[Memory {r['id']} (L{r['level']})] Tags: [{tags_str}]\n{r['information']}")

        memories_text = "\n\n---\n\n".join(parts)
        cjk_instr = CJK_INSTRUCTION if self.cjk_enabled else ""
        prompt = L1_TO_L2_TEMPLATE.format(source_level=source_level, memories_text=memories_text, cjk_instruction=cjk_instr)

        result = self._llm_call(prompt)
        return self._store_memories(result, level=out_level, source_ids=[r["id"] for r in rows])

    # ── Recall ───────────────────────────────────────────────────

    def recall(self, tags: list[str], max_tokens: int = 200_000,
               query_text: str = "") -> str:
        """Recall memories matching tags within token budget.

        Three-phase approach:
        1. Direct tag match (LIKE on tags column) — fast, no LLM needed
        2. CJK count-sort: translate query to Chinese tags, score memories by
           how many stored tags match, return highest-scoring results
        3. FTS5 content search fallback (when nothing matches at all)

        When CJK is enabled, stored tags and information are in Chinese.
        The translation step bridges English queries to Chinese memory content.

        Args:
            tags: List of tags to match against stored memory tags.
            max_tokens: Token budget for results.
            query_text: Original query text (used for CJK translation + FTS5).
        """
        if not tags:
            return ""

        budget_chars = max_tokens * 4  # rough: 4 chars per token
        scored: list[tuple[int, int, str]] = []  # (score, id, info)

        # Phase 1: Direct tag match
        conditions = " OR ".join(f"tags LIKE ?" for _ in tags)
        params = [f"%{t}%" for t in tags]

        rows = self.conn.execute(
            f"SELECT id, level, information FROM memories WHERE ({conditions}) "
            "ORDER BY level DESC, created_at DESC",
            params,
        ).fetchall()

        if rows:
            for r in rows:
                scored.append((1, r["id"], f"[L{r['level']}] {r['information']}"))

        # Phase 2: CJK count-sort
        cjk_tags: list[str] = []
        if self.cjk_enabled and query_text:
            cjk_tags = self._translate_query_to_tags(query_text)
            if cjk_tags:
                cjk_scored = self._count_sort_recall(cjk_tags)
                seen_ids = {s[1] for s in scored}
                for s in cjk_scored:
                    if s[1] not in seen_ids:
                        scored.append(s)
                        seen_ids.add(s[1])

        # Phase 3: FTS5 content search fallback
        if not scored and query_text:
            try:
                # Use translated Chinese tags for FTS5 when CJK is enabled
                # (stored information is in Chinese, English terms won't match)
                search_tags = cjk_tags if self.cjk_enabled and cjk_tags else tags
                fts_query = " OR ".join(f'"{t}"' for t in search_tags if len(t) >= 2)
                if fts_query:
                    rows = self.conn.execute(
                        f"""
                        SELECT m.id, m.level, m.information
                        FROM memories m
                        WHERE m.rowid IN (
                            SELECT rowid FROM memories_fts
                            WHERE memories_fts MATCH ?
                        )
                        ORDER BY m.level DESC, m.created_at DESC
                        """,
                        (fts_query,),
                    ).fetchall()
                    for r in rows:
                        scored.append((0, r["id"], f"[L{r['level']}] {r['information']}"))
            except Exception as e:
                logger.debug("FTS5 fallback search failed: %s", e)

        # Sort by score descending, then by id (newer first)
        scored.sort(key=lambda x: (-x[0], -x[1]))

        # Accumulate within budget
        accumulated = []
        total_chars = 0
        for _, _, info in scored:
            if total_chars + len(info) > budget_chars and accumulated:
                break
            accumulated.append(info)
            total_chars += len(info)

        return "\n\n".join(accumulated)

    def _translate_query_to_tags(self, query_text: str) -> list[str]:
        """Use LLM to translate an English query into Chinese subject tags.

        Returns list of Chinese tag strings, or empty list on failure.
        Uses max_retries=1 to fail fast — this is a lookup, not a write path.
        """
        prompt = (
            "Given this query, produce as many Chinese (中文) subject tags as appropriate "
            "to cover the subjects. Output ONLY a JSON array of strings.\n\n"
            f"Query: {query_text}\n\n"
            "Chinese tags:"
        )
        result = self._llm_call(prompt, max_retries=1)
        if not result:
            return []

        # Handle both our standard format and bare JSON arrays
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            # Try subjects key first, then memories tags
            subjects = result.get("subjects", [])
            if subjects:
                return subjects
            for mem in result.get("memories", []):
                return mem.get("tags", [])
        return []

    def _count_sort_recall(self, query_tags: list[str]) -> list[tuple[int, int, str]]:
        """Score all memories by how many of their stored tags match query tags.

        Returns list of (score, memory_id, formatted_info) sorted by score desc.
        Only returns memories with score > 0.
        """
        if not query_tags:
            return []

        # Fetch all memories (we'll score them in Python)
        rows = self.conn.execute(
            "SELECT id, level, tags, information FROM memories "
            "ORDER BY level DESC, created_at DESC"
        ).fetchall()

        if not rows:
            return []

        scored: list[tuple[int, int, str]] = []
        for r in rows:
            try:
                stored_tags = json.loads(r["tags"]) if r["tags"] else []
            except (json.JSONDecodeError, TypeError):
                stored_tags = []

            # Count matches: case-insensitive substring match
            match_count = 0
            for qt in query_tags:
                qt_lower = qt.lower()
                for st in stored_tags:
                    if qt_lower in st.lower() or st.lower() in qt_lower:
                        match_count += 1
                        break  # one match per query tag is enough

            if match_count > 0:
                scored.append((
                    match_count,
                    r["id"],
                    f"[L{r['level']}] {r['information']}",
                ))

        # Sort by score descending
        scored.sort(key=lambda x: (-x[0], -x[1]))
        return scored

    # ── Status ───────────────────────────────────────────────────

    def status(self) -> dict:
        """Return stats: command count, chunk count, memory counts by level,
        and LLM backend health indicators."""
        cmd_count = self.conn.execute("SELECT COUNT(*) FROM commands").fetchone()[0]
        unassigned = self.conn.execute(
            "SELECT COUNT(*) FROM commands WHERE chunk_number = -1"
        ).fetchone()[0]

        chunks = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

        level_rows = self.conn.execute(
            "SELECT level, COUNT(*) as cnt FROM memories GROUP BY level ORDER BY level"
        ).fetchall()
        by_level = {str(r["level"]): r["cnt"] for r in level_rows}

        return {
            "commands": cmd_count,
            "unassigned": unassigned,
            "chunks": chunks,
            "memories_by_level": by_level,
            "total_memories": sum(by_level.values()),
            "llm_failure_count": self._llm_failure_count,
            "last_llm_error": self._last_llm_error,
        }

    # ── Internal helpers ─────────────────────────────────────────

    def _llm_call(self, prompt: str, max_retries: int = 3) -> dict | None:
        """Call the LLM and enforce valid JSON output.

        Retries up to max_retries times with exponential backoff on failure.
        Tracks failure count for visibility via status().
        """
        messages = [{"role": "user", "content": prompt}]

        for attempt in range(1, max_retries + 1):
            text = ""
            try:
                # 1. Try external backend first (e.g., Lycus auxiliary_client)
                if callable(self._llm_backend):
                    resp = self._llm_backend(messages, temperature=0.1)
                    text = resp.choices[0].message.content or ""
                # 2. Fall back to built-in OpenAI client
                else:
                    if self._client is None:
                        self._client = OpenAI(**self._client_kwargs)
                    resp = self._client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=0.1,
                    )
                    text = resp.choices[0].message.content or ""
            except Exception as e:
                self._llm_failure_count += 1
                self._last_llm_error = str(e)
                logger.error(
                    "LLM call failed (attempt %d/%d): %s",
                    attempt, max_retries, e,
                )
                if attempt < max_retries:
                    time.sleep(min(2 ** (attempt - 1), 10))  # exponential backoff
                    continue

            # Validate JSON output
            result = enforce_json(text)
            if result is not None:
                # Success — reset failure counter
                self._llm_failure_count = max(0, self._llm_failure_count - 1)
                return result

            # JSON validation failed — treat as retryable
            self._llm_failure_count += 1
            self._last_llm_error = "JSON validation failed"
            logger.warning(
                "LLM returned invalid JSON (attempt %d/%d)",
                attempt, max_retries,
            )
            if attempt < max_retries:
                time.sleep(min(2 ** (attempt - 1), 10))

        # All retries exhausted
        logger.error(
            "LLM call exhausted all %d retries. Last error: %s",
            max_retries, self._last_llm_error,
        )
        return None

    def _store_memories(self, result: dict | None, level: int,
                        source_ids: list[int]) -> list[int]:
        """Store validated memories into the database. Returns new memory ids.

        Raises CompressionError if the LLM backend failed entirely,
        so callers can distinguish "LLM broken" from "LLM returned nothing useful".
        """
        if result is None:
            raise CompressionError(
                f"LLM backend failed for level {level} compression. "
                f"Failure count: {self._llm_failure_count}, last error: {self._last_llm_error}"
            )

        memories = result.get("memories", [])
        if not memories:
            logger.info("LLM returned 0 memories for level %d (not an error — nothing to extract)", level)
            return []

        stored = []
        for mem in memories:
            tags_json = json.dumps(mem.get("tags", []))
            info = mem.get("information", "")
            src_json = json.dumps(source_ids)

            cur = self.conn.execute(
                "INSERT INTO memories (level, tags, information, source_chunk_ids) VALUES (?, ?, ?, ?)",
                (level, tags_json, info, src_json),
            )
            stored.append(cur.lastrowid)

        self.conn.commit()
        logger.info("Stored %d memories at level %d", len(stored), level)
        return stored

    def close(self):
        """Close database connection."""
        self.conn.close()
