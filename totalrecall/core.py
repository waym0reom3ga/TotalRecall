"""TotalRecall — recursive memory compression with LLM distillation."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Callable

from openai import OpenAI

from .schema import init_db
from .templates import L0_TO_L1_TEMPLATE, L1_TO_L2_TEMPLATE, CJK_INSTRUCTION
from .validate import enforce_json

logger = logging.getLogger(__name__)


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

    def recall(self, tags: list[str], max_tokens: int = 200_000) -> str:
        """Recall memories matching tags within token budget."""
        if not tags:
            return ""

        # Build OR query for JSON array containment
        conditions = " OR ".join(f"tags LIKE ?" for _ in tags)
        params = [f"%{t}%" for t in tags]

        rows = self.conn.execute(
            f"SELECT id, level, information FROM memories WHERE ({conditions}) "
            "ORDER BY level DESC, created_at DESC",
            params,
        ).fetchall()

        budget_chars = max_tokens * 4  # rough: 4 chars per token
        accumulated = []
        total_chars = 0

        for r in rows:
            info = r["information"]
            if total_chars + len(info) > budget_chars and accumulated:
                break
            accumulated.append(f"[L{r['level']}] {info}")
            total_chars += len(info)

        return "\n\n".join(accumulated)

    # ── Status ───────────────────────────────────────────────────

    def status(self) -> dict:
        """Return stats: command count, chunk count, memory counts by level."""
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
        }

    # ── Internal helpers ─────────────────────────────────────────

    def _llm_call(self, prompt: str) -> dict | None:
        """Call the LLM and enforce valid JSON output."""
        messages = [{"role": "user", "content": prompt}]

        # 1. Try external backend first (e.g., Lycus auxiliary_client)
        if callable(self._llm_backend):
            try:
                resp = self._llm_backend(messages, temperature=0.1)
                text = resp.choices[0].message.content or ""
            except Exception as e:
                logger.error("External LLM backend failed: %s", e)
                return None

        # 2. Fall back to built-in OpenAI client
        else:
            if self._client is None:
                try:
                    self._client = OpenAI(**self._client_kwargs)
                except Exception as e:
                    logger.error("Cannot create OpenAI client: %s", e)
                    return None

            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.1,
                )
                text = resp.choices[0].message.content or ""
            except Exception as e:
                logger.error("LLM call failed: %s", e)
                return None

        result = enforce_json(text)
        if result is None:
            logger.error("Failed to get valid compression output")
        return result

    def _store_memories(self, result: dict | None, level: int,
                        source_ids: list[int]) -> list[int]:
        """Store validated memories into the database. Returns new memory ids."""
        if not result or "memories" not in result:
            return []

        stored = []
        for mem in result["memories"]:
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
