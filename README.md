# TotalRecall

Recursive memory compression system with LLM-powered distillation.

## Overview

TotalRecall captures raw agent interactions, chunks them into sessions, and compresses them through recursive LLM distillation cycles — producing increasingly abstract, high-signal memories organized by tags and compression level.

### Architecture

- **memory_log.db** (Layer 0): Raw command log with session tracking
- **total_recall.db** (Layer 1+): Compressed memories with tag-based recall

### Compression Flow

```
L0 commands → chunk → LLM distillation → L1 memories
L1 memories → batch → LLM distillation → L2 memories
... repeat recursively ...
```

Higher levels = more abstract, cross-cutting insights. Mixed-level inputs fall back to baseline L1.

## Installation

```bash
cd ~/compiled/TotalRecall
uv sync
```

Requires Python 3.11+ and an OpenAI-compatible API endpoint.

## Usage

### CLI

```bash
# Ingest a command
totalrecall ingest --session-id "sess-001" --input "ls -la" --output "file listing..."

# Assign unassigned commands to a chunk
totalrecall chunk

# Compress L0 chunk → L1 memories
totalrecall compress --chunk 1

# Compress existing memories to next level
totalrecall compress --memories 1,2,3

# Recall by tags
totalrecall recall --tags "python-sqlite,deploy-error"

# Check status
totalrecall status
```

### Library API

```python
from totalrecall import TotalRecall

tr = TotalRecall(
    db_dir="~/.totalrecall",
    model="gpt-4o-mini",
    base_url="https://api.openai.com/v1",  # or any compatible endpoint
)

# Ingest commands
cmd_id = tr.ingest("session-abc", "user input", "agent output")

# Chunk and compress
chunk_num = tr.assign_chunk()
if chunk_num:
    memory_ids = tr.compress_chunk(chunk_num)

# Recall
result = tr.recall(["python-sqlite"], max_tokens=200_000)
print(result)

# Status
print(tr.status())

tr.close()
```

## Configuration

| Env Var | Purpose |
|---------|---------|
| `OPENAI_API_KEY` | API key (fallback if not passed via CLI/constructor) |

CLI flags override environment: `--model`, `--base-url`, `--api-key`.

## Database Schema

### memory_log.db
- **commands**: id, timestamp, session_id, chunk_number, layer, input, output, error_log
- **chunks**: chunk_number, created_at

### total_recall.db
- **memories**: id, level (≥1), tags (JSON array), information, created_at, source_chunk_ids (JSON array)
- Indexed on tags and (level DESC, created_at DESC)

## Token Budgeting

Recall uses a rough 4 chars/token estimate. Memories are accumulated from highest level / most recent until the budget is reached.
