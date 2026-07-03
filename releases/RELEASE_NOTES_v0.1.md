# TotalRecall v0.1 — "The First Step"

![Banner](banner.png)

Initial release of TotalRecall, an independent recursive memory compression system with LLM-powered distillation.

## Key Features

- **Two-database architecture**: `memory_log.db` (L0 raw commands) + `total_recall.db` (L1+ compressed memories)
- **Recursive compression cycles**: L0→L1→L2+ with layer inheritance rules
- **Tag-based recall** with token budgeting (~4 chars/token estimate)
- **CJK encoding mode** for ~40% token savings with Qwen models (default enabled)
- **Strict JSON enforcement** during LLM compression (3 retries + schema validation + fallback)
- **CLI interface** and Python library API
