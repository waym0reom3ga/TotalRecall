"""LLM compression prompt templates for TotalRecall."""


CJK_INSTRUCTION = """\
CJK ENCODING: Encode all tag values and information fields in Chinese (中文). The model's internal tokenization is more efficient for Chinese text (~40% fewer tokens). When these memories are later recalled, the agent will always reply to the user in their own language — only the stored tags and information should be in Chinese."""


L0_TO_L1_TEMPLATE = """\
You are a memory distillation engine. Below is a series of commands and their outputs from an AI agent session. Your task is to extract the most useful information as structured memories.

Return ONLY valid JSON with this exact structure:
{{
  "subjects": ["main subject 1", "main subject 2"],
  "memories": [
    {{"tags": ["tag1", "tag2"], "information": "concise but complete summary of useful information"}},
    ...
  ]
}}

Rules:
- Tags should be specific and searchable (e.g., "python-sqlite", "deploy-error", "user-preference")
- Information should capture facts, decisions, errors resolved, patterns observed
- Skip trivial or ephemeral details; keep only what would be useful to recall later
- Be concise but complete — every word in information should carry value
{cjk_instruction}

Commands and outputs:
{commands_text}
"""


L1_TO_L2_TEMPLATE = """\
You are a recursive memory distillation engine. Below is a series of compressed memory chunks from previous compression cycles. Your task is to further distill these into higher-level memories, preserving the most valuable information while reducing redundancy.

Return ONLY valid JSON with this exact structure:
{{
  "subjects": ["distilled subject 1"],
  "memories": [
    {{"tags": ["tag1", "tag2"], "information": "highly condensed cross-cutting insight or pattern"}},
    ...
  ]
}}

Rules:
- Merge overlapping information from different source memories
- Extract patterns and insights that emerge across multiple memories
- Tags should reflect the synthesized subject matter
- Information at this level should be high-signal: principles, recurring patterns, critical knowledge
- Be more aggressive about compression than at lower levels
{cjk_instruction}

Compressed memory chunks (level {source_level}):
{memories_text}
"""
