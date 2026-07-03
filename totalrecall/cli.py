"""CLI entry point for TotalRecall."""

import argparse
import json
import logging
import sys
from pathlib import Path

from totalrecall.core import TotalRecall


def main():
    parser = argparse.ArgumentParser(
        prog="totalrecall",
        description="Recursive memory compression system with LLM distillation",
    )
    parser.add_argument("--db-dir", default="~/.totalrecall", help="Database directory")
    parser.add_argument("--model", default=None, help="LLM model name")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible API base URL")
    parser.add_argument("--api-key", default=None, help="API key (falls back to OPENAI_API_KEY)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")

    sub = parser.add_subparsers(dest="command", required=True)

    # ingest
    p_ingest = sub.add_parser("ingest", help="Append a command to the log")
    p_ingest.add_argument("--session-id", required=True, help="Session identifier")
    p_ingest.add_argument("--input", required=True, help="Input text")
    p_ingest.add_argument("--output", default="", help="Output text")
    p_ingest.add_argument("--error-log", default="", help="Error log text")

    # chunk
    sub.add_parser("chunk", help="Assign unassigned commands to a new chunk")

    # compress
    p_compress = sub.add_parser("compress", help="Compress a chunk or memories")
    p_compress.add_argument("--chunk", type=int, default=None, help="Chunk number to compress (L0→L1)")
    p_compress.add_argument("--memories", default=None, help="Comma-separated memory ids to compress (L1+→L2+)")

    # recall
    p_recall = sub.add_parser("recall", help="Recall memories by tags")
    p_recall.add_argument("--tags", required=True, help="Comma-separated query tags")
    p_recall.add_argument("--max-tokens", type=int, default=200_000, help="Token budget for recall")

    # status
    sub.add_parser("status", help="Show database statistics")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(name)s %(levelname)s %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    tr = TotalRecall(
        db_dir=args.db_dir,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
    )

    try:
        if args.command == "ingest":
            cmd_id = tr.ingest(args.session_id, args.input, args.output, args.error_log)
            print(f"Stored command id={cmd_id}")

        elif args.command == "chunk":
            chunk_num = tr.assign_chunk()
            if chunk_num is None:
                print("No unassigned commands to chunk")
            else:
                print(f"Created chunk {chunk_num}")

        elif args.command == "compress":
            if args.chunk is not None:
                ids = tr.compress_chunk(args.chunk)
                print(f"Compressed chunk {args.chunk} → memory ids: {ids}")
            elif args.memories:
                mem_ids = [int(x.strip()) for x in args.memories.split(",")]
                new_ids = tr.compress_memories(mem_ids)
                print(f"Compressed memories {mem_ids} → new ids: {new_ids}")
            else:
                parser.error("Provide --chunk or --memories")

        elif args.command == "recall":
            tags = [t.strip() for t in args.tags.split(",")]
            result = tr.recall(tags, max_tokens=args.max_tokens)
            if result:
                print(result)
            else:
                print("No matching memories found", file=sys.stderr)

        elif args.command == "status":
            stats = tr.status()
            print(json.dumps(stats, indent=2))

    finally:
        tr.close()


if __name__ == "__main__":
    main()
