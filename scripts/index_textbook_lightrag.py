#!/usr/bin/env python3
"""Run resumable local textbook indexing with the pinned LightRAG package."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import sys

from material_graph.knowledge.textbook_lightrag import (
    DEFAULT_WORKSPACE,
    LocalTextbookIndexSummary,
    LocalTextbookLightRAGSettings,
    TextbookLightRAGError,
    index_local_textbook_fragments,
)


ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fragments",
        type=Path,
        default=ROOT / "data/runtime/textbook-corpus/fragments.jsonl",
    )
    parser.add_argument(
        "--working-dir",
        type=Path,
        default=ROOT / "data/runtime/textbook-graphrag",
    )
    parser.add_argument(
        "--embedding-binding",
        type=Path,
        default=ROOT / "config/knowledge/embedding-binding.v1.json",
    )
    parser.add_argument(
        "--llm-pool-binding",
        type=Path,
        default=ROOT / "config/knowledge/textbook-llm-pool.v1.json",
    )
    parser.add_argument("--workspace", default=DEFAULT_WORKSPACE)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--embedding-concurrency", type=int, default=16)
    parser.add_argument("--insert-concurrency", type=int, default=88)
    parser.add_argument("--entity-extract-max-gleaning", type=int, default=0)
    parser.add_argument(
        "--text-extraction",
        action="store_true",
        help="disable JSON-mode entity extraction for compatibility diagnostics",
    )
    return parser


def _emit(summary: LocalTextbookIndexSummary) -> None:
    sys.stdout.write(json.dumps(asdict(summary), ensure_ascii=False, sort_keys=True) + "\n")
    sys.stdout.flush()


async def _run(arguments: argparse.Namespace) -> int:
    settings = LocalTextbookLightRAGSettings(
        fragments_path=arguments.fragments.resolve(),
        working_dir=arguments.working_dir.resolve(),
        embedding_binding_path=arguments.embedding_binding.resolve(),
        llm_pool_binding_path=arguments.llm_pool_binding.resolve(),
        workspace=arguments.workspace,
        batch_size=arguments.batch_size,
        limit=arguments.limit,
        embedding_concurrency=arguments.embedding_concurrency,
        insert_concurrency=arguments.insert_concurrency,
        entity_extract_max_gleaning=arguments.entity_extract_max_gleaning,
        entity_extraction_use_json=not arguments.text_extraction,
    )
    summary = await index_local_textbook_fragments(settings, progress_callback=_emit)
    return 0 if summary.status == "completed" else 2


def main() -> int:
    try:
        return asyncio.run(_run(_parser().parse_args()))
    except (OSError, ValueError, TextbookLightRAGError):
        sys.stderr.write("error:textbook_lightrag_index_failed\n")
        return 2
    except KeyboardInterrupt:
        sys.stderr.write("error:textbook_lightrag_index_interrupted\n")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
