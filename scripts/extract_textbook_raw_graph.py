#!/usr/bin/env python3
"""Extract the textbook raw graph through the capacity-weighted LLM pool."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import sys

from material_graph.knowledge.textbook_lightrag import TextbookLightRAGError
from material_graph.knowledge.textbook_raw_graph import (
    RawGraphExtractionSettings,
    RawGraphExtractionSummary,
    extract_raw_textbook_graph,
)


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "data/runtime/textbook-raw-graph-v2"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fragments",
        type=Path,
        default=ROOT / "data/runtime/textbook-corpus/fragments.jsonl",
    )
    parser.add_argument("--output", type=Path, default=RUNTIME / "extractions.jsonl")
    parser.add_argument("--state", type=Path, default=RUNTIME / "extraction-state.json")
    parser.add_argument("--failures", type=Path, default=RUNTIME / "failures.jsonl")
    parser.add_argument(
        "--provider-audit",
        type=Path,
        default=RUNTIME / "provider-calls.jsonl",
    )
    parser.add_argument(
        "--llm-pool-binding",
        type=Path,
        default=ROOT / "config/knowledge/textbook-llm-pool.v1.json",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--queue-size", type=int, default=512)
    parser.add_argument("--parse-attempts", type=int, default=2)
    parser.add_argument("--max-output-tokens", type=int, default=3_072)
    parser.add_argument("--sync-every", type=int, default=32)
    return parser


def _emit(summary: RawGraphExtractionSummary) -> None:
    sys.stdout.write(json.dumps(asdict(summary), ensure_ascii=False, sort_keys=True) + "\n")
    sys.stdout.flush()


async def _run(arguments: argparse.Namespace) -> int:
    settings = RawGraphExtractionSettings(
        fragments_path=arguments.fragments.resolve(),
        output_path=arguments.output.resolve(),
        state_path=arguments.state.resolve(),
        failure_path=arguments.failures.resolve(),
        provider_audit_path=arguments.provider_audit.resolve(),
        llm_pool_binding_path=arguments.llm_pool_binding.resolve(),
        limit=arguments.limit,
        queue_size=arguments.queue_size,
        parse_attempts=arguments.parse_attempts,
        max_output_tokens=arguments.max_output_tokens,
        sync_every=arguments.sync_every,
    )
    summary = await extract_raw_textbook_graph(settings, progress_callback=_emit)
    return 0 if summary.status == "completed" else 2


def main() -> int:
    try:
        return asyncio.run(_run(_parser().parse_args()))
    except (OSError, ValueError, TextbookLightRAGError):
        sys.stderr.write("error:textbook_raw_graph_extraction_failed\n")
        return 2
    except KeyboardInterrupt:
        sys.stderr.write("error:textbook_raw_graph_extraction_interrupted\n")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
