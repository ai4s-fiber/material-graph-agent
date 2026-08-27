#!/usr/bin/env python3
"""Index the canonical textbook custom-KG with the canonical binding policy."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import sys

from material_graph.knowledge.textbook_custom_kg import (
    TextbookCustomKGIndexSettings,
    index_textbook_custom_kg,
)
from material_graph.knowledge.textbook_lightrag import TextbookLightRAGError


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "data/runtime/textbook-graphrag-custom"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--custom-kg",
        type=Path,
        default=ROOT / "data/runtime/textbook-graph-bundle/lightrag-custom-kg.json",
    )
    parser.add_argument(
        "--fragments",
        type=Path,
        default=ROOT / "data/runtime/textbook-corpus/fragments.jsonl",
    )
    parser.add_argument("--working-dir", type=Path, default=RUNTIME)
    parser.add_argument(
        "--deployment-dir",
        type=Path,
        default=ROOT / "data/runtime/textbook-deployment-bundles",
    )
    parser.add_argument(
        "--embedding-binding",
        type=Path,
        default=ROOT / "config/knowledge/embedding-binding.v1.json",
    )
    parser.add_argument(
        "--failover-policy",
        type=Path,
        default=ROOT / "config/knowledge/embedding-failover.v1.json",
    )
    parser.add_argument("--state", type=Path, default=RUNTIME / "index-state.json")
    return parser


async def _run(arguments: argparse.Namespace) -> int:
    summary = await index_textbook_custom_kg(
        TextbookCustomKGIndexSettings(
            custom_kg_path=arguments.custom_kg.resolve(),
            fragments_path=arguments.fragments.resolve(),
            working_dir=arguments.working_dir.resolve(),
            deployment_dir=arguments.deployment_dir.resolve(),
            primary_embedding_binding_path=arguments.embedding_binding.resolve(),
            failover_policy_path=arguments.failover_policy.resolve(),
            state_path=arguments.state.resolve(),
        )
    )
    sys.stdout.write(json.dumps(asdict(summary), ensure_ascii=False, sort_keys=True) + "\n")
    return 0


def main() -> int:
    try:
        return asyncio.run(_run(_parser().parse_args()))
    except (OSError, ValueError, TextbookLightRAGError):
        sys.stderr.write("error:textbook_custom_kg_index_failed\n")
        return 2
    except KeyboardInterrupt:
        sys.stderr.write("error:textbook_custom_kg_index_interrupted\n")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
