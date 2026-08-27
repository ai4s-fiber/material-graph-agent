#!/usr/bin/env python3
"""Build canonical textbook graph artifacts from completed raw extractions."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

from material_graph.knowledge.textbook_graph_bundle import (
    TextbookGraphBundleError,
    TextbookGraphBundleSettings,
    build_textbook_graph_bundle,
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
        "--extractions",
        type=Path,
        default=ROOT / "data/runtime/textbook-raw-graph-v2/extractions.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/runtime/textbook-graph-bundle",
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    try:
        summary = build_textbook_graph_bundle(
            TextbookGraphBundleSettings(
                fragments_path=arguments.fragments.resolve(),
                extractions_path=arguments.extractions.resolve(),
                output_dir=arguments.output_dir.resolve(),
                require_complete=not arguments.allow_incomplete,
            )
        )
    except (OSError, ValueError, TextbookGraphBundleError):
        sys.stderr.write("error:textbook_graph_bundle_failed\n")
        return 2
    sys.stdout.write(json.dumps(asdict(summary), ensure_ascii=False, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
