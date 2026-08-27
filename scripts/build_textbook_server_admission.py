#!/usr/bin/env python3
"""Build a verified raw-free textbook bundle for server admission."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

from material_graph.knowledge.textbook_server_admission import (
    TextbookServerAdmissionError,
    TextbookServerAdmissionSettings,
    build_textbook_server_admission,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment-bundle", type=Path, required=True)
    parser.add_argument("--portable-archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--vector-mode",
        choices=("copy",),
        default="copy",
        help="Production admission bundles always contain independent vector copies.",
    )
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    try:
        summary = build_textbook_server_admission(
            TextbookServerAdmissionSettings(
                deployment_bundle=arguments.deployment_bundle.resolve(),
                portable_archive=arguments.portable_archive.resolve(),
                output_dir=arguments.output_dir.resolve(),
                vector_mode=arguments.vector_mode,
            )
        )
    except (OSError, RuntimeError, ValueError, TextbookServerAdmissionError):
        sys.stderr.write("error:textbook_server_admission_failed\n")
        return 2
    sys.stdout.write(json.dumps(asdict(summary), ensure_ascii=False, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
