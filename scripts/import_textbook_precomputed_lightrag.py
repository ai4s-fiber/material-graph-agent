#!/usr/bin/env python3
"""Import a verified textbook vector archive without embedding API calls."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import MutableMapping
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

from material_graph.knowledge.textbook_precomputed_lightrag import (
    TextbookPrecomputedImportError,
    TextbookPrecomputedImportSettings,
    import_textbook_precomputed_lightrag,
)


ROOT = Path(__file__).resolve().parents[1]
GENERATION = "glm_embedding_3_1024_halfvec_v1"
MAX_POSTGRES_PASSWORD_BYTES = 16_384


class PostgresPasswordFileError(ValueError):
    """Secret-free failure while hydrating PostgreSQL file credentials."""


def _hydrate_postgres_password(
    environ: MutableMapping[str, str] | None = None,
) -> None:
    """Load an optional PostgreSQL password file into this process only."""

    resolved = os.environ if environ is None else environ
    raw_path = resolved.get("POSTGRES_PASSWORD_FILE")
    if not raw_path:
        return
    if "POSTGRES_PASSWORD" in resolved:
        raise PostgresPasswordFileError(
            "POSTGRES_PASSWORD must be supplied through the file-backed secret"
        )
    path = Path(raw_path)
    if path.is_symlink() or not path.is_file():
        raise PostgresPasswordFileError("POSTGRES_PASSWORD_FILE must reference a regular file")
    try:
        with path.open("rb") as stream:
            payload = stream.read(MAX_POSTGRES_PASSWORD_BYTES + 1)
    except OSError:
        raise PostgresPasswordFileError("POSTGRES_PASSWORD_FILE is unavailable") from None
    if (
        not payload
        or len(payload) > MAX_POSTGRES_PASSWORD_BYTES
        or b"\x00" in payload
        or b"\r" in payload
        or b"\n" in payload
    ):
        raise PostgresPasswordFileError("POSTGRES_PASSWORD_FILE has an invalid value")
    try:
        password = payload.decode("utf-8")
    except UnicodeDecodeError:
        raise PostgresPasswordFileError("POSTGRES_PASSWORD_FILE has an invalid value") from None
    resolved["POSTGRES_PASSWORD"] = password
    resolved.pop("POSTGRES_PASSWORD_FILE", None)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=ROOT / "data/runtime/textbook-deployment-bundles" / GENERATION,
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=ROOT / "data/runtime/textbook-portable-embeddings" / GENERATION,
    )
    parser.add_argument(
        "--working-dir",
        type=Path,
        default=ROOT / "data/runtime/textbook-precomputed-import",
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=ROOT / "data/runtime/textbook-precomputed-import/precomputed-import-state.json",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    return parser


async def _run(arguments: argparse.Namespace) -> int:
    summary = await import_textbook_precomputed_lightrag(
        TextbookPrecomputedImportSettings(
            bundle_dir=arguments.bundle_dir.resolve(),
            archive_dir=arguments.archive_dir.resolve(),
            working_dir=arguments.working_dir.resolve(),
            state_path=arguments.state.resolve(),
            batch_size=arguments.batch_size,
        )
    )
    sys.stdout.write(json.dumps(asdict(summary), ensure_ascii=False, sort_keys=True) + "\n")
    return 0


def main() -> int:
    try:
        _hydrate_postgres_password()
        return asyncio.run(_run(_parser().parse_args()))
    except (
        OSError,
        ValueError,
        PostgresPasswordFileError,
        TextbookPrecomputedImportError,
    ):
        sys.stderr.write("error:textbook_precomputed_lightrag_import_failed\n")
        return 2
    except KeyboardInterrupt:
        sys.stderr.write("error:textbook_precomputed_lightrag_import_interrupted\n")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
