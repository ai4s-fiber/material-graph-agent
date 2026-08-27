#!/usr/bin/env python3
"""Import a verified textbook provenance bundle into production PostgreSQL."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import sys

from material_graph.knowledge.postgres import (
    PostgresCheckpointRepository,
    PostgresEvidenceRepository,
    PostgresLightRAGSourceMappingStore,
    PostgresSourceCatalogRepository,
    create_psycopg_async_pool,
    create_psycopg_sync_pool,
)
from material_graph.knowledge.textbook_deployment_import import (
    TextbookDeploymentImportError,
    TextbookDeploymentImportSettings,
    import_textbook_deployment_bundle,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--database-url-file", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    return parser


async def _run(arguments: argparse.Namespace) -> int:
    secret_file = arguments.database_url_file.resolve()
    if not secret_file.is_file():
        raise ValueError("database URL secret file is unavailable")
    dsn = secret_file.read_text(encoding="utf-8").strip()
    if not dsn:
        raise ValueError("database URL secret file is empty")

    sync_pool = create_psycopg_sync_pool(dsn, min_size=1, max_size=2)
    async_pool = create_psycopg_async_pool(dsn, min_size=1, max_size=8)
    sync_pool.open(wait=True)
    await async_pool.open(wait=True)
    try:
        summary = await import_textbook_deployment_bundle(
            TextbookDeploymentImportSettings(
                bundle_dir=arguments.bundle_dir.resolve(),
                batch_size=arguments.batch_size,
                fragment_source="derived_chunks",
            ),
            catalog=PostgresSourceCatalogRepository(sync_pool),
            checkpoints=PostgresCheckpointRepository(async_pool),
            evidence=PostgresEvidenceRepository(async_pool),
            mappings=PostgresLightRAGSourceMappingStore(async_pool),
        )
    finally:
        await async_pool.close()
        sync_pool.close()
    sys.stdout.write(json.dumps(asdict(summary), ensure_ascii=False, sort_keys=True) + "\n")
    return 0


def main() -> int:
    try:
        return asyncio.run(_run(_parser().parse_args()))
    except (OSError, RuntimeError, ValueError, TextbookDeploymentImportError):
        sys.stderr.write("error:textbook_deployment_import_failed\n")
        return 2
    except KeyboardInterrupt:
        sys.stderr.write("error:textbook_deployment_import_interrupted\n")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
