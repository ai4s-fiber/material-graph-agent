#!/usr/bin/env python3
"""Explicitly migrate a legacy GLM deployment pair to the strict provenance contract."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

from material_graph.knowledge.textbook_server_admission import (
    TextbookProvenanceContractMigrationSettings,
    TextbookServerAdmissionError,
    migrate_textbook_provenance_contract,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment-bundle", type=Path, required=True)
    parser.add_argument("--portable-archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--enable-legacy-contract-migration",
        action="store_true",
        help="Required acknowledgement; migration is disabled without this flag.",
    )
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    try:
        summary = migrate_textbook_provenance_contract(
            TextbookProvenanceContractMigrationSettings(
                deployment_bundle=arguments.deployment_bundle.resolve(),
                portable_archive=arguments.portable_archive.resolve(),
                output_dir=arguments.output_dir.resolve(),
                enable_legacy_contract_migration=(arguments.enable_legacy_contract_migration),
            )
        )
    except (OSError, RuntimeError, ValueError, TextbookServerAdmissionError):
        sys.stderr.write("error:textbook_provenance_contract_migration_failed\n")
        return 2
    sys.stdout.write(json.dumps(asdict(summary), ensure_ascii=False, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
