"""Apply ordered SQL migrations using a DSN read from a protected file."""
from __future__ import annotations

import argparse
from pathlib import Path


def apply_migrations(directory: Path, dsn_file: Path) -> None:
    import psycopg

    dsn = dsn_file.read_text(encoding="utf-8").strip()
    if not dsn:
        raise RuntimeError("database DSN secret is empty")
    with psycopg.connect(dsn, autocommit=True) as connection:
        for migration in sorted(directory.glob("[0-9][0-9][0-9][0-9]_*.sql")):
            connection.execute(migration.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=Path("migrations"))
    parser.add_argument("--dsn-file", type=Path, default=Path("/run/secrets/postgres_dsn"))
    args = parser.parse_args()
    apply_migrations(args.directory, args.dsn_file)


if __name__ == "__main__":
    main()
