#!/usr/bin/env sh
set -eu

apply_migration() {
  migration="$1"
  python - "$migration" <<'PY'
import pathlib
import sys

import psycopg

dsn = pathlib.Path("/run/secrets/postgres_dsn").read_text(encoding="utf-8").strip()
migration = pathlib.Path(sys.argv[1])
version = migration.name.removesuffix(".sql")
sql = migration.read_text(encoding="utf-8")
superseded_by = {
    "0002_runtime_status_contract": {"0003_runtime_outcome_statuses"},
}
with psycopg.connect(dsn) as connection:
    migrations_table = connection.execute(
        "SELECT to_regclass('public.schema_migrations')"
    ).fetchone()[0]
    if migrations_table is not None:
        applied_versions = {
            row[0]
            for row in connection.execute(
                "SELECT version FROM public.schema_migrations"
            ).fetchall()
        }
        if version in applied_versions:
            print(f"migration {version} already applied; skipping")
            raise SystemExit(0)
        replacements = superseded_by.get(version, set())
        if replacements.intersection(applied_versions):
            connection.execute(
                """
                INSERT INTO public.schema_migrations(version)
                VALUES (%s)
                ON CONFLICT DO NOTHING
                """,
                (version,),
            )
            connection.commit()
            print(
                f"migration {version} is superseded by "
                f"{sorted(replacements.intersection(applied_versions))}; recording and skipping"
            )
            raise SystemExit(0)
    connection.execute(sql)
PY
}

# These migrations have explicit dependencies: the durable worker queue belongs
# to the knowledge schema and is installed before AGE facts are enabled.
for name in 0001_runtime.sql knowledge_0001.sql knowledge_0002.sql knowledge_0003.sql knowledge_0004.sql knowledge_0005.sql provider_0001.sql graph_admission_0001.sql age_0001.sql; do
  migration="/app/migrations/$name"
  [ ! -f "$migration" ] || apply_migration "$migration"
done

# Apply later forward migrations without ever auto-running rollback files.
for migration in /app/migrations/*.sql; do
  case "$migration" in
    *.down.sql|*/0001_runtime.sql|*/knowledge_0001.sql|*/knowledge_0002.sql|*/knowledge_0003.sql|*/knowledge_0004.sql|*/knowledge_0005.sql|*/provider_0001.sql|*/graph_admission_0001.sql|*/age_0001.sql) continue ;;
  esac
  apply_migration "$migration"
done

python - <<'PY'
import pathlib
from langgraph.checkpoint.postgres import PostgresSaver

dsn = pathlib.Path("/run/secrets/postgres_dsn").read_text(encoding="utf-8").strip()
with PostgresSaver.from_conn_string(dsn) as checkpointer:
    checkpointer.setup()
PY
