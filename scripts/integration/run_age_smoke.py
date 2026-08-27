"""Run the explicitly enabled Apache AGE integration smoke without echoing its DSN."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ENABLE_ENV = "MATERIAL_GRAPH_RUN_AGE_SMOKE"
DSN_ENV = "MATERIAL_GRAPH_AGE_DSN"
TEST_PATH = "tests/integration/test_age_writer_integration.py"


def main() -> int:
    if os.environ.get(ENABLE_ENV) != "1":
        print("AGE smoke is disabled; set the explicit enable flag.", file=sys.stderr)
        return 2
    if not os.environ.get(DSN_ENV, "").strip():
        print("AGE smoke DSN is missing.", file=sys.stderr)
        return 2

    root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--maxfail=1",
            TEST_PATH,
        ],
        cwd=root,
        env=os.environ.copy(),
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
