#!/usr/bin/env python3
"""Reject dependency licenses that require a release decision."""
from __future__ import annotations

import json
import sys
from pathlib import Path

DENIED = {"AGPL-3.0", "AGPL-3.0-only", "SSPL-1.0", "BUSL-1.1", "UNKNOWN"}
rows = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
violations = [(row.get("Name"), row.get("License")) for row in rows if any(item in str(row.get("License", "")).upper() for item in DENIED)]
if violations:
    raise SystemExit(f"licenses require review: {violations}")
print("license policy passed")
