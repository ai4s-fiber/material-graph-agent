#!/usr/bin/env python3
"""Finish the local textbook GraphRAG pipeline after a running extraction exits."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import ctypes
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "data/runtime"
FRAGMENTS = RUNTIME / "textbook-corpus/fragments.jsonl"
EXTRACTIONS = RUNTIME / "textbook-raw-graph-v2/extractions.jsonl"
EXTRACTION_STATE = RUNTIME / "textbook-raw-graph-v2/extraction-state.json"
PIPELINE_DIR = RUNTIME / "textbook-pipeline"
PIPELINE_STATE = PIPELINE_DIR / "pipeline-state.json"
PIPELINE_LOG = PIPELINE_DIR / "pipeline.log"
_PIPELINE_CONTEXT: dict[str, Any] = {
    "run_id": uuid4().hex,
    "state_authority": "generation_state",
    "manifest_authority": "archive_manifest",
}


class TextbookPipelineError(RuntimeError):
    """Stable background-pipeline failure."""


@dataclass(frozen=True, slots=True)
class ExtractionInventory:
    fragments: int
    extractions: int

    @property
    def missing(self) -> int:
        return self.fragments - self.extractions


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-pid", type=int, required=True)
    parser.add_argument("--recovery-attempts", type=int, default=3)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument(
        "--llm-pool-binding",
        type=Path,
        default=ROOT / "config/knowledge/textbook-llm-pool.v1.json",
    )
    parser.add_argument(
        "--embedding-binding",
        type=Path,
        default=ROOT / "config/knowledge/embedding-binding.v1.json",
    )
    parser.add_argument(
        "--embedding-failover-policy",
        type=Path,
        default=ROOT / "config/knowledge/embedding-failover.v1.json",
    )
    parser.add_argument("--embedding-flush-items", type=int, default=512)
    parser.add_argument("--embedding-max-async", type=int)
    return parser


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _safe_workspace(generation_id: str) -> str:
    workspace = re.sub(r"[^A-Za-z0-9_]+", "_", generation_id).strip("_")
    if not workspace:
        raise TextbookPipelineError("embedding generation cannot form a workspace")
    return workspace


def _embedding_context(binding_path: Path) -> dict[str, Any]:
    try:
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
        generation_id = str(binding["generation_id"])
        provider = str(binding["provider"])
        model = str(binding["model"])
        dimensions = int(binding["dimensions"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise TextbookPipelineError("embedding binding metadata is invalid") from error
    if not all((generation_id, provider, model)) or dimensions <= 0:
        raise TextbookPipelineError("embedding binding metadata is invalid")

    workspace = _safe_workspace(generation_id)
    archive_dir = RUNTIME / "textbook-portable-embeddings" / workspace
    deployment_dir = RUNTIME / "textbook-deployment-bundles" / workspace
    context: dict[str, Any] = {
        "embedding_binding_path": binding_path.relative_to(ROOT).as_posix()
        if binding_path.is_relative_to(ROOT)
        else binding_path.as_posix(),
        "generation_id": generation_id,
        "provider": provider,
        "model": model,
        "dimensions": dimensions,
        "embedding_state_path": (archive_dir / "embedding-state.json").relative_to(ROOT).as_posix(),
        "archive_manifest_path": (archive_dir / "archive-manifest.json")
        .relative_to(ROOT)
        .as_posix(),
        "deployment_manifest_path": (deployment_dir / "manifest.json").relative_to(ROOT).as_posix(),
    }
    manifest = deployment_dir / "manifest.json"
    if manifest.is_file():
        context["bundle_manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    return context


def _stage_status(stage: str) -> str:
    if stage == "completed":
        return "completed"
    if stage in {"failed", "interrupted"}:
        return stage
    return "running"


def _publish(stage: str, **details: Any) -> None:
    payload = {
        "schema_version": 1,
        **_PIPELINE_CONTEXT,
        "stage": stage,
        "status": _stage_status(stage),
        "updated_at_unix": time.time(),
        **details,
    }
    _atomic_json(PIPELINE_STATE, payload)
    PIPELINE_DIR.mkdir(parents=True, exist_ok=True)
    with PIPELINE_LOG.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _wait_for_windows_process(process_id: int, poll_seconds: float) -> None:
    if process_id <= 0:
        raise TextbookPipelineError("wait process id is invalid")
    synchronize = 0x00100000
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(synchronize, False, process_id)
    if not handle:
        return
    try:
        wait_timeout = 0x00000102
        while kernel32.WaitForSingleObject(handle, int(poll_seconds * 1000)) == wait_timeout:
            _publish("waiting_initial_extraction", process_id=process_id)
    finally:
        kernel32.CloseHandle(handle)


def _jsonl_ids(path: Path, field: str) -> set[str]:
    values: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                value = str(payload[field])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise TextbookPipelineError(
                    f"runtime JSONL is invalid at line {line_number}"
                ) from error
            if value in values:
                raise TextbookPipelineError("runtime JSONL contains duplicate identities")
            values.add(value)
    return values


def _inventory() -> ExtractionInventory:
    fragment_ids = _jsonl_ids(FRAGMENTS, "fragment_id")
    extraction_ids = _jsonl_ids(EXTRACTIONS, "fragment_id")
    if not extraction_ids <= fragment_ids:
        raise TextbookPipelineError("raw extraction references unknown fragments")
    return ExtractionInventory(
        fragments=len(fragment_ids),
        extractions=len(extraction_ids),
    )


def _run(
    command: Sequence[str],
    *,
    stage: str,
    accepted_returncodes: frozenset[int] = frozenset({0}),
) -> None:
    _publish(stage, command=Path(command[1]).name if len(command) > 1 else "python")
    completed = subprocess.run(
        list(command),
        cwd=ROOT,
        env=os.environ.copy(),
        check=False,
    )
    if completed.returncode not in accepted_returncodes:
        raise TextbookPipelineError(f"{stage} exited with code {completed.returncode}")


def _recover_extractions(
    max_attempts: int,
    *,
    llm_pool_binding: Path,
) -> ExtractionInventory:
    previous = _inventory()
    for attempt in range(1, max_attempts + 1):
        if previous.missing == 0:
            return previous
        _publish(
            "recovering_extractions",
            attempt=attempt,
            fragments=previous.fragments,
            extractions=previous.extractions,
            missing=previous.missing,
        )
        _run(
            (
                sys.executable,
                "scripts/extract_textbook_raw_graph.py",
                "--max-output-tokens",
                "4096",
                "--parse-attempts",
                "3",
                "--llm-pool-binding",
                str(llm_pool_binding),
            ),
            stage="recovery_extractor_running",
            # The extractor uses exit code 2 for a completed pass that still
            # has per-fragment parse failures. Inventory reconciliation below
            # decides whether the pass made progress and whether another
            # bounded recovery attempt is required.
            accepted_returncodes=frozenset({0, 2}),
        )
        current = _inventory()
        if current.extractions <= previous.extractions and current.missing:
            raise TextbookPipelineError("raw extraction recovery made no progress")
        previous = current
    if previous.missing:
        raise TextbookPipelineError("raw extraction recovery attempts were exhausted")
    return previous


def main() -> int:
    arguments = _parser().parse_args()
    if arguments.recovery_attempts <= 0 or arguments.poll_seconds <= 0:
        sys.stderr.write("error:textbook_pipeline_invalid_arguments\n")
        return 2
    try:
        _PIPELINE_CONTEXT.update(_embedding_context(arguments.embedding_binding.resolve()))
        _publish(
            "waiting_initial_extraction",
            process_id=arguments.wait_pid,
        )
        _wait_for_windows_process(arguments.wait_pid, arguments.poll_seconds)
        inventory = _recover_extractions(
            arguments.recovery_attempts,
            llm_pool_binding=arguments.llm_pool_binding.resolve(),
        )
        _publish(
            "extractions_completed",
            fragments=inventory.fragments,
            extractions=inventory.extractions,
            missing=inventory.missing,
        )
        _run(
            (sys.executable, "scripts/build_textbook_graph_bundle.py"),
            stage="graph_bundle_running",
        )
        embedding_command = [
            sys.executable,
            "scripts/build_textbook_embedding_bundle.py",
            "--embedding-binding",
            str(arguments.embedding_binding.resolve()),
            "--failover-policy",
            str(arguments.embedding_failover_policy.resolve()),
            "--flush-items",
            str(arguments.embedding_flush_items),
        ]
        if arguments.embedding_max_async is not None:
            embedding_command.extend(("--max-async", str(arguments.embedding_max_async)))
        _run(embedding_command, stage="embedding_bundle_running")
        _publish(
            "completed",
            fragments=inventory.fragments,
            extractions=inventory.extractions,
            missing=0,
        )
        return 0
    except (OSError, ValueError, TextbookPipelineError) as error:
        _publish("failed", error_type=type(error).__name__)
        sys.stderr.write("error:textbook_pipeline_failed\n")
        return 2
    except KeyboardInterrupt:
        _publish("interrupted")
        sys.stderr.write("error:textbook_pipeline_interrupted\n")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
