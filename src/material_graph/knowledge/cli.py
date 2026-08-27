"""Fail-closed verification CLI for non-secret knowledge configuration."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NoReturn
from urllib.parse import urlsplit
from uuid import uuid4

from .bindings import EmbeddingBinding, ProviderBindings
from .lightrag_runtime import (
    GenerationReleaseContract,
    LightRAGRuntimeConfig,
    LightRAGRuntimeConfigurationError,
    LightRAGStorageConfig,
)
from .policy import CorpusPolicy
from .textbook_chunking import TextbookChunkingPolicy
from .textbook_corpus import TextbookCorpusError
from .textbook_import import (
    PreparedTextbookCorpus,
    iter_fragment_jsonl,
    prepare_textbook_corpus,
)


_ENVIRONMENT_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_CREDENTIAL_KEY_MARKERS = (
    "APIKEY",
    "PASSWORD",
    "PASSWD",
    "SECRET",
    "CREDENTIAL",
    "AUTHORIZATION",
    "ACCESSTOKEN",
    "REFRESHTOKEN",
    "BEARERTOKEN",
    "PRIVATEKEY",
    "CLIENTSECRET",
    "SESSIONCOOKIE",
    "DEVICETOKEN",
)
_CREDENTIAL_KEY_SUFFIXES = (
    "_AUTH",
    "_AUTH_FILE",
    "_COOKIE",
    "_DEVICE_ID",
    "_DID",
    "_DID_FILE",
    "_KEY",
    "_KEY_FILE",
    "_PASS",
    "_SID",
    "_TOKEN",
    "_TOKEN_FILE",
    "_USER",
    "_USERNAME",
)
_CREDENTIAL_VALUE_PREFIXES = (
    "sk-",
    "sk_",
    "ghp_",
    "gho_",
    "ghs_",
    "github_pat_",
    "glpat-",
    "xoxb-",
    "xoxp-",
    "xoxa-",
    "xoxr-",
    "akia",
    "aiza",
)
_ASSIGNMENT_CREDENTIAL = re.compile(
    r"(?:^|[;,&\s])(?:api[_-]?key|password|passwd|pwd|secret|credential|"
    r"access[_-]?token|refresh[_-]?token)\s*[:=]\s*\S+",
    re.IGNORECASE,
)
_JWT_VALUE = re.compile(r"^eyJ[A-Za-z0-9_-]{4,}\.[A-Za-z0-9_-]{4,}(?:\.[A-Za-z0-9_-]{4,})?$")


class _CliFailure(Exception):
    """Internal control flow that carries only a stable public error code."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:  # noqa: ARG002
        raise _CliFailure("usage_invalid")


def _parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(prog="material-graph-knowledge")
    commands = parser.add_subparsers(dest="command", required=True)

    verify_policy = commands.add_parser("verify-policy")
    verify_policy.add_argument("--config", required=True)

    verify_bindings = commands.add_parser("verify-bindings")
    verify_bindings.add_argument("--embedding", required=True)
    verify_bindings.add_argument("--reranker", required=True)

    verify_runtime = commands.add_parser("verify-runtime")
    verify_runtime.add_argument("--env", required=True)
    verify_runtime.add_argument("--embedding", required=True)
    verify_runtime.add_argument("--reranker", required=True)

    verify_generation_release = commands.add_parser("verify-generation-release")
    verify_generation_release.add_argument("--contract", required=True)

    audit_textbooks = commands.add_parser("audit-textbook-corpus")
    audit_textbooks.add_argument("--root", required=True)
    _add_textbook_chunking_arguments(audit_textbooks)

    prepare_textbooks = commands.add_parser("prepare-textbook-corpus")
    prepare_textbooks.add_argument("--root", required=True)
    prepare_textbooks.add_argument("--output", required=True)
    prepare_textbooks.add_argument("--embedding", required=True)
    _add_textbook_chunking_arguments(prepare_textbooks)
    return parser


def _add_textbook_chunking_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--target-chars", type=int, default=2_400)
    parser.add_argument("--max-chars", type=int, default=4_000)
    parser.add_argument("--overlap-chars", type=int, default=200)
    parser.add_argument("--min-content-chars", type=int, default=20)


def _has_credential_key(key: str) -> bool:
    normalized = re.sub(r"[^A-Z0-9]+", "_", key.upper()).strip("_")
    compact = normalized.replace("_", "")
    if any(marker in compact for marker in _CREDENTIAL_KEY_MARKERS):
        return True
    return normalized in {"AUTH", "COOKIE", "DID", "KEY", "PASS", "SID", "TOKEN", "USER"} or (
        normalized.endswith(_CREDENTIAL_KEY_SUFFIXES)
    )


def _has_credential_value(value: str) -> bool:
    candidate = value.strip()
    lowered = candidate.casefold()
    if lowered.startswith(("bearer ", "basic ")):
        return True
    if lowered.startswith(_CREDENTIAL_VALUE_PREFIXES):
        return True
    if _JWT_VALUE.fullmatch(candidate) or _ASSIGNMENT_CREDENTIAL.search(candidate):
        return True
    if "-----begin " in lowered and "private key-----" in lowered:
        return True
    if any(
        marker in lowered
        for marker in ("client-secret", "credential", "password", "private-key", "secret")
    ):
        return True
    if (
        len(candidate) >= 10
        and not any(character.isspace() for character in candidate)
        and any(character.islower() for character in candidate)
        and any(character.isupper() for character in candidate)
        and any(character.isdigit() for character in candidate)
        and any(character in "!@#$%^&*?.:" for character in candidate)
    ):
        return True
    try:
        parsed = urlsplit(candidate)
    except ValueError:
        return False
    return parsed.username is not None or parsed.password is not None


def _unquote_environment_value(value: str) -> str:
    if not value.startswith(("'", '"')):
        return value
    quote = value[0]
    if len(value) < 2 or not value.endswith(quote):
        raise _CliFailure("env_malformed")
    return value[1:-1]


def _load_environment(path: str | Path) -> dict[str, str]:
    try:
        content = Path(path).read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError):
        raise _CliFailure("env_unreadable") from None

    environment: dict[str, str] = {}
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, raw_value = line.partition("=")
        if (
            not separator
            or key != key.strip()
            or not _ENVIRONMENT_KEY.fullmatch(key)
            or "\x00" in raw_value
        ):
            raise _CliFailure("env_malformed")
        if key in environment:
            raise _CliFailure("env_duplicate_key")
        if _has_credential_key(key):
            raise _CliFailure("env_credential_key")
        value = _unquote_environment_value(raw_value.strip())
        if _has_credential_value(value):
            raise _CliFailure("env_credential_value")
        environment[key] = value
    return environment


def _load_policy(path: str | Path) -> CorpusPolicy:
    try:
        return CorpusPolicy.load(path)
    except Exception:
        raise _CliFailure("policy_invalid") from None


def _load_bindings(embedding_path: str | Path, reranker_path: str | Path) -> ProviderBindings:
    try:
        return ProviderBindings.load(
            embedding_path=embedding_path,
            reranker_path=reranker_path,
        )
    except Exception:
        raise _CliFailure("bindings_invalid") from None


def _load_embedding(path: str | Path) -> EmbeddingBinding:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return EmbeddingBinding.model_validate(payload)
    except Exception:
        raise _CliFailure("embedding_invalid") from None


def _load_generation_release(path: str | Path) -> GenerationReleaseContract:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return GenerationReleaseContract.model_validate(payload)
    except Exception:
        raise _CliFailure("generation_release_invalid") from None


def _textbook_chunking_policy(arguments: argparse.Namespace) -> TextbookChunkingPolicy:
    try:
        return TextbookChunkingPolicy(
            target_chars=arguments.target_chars,
            max_chars=arguments.max_chars,
            overlap_chars=arguments.overlap_chars,
            min_content_chars=arguments.min_content_chars,
        )
    except (TypeError, ValueError):
        raise _CliFailure("chunk_policy_invalid") from None


def _prepare_textbooks(
    arguments: argparse.Namespace,
    *,
    embedding_generation_id: str,
) -> PreparedTextbookCorpus:
    try:
        return prepare_textbook_corpus(
            arguments.root,
            embedding_generation_id=embedding_generation_id,
            chunking_policy=_textbook_chunking_policy(arguments),
        )
    except TextbookCorpusError:
        raise _CliFailure("corpus_invalid") from None
    except (OSError, UnicodeError, ValueError):
        raise _CliFailure("corpus_invalid") from None


def _textbook_summary(
    prepared: PreparedTextbookCorpus,
    *,
    status: str,
) -> dict[str, object]:
    source_families: dict[str, int] = {}
    for document in prepared.inventory.documents:
        source_families[document.source_family] = source_families.get(document.source_family, 0) + 1
    return {
        "corpus_digest": prepared.corpus_digest,
        "discovered_documents": prepared.discovered_document_count,
        "duplicate_documents": prepared.duplicate_document_count,
        "fragments": prepared.fragment_count,
        "logical_books": prepared.logical_book_count,
        "source_bytes": prepared.inventory.total_source_bytes,
        "source_families": dict(sorted(source_families.items())),
        "status": status,
        "total_fragment_chars": prepared.total_fragment_chars,
        "unique_documents": prepared.unique_document_count,
    }


def _validate_textbook_output(root: str | Path, output: str | Path) -> Path:
    try:
        source_root = Path(root).resolve(strict=True)
        resolved_output = Path(output).resolve(strict=False)
    except OSError:
        raise _CliFailure("output_invalid") from None
    if resolved_output == source_root or resolved_output.is_relative_to(source_root):
        raise _CliFailure("output_invalid")
    return resolved_output


def _write_textbook_jsonl(
    prepared: PreparedTextbookCorpus,
    output: Path,
) -> int:
    temporary = output.with_name(f"{output.name}.tmp-{uuid4().hex}")
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            for line in iter_fragment_jsonl(prepared):
                stream.write(line)
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(output)
        return output.stat().st_size
    except OSError:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise _CliFailure("output_invalid") from None


def _policy_summary(policy: CorpusPolicy) -> dict[str, object]:
    return {
        "derived_data_hard_cap_bytes": policy.derived_data_hard_cap_bytes,
        "schema_version": policy.schema_version,
        "server_capacity_gb": policy.server_capacity_gb,
        "source_count": len(policy.sources),
        "source_ids": [source.root_id for source in policy.sources],
        "spool": {
            "max_active_objects": policy.spool.max_active_objects,
            "max_object_bytes": policy.spool.max_object_bytes,
            "max_total_bytes": policy.spool.max_total_bytes,
        },
        "status": "valid",
    }


def _bindings_summary(bindings: ProviderBindings) -> dict[str, object]:
    embedding = bindings.embedding
    reranker = bindings.reranker
    return {
        "embedding": {
            "batch_size": embedding.batch_size,
            "dimensions": embedding.dimensions,
            "generation_id": embedding.generation_id,
            "max_async": embedding.max_async,
            "model": embedding.model,
            "provider": embedding.provider,
        },
        "reranker": {
            "fallbacks": reranker.fallbacks,
            "max_async": reranker.max_async,
            "model": reranker.model,
            "provider": reranker.provider,
            "top_n": reranker.top_n,
        },
        "status": "valid",
    }


def _format_number(value: float | int) -> str:
    numeric = float(value)
    return str(int(numeric)) if numeric.is_integer() else str(numeric)


def _expected_reranker_environment(bindings: ProviderBindings) -> dict[str, str]:
    reranker = bindings.reranker
    return {
        "MAX_ASYNC_RERANK": str(reranker.max_async),
        "MIN_RERANK_SCORE": str(reranker.minimum_score),
        "RERANK_BINDING": reranker.binding,
        "RERANK_BINDING_HOST": reranker.endpoint,
        "RERANK_MODEL": reranker.model,
        "RERANK_TIMEOUT": _format_number(reranker.timeout_seconds),
    }


def _verify_runtime(
    environment: Mapping[str, str],
    bindings: ProviderBindings,
) -> LightRAGRuntimeConfig:
    try:
        storage = LightRAGStorageConfig.from_environment(environment)
    except LightRAGRuntimeConfigurationError:
        raise _CliFailure("runtime_storage_invalid") from None
    try:
        runtime = LightRAGRuntimeConfig(embedding=bindings.embedding, storage=storage)
    except Exception:
        raise _CliFailure("runtime_binding_mismatch") from None

    expected = runtime.to_native_environment()
    expected.update(_expected_reranker_environment(bindings))
    if any(environment.get(key) != value for key, value in expected.items()):
        raise _CliFailure("runtime_binding_mismatch")
    return runtime


def _runtime_summary(
    runtime: LightRAGRuntimeConfig,
    bindings: ProviderBindings,
) -> dict[str, object]:
    return {
        "embedding": {
            "dimensions": bindings.embedding.dimensions,
            "generation_id": bindings.embedding.generation_id,
            "max_async": bindings.embedding.max_async,
            "model": bindings.embedding.model,
        },
        "reranker": {
            "max_async": bindings.reranker.max_async,
            "model": bindings.reranker.model,
        },
        "status": "valid",
        "storage": {
            "doc_status": runtime.storage.doc_status_storage,
            "graph": runtime.storage.graph_storage,
            "kv": runtime.storage.kv_storage,
            "vector": runtime.storage.vector_storage,
        },
        "workspace": runtime.storage.workspace,
    }


def _generation_release_summary(
    release: GenerationReleaseContract,
) -> dict[str, object]:
    return {
        "dimensions": release.dimensions,
        "fragment_count": release.fragment_count,
        "generation_id": release.generation_id,
        "promotion_from": release.promotion.from_generation_id,
        "promotion_to": release.promotion.to_generation_id,
        "quality_passed": release.quality_report.passed,
        "rollback_to": release.rollback.to_generation_id,
        "status": "valid",
    }


def _dispatch(arguments: argparse.Namespace) -> dict[str, object]:
    if arguments.command == "verify-policy":
        return _policy_summary(_load_policy(arguments.config))
    if arguments.command == "verify-bindings":
        bindings = _load_bindings(arguments.embedding, arguments.reranker)
        return _bindings_summary(bindings)
    if arguments.command == "verify-runtime":
        bindings = _load_bindings(arguments.embedding, arguments.reranker)
        environment = _load_environment(arguments.env)
        runtime = _verify_runtime(environment, bindings)
        return _runtime_summary(runtime, bindings)
    if arguments.command == "verify-generation-release":
        return _generation_release_summary(_load_generation_release(arguments.contract))
    if arguments.command == "audit-textbook-corpus":
        prepared = _prepare_textbooks(arguments, embedding_generation_id="audit-unbound")
        return _textbook_summary(prepared, status="valid")
    if arguments.command == "prepare-textbook-corpus":
        output = _validate_textbook_output(arguments.root, arguments.output)
        embedding = _load_embedding(arguments.embedding)
        prepared = _prepare_textbooks(
            arguments,
            embedding_generation_id=embedding.generation_id,
        )
        output_bytes = _write_textbook_jsonl(prepared, output)
        summary = _textbook_summary(prepared, status="prepared")
        summary["embedding_generation_id"] = embedding.generation_id
        summary["output_bytes"] = output_bytes
        return summary
    raise _CliFailure("usage_invalid")


def main(argv: Sequence[str] | None = None) -> int:
    """Validate one configuration surface and emit only a safe JSON summary."""

    try:
        arguments = _parser().parse_args(argv)
        summary = _dispatch(arguments)
    except _CliFailure as failure:
        sys.stderr.write(f"error:{failure.code}\n")
        return 2
    except Exception:
        sys.stderr.write("error:internal_error\n")
        return 2
    sys.stdout.write(json.dumps(summary, ensure_ascii=False, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
