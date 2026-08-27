"""Fail-closed LightRAG runtime translation and startup validation."""

from __future__ import annotations

import math
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from hashlib import sha256
import json
from numbers import Real
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .bindings import EmbeddingBinding


PostgresProbe = Callable[[str], Awaitable["LightRAGPostgresSnapshot"]]
EmbeddingProbe = Callable[[EmbeddingBinding], Awaitable[Sequence[float]]]

_STORAGE_ENV_FIELDS = {
    "kv_storage": "LIGHTRAG_KV_STORAGE",
    "doc_status_storage": "LIGHTRAG_DOC_STATUS_STORAGE",
    "graph_storage": "LIGHTRAG_GRAPH_STORAGE",
    "vector_storage": "LIGHTRAG_VECTOR_STORAGE",
    "workspace": "WORKSPACE",
    "postgres_workspace": "POSTGRES_WORKSPACE",
}
_WORKSPACE_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_-]*$"
_MIN_PGVECTOR_VERSION = (0, 7, 0)
_SHA256_PATTERN = r"^sha256:[0-9a-f]{64}$"
_QWEN_GENERATION_ID = "qwen3-embedding-4b-2560-bf16-v1"
_GLM_GENERATION_ID = "glm-embedding-3-1024-halfvec-v1"


def _canonical_digest(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{sha256(payload).hexdigest()}"


class LightRAGRuntimeConfigurationError(ValueError):
    """Safe configuration failure containing only a stable error code."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(f"LightRAG runtime configuration failed: {code}")


class LightRAGStartupValidationError(RuntimeError):
    """Safe fail-closed startup error; provider/database details stay in their probes."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(f"LightRAG startup validation failed: {code}")


class LightRAGGenerationContractError(ValueError):
    """Fail-closed generation promotion contract error with no input echo."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(f"LightRAG generation contract failed: {code}")


class GenerationQualityReport(BaseModel):
    """Secret-free shadow benchmark evidence required before promotion."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    benchmark_sha256: str = Field(pattern=_SHA256_PATTERN)
    corpus_coverage: float = Field(ge=0, le=1)
    query_coverage: float = Field(ge=0, le=1)
    recall_at_k: float = Field(ge=0, le=1)
    ndcg_at_k: float = Field(ge=0, le=1)
    p95_latency_ms: float = Field(ge=0)
    storage_bytes: int = Field(ge=0)
    citation_provenance_coverage: float = Field(ge=0, le=1)
    mixed_generation_hits: int = Field(ge=0)
    passed: bool
    failed_gates: tuple[str, ...] = ()

    @model_validator(mode="after")
    def pass_state_is_consistent(self) -> "GenerationQualityReport":
        if len(self.failed_gates) != len(set(self.failed_gates)):
            raise ValueError("quality report failed gates must be unique")
        if self.passed and self.failed_gates:
            raise ValueError("passed quality report cannot contain failed gates")
        if not self.passed and not self.failed_gates:
            raise ValueError("failed quality report must name failed gates")
        if self.mixed_generation_hits and self.passed:
            raise ValueError("mixed generations cannot pass quality gates")
        if self.citation_provenance_coverage != 1.0 and self.passed:
            raise ValueError("incomplete citation provenance cannot pass quality gates")
        if self.passed and (
            self.corpus_coverage < 0.99
            or self.query_coverage < 0.99
            or self.recall_at_k < 0.80
            or self.ndcg_at_k < 0.80
            or self.p95_latency_ms > 500.0
            or self.storage_bytes > 85_899_345_920
        ):
            raise ValueError("quality report does not satisfy the production floor")
        return self

    @property
    def report_sha256(self) -> str:
        return _canonical_digest(self.model_dump(mode="json"))


class GenerationAliasTransition(BaseModel):
    """Compare-and-swap input for one atomic active-generation alias change."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    operation: Literal["promote", "rollback"]
    alias_name: str = Field(min_length=1, max_length=200, pattern=_WORKSPACE_PATTERN)
    expected_alias_version: int = Field(ge=0)
    from_generation_id: str = Field(min_length=1, max_length=200, pattern=_WORKSPACE_PATTERN)
    to_generation_id: str = Field(min_length=1, max_length=200, pattern=_WORKSPACE_PATTERN)
    generation_contract_sha256: str = Field(pattern=_SHA256_PATTERN)
    quality_report_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def transition_changes_generation(self) -> "GenerationAliasTransition":
        if self.from_generation_id == self.to_generation_id:
            raise ValueError("alias transition must change generation")
        return self

    @property
    def transition_sha256(self) -> str:
        return _canonical_digest(self.model_dump(mode="json"))


class GenerationReleaseContract(BaseModel):
    """Immutable Qwen generation evidence plus reversible alias transitions."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        validate_by_alias=True,
        serialize_by_alias=True,
    )

    schema_name: Literal["omnimat.knowledge-generation-release.v1"] = Field(
        default="omnimat.knowledge-generation-release.v1",
        alias="schema",
    )
    generation_id: Literal["qwen3-embedding-4b-2560-bf16-v1"]
    workspace: Literal["qwen3-embedding-4b-2560-bf16-v1"]
    dimensions: Literal[2560]
    fragment_count: int = Field(gt=0)
    content_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    source_mappings_sha256: str = Field(pattern=_SHA256_PATTERN)
    binding_sha256: str = Field(pattern=_SHA256_PATTERN)
    model_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    generation_contract_sha256: str = Field(pattern=_SHA256_PATTERN)
    quality_report: GenerationQualityReport
    promotion: GenerationAliasTransition
    rollback: GenerationAliasTransition

    @property
    def expected_generation_contract_sha256(self) -> str:
        return _canonical_digest(
            {
                "binding_sha256": self.binding_sha256,
                "content_manifest_sha256": self.content_manifest_sha256,
                "dimensions": self.dimensions,
                "fragment_count": self.fragment_count,
                "generation_id": self.generation_id,
                "model_manifest_sha256": self.model_manifest_sha256,
                "source_mappings_sha256": self.source_mappings_sha256,
                "workspace": self.workspace,
            }
        )

    @model_validator(mode="after")
    def release_is_promotable_and_reversible(self) -> "GenerationReleaseContract":
        if self.generation_contract_sha256 != self.expected_generation_contract_sha256:
            raise LightRAGGenerationContractError("generation_digest_mismatch")
        if not self.quality_report.passed:
            raise LightRAGGenerationContractError("quality_gates_failed")
        if self.promotion.operation != "promote" or self.rollback.operation != "rollback":
            raise LightRAGGenerationContractError("transition_operation_invalid")
        if self.promotion.alias_name != self.rollback.alias_name:
            raise LightRAGGenerationContractError("alias_name_mismatch")
        if (
            self.promotion.from_generation_id != _GLM_GENERATION_ID
            or self.promotion.to_generation_id != _QWEN_GENERATION_ID
            or self.rollback.from_generation_id != _QWEN_GENERATION_ID
            or self.rollback.to_generation_id != _GLM_GENERATION_ID
        ):
            raise LightRAGGenerationContractError("transition_generation_invalid")
        if self.rollback.expected_alias_version != self.promotion.expected_alias_version + 1:
            raise LightRAGGenerationContractError("transition_version_invalid")
        expected_quality = self.quality_report.report_sha256
        for transition in (self.promotion, self.rollback):
            if transition.generation_contract_sha256 != self.generation_contract_sha256:
                raise LightRAGGenerationContractError("transition_generation_digest_mismatch")
            if transition.quality_report_sha256 != expected_quality:
                raise LightRAGGenerationContractError("transition_quality_digest_mismatch")
        return self


def atomic_generation_alias_transition_sql() -> str:
    """Return the fixed PostgreSQL CAS contract used for promotion and rollback.

    Callers bind every ``%(...)s`` value; no identifier or SQL fragment is
    interpolated.  The row lock, version comparison, generation admission, and
    update occur in one statement/transaction.
    """

    return """
WITH locked AS (
    SELECT alias_name, active_generation_id, version
      FROM knowledge_embedding_generation_aliases
     WHERE alias_name = %(alias_name)s
     FOR UPDATE
), admitted AS (
    SELECT generation_id
      FROM knowledge_embedding_generations
     WHERE generation_id = %(to_generation_id)s
       AND generation_contract_sha256 = %(generation_contract_sha256)s
       AND quality_report_sha256 = %(quality_report_sha256)s
       AND status = 'admitted'
), updated AS (
    UPDATE knowledge_embedding_generation_aliases AS aliases
       SET active_generation_id = %(to_generation_id)s,
           rollback_generation_id = %(from_generation_id)s,
           candidate_generation_id = NULL,
           shadow_generation_id = NULL,
           version = locked.version + 1
      FROM locked, admitted
     WHERE aliases.alias_name = locked.alias_name
       AND locked.active_generation_id = %(from_generation_id)s
       AND locked.version = %(expected_alias_version)s
    RETURNING aliases.alias_name, aliases.active_generation_id, aliases.version
)
SELECT alias_name, active_generation_id, version FROM updated
""".strip()


class LightRAGStorageConfig(BaseModel):
    """Only the allowlisted, non-secret PostgreSQL storage bindings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kv_storage: Literal["PGKVStorage"]
    doc_status_storage: Literal["PGDocStatusStorage"]
    graph_storage: Literal["PGGraphStorage"]
    vector_storage: Literal["PGVectorStorage"]
    workspace: str = Field(min_length=1, pattern=_WORKSPACE_PATTERN)
    postgres_workspace: str = Field(min_length=1, pattern=_WORKSPACE_PATTERN)

    @model_validator(mode="after")
    def postgres_workspace_matches_workspace(self) -> "LightRAGStorageConfig":
        if self.postgres_workspace != self.workspace:
            raise ValueError("POSTGRES_WORKSPACE must identify WORKSPACE exactly")
        return self

    @classmethod
    def from_environment(cls, environment: Mapping[str, str]) -> "LightRAGStorageConfig":
        missing = [
            environment_name
            for environment_name in _STORAGE_ENV_FIELDS.values()
            if environment_name not in environment
        ]
        if missing:
            raise LightRAGRuntimeConfigurationError("storage_contract_missing")
        payload = {
            field_name: environment[environment_name]
            for field_name, environment_name in _STORAGE_ENV_FIELDS.items()
        }
        try:
            return cls.model_validate(payload)
        except ValidationError:
            raise LightRAGRuntimeConfigurationError("storage_contract_invalid") from None

    def to_native_environment(self) -> dict[str, str]:
        return {
            environment_name: str(getattr(self, field_name))
            for field_name, environment_name in _STORAGE_ENV_FIELDS.items()
        }


def workspace_for_generation(generation_id: str) -> str:
    """Return the stable LightRAG workspace name for one embedding generation."""

    normalized = re.sub(r"[^A-Za-z0-9_-]+", "_", generation_id).strip("_")
    if not normalized or not re.fullmatch(_WORKSPACE_PATTERN, normalized):
        raise LightRAGRuntimeConfigurationError("embedding_generation_workspace_invalid")
    return normalized


def _format_number(value: float | int) -> str:
    numeric = float(value)
    return str(int(numeric)) if numeric.is_integer() else str(numeric)


class LightRAGRuntimeConfig(BaseModel):
    """Non-secret runtime configuration that can safely enter checkpoints."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    embedding: EmbeddingBinding
    storage: LightRAGStorageConfig

    @model_validator(mode="after")
    def workspace_matches_embedding_generation(self) -> "LightRAGRuntimeConfig":
        expected = workspace_for_generation(self.embedding.generation_id)
        if self.storage.workspace != expected:
            raise ValueError("WORKSPACE must identify the embedding generation")
        return self

    def to_native_environment(self) -> dict[str, str]:
        binding = self.embedding
        translated = {
            "EMBEDDING_BINDING": binding.binding,
            "EMBEDDING_BINDING_HOST": binding.base_url,
            "EMBEDDING_MODEL": binding.model,
            "EMBEDDING_DIM": str(binding.dimensions),
            "EMBEDDING_SEND_DIM": str(binding.send_dimensions).lower(),
            "EMBEDDING_TOKEN_LIMIT": str(binding.max_input_tokens),
            "EMBEDDING_USE_BASE64": str(binding.use_base64).lower(),
            "EMBEDDING_ASYMMETRIC": str(binding.asymmetric).lower(),
            "EMBEDDING_DOCUMENT_PREFIX": binding.document_prefix,
            "EMBEDDING_QUERY_PREFIX": binding.query_prefix,
            "EMBEDDING_FUNC_MAX_ASYNC": str(binding.max_async),
            "EMBEDDING_BATCH_NUM": str(binding.batch_size),
            "EMBEDDING_TIMEOUT": _format_number(binding.timeout_seconds),
            "POSTGRES_VECTOR_INDEX_TYPE": binding.postgres_vector_index_type,
        }
        translated.update(self.storage.to_native_environment())
        return translated

    @classmethod
    def from_environment(
        cls,
        *,
        embedding: EmbeddingBinding,
        environment: Mapping[str, str],
    ) -> "LightRAGRuntimeConfig":
        """Build the frozen runtime only when every native binding is exact."""

        storage = LightRAGStorageConfig.from_environment(environment)
        try:
            runtime = cls(embedding=embedding, storage=storage)
        except ValidationError:
            raise LightRAGRuntimeConfigurationError("runtime_contract_invalid") from None
        expected = runtime.to_native_environment()
        if any(environment.get(name) != value for name, value in expected.items()):
            raise LightRAGRuntimeConfigurationError("native_binding_mismatch")
        return runtime


class LightRAGPostgresSnapshot(BaseModel):
    """Minimal result of live pgvector/column/index introspection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    workspace: str = Field(min_length=1)
    pgvector_version: str = Field(min_length=1)
    vector_type: str = Field(min_length=1)
    vector_dimensions: int = Field(gt=0)
    operator_class: str = Field(min_length=1)
    index_method: str = Field(min_length=1)
    index_present: bool


class LightRAGStartupReport(BaseModel):
    """Secret-free proof that all required startup gates passed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    workspace: str
    embedding_generation_id: str
    pgvector_version: str
    vector_dimensions: int
    canary_dimensions: int


def _parse_pgvector_version(value: str) -> tuple[int, int, int] | None:
    match = re.fullmatch(r"\s*(\d+)\.(\d+)(?:\.(\d+))?\s*", value)
    if match is None:
        return None
    return (
        int(match.group(1)),
        int(match.group(2)),
        int(match.group(3) or 0),
    )


class LightRAGStartupValidator:
    """Run live PostgreSQL and provider probes before accepting any insertion."""

    def __init__(
        self,
        *,
        postgres_probe: PostgresProbe,
        embedding_probe: EmbeddingProbe,
    ) -> None:
        self._postgres_probe = postgres_probe
        self._embedding_probe = embedding_probe

    async def validate(self, runtime: LightRAGRuntimeConfig) -> LightRAGStartupReport:
        snapshot = await self._probe_postgres(runtime.storage.workspace)
        self._validate_postgres(snapshot, runtime)

        vector = await self._probe_embedding(runtime.embedding)
        self._validate_embedding_vector(vector, runtime.embedding.dimensions)

        return LightRAGStartupReport(
            workspace=runtime.storage.workspace,
            embedding_generation_id=runtime.embedding.generation_id,
            pgvector_version=snapshot.pgvector_version,
            vector_dimensions=snapshot.vector_dimensions,
            canary_dimensions=len(vector),
        )

    async def _probe_postgres(self, workspace: str) -> LightRAGPostgresSnapshot:
        try:
            snapshot = await self._postgres_probe(workspace)
            return LightRAGPostgresSnapshot.model_validate(snapshot)
        except Exception:
            raise LightRAGStartupValidationError("postgres_probe_unavailable") from None

    async def _probe_embedding(self, binding: EmbeddingBinding) -> list[float]:
        try:
            return list(await self._embedding_probe(binding))
        except Exception:
            raise LightRAGStartupValidationError("embedding_canary_unavailable") from None

    @staticmethod
    def _validate_postgres(
        snapshot: LightRAGPostgresSnapshot,
        runtime: LightRAGRuntimeConfig,
    ) -> None:
        if snapshot.workspace != runtime.storage.workspace:
            raise LightRAGStartupValidationError("postgres_workspace_mismatch")

        version = _parse_pgvector_version(snapshot.pgvector_version)
        if version is None:
            raise LightRAGStartupValidationError("pgvector_version_invalid")
        if version < _MIN_PGVECTOR_VERSION:
            raise LightRAGStartupValidationError("pgvector_version_unsupported")

        if snapshot.vector_type.casefold() != "halfvec":
            raise LightRAGStartupValidationError("vector_type_mismatch")
        if snapshot.vector_dimensions != runtime.embedding.dimensions:
            raise LightRAGStartupValidationError("vector_dimensions_mismatch")
        if snapshot.operator_class.casefold() != "halfvec_cosine_ops":
            raise LightRAGStartupValidationError("vector_operator_class_mismatch")
        if not snapshot.index_present or snapshot.index_method.casefold() != "hnsw":
            raise LightRAGStartupValidationError("hnsw_index_missing")

    @staticmethod
    def _validate_embedding_vector(vector: Sequence[float], expected_dimensions: int) -> None:
        if len(vector) != expected_dimensions:
            raise LightRAGStartupValidationError("embedding_dimension_mismatch")
        squared_norm = 0.0
        for value in vector:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise LightRAGStartupValidationError("embedding_non_finite")
            if not math.isfinite(float(value)):
                raise LightRAGStartupValidationError("embedding_non_finite")
            squared_norm += float(value) * float(value)
        norm = math.sqrt(squared_norm)
        if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=0.005):
            raise LightRAGStartupValidationError("embedding_not_normalized")
