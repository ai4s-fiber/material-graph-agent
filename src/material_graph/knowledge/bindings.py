"""Immutable, non-secret embedding and reranker bindings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


_SECRET_FIELD_NAMES = frozenset(
    {"api_key", "authorization", "credential", "password", "secret", "token"}
)
_GLM_QUERY_PREFIX = (
    "为材料科学研究问题检索包含可核验材料组成、制备工艺、测试条件、性能值和来源定位的证据段落。查询："
)
_QWEN_QUERY_PREFIX = (
    "Instruct: 为材料科学研究问题检索包含可核验材料组成、制备工艺、测试条件、"
    "性能值和来源定位的证据段落。\nQuery: "
)
_QWEN_MODEL_MANIFEST_PATH = (
    "/home/cyy/dhu-zh-workspace/omnimat/evidence/embedding/"
    "qwen3-embedding-4b-model-manifest.v1.json"
)
_QWEN_REQUIRED_MODEL_ARTIFACTS = frozenset(
    {
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "model.safetensors.index.json",
        "config.json",
        "tokenizer.json",
    }
)
_SHA256_PATTERN = r"^sha256:[0-9a-f]{64}$"


def _secret_field_paths(value: object, path: tuple[str, ...] = ()) -> list[str]:
    matches: list[str] = []
    if isinstance(value, dict):
        for raw_key, nested in value.items():
            key = str(raw_key)
            normalized = key.strip().casefold().replace("-", "_")
            if normalized in _SECRET_FIELD_NAMES or any(
                normalized.endswith(f"_{suffix}") for suffix in _SECRET_FIELD_NAMES
            ):
                matches.append(".".join((*path, key)))
            matches.extend(_secret_field_paths(nested, (*path, key)))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            matches.extend(_secret_field_paths(nested, (*path, str(index))))
    return matches


class ResolvedEmbeddingArtifact(BaseModel):
    """One immutable model file recorded by the resolved Qwen binding."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    path: str = Field(min_length=1)
    type: Literal["file"]
    sha256: str = Field(pattern=_SHA256_PATTERN)
    size_bytes: int = Field(gt=0)


class ResolvedEmbeddingArtifactIntegrity(BaseModel):
    """Fail-closed artifact identity required by the local Qwen generation."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    status: Literal["resolved"]
    required_model_manifest_path: Literal[
        "/home/cyy/dhu-zh-workspace/omnimat/evidence/embedding/"
        "qwen3-embedding-4b-model-manifest.v1.json"
    ]
    required_model_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    model_directory_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    runtime_environment_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    required_model_artifacts: list[ResolvedEmbeddingArtifact] = Field(min_length=1)

    @model_validator(mode="after")
    def required_artifact_inventory_is_exact(self) -> "ResolvedEmbeddingArtifactIntegrity":
        paths = [artifact.path for artifact in self.required_model_artifacts]
        if len(paths) != len(set(paths)) or frozenset(paths) != _QWEN_REQUIRED_MODEL_ARTIFACTS:
            raise ValueError("resolved Qwen model artifact inventory is invalid")
        return self


class EmbeddingBinding(BaseModel):
    """One of the two immutable, non-mixable production embedding generations."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    schema_version: Literal[1] = 1
    provider: str = Field(min_length=1)
    binding: Literal["openai"] = "openai"
    base_url: str
    model: str = Field(min_length=1)
    dimensions: int = Field(default=1024, gt=0)
    send_dimensions: bool = True
    use_base64: bool = False
    asymmetric: Literal[True] = True
    document_prefix: Literal["NO_PREFIX"] = "NO_PREFIX"
    query_prefix: str = Field(default=_GLM_QUERY_PREFIX, min_length=1)
    max_input_tokens: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    max_async: int = Field(gt=0)
    timeout_seconds: float = Field(gt=0)
    normalize: Literal[True] = True
    distance: Literal["cosine"] = "cosine"
    postgres_vector_index_type: Literal["HNSW_HALFVEC"] = "HNSW_HALFVEC"
    generation_id: str = Field(min_length=1)
    precision: Literal["bf16"] | None = None
    artifact_integrity: ResolvedEmbeddingArtifactIntegrity | None = None

    @model_validator(mode="before")
    @classmethod
    def reject_secret_fields(cls, value: Any) -> Any:
        secret_paths = _secret_field_paths(value)
        if secret_paths:
            raise ValueError(f"embedding binding contains secret fields: {sorted(secret_paths)}")
        if isinstance(value, dict) and "use_base64" not in value:
            return {
                **value,
                "use_base64": value.get("provider") != "vllm_openai_compatible",
            }
        return value

    @model_validator(mode="after")
    def generation_contract_is_exact(self) -> "EmbeddingBinding":
        contract = (
            self.provider,
            self.base_url,
            self.model,
            self.dimensions,
            self.send_dimensions,
            self.use_base64,
            self.query_prefix,
            self.max_input_tokens,
            self.normalize,
            self.distance,
            self.postgres_vector_index_type,
            self.generation_id,
            self.precision,
            self.artifact_integrity is not None,
        )
        glm_contract = (
            "glm_openai_compatible",
            "https://open.bigmodel.cn/api/paas/v4",
            "embedding-3",
            1024,
            True,
            True,
            _GLM_QUERY_PREFIX,
            8192,
            True,
            "cosine",
            "HNSW_HALFVEC",
            "glm-embedding-3-1024-halfvec-v1",
            None,
            False,
        )
        qwen_contract = (
            "vllm_openai_compatible",
            "http://host.docker.internal:31001/v1",
            "qwen3-embedding-4b",
            2560,
            False,
            False,
            _QWEN_QUERY_PREFIX,
            8192,
            True,
            "cosine",
            "HNSW_HALFVEC",
            "qwen3-embedding-4b-2560-bf16-v1",
            "bf16",
            True,
        )
        glm_markers = (
            self.provider == "glm_openai_compatible",
            self.model == "embedding-3",
            self.generation_id == "glm-embedding-3-1024-halfvec-v1",
        )
        qwen_markers = (
            self.provider == "vllm_openai_compatible",
            self.model == "qwen3-embedding-4b",
            self.generation_id == "qwen3-embedding-4b-2560-bf16-v1",
        )
        identifies_glm = any(glm_markers)
        identifies_qwen = any(qwen_markers)
        if identifies_glm and (identifies_qwen or contract != glm_contract):
            raise ValueError("GLM embedding generation contract is invalid")
        if identifies_qwen and (identifies_glm or contract != qwen_contract):
            raise ValueError("Qwen embedding generation contract is invalid")
        if not identifies_qwen and self.artifact_integrity is not None:
            raise ValueError("artifact integrity is only valid for the Qwen generation")
        if (
            self.artifact_integrity is not None
            and self.artifact_integrity.required_model_manifest_path
            != _QWEN_MODEL_MANIFEST_PATH
        ):
            raise ValueError("resolved Qwen model manifest path is invalid")
        return self


class RerankerBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    provider: str
    binding: Literal["cohere"] = "cohere"
    endpoint: str
    model: str
    fallbacks: list[str] = Field(default_factory=list)
    top_n: int = Field(gt=0)
    minimum_score: float = Field(ge=0, le=1)
    max_async: int = Field(gt=0)
    timeout_seconds: float = Field(gt=0)


class ProviderBindings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    embedding: EmbeddingBinding
    reranker: RerankerBinding

    @classmethod
    def load(
        cls,
        *,
        embedding_path: str | Path,
        reranker_path: str | Path,
    ) -> "ProviderBindings":
        embedding_payload = json.loads(Path(embedding_path).read_text(encoding="utf-8"))
        reranker_payload = json.loads(Path(reranker_path).read_text(encoding="utf-8"))
        for payload in (embedding_payload, reranker_payload):
            secret_paths = _secret_field_paths(payload)
            if secret_paths:
                raise ValueError(f"provider binding contains secret fields: {sorted(secret_paths)}")
        return cls(
            embedding=EmbeddingBinding.model_validate(embedding_payload),
            reranker=RerankerBinding.model_validate(reranker_payload),
        )
