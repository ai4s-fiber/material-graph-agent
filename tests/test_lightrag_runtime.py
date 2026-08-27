from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Awaitable
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from material_graph.knowledge.bindings import EmbeddingBinding, ProviderBindings
from material_graph.knowledge.lightrag_runtime import (
    LightRAGPostgresSnapshot,
    LightRAGRuntimeConfig,
    LightRAGRuntimeConfigurationError,
    LightRAGStartupValidationError,
    LightRAGStartupValidator,
    LightRAGStorageConfig,
)


CONFIG_DIR = Path("config/knowledge")
STORAGE_ENV = {
    "LIGHTRAG_KV_STORAGE": "PGKVStorage",
    "LIGHTRAG_DOC_STATUS_STORAGE": "PGDocStatusStorage",
    "LIGHTRAG_GRAPH_STORAGE": "PGGraphStorage",
    "LIGHTRAG_VECTOR_STORAGE": "PGVectorStorage",
    "WORKSPACE": "glm-embedding-3-1024-halfvec-v1",
    "POSTGRES_WORKSPACE": "glm-embedding-3-1024-halfvec-v1",
}


def _binding() -> EmbeddingBinding:
    return ProviderBindings.load(
        embedding_path=CONFIG_DIR / "embedding-binding.v1.json",
        reranker_path=CONFIG_DIR / "reranker-binding.v1.json",
    ).embedding


def _runtime(
    *,
    storage_environment: dict[str, str] | None = None,
) -> LightRAGRuntimeConfig:
    storage = LightRAGStorageConfig.from_environment(storage_environment or STORAGE_ENV)
    return LightRAGRuntimeConfig(embedding=_binding(), storage=storage)


def _snapshot(**overrides: Any) -> LightRAGPostgresSnapshot:
    payload: dict[str, Any] = {
        "workspace": STORAGE_ENV["WORKSPACE"],
        "pgvector_version": "0.7.0",
        "vector_type": "halfvec",
        "vector_dimensions": 1024,
        "operator_class": "halfvec_cosine_ops",
        "index_method": "hnsw",
        "index_present": True,
    }
    payload.update(overrides)
    return LightRAGPostgresSnapshot.model_validate(payload)


def _run(awaitable: Awaitable[object]) -> object:
    return asyncio.run(awaitable)


def test_embedding_binding_freezes_native_qwen_requirements() -> None:
    binding = _binding()

    assert binding.dimensions == 1024
    assert binding.send_dimensions is True
    assert binding.use_base64 is True
    assert binding.asymmetric is True
    assert binding.document_prefix == "NO_PREFIX"
    assert binding.query_prefix == (
        "为材料科学研究问题检索包含可核验材料组成、制备工艺、测试条件、"
        "性能值和来源定位的证据段落。查询："
    )
    assert binding.normalize is True
    assert binding.distance == "cosine"
    assert binding.postgres_vector_index_type == "HNSW_HALFVEC"

    with pytest.raises(ValidationError):
        binding.dimensions = 1536


def test_binding_to_native_environment_is_exact_and_secret_free() -> None:
    environment = dict(STORAGE_ENV)
    api_key = "runtime-secret-must-not-be-copied"
    environment["EMBEDDING_BINDING_API_KEY"] = api_key
    runtime = _runtime(storage_environment=environment)

    native = runtime.to_native_environment()
    expected_embedding = {
        "EMBEDDING_BINDING": "openai",
        "EMBEDDING_BINDING_HOST": "https://open.bigmodel.cn/api/paas/v4",
        "EMBEDDING_MODEL": "embedding-3",
        "EMBEDDING_DIM": "1024",
        "EMBEDDING_SEND_DIM": "true",
        "EMBEDDING_TOKEN_LIMIT": "8192",
        "EMBEDDING_USE_BASE64": "true",
        "EMBEDDING_ASYMMETRIC": "true",
        "EMBEDDING_DOCUMENT_PREFIX": "NO_PREFIX",
        "EMBEDDING_QUERY_PREFIX": (
            "为材料科学研究问题检索包含可核验材料组成、制备工艺、测试条件、"
            "性能值和来源定位的证据段落。查询："
        ),
        "EMBEDDING_FUNC_MAX_ASYNC": "32",
        "EMBEDDING_BATCH_NUM": "32",
        "EMBEDDING_TIMEOUT": "90",
        "POSTGRES_VECTOR_INDEX_TYPE": "HNSW_HALFVEC",
    }

    assert {key: native[key] for key in expected_embedding} == expected_embedding
    assert {key: native[key] for key in STORAGE_ENV} == STORAGE_ENV
    assert not any(
        "KEY" in key or "TOKEN" in key and key != "EMBEDDING_TOKEN_LIMIT" for key in native
    )
    assert api_key not in json.dumps(native, ensure_ascii=False)
    assert api_key not in runtime.model_dump_json()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("LIGHTRAG_KV_STORAGE", "JsonKVStorage"),
        ("LIGHTRAG_DOC_STATUS_STORAGE", "JsonDocStatusStorage"),
        ("LIGHTRAG_GRAPH_STORAGE", "NetworkXStorage"),
        ("LIGHTRAG_VECTOR_STORAGE", "NanoVectorDBStorage"),
        ("WORKSPACE", ""),
        ("POSTGRES_WORKSPACE", ""),
    ],
)
def test_storage_bindings_fail_closed_on_missing_or_wrong_values(field: str, value: str) -> None:
    environment = dict(STORAGE_ENV)
    environment[field] = value

    with pytest.raises(LightRAGRuntimeConfigurationError, match="storage_contract_invalid"):
        LightRAGStorageConfig.from_environment(environment)

    environment = dict(STORAGE_ENV)
    environment.pop(field)
    with pytest.raises(LightRAGRuntimeConfigurationError, match="storage_contract_missing"):
        LightRAGStorageConfig.from_environment(environment)


def test_workspace_must_be_the_embedding_generation_workspace() -> None:
    environment = dict(STORAGE_ENV)
    environment["WORKSPACE"] = "shared-default"
    environment["POSTGRES_WORKSPACE"] = "shared-default"

    with pytest.raises(ValidationError, match="embedding generation"):
        _runtime(storage_environment=environment)


def test_postgres_workspace_must_preserve_the_unsanitized_workspace() -> None:
    environment = dict(STORAGE_ENV)
    environment["POSTGRES_WORKSPACE"] = "glm_embedding_3_1024_halfvec_v1"

    with pytest.raises(
        LightRAGRuntimeConfigurationError,
        match="storage_contract_invalid",
    ):
        LightRAGStorageConfig.from_environment(environment)


def test_successful_startup_validation_returns_secret_free_report() -> None:
    runtime = _runtime()
    api_key = "probe-owned-secret"

    async def postgres_probe(workspace: str) -> LightRAGPostgresSnapshot:
        assert workspace == STORAGE_ENV["WORKSPACE"]
        return _snapshot()

    async def embedding_probe(binding: EmbeddingBinding) -> list[float]:
        assert binding.model == "embedding-3"
        return [1 / math.sqrt(binding.dimensions)] * binding.dimensions

    report = _run(
        LightRAGStartupValidator(
            postgres_probe=postgres_probe,
            embedding_probe=embedding_probe,
        ).validate(runtime)
    )

    assert report.workspace == STORAGE_ENV["WORKSPACE"]
    assert report.pgvector_version == "0.7.0"
    assert report.vector_dimensions == 1024
    assert report.canary_dimensions == 1024
    assert report.embedding_generation_id == runtime.embedding.generation_id
    assert api_key not in report.model_dump_json()


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        ({"workspace": "wrong-workspace"}, "postgres_workspace_mismatch"),
        ({"pgvector_version": "0.6.2"}, "pgvector_version_unsupported"),
        ({"pgvector_version": "not-a-version"}, "pgvector_version_invalid"),
        ({"vector_type": "vector"}, "vector_type_mismatch"),
        ({"vector_dimensions": 1536}, "vector_dimensions_mismatch"),
        ({"operator_class": "vector_cosine_ops"}, "vector_operator_class_mismatch"),
        ({"index_method": "ivfflat"}, "hnsw_index_missing"),
        ({"index_present": False}, "hnsw_index_missing"),
    ],
)
def test_postgres_schema_mismatches_fail_closed(
    overrides: dict[str, object],
    code: str,
) -> None:
    async def postgres_probe(_: str) -> LightRAGPostgresSnapshot:
        return _snapshot(**overrides)

    async def embedding_probe(binding: EmbeddingBinding) -> list[float]:
        return [0.1] * binding.dimensions

    validator = LightRAGStartupValidator(
        postgres_probe=postgres_probe,
        embedding_probe=embedding_probe,
    )

    with pytest.raises(LightRAGStartupValidationError) as raised:
        _run(validator.validate(_runtime()))

    assert raised.value.code == code
    assert code in str(raised.value)


@pytest.mark.parametrize(
    ("vector_factory", "code"),
    [
        (lambda binding: [0.1] * (binding.dimensions - 1), "embedding_dimension_mismatch"),
        (
            lambda binding: [0.1] * (binding.dimensions - 1) + [math.nan],
            "embedding_non_finite",
        ),
        (
            lambda binding: [0.1] * (binding.dimensions - 1) + [math.inf],
            "embedding_non_finite",
        ),
        (lambda binding: [0.1] * binding.dimensions, "embedding_not_normalized"),
    ],
)
def test_embedding_canary_must_have_exact_finite_dimensions(
    vector_factory: Any,
    code: str,
) -> None:
    async def postgres_probe(_: str) -> LightRAGPostgresSnapshot:
        return _snapshot()

    async def embedding_probe(binding: EmbeddingBinding) -> list[float]:
        return vector_factory(binding)

    validator = LightRAGStartupValidator(
        postgres_probe=postgres_probe,
        embedding_probe=embedding_probe,
    )

    with pytest.raises(LightRAGStartupValidationError) as raised:
        _run(validator.validate(_runtime()))

    assert raised.value.code == code


@pytest.mark.parametrize(
    ("failing_probe", "code"),
    [
        ("postgres", "postgres_probe_unavailable"),
        ("embedding", "embedding_canary_unavailable"),
    ],
)
def test_probe_failures_do_not_echo_credentials(failing_probe: str, code: str) -> None:
    secret = "runtime-credential-do-not-echo-this-value"

    async def postgres_probe(_: str) -> LightRAGPostgresSnapshot:
        if failing_probe == "postgres":
            raise RuntimeError(secret)
        return _snapshot()

    async def embedding_probe(binding: EmbeddingBinding) -> list[float]:
        if failing_probe == "embedding":
            raise RuntimeError(secret)
        return [0.1] * binding.dimensions

    validator = LightRAGStartupValidator(
        postgres_probe=postgres_probe,
        embedding_probe=embedding_probe,
    )

    with pytest.raises(LightRAGStartupValidationError) as raised:
        _run(validator.validate(_runtime()))

    assert raised.value.code == code
    assert secret not in str(raised.value)
    assert raised.value.__cause__ is None


def test_binding_loader_rejects_nested_secret_keys_without_echoing_values(tmp_path: Path) -> None:
    embedding_payload = json.loads(
        (CONFIG_DIR / "embedding-binding.v1.json").read_text(encoding="utf-8")
    )
    secret = "nested-provider-secret"
    embedding_payload["metadata"] = {"auth": {"api_key": secret}}
    embedding_path = tmp_path / "embedding.json"
    embedding_path.write_text(json.dumps(embedding_payload), encoding="utf-8")

    with pytest.raises(ValueError) as raised:
        ProviderBindings.load(
            embedding_path=embedding_path,
            reranker_path=CONFIG_DIR / "reranker-binding.v1.json",
        )

    assert "provider binding contains secret fields" in str(raised.value)
    assert secret not in str(raised.value)
