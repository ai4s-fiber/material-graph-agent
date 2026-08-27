from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from material_graph.knowledge.bindings import EmbeddingBinding
from material_graph.knowledge import textbook_embedding_bundle as embedding_bundle_module
from material_graph.knowledge.textbook_embedding_bundle import (
    TextbookEmbeddingArchiveError,
    TextbookEmbeddingArchiveSettings,
    build_textbook_embedding_archive,
    embedding_text,
)


ROOT = Path(__file__).resolve().parents[1]
PRIMARY = ROOT / "config/knowledge/embedding-binding.v1.json"


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    artifacts = {
        "custom_kg_chunks": bundle / "custom-kg-chunks.jsonl",
        "custom_kg_entities": bundle / "custom-kg-entities.jsonl",
        "custom_kg_relationships": bundle / "custom-kg-relationships.jsonl",
    }
    _write_jsonl(
        artifacts["custom_kg_chunks"],
        [
            {
                "content": "PET 经牵伸形成取向结构。",
                "source_id": "fragment-1",
                "file_path": "mg_fragment_1.txt",
                "chunk_order_index": 0,
            }
        ],
    )
    _write_jsonl(
        artifacts["custom_kg_entities"],
        [
            {
                "entity_name": "PET",
                "entity_type": "Material",
                "description": "聚酯",
                "source_id": "fragment-1",
                "file_path": "mg_fragment_1.txt",
            }
        ],
    )
    _write_jsonl(
        artifacts["custom_kg_relationships"],
        [
            {
                "src_id": "PET",
                "tgt_id": "取向结构",
                "description": "牵伸促使PET形成取向结构",
                "keywords": "材料科学,教材知识,Material-Structure",
                "weight": 1.0,
                "source_id": "fragment-1",
                "file_path": "mg_fragment_1.txt",
            }
        ],
    )
    binding = EmbeddingBinding.model_validate_json(PRIMARY.read_text(encoding="utf-8"))
    manifest = {
        "schema_version": 1,
        "embedding": {
            "generation_id": binding.generation_id,
            "provider": binding.provider,
            "model": binding.model,
            "dimensions": binding.dimensions,
        },
        "counts": {
            "custom_kg_chunks": 1,
            "custom_kg_entities": 1,
            "custom_kg_relationships": 1,
        },
        "artifacts": {
            name: {
                "path": path.name,
                "sha256": _digest(path),
                "bytes": path.stat().st_size,
            }
            for name, path in artifacts.items()
        },
    }
    (bundle / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return bundle


def test_embedding_text_matches_lightrag_custom_kg_contract() -> None:
    assert embedding_text("chunk", {"content": "PET"}) == "PET"
    assert (
        embedding_text(
            "entity",
            {"entity_name": "PET", "description": "聚酯"},
        )
        == "PET\n聚酯"
    )
    assert (
        embedding_text(
            "relationship",
            {
                "keywords": "材料科学,教材知识,Material-Property",
                "src_id": "PET",
                "tgt_id": "拉伸强度",
                "description": "PET具有拉伸强度",
            },
        )
        == "材料科学,教材知识,Material-Property\tPET\n拉伸强度\nPET具有拉伸强度"
    )


@pytest.mark.asyncio
async def test_embedding_archive_is_resumable_without_duplicate_api_calls(
    tmp_path: Path,
) -> None:
    binding = EmbeddingBinding.model_validate_json(PRIMARY.read_text(encoding="utf-8"))
    settings = TextbookEmbeddingArchiveSettings(
        bundle_dir=_bundle(tmp_path),
        output_dir=tmp_path / "archive",
        flush_items=32,
    )
    calls: list[list[str]] = []

    async def embedder(
        active_binding: EmbeddingBinding,
        api_key: str,
        texts: list[str],
    ) -> np.ndarray:
        assert active_binding == binding
        assert api_key == "primary-secret"
        calls.append(texts)
        values = np.ones((len(texts), binding.dimensions), dtype=np.float32)
        values[:, 0] = np.arange(1, len(texts) + 1, dtype=np.float32)
        return values

    first = await build_textbook_embedding_archive(
        settings,
        binding,
        "primary-secret",
        embedder=embedder,
    )
    first_call_count = len(calls)
    second = await build_textbook_embedding_archive(
        settings,
        binding,
        "primary-secret",
        embedder=embedder,
    )

    assert first.vector_count == second.vector_count == 3
    assert first.item_count == second.item_count == 3
    assert first_call_count == 3
    assert len(calls) == first_call_count
    assert (settings.output_dir / "vectors.f16.bin").stat().st_size == (
        first.vector_count * binding.dimensions * 2
    )


@pytest.mark.asyncio
async def test_embedding_archive_reuses_vector_for_duplicate_text_without_item_collision(
    tmp_path: Path,
) -> None:
    binding = EmbeddingBinding.model_validate_json(PRIMARY.read_text(encoding="utf-8"))
    bundle = _bundle(tmp_path)
    chunks_path = bundle / "custom-kg-chunks.jsonl"
    _write_jsonl(
        chunks_path,
        [
            {
                "content": "相同教材片段",
                "source_id": "fragment-1",
                "file_path": "mg_fragment_1.txt",
                "chunk_order_index": 0,
            },
            {
                "content": "相同教材片段",
                "source_id": "fragment-2",
                "file_path": "mg_fragment_2.txt",
                "chunk_order_index": 1,
            },
        ],
    )
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["counts"]["custom_kg_chunks"] = 2
    manifest["artifacts"]["custom_kg_chunks"] = {
        "path": chunks_path.name,
        "sha256": _digest(chunks_path),
        "bytes": chunks_path.stat().st_size,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )

    async def embedder(
        active_binding: EmbeddingBinding,
        api_key: str,
        texts: list[str],
    ) -> np.ndarray:
        return np.ones((len(texts), active_binding.dimensions), dtype=np.float32)

    summary = await build_textbook_embedding_archive(
        TextbookEmbeddingArchiveSettings(
            bundle_dir=bundle,
            output_dir=tmp_path / "duplicate-text-archive",
            flush_items=32,
        ),
        binding,
        "primary-secret",
        embedder=embedder,
    )

    assert summary.item_count == 4
    assert summary.vector_count == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["wrong_dimension", "nan"])
async def test_embedding_archive_rejects_invalid_vectors(
    tmp_path: Path,
    mode: str,
) -> None:
    binding = EmbeddingBinding.model_validate_json(PRIMARY.read_text(encoding="utf-8"))
    settings = TextbookEmbeddingArchiveSettings(
        bundle_dir=_bundle(tmp_path),
        output_dir=tmp_path / f"archive-{mode}",
    )

    async def embedder(
        active_binding: EmbeddingBinding,
        api_key: str,
        texts: list[str],
    ) -> np.ndarray:
        dimensions = binding.dimensions - 1 if mode == "wrong_dimension" else binding.dimensions
        values = np.ones((len(texts), dimensions), dtype=np.float32)
        if mode == "nan":
            values[0, 0] = np.nan
        return values

    with pytest.raises(TextbookEmbeddingArchiveError, match="embedding"):
        await build_textbook_embedding_archive(
            settings,
            binding,
            "primary-secret",
            embedder=embedder,
        )
    assert not (settings.output_dir / "archive-manifest.json").exists()
    state = json.loads((settings.output_dir / "embedding-state.json").read_text(encoding="utf-8"))
    assert state["status"] == "failed"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [RuntimeError("HTTP 429 TPM limit reached"), TimeoutError("request timed out")],
)
async def test_embedding_archive_retries_transient_batch_failures(
    tmp_path: Path,
    failure: Exception,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = EmbeddingBinding.model_validate_json(PRIMARY.read_text(encoding="utf-8"))
    settings = TextbookEmbeddingArchiveSettings(
        bundle_dir=_bundle(tmp_path),
        output_dir=tmp_path / "transient-retry-archive",
        retry_max_attempts=3,
        retry_backoff_base_seconds=0.25,
        retry_backoff_max_seconds=0.75,
    )
    attempts: dict[str, int] = {}
    delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(embedding_bundle_module.asyncio, "sleep", fake_sleep)

    async def embedder(
        active_binding: EmbeddingBinding,
        api_key: str,
        texts: list[str],
    ) -> np.ndarray:
        key = texts[0]
        attempts[key] = attempts.get(key, 0) + 1
        if attempts[key] < 3:
            raise failure
        return np.ones((len(texts), active_binding.dimensions), dtype=np.float32)

    summary = await build_textbook_embedding_archive(
        settings,
        binding,
        "primary-secret",
        embedder=embedder,
    )

    assert summary.status == "completed"
    assert sorted(attempts.values()) == [3, 3, 3]
    assert sorted(delays) == [0.25, 0.25, 0.25, 0.5, 0.5, 0.5]


@pytest.mark.asyncio
async def test_embedding_archive_does_not_retry_terminal_quota_failure(
    tmp_path: Path,
) -> None:
    binding = EmbeddingBinding.model_validate_json(PRIMARY.read_text(encoding="utf-8"))
    settings = TextbookEmbeddingArchiveSettings(
        bundle_dir=_bundle(tmp_path),
        output_dir=tmp_path / "terminal-quota-archive",
        retry_max_attempts=4,
        retry_backoff_base_seconds=0,
        retry_backoff_max_seconds=0,
    )
    failure = RuntimeError("insufficient_balance")
    calls = 0

    async def embedder(
        active_binding: EmbeddingBinding,
        api_key: str,
        texts: list[str],
    ) -> np.ndarray:
        del active_binding, api_key, texts
        nonlocal calls
        calls += 1
        raise failure

    with pytest.raises(RuntimeError) as captured:
        await build_textbook_embedding_archive(
            settings,
            binding,
            "primary-secret",
            embedder=embedder,
        )

    assert captured.value is failure
    assert calls == 1
    state = json.loads((settings.output_dir / "embedding-state.json").read_text(encoding="utf-8"))
    assert state["status"] == "failed"
