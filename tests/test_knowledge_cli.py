from __future__ import annotations

import json
from pathlib import Path

import pytest

from material_graph.knowledge.cli import main


ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "config/knowledge/corpus-policy.v1.json"
EMBEDDING = ROOT / "config/knowledge/embedding-binding.v1.json"
RERANKER = ROOT / "config/knowledge/reranker-binding.v1.json"
RUNTIME_ENV = ROOT / "deploy/config/ingestion-runtime.env"


def _run(arguments: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str, str]:
    status = main(arguments)
    captured = capsys.readouterr()
    return status, captured.out, captured.err


def _runtime_arguments(environment: Path) -> list[str]:
    return [
        "verify-runtime",
        "--env",
        str(environment),
        "--embedding",
        str(EMBEDDING),
        "--reranker",
        str(RERANKER),
    ]


def test_verify_policy_emits_path_free_json_summary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    status, stdout, stderr = _run(
        ["verify-policy", "--config", str(POLICY)],
        capsys,
    )

    summary = json.loads(stdout)
    assert status == 0
    assert stderr == ""
    assert summary == {
        "derived_data_hard_cap_bytes": 85_899_345_920,
        "schema_version": 1,
        "server_capacity_gb": 130.0,
        "source_count": 3,
        "source_ids": ["document_data_1", "data_2", "data_3"],
        "spool": {
            "max_active_objects": 4,
            "max_object_bytes": 1_073_741_824,
            "max_total_bytes": 8_589_934_592,
        },
        "status": "valid",
    }
    assert str(POLICY) not in stdout
    assert "password" not in stdout.casefold()


def test_verify_bindings_emits_only_safe_provider_summary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    status, stdout, stderr = _run(
        [
            "verify-bindings",
            "--embedding",
            str(EMBEDDING),
            "--reranker",
            str(RERANKER),
        ],
        capsys,
    )

    summary = json.loads(stdout)
    assert status == 0
    assert stderr == ""
    assert summary["status"] == "valid"
    assert summary["embedding"] == {
        "batch_size": 32,
        "dimensions": 1024,
        "generation_id": "glm-embedding-3-1024-halfvec-v1",
        "max_async": 32,
        "model": "embedding-3",
        "provider": "glm_openai_compatible",
    }
    assert summary["reranker"]["model"] == "Qwen/Qwen3-Reranker-8B"
    assert summary["reranker"]["max_async"] == 8
    assert "https://" not in stdout
    assert str(EMBEDDING) not in stdout


def test_verify_runtime_checks_storage_and_frozen_bindings(
    capsys: pytest.CaptureFixture[str],
) -> None:
    status, stdout, stderr = _run(_runtime_arguments(RUNTIME_ENV), capsys)

    summary = json.loads(stdout)
    assert status == 0
    assert stderr == ""
    assert summary == {
        "embedding": {
            "dimensions": 1024,
            "generation_id": "glm-embedding-3-1024-halfvec-v1",
            "max_async": 32,
            "model": "embedding-3",
        },
        "reranker": {"max_async": 8, "model": "Qwen/Qwen3-Reranker-8B"},
        "status": "valid",
        "storage": {
            "doc_status": "PGDocStatusStorage",
            "graph": "PGGraphStorage",
            "kv": "PGKVStorage",
            "vector": "PGVectorStorage",
        },
        "workspace": "glm-embedding-3-1024-halfvec-v1",
    }
    assert str(RUNTIME_ENV) not in stdout
    assert "siliconflow.cn" not in stdout


def test_runtime_binding_mismatch_uses_stable_code_without_echo(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret_marker = "unsafe-model-value-must-not-be-echoed"
    environment = RUNTIME_ENV.read_text(encoding="utf-8").replace(
        "EMBEDDING_MODEL=embedding-3",
        f"EMBEDDING_MODEL={secret_marker}",
    )
    path = tmp_path / "mismatch.env"
    path.write_text(environment, encoding="utf-8")

    status, stdout, stderr = _run(_runtime_arguments(path), capsys)

    assert status == 2
    assert stdout == ""
    assert stderr == "error:runtime_binding_mismatch\n"
    assert secret_marker not in stderr
    assert str(path) not in stderr


@pytest.mark.parametrize(
    ("extra_line", "expected_code"),
    [
        ("OPENAI_API_KEY=do-not-echo-key-value", "env_credential_key"),
        ("SYNOLOGY_DID=do-not-echo-device-token", "env_credential_key"),
        ("SYNOLOGY_DEVICE_ID=do-not-echo-device-token", "env_credential_key"),
        ("SAFE_SETTING=sk-do-not-echo-value", "env_credential_value"),
        ("SAFE_SETTING=Mixed12345!", "env_credential_value"),
        ("WORKSPACE=duplicate-do-not-echo", "env_duplicate_key"),
        ("MALFORMED LINE do-not-echo", "env_malformed"),
        ('SAFE_SETTING="unterminated-do-not-echo', "env_malformed"),
    ],
)
def test_dotenv_failures_never_echo_input(
    extra_line: str,
    expected_code: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "unsafe.env"
    path.write_text(
        RUNTIME_ENV.read_text(encoding="utf-8") + f"\n{extra_line}\n",
        encoding="utf-8",
    )

    status, stdout, stderr = _run(_runtime_arguments(path), capsys)

    assert status == 2
    assert stdout == ""
    assert stderr == f"error:{expected_code}\n"
    assert "do-not-echo" not in stderr
    assert str(path) not in stderr


def test_quoted_non_secret_dotenv_values_are_supported(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    environment = RUNTIME_ENV.read_text(encoding="utf-8").replace(
        "WORKSPACE=glm-embedding-3-1024-halfvec-v1",
        'WORKSPACE="glm-embedding-3-1024-halfvec-v1"',
    )
    path = tmp_path / "quoted.env"
    path.write_text(environment, encoding="utf-8")

    status, stdout, stderr = _run(_runtime_arguments(path), capsys)

    assert status == 0
    assert json.loads(stdout)["status"] == "valid"
    assert stderr == ""


def test_invalid_policy_and_binding_files_use_safe_codes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "invalid-content-do-not-echo"
    invalid = tmp_path / "invalid.json"
    invalid.write_text(secret, encoding="utf-8")

    status, stdout, stderr = _run(
        ["verify-policy", "--config", str(invalid)],
        capsys,
    )
    assert (status, stdout, stderr) == (2, "", "error:policy_invalid\n")

    status, stdout, stderr = _run(
        [
            "verify-bindings",
            "--embedding",
            str(invalid),
            "--reranker",
            str(RERANKER),
        ],
        capsys,
    )
    assert (status, stdout, stderr) == (2, "", "error:bindings_invalid\n")
    assert secret not in stderr


def test_missing_storage_and_bad_usage_return_stable_codes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    environment = RUNTIME_ENV.read_text(encoding="utf-8").replace(
        "LIGHTRAG_KV_STORAGE=PGKVStorage\n",
        "",
    )
    path = tmp_path / "missing-storage.env"
    path.write_text(environment, encoding="utf-8")

    status, stdout, stderr = _run(_runtime_arguments(path), capsys)
    assert (status, stdout, stderr) == (2, "", "error:runtime_storage_invalid\n")

    status, stdout, stderr = _run(["verify-policy"], capsys)
    assert (status, stdout, stderr) == (2, "", "error:usage_invalid\n")


def test_unreadable_environment_and_workspace_mismatch_are_safe(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing = tmp_path / "missing.env"
    status, stdout, stderr = _run(_runtime_arguments(missing), capsys)
    assert (status, stdout, stderr) == (2, "", "error:env_unreadable\n")

    environment = RUNTIME_ENV.read_text(encoding="utf-8").replace(
        "WORKSPACE=glm-embedding-3-1024-halfvec-v1",
        "WORKSPACE=wrong-generation",
    )
    path = tmp_path / "workspace.env"
    path.write_text(environment, encoding="utf-8")
    status, stdout, stderr = _run(_runtime_arguments(path), capsys)
    assert (status, stdout, stderr) == (2, "", "error:runtime_binding_mismatch\n")


def _write_textbook_fixture(root: Path) -> None:
    stem = "聚合物加工__part01"
    path = root / "source_hu" / "第1批" / stem / f"{stem}.md"
    path.parent.mkdir(parents=True)
    path.write_text(
        "# 聚合物加工__part01\n\n"
        "<!-- PAGE 5 -->\n\n"
        "### 挤出成形\n\n"
        + "挤出温度、螺杆转速与熔体压力共同决定制品质量。" * 30,
        encoding="utf-8",
    )


def test_audit_textbook_corpus_emits_counts_without_source_paths(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    corpus = tmp_path / "private-corpus-name"
    _write_textbook_fixture(corpus)

    status, stdout, stderr = _run(
        ["audit-textbook-corpus", "--root", str(corpus)],
        capsys,
    )

    summary = json.loads(stdout)
    assert status == 0
    assert stderr == ""
    assert summary["status"] == "valid"
    assert summary["discovered_documents"] == 1
    assert summary["unique_documents"] == 1
    assert summary["duplicate_documents"] == 0
    assert summary["logical_books"] == 1
    assert summary["fragments"] >= 1
    assert summary["source_families"] == {"source_hu_markdown": 1}
    assert str(corpus) not in stdout
    assert "private-corpus-name" not in stdout


def test_prepare_textbook_corpus_writes_atomic_jsonl_with_bound_generation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    corpus = tmp_path / "corpus"
    output = tmp_path / "runtime" / "fragments.jsonl"
    _write_textbook_fixture(corpus)

    status, stdout, stderr = _run(
        [
            "prepare-textbook-corpus",
            "--root",
            str(corpus),
            "--output",
            str(output),
            "--embedding",
            str(EMBEDDING),
        ],
        capsys,
    )

    summary = json.loads(stdout)
    lines = output.read_text(encoding="utf-8").splitlines()
    assert status == 0
    assert stderr == ""
    assert summary["status"] == "prepared"
    assert summary["embedding_generation_id"] == "glm-embedding-3-1024-halfvec-v1"
    assert summary["fragments"] == len(lines)
    assert all(
        json.loads(line)["embedding_generation_id"]
        == "glm-embedding-3-1024-halfvec-v1"
        for line in lines
    )
    assert not list(output.parent.glob("*.tmp-*"))
    assert str(corpus) not in stdout
    assert str(output) not in stdout


def test_textbook_cli_failures_are_stable_and_do_not_write_into_sources(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing = tmp_path / "missing-private-name"
    status, stdout, stderr = _run(
        ["audit-textbook-corpus", "--root", str(missing)],
        capsys,
    )
    assert (status, stdout, stderr) == (2, "", "error:corpus_invalid\n")
    assert str(missing) not in stderr

    corpus = tmp_path / "corpus"
    _write_textbook_fixture(corpus)
    unsafe_output = corpus / "generated" / "fragments.jsonl"
    status, stdout, stderr = _run(
        [
            "prepare-textbook-corpus",
            "--root",
            str(corpus),
            "--output",
            str(unsafe_output),
            "--embedding",
            str(EMBEDDING),
        ],
        capsys,
    )
    assert (status, stdout, stderr) == (2, "", "error:output_invalid\n")
    assert not unsafe_output.exists()
