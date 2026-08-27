from __future__ import annotations

import asyncio
import inspect
import json
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import UUID

import httpx
import pytest
from pydantic import ValidationError

from material_graph.knowledge.extraction import (
    EvidenceFactExtractionPipeline,
    FactExtractionPolicy,
    FactExtractor,
    FactExtractorProviderError,
    FactExtractorRequest,
    InMemoryFactExtractionCheckpointRepository,
    export_extractor_payload_json_schema,
)
from material_graph.knowledge.facts import ExtractionProvenance
from material_graph.knowledge.models import EvidenceFragment, SourceLocator
from material_graph.knowledge.openai_fact_extractor import (
    FileOpenAIAPIKeyProvider,
    InMemoryOpenAIFactTraceSink,
    OpenAIFactExtractor,
    OpenAIFactExtractorBinding,
    OpenAIFactExtractorConfigurationError,
    OpenAIFactExtractorError,
    build_openai_fact_idempotency_key,
)
from material_graph.providers.coordination import (
    AsyncProviderCoordinator,
    InMemoryProviderRuntime,
    ProviderRuntimePolicy,
    SyncProviderCoordinator,
)


FRAGMENT_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
SOURCE_ID = UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
RUNTIME_CREDENTIAL = "runtime-credential-value"


def _binding(**updates: object) -> OpenAIFactExtractorBinding:
    payload: dict[str, object] = {
        "schema_version": 1,
        "provider": "openai_compatible",
        "binding": "responses_json_schema",
        "base_url": "https://llm.example.test/v1",
        "model": "gpt-5.6",
        "model_version": "gpt-5.6-2026-07-27",
        "schema_generation_id": "material-fact-extractor-json-schema-v1",
        "timeouts": {
            "connect_seconds": 5,
            "read_seconds": 120,
            "write_seconds": 30,
            "pool_seconds": 5,
        },
    }
    payload.update(updates)
    return OpenAIFactExtractorBinding.model_validate(payload)


def _request(
    *, text: str = "Material MX-17 had tensile strength 125 MPa at 23 degC."
) -> FactExtractorRequest:
    return FactExtractorRequest(
        fragment_id=FRAGMENT_ID,
        source_id=SOURCE_ID,
        content_sha256=sha256(text.encode("utf-8")).hexdigest(),
        text=text,
        source_uri=f"source://document_data_1/{SOURCE_ID}?page=7&block_index=3",
        evidence_anchor={"page": 7, "block_index": 3},
        extraction=ExtractionProvenance(
            extractor_name="openai-compatible-fact-extractor",
            extractor_version="1.0.0",
            generation_id="gpt-5.6-fact-extraction-v1",
            model_name="gpt-5.6",
            model_version="gpt-5.6-2026-07-27",
        ),
        output_json_schema=export_extractor_payload_json_schema(),
    )


def _fact_payload() -> dict[str, Any]:
    material = {
        "entity_type": "material",
        "canonical_name": "Material MX-17",
        "aliases": ["MX17"],
        "identifiers": {"laboratory_id": "MX-17"},
    }
    return {
        "entities": [material],
        "relations": [],
        "observations": [
            {
                "subject": material,
                "subject_role": "material",
                "property_name": "tensile strength",
                "value": 125.0,
                "unit": "MPa",
                "test_method": "ISO 527",
                "test_conditions": [{"name": "temperature", "value": 23.0, "unit": "degC"}],
                "process_conditions": [],
                "evidence_role": "supports",
                "confidence": 0.93,
                "evidence_quality": "high",
                "assertion_status": "affirmed",
            }
        ],
    }


def _chat_response(
    content: str | None = None,
    *,
    response_id: str = "chatcmpl-provider-response-001",
    response_model: str = "gpt-5.6-2026-07-27",
    finish_reason: str = "stop",
    message_updates: dict[str, object] | None = None,
) -> dict[str, object]:
    message: dict[str, object] = {
        "role": "assistant",
        "content": content if content is not None else json.dumps(_fact_payload()),
    }
    if message_updates:
        message.update(message_updates)
    return {
        "id": response_id,
        "model": response_model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 200},
    }


def _responses_response(
    content: str | None = None,
    *,
    response_id: str = "resp-provider-response-001",
    response_model: str = "gpt-5.6-2026-07-27",
    status: str = "completed",
    top_level: bool = False,
    output: list[object] | None = None,
) -> dict[str, object]:
    rendered = content if content is not None else json.dumps(_fact_payload())
    envelope: dict[str, object] = {
        "id": response_id,
        "model": response_model,
        "status": status,
        "output": (
            output
            if output is not None
            else [
                {
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": rendered}],
                }
            ]
        ),
    }
    if top_level:
        envelope["output_text"] = rendered
    return envelope


class RecordingSecretProvider:
    def __init__(self, value: str = RUNTIME_CREDENTIAL) -> None:
        self.value = value
        self.calls = 0

    def __call__(self) -> str:
        self.calls += 1
        return self.value

    def __repr__(self) -> str:
        return "RecordingSecretProvider()"


def _run(coro: object) -> Any:
    return asyncio.run(coro)  # type: ignore[arg-type]


def test_binding_file_is_nonsecret_pinned_and_loadable() -> None:
    path = Path("config/knowledge/fact-extractor-binding.v1.json")
    raw = path.read_text(encoding="utf-8")
    binding = OpenAIFactExtractorBinding.load(path)

    assert binding.model == "deepseek-v4-flash"
    assert binding.model_version == "deepseek-v4-flash-0731-observed-2026-08-26"
    assert binding.base_url == "https://api.deepseek.com"
    assert binding.binding == "responses_json_schema"
    assert binding.schema_generation_id == "material-fact-extractor-json-schema-v1"
    assert "api_key" not in raw.casefold()
    assert "password" not in raw.casefold()
    assert "credential" not in raw.casefold()


def test_binding_loader_rejects_secrets_duplicate_keys_and_unsafe_urls(tmp_path: Path) -> None:
    default_payload = _binding().model_dump(mode="python")
    default_payload.pop("binding")
    assert (
        OpenAIFactExtractorBinding.model_validate(default_payload).binding
        == "responses_json_schema"
    )
    with pytest.raises(ValidationError):
        _binding(binding="automatic_fallback")

    secret_path = tmp_path / "secret-binding.json"
    secret_path.write_text(
        json.dumps({**_binding().model_dump(mode="json"), "api_key": "redacted"}),
        encoding="utf-8",
    )
    with pytest.raises(OpenAIFactExtractorConfigurationError) as secret_error:
        OpenAIFactExtractorBinding.load(secret_path)
    assert str(secret_error.value) == "openai_fact_extractor.invalid_configuration"

    duplicate_path = tmp_path / "duplicate.json"
    duplicate_path.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    with pytest.raises(OpenAIFactExtractorConfigurationError):
        OpenAIFactExtractorBinding.load(duplicate_path)

    for base_url in (
        "http://llm.example.test/v1",
        "https://user:pass@llm.example.test/v1",
        "https://llm.example.test/v1?token=redacted",
    ):
        with pytest.raises(ValidationError):
            _binding(base_url=base_url)


def test_structured_output_request_sends_only_bounded_text_and_narrow_schema() -> None:
    async def scenario() -> None:
        secret_provider = RecordingSecretProvider()
        trace_sink = InMemoryOpenAIFactTraceSink()
        fact_request = _request()

        async def handler(request: httpx.Request) -> httpx.Response:
            assert request.url == httpx.URL("https://llm.example.test/v1/responses")
            assert request.headers["authorization"] == f"Bearer {RUNTIME_CREDENTIAL}"
            assert request.headers["idempotency-key"] == build_openai_fact_idempotency_key(
                fact_request,
                _binding(),
            )
            payload = json.loads(request.content)
            assert payload["model"] == "gpt-5.6"
            assert payload["stream"] is False
            assert payload["input"] == [
                {
                    "role": "system",
                    "content": OpenAIFactExtractor.SYSTEM_INSTRUCTION,
                },
                {"role": "user", "content": fact_request.text},
            ]
            response_format = payload["text"]["format"]
            assert response_format["type"] == "json_schema"
            assert response_format["strict"] is True
            assert response_format["schema"] == export_extractor_payload_json_schema()
            assert payload["max_output_tokens"] == 8192
            assert "messages" not in payload
            assert "response_format" not in payload
            assert "max_completion_tokens" not in payload
            rendered = json.dumps(payload, ensure_ascii=False)
            assert str(FRAGMENT_ID) not in rendered
            assert str(SOURCE_ID) not in rendered
            assert fact_request.source_uri not in rendered
            assert "evidence_anchor" not in rendered
            assert RUNTIME_CREDENTIAL not in rendered
            return httpx.Response(200, json=_responses_response(top_level=True))

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            extractor = OpenAIFactExtractor(
                http,
                binding=_binding(),
                api_key_provider=secret_provider,
                trace_sink=trace_sink,
            )
            output = await extractor.extract(fact_request)
            rendered_repr = repr(extractor)

        assert output == _fact_payload()
        assert isinstance(extractor, FactExtractor)
        assert secret_provider.calls == 1
        assert RUNTIME_CREDENTIAL not in rendered_repr
        traces = await trace_sink.list_all()
        assert len(traces) == 1
        trace = traces[0]
        assert trace.fragment_id == FRAGMENT_ID
        assert trace.content_sha256 == fact_request.content_sha256
        assert trace.model == "gpt-5.6"
        assert trace.model_version == "gpt-5.6-2026-07-27"
        assert trace.schema_generation_id == "material-fact-extractor-json-schema-v1"
        assert trace.response_id_sha256 == sha256(b"resp-provider-response-001").hexdigest()
        assert "resp-provider-response-001" not in trace.model_dump_json()
        assert RUNTIME_CREDENTIAL not in trace.model_dump_json()

    _run(scenario())


def test_responses_accepts_top_level_output_text_without_nested_message() -> None:
    async def scenario() -> None:
        response = _responses_response(top_level=True, output=[])
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json=response))
        ) as http:
            output = await OpenAIFactExtractor(
                http,
                binding=_binding(),
                api_key_provider=RecordingSecretProvider(),
                trace_sink=InMemoryOpenAIFactTraceSink(),
            ).extract(_request())
        assert output == _fact_payload()

    _run(scenario())


def test_explicit_chat_completions_binding_preserves_legacy_protocol() -> None:
    async def scenario() -> None:
        fact_request = _request()
        binding = _binding(binding="chat_completions_json_schema")

        async def handler(request: httpx.Request) -> httpx.Response:
            assert request.url == httpx.URL("https://llm.example.test/v1/chat/completions")
            payload = json.loads(request.content)
            assert payload["messages"][1] == {"role": "user", "content": fact_request.text}
            assert payload["response_format"]["json_schema"]["strict"] is True
            assert payload["max_tokens"] == 8192
            assert payload["stream"] is False
            assert "input" not in payload
            assert "text" not in payload
            assert "max_output_tokens" not in payload
            assert "max_completion_tokens" not in payload
            return httpx.Response(200, json=_chat_response())

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            output = await OpenAIFactExtractor(
                http,
                binding=binding,
                api_key_provider=RecordingSecretProvider(),
                trace_sink=InMemoryOpenAIFactTraceSink(),
            ).extract(fact_request)
        assert output == _fact_payload()
        assert build_openai_fact_idempotency_key(
            fact_request, binding
        ) != build_openai_fact_idempotency_key(fact_request, _binding())

    _run(scenario())


def test_responses_mode_never_falls_back_to_chat_completions() -> None:
    async def scenario() -> None:
        paths: list[str] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            paths.append(request.url.path)
            return httpx.Response(502)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            extractor = OpenAIFactExtractor(
                http,
                binding=_binding(),
                api_key_provider=RecordingSecretProvider(),
                trace_sink=InMemoryOpenAIFactTraceSink(),
                max_attempts=1,
            )
            with pytest.raises(OpenAIFactExtractorError) as captured:
                await extractor.extract(_request())

        assert captured.value.code == "openai_fact_extractor.unavailable"
        assert paths == ["/v1/responses"]

    _run(scenario())


def test_adapter_runs_inside_review_gated_fact_extraction_pipeline() -> None:
    async def scenario() -> None:
        trace_sink = InMemoryOpenAIFactTraceSink()
        fragment = EvidenceFragment(
            fragment_id=FRAGMENT_ID,
            source_id=SOURCE_ID,
            text="Material MX-17 had tensile strength 125 MPa at 23 degC.",
            locator=SourceLocator(
                root_id="document_data_1",
                relative_path="private/nas/material-paper.pdf",
                page=7,
                block_index=3,
            ),
            retention_reason="active evidence gap",
            parser_name="mineru",
            parser_version="3.4.4",
            embedding_generation_id="qwen3-embedding-v1",
        )
        provenance = ExtractionProvenance(
            extractor_name="openai-compatible-fact-extractor",
            extractor_version="1.0.0",
            generation_id="gpt-5.6-fact-extraction-v1",
            model_name="gpt-5.6",
            model_version="gpt-5.6-2026-07-27",
        )

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json=_responses_response()))
        ) as http:
            adapter = OpenAIFactExtractor(
                http,
                binding=_binding(),
                api_key_provider=RecordingSecretProvider(),
                trace_sink=trace_sink,
            )
            pipeline = EvidenceFactExtractionPipeline(
                extractor=adapter,
                extraction=provenance,
                checkpoints=InMemoryFactExtractionCheckpointRepository(),
                policy=FactExtractionPolicy(
                    max_attempts=1,
                    retry_base_seconds=0,
                    retry_max_seconds=0,
                ),
            )
            result = await pipeline.extract(fragment)

        assert result.review_status == "pending_review"
        assert result.batch.observations[0].unit == "MPa"
        assert result.batch.observations[0].evidence[0].fragment_id == FRAGMENT_ID
        assert len(await trace_sink.list_all()) == 1

    _run(scenario())


def test_429_and_5xx_retry_with_bounded_backoff_then_trace_attempt_count() -> None:
    async def scenario() -> None:
        responses = [
            httpx.Response(429, headers={"Retry-After": "2"}),
            httpx.Response(503),
            httpx.Response(200, json=_responses_response()),
        ]
        calls = 0
        sleeps: list[float] = []
        trace_sink = InMemoryOpenAIFactTraceSink()

        async def handler(_: httpx.Request) -> httpx.Response:
            nonlocal calls
            response = responses[calls]
            calls += 1
            return response

        async def record_sleep(seconds: float) -> None:
            sleeps.append(seconds)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            extractor = OpenAIFactExtractor(
                http,
                binding=_binding(),
                api_key_provider=RecordingSecretProvider(),
                trace_sink=trace_sink,
                max_attempts=3,
                retry_base_seconds=0.25,
                retry_max_seconds=5,
                sleep=record_sleep,
            )
            output = await extractor.extract(_request())

        assert output == _fact_payload()
        assert calls == 3
        assert sleeps == [2.0, 0.5]
        assert (await trace_sink.list_all())[0].attempts == 3

    _run(scenario())


@pytest.mark.parametrize(
    ("status_code", "expected_code"),
    [
        (400, "openai_fact_extractor.request_rejected"),
        (401, "openai_fact_extractor.authentication"),
        (403, "openai_fact_extractor.authentication"),
        (422, "openai_fact_extractor.request_rejected"),
    ],
)
def test_nonretryable_http_errors_are_stable_and_do_not_read_provider_detail(
    status_code: int,
    expected_code: str,
) -> None:
    async def scenario() -> None:
        calls = 0
        marker = "provider-detail-must-not-leak"

        async def handler(_: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(status_code, json={"error": {"message": marker}})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            extractor = OpenAIFactExtractor(
                http,
                binding=_binding(),
                api_key_provider=RecordingSecretProvider(),
                trace_sink=InMemoryOpenAIFactTraceSink(),
                max_attempts=4,
            )
            with pytest.raises(OpenAIFactExtractorError) as captured:
                await extractor.extract(_request())

        assert calls == 1
        assert captured.value.code == expected_code
        assert str(captured.value) == expected_code
        assert marker not in str(captured.value)
        assert captured.value.retryable is False
        assert isinstance(captured.value, FactExtractorProviderError)

    _run(scenario())


@pytest.mark.parametrize(
    ("exception_type", "expected_code"),
    [
        (httpx.ReadTimeout, "openai_fact_extractor.timeout"),
        (httpx.ConnectError, "openai_fact_extractor.transport"),
    ],
)
def test_transport_failures_retry_then_raise_without_detail(
    exception_type: type[httpx.TransportError],
    expected_code: str,
) -> None:
    async def scenario() -> None:
        calls = 0
        sleeps: list[float] = []
        marker = "transport-detail-must-not-leak"

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            raise exception_type(marker, request=request)

        async def record_sleep(seconds: float) -> None:
            sleeps.append(seconds)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            extractor = OpenAIFactExtractor(
                http,
                binding=_binding(),
                api_key_provider=RecordingSecretProvider(),
                trace_sink=InMemoryOpenAIFactTraceSink(),
                max_attempts=2,
                retry_base_seconds=0.25,
                sleep=record_sleep,
            )
            with pytest.raises(OpenAIFactExtractorError) as captured:
                await extractor.extract(_request())

        assert calls == 2
        assert sleeps == [0.25]
        assert captured.value.code == expected_code
        assert captured.value.retryable is True
        assert captured.value.attempts == 2
        assert marker not in str(captured.value)

    _run(scenario())


@pytest.mark.parametrize(
    ("responses", "expected_code"),
    [
        ([httpx.Response(429), httpx.Response(429)], "openai_fact_extractor.rate_limited"),
        ([httpx.Response(500), httpx.Response(503)], "openai_fact_extractor.unavailable"),
    ],
)
def test_retryable_http_exhaustion_is_finite(
    responses: list[httpx.Response],
    expected_code: str,
) -> None:
    async def scenario() -> None:
        calls = 0

        async def handler(_: httpx.Request) -> httpx.Response:
            nonlocal calls
            response = responses[calls]
            calls += 1
            return response

        async def no_wait(_: float) -> None:
            return None

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            extractor = OpenAIFactExtractor(
                http,
                binding=_binding(),
                api_key_provider=RecordingSecretProvider(),
                trace_sink=InMemoryOpenAIFactTraceSink(),
                max_attempts=2,
                sleep=no_wait,
            )
            with pytest.raises(OpenAIFactExtractorError) as captured:
                await extractor.extract(_request())

        assert calls == 2
        assert captured.value.code == expected_code
        assert captured.value.retryable is True
        assert captured.value.attempts == 2

    _run(scenario())


@pytest.mark.parametrize(
    ("response", "expected_code"),
    [
        (httpx.Response(200, content=b"not-json"), "openai_fact_extractor.invalid_response"),
        (httpx.Response(200, json=["not", "object"]), "openai_fact_extractor.invalid_response"),
        (
            httpx.Response(
                200,
                json=_responses_response("```json\n{}\n```"),
            ),
            "openai_fact_extractor.markdown_forbidden",
        ),
        (
            httpx.Response(
                200,
                json=_responses_response(output=[{"type": "function_call", "name": "unsafe-tool"}]),
            ),
            "openai_fact_extractor.tool_calls_forbidden",
        ),
        (
            httpx.Response(200, json=_responses_response("not-json")),
            "openai_fact_extractor.invalid_response",
        ),
        (
            httpx.Response(200, json=_responses_response(status="incomplete")),
            "openai_fact_extractor.incomplete",
        ),
        (
            httpx.Response(200, json=_responses_response(status="failed")),
            "openai_fact_extractor.incomplete",
        ),
        (
            httpx.Response(
                200,
                json=_responses_response(
                    output=[
                        {
                            "type": "message",
                            "status": "completed",
                            "role": "assistant",
                            "content": [{"type": "refusal", "refusal": "not allowed"}],
                        }
                    ]
                ),
            ),
            "openai_fact_extractor.refused",
        ),
        (
            httpx.Response(
                200,
                json={
                    **_responses_response(top_level=True),
                    "output_text": '{"entities":[],"relations":[],"observations":[]}',
                },
            ),
            "openai_fact_extractor.invalid_response",
        ),
        (
            httpx.Response(
                200,
                json=_responses_response(output=[{"type": "unknown_output"}]),
            ),
            "openai_fact_extractor.invalid_response",
        ),
    ],
)
def test_unsafe_or_nonjson_responses_are_rejected_without_retry(
    response: httpx.Response,
    expected_code: str,
) -> None:
    async def scenario() -> None:
        calls = 0

        async def handler(_: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return response

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            extractor = OpenAIFactExtractor(
                http,
                binding=_binding(),
                api_key_provider=RecordingSecretProvider(),
                trace_sink=InMemoryOpenAIFactTraceSink(),
                max_attempts=5,
            )
            with pytest.raises(OpenAIFactExtractorError) as captured:
                await extractor.extract(_request())

        assert calls == 1
        assert captured.value.code == expected_code
        assert captured.value.retryable is False

    _run(scenario())


def test_duplicate_keys_nonfinite_content_and_multiple_messages_are_rejected() -> None:
    duplicate_content = '{"entities":[],"entities":[],"relations":[],"observations":[]}'
    nonfinite_content = '{"entities":[],"relations":[],"observations":[NaN]}'
    multiple_messages = _responses_response()
    multiple_messages["output"] = [
        *multiple_messages["output"],  # type: ignore[misc]
        deepcopy(multiple_messages["output"][0]),  # type: ignore[index]
    ]
    responses = [
        httpx.Response(200, json=_responses_response(duplicate_content)),
        httpx.Response(200, json=_responses_response(nonfinite_content)),
        httpx.Response(200, json=multiple_messages),
    ]

    async def scenario(response: httpx.Response) -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: response)) as http:
            extractor = OpenAIFactExtractor(
                http,
                binding=_binding(),
                api_key_provider=RecordingSecretProvider(),
                trace_sink=InMemoryOpenAIFactTraceSink(),
            )
            with pytest.raises(OpenAIFactExtractorError, match="invalid_response"):
                await extractor.extract(_request())

    for response in responses:
        _run(scenario(response))


def test_response_stream_size_limit_uses_content_length_and_decoded_chunks() -> None:
    async def scenario(response: httpx.Response) -> OpenAIFactExtractorError:
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: response)) as http:
            extractor = OpenAIFactExtractor(
                http,
                binding=_binding(),
                api_key_provider=RecordingSecretProvider(),
                trace_sink=InMemoryOpenAIFactTraceSink(),
                max_response_bytes=1024,
            )
            with pytest.raises(OpenAIFactExtractorError) as captured:
                await extractor.extract(_request())
        return captured.value

    declared = httpx.Response(
        200,
        headers={"Content-Length": "5000"},
        content=b"{}",
    )
    streamed = httpx.Response(200, content=b"x" * 2048)

    for response in (declared, streamed):
        error = _run(scenario(response))
        assert error.code == "openai_fact_extractor.response_too_large"
        assert error.retryable is False


def test_request_text_and_schema_are_validated_before_secret_or_network() -> None:
    async def scenario(request: FactExtractorRequest, *, max_input_bytes: int) -> None:
        calls = 0
        secret_provider = RecordingSecretProvider()

        async def handler(_: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(200, json=_responses_response())

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            extractor = OpenAIFactExtractor(
                http,
                binding=_binding(),
                api_key_provider=secret_provider,
                trace_sink=InMemoryOpenAIFactTraceSink(),
                max_input_bytes=max_input_bytes,
            )
            with pytest.raises(OpenAIFactExtractorError) as captured:
                await extractor.extract(request)

        assert captured.value.code == "openai_fact_extractor.invalid_request"
        assert calls == 0
        assert secret_provider.calls == 0

    oversized = _request(text="x" * 1025)
    _run(scenario(oversized, max_input_bytes=1024))

    schema_payload = _request().model_dump(mode="python")
    schema_payload["output_json_schema"] = {
        **schema_payload["output_json_schema"],
        "title": "provider-controlled-schema",
    }
    changed_schema = FactExtractorRequest.model_validate(schema_payload)
    _run(scenario(changed_schema, max_input_bytes=4096))


def test_secret_provider_failure_or_blank_value_is_stable_and_pretransport() -> None:
    class FailingProvider:
        def __call__(self) -> str:
            raise RuntimeError("vault-detail-must-not-leak")

    async def scenario(provider: object) -> OpenAIFactExtractorError:
        calls = 0

        async def handler(_: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(200, json=_responses_response())

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            extractor = OpenAIFactExtractor(
                http,
                binding=_binding(),
                api_key_provider=provider,  # type: ignore[arg-type]
                trace_sink=InMemoryOpenAIFactTraceSink(),
            )
            with pytest.raises(OpenAIFactExtractorError) as captured:
                await extractor.extract(_request())
        assert calls == 0
        return captured.value

    for provider in (FailingProvider(), RecordingSecretProvider(" ")):
        error = _run(scenario(provider))
        assert error.code == "openai_fact_extractor.credential_unavailable"
        assert "vault-detail" not in str(error)
        assert error.retryable is False


def test_file_api_key_provider_reads_only_small_single_secret_and_hides_path(
    tmp_path: Path,
) -> None:
    secret_path = tmp_path / "openai-secret.txt"
    secret_path.write_text(f"{RUNTIME_CREDENTIAL}\n", encoding="utf-8")
    provider = FileOpenAIAPIKeyProvider(secret_path)

    assert provider() == RUNTIME_CREDENTIAL
    assert RUNTIME_CREDENTIAL not in repr(provider)
    assert str(secret_path) not in repr(provider)

    secret_path.write_text("first\nsecond", encoding="utf-8")
    with pytest.raises(ValueError):
        provider()

    secret_path.write_text("x" * 20_000, encoding="utf-8")
    with pytest.raises(ValueError):
        provider()


def test_constructor_has_no_raw_api_key_parameter_and_rejects_unsafe_limits() -> None:
    signature = inspect.signature(OpenAIFactExtractor)
    assert "api_key" not in signature.parameters

    async def scenario() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json=_responses_response()))
        ) as http:
            base: dict[str, object] = {
                "http": http,
                "binding": _binding(),
                "api_key_provider": RecordingSecretProvider(),
                "trace_sink": InMemoryOpenAIFactTraceSink(),
            }
            for update in (
                {"max_attempts": 0},
                {"retry_base_seconds": -1},
                {"retry_max_seconds": -1},
                {"max_input_bytes": 100},
                {"max_response_bytes": 100},
                {"max_completion_tokens": 0},
            ):
                with pytest.raises(ValueError):
                    OpenAIFactExtractor(**{**base, **update})  # type: ignore[arg-type]

    _run(scenario())


def test_trace_sink_is_idempotent_conflict_safe_and_defensive() -> None:
    async def scenario() -> None:
        sink = InMemoryOpenAIFactTraceSink()

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json=_responses_response()))
        ) as http:
            extractor = OpenAIFactExtractor(
                http,
                binding=_binding(),
                api_key_provider=RecordingSecretProvider(),
                trace_sink=sink,
            )
            await extractor.extract(_request())
            await extractor.extract(_request())

        first = await sink.list_all()
        second = await sink.list_all()
        assert len(first) == 1
        assert first == second
        assert first is not second

    _run(scenario())


def test_trace_sink_failure_is_stable_and_does_not_return_untraced_output() -> None:
    class FailingSink:
        async def record(self, trace: object) -> None:
            del trace
            raise RuntimeError("trace-store-detail-must-not-leak")

    async def scenario() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json=_responses_response()))
        ) as http:
            extractor = OpenAIFactExtractor(
                http,
                binding=_binding(),
                api_key_provider=RecordingSecretProvider(),
                trace_sink=FailingSink(),
            )
            with pytest.raises(OpenAIFactExtractorError) as captured:
                await extractor.extract(_request())

        assert captured.value.code == "openai_fact_extractor.trace_unavailable"
        assert captured.value.retryable is True
        assert "trace-store-detail" not in str(captured.value)

    _run(scenario())


def test_response_identifier_and_model_are_hashed_not_persisted() -> None:
    async def scenario() -> None:
        response_id = "provider-response-private-identifier"
        response_model = "provider-deployment-private-name"
        trace_sink = InMemoryOpenAIFactTraceSink()

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    200,
                    json=_responses_response(
                        response_id=response_id,
                        response_model=response_model,
                    ),
                )
            )
        ) as http:
            extractor = OpenAIFactExtractor(
                http,
                binding=_binding(),
                api_key_provider=RecordingSecretProvider(),
                trace_sink=trace_sink,
            )
            await extractor.extract(_request())

        rendered = (await trace_sink.list_all())[0].model_dump_json()
        assert response_id not in rendered
        assert response_model not in rendered
        assert sha256(response_id.encode()).hexdigest() in rendered
        assert sha256(response_model.encode()).hexdigest() in rendered

    _run(scenario())


def test_invalid_retry_after_falls_back_to_exponential_delay() -> None:
    async def scenario() -> None:
        calls = 0
        sleeps: list[float] = []

        async def handler(_: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            if calls == 1:
                return httpx.Response(429, headers={"Retry-After": "not-a-date"})
            return httpx.Response(200, json=_responses_response())

        async def record_sleep(seconds: float) -> None:
            sleeps.append(seconds)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            extractor = OpenAIFactExtractor(
                http,
                binding=_binding(),
                api_key_provider=RecordingSecretProvider(),
                trace_sink=InMemoryOpenAIFactTraceSink(),
                retry_base_seconds=0.25,
                sleep=record_sleep,
            )
            await extractor.extract(_request())

        assert sleeps == [0.25]

    _run(scenario())


def test_fact_extractor_uses_bulk_lease_per_attempt_and_releases_before_backoff() -> None:
    async def scenario() -> None:
        runtime = InMemoryProviderRuntime()
        policy = ProviderRuntimePolicy(
            total_slots=8,
            reserved_interactive_slots=2,
            bulk_initial_slots=4,
            bulk_hard_max=6,
            requests_per_minute=100,
            tokens_per_minute=100_000,
            reserved_interactive_requests=2,
            reserved_interactive_tokens=100,
        )
        coordinator = AsyncProviderCoordinator(
            runtime.async_store(),
            provider_scope="provider:fact-test",
            policy=policy,
            owner_id="worker:fact-test",
        )
        observer = SyncProviderCoordinator(
            runtime.sync_store(),
            provider_scope="provider:fact-test",
            policy=policy,
            owner_id="api:fact-observer",
        )
        responses = [
            httpx.Response(429, headers={"Retry-After": "0.01"}),
            httpx.Response(200, json=_responses_response()),
        ]
        active_during_http: list[int] = []

        async def handler(_: httpx.Request) -> httpx.Response:
            active_during_http.append(observer.snapshot()["active_bulk"])
            return responses.pop(0)

        sleeps: list[float] = []

        async def backoff(seconds: float) -> None:
            assert observer.snapshot()["active_bulk"] == 0
            sleeps.append(seconds)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            extractor = OpenAIFactExtractor(
                http,
                binding=_binding(),
                api_key_provider=RecordingSecretProvider(),
                trace_sink=InMemoryOpenAIFactTraceSink(),
                coordinator=coordinator,
                sleep=backoff,
            )
            output = await extractor.extract(_request())

        assert output == _fact_payload()
        assert active_during_http == [1, 1]
        assert sleeps == [0.01]
        assert observer.snapshot()["active_bulk"] == 0
        assert observer.snapshot()["bulk_limit"] == 2

    _run(scenario())
