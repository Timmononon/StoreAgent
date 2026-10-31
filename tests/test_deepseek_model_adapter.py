from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Literal

import httpx
import pytest
from openai import BadRequestError
from pydantic import BaseModel, ValidationError

from backend.agent.models import ModelRegistry
from backend.core.settings import ModelSettings
from backend.model_control import (
    ModelRole,
    ModelRuntimeSpec,
    build_model_catalog_snapshot,
)
from backend.providers import (
    ProviderCallContext,
    ProviderCode,
    ProviderError,
    ProviderExecutor,
    ProviderOperation,
    ProviderPolicy,
)


class ProbeDecision(BaseModel):
    status: Literal["ok"]


@pytest.fixture
async def model_transport(monkeypatch):
    requests = []
    response_state = {"status": "ok", "reject": False, "empty_responses": 0}
    created_models = []

    def reject_network(*args, **kwargs):
        raise AssertionError("This regression test must not use the network")

    async def reject_async_network(*args, **kwargs):
        raise AssertionError("This regression test must not use the network")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", reject_network)
    monkeypatch.setattr(
        httpx.AsyncHTTPTransport, "handle_async_request", reject_async_network
    )

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        deepseek = request.url.host == "api.deepseek.com"
        incompatible = deepseek and (
            body.get("response_format", {}).get("type") == "json_schema"
            or body.get("thinking") != {"type": "disabled"}
        )
        if incompatible or response_state["reject"]:
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": "Unsupported request",
                        "type": "invalid_request_error",
                    }
                },
            )
        message = {"role": "assistant", "content": "OK"}
        finish_reason = "stop"
        if response_state["empty_responses"]:
            response_state["empty_responses"] -= 1
        elif body.get("tools"):
            name = body["tools"][0]["function"]["name"]
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_probe",
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(
                                {"status": response_state["status"]}
                            ),
                        },
                    }
                ],
            }
            finish_reason = "tool_calls"
        elif body.get("response_format"):
            message["content"] = json.dumps({"status": response_state["status"]})
        envelope = {"id": "chatcmpl-offline", "model": "deepseek-flash", "created": 0}
        if body.get("stream"):
            chunk = {
                **envelope,
                "object": "chat.completion.chunk",
                "choices": [
                    {"index": 0, "delta": message, "finish_reason": finish_reason}
                ],
            }
            return httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream"},
                content=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n",
            )
        return httpx.Response(
            200,
            json={
                **envelope,
                "object": "chat.completion",
                "choices": [
                    {"index": 0, "message": message, "finish_reason": finish_reason}
                ],
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(respond)
        ) as async_client:

            def create_model(
                role=ModelRole.FAST, base_url="https://api.deepseek.com", snapshot=None
            ):
                settings = SimpleNamespace(
                    models=ModelSettings(
                        _env_file=None,
                        ARK_API_KEY="offline-test-key",
                        BASE_URL=base_url,
                        MODEL="deepseek-flash",
                        FAST_MODEL="deepseek-flash",
                        GRADE_MODEL="deepseek-flash",
                        EVALUATION_MODEL="deepseek-flash",
                        MODEL_TIMEOUT_SECONDS=9.5,
                    )
                )
                original = ModelRegistry(settings=settings).get(role, snapshot=snapshot)
                root_client = original.root_client.with_options(http_client=client)
                root_async_client = original.root_async_client.with_options(
                    http_client=async_client
                )
                model = original.model_copy(
                    update={
                        "root_client": root_client,
                        "root_async_client": root_async_client,
                        "client": root_client.chat.completions,
                        "async_client": root_async_client.chat.completions,
                    }
                )
                created_models.append(model)
                return model

            try:
                yield create_model, requests, response_state
            finally:
                for model in created_models:
                    model.root_client.close()
                    await model.root_async_client.close()


@pytest.mark.parametrize(
    "role", [ModelRole.FAST, ModelRole.GRADER, ModelRole.EVALUATOR]
)
async def test_deepseek_roles_return_validated_structured_output(model_transport, role):
    create_model, requests, _ = model_transport
    model = create_model(role)

    result = await model.with_structured_output(ProbeDecision).ainvoke(
        "Set status to ok."
    )

    assert result == ProbeDecision(status="ok")
    assert len(requests) == 1
    assert "response_format" not in requests[0]
    assert requests[0]["thinking"] == {"type": "disabled"}
    assert requests[0]["tool_choice"]["function"]["name"] == "ProbeDecision"
    assert model.root_client.max_retries == 0
    assert model.root_async_client.max_retries == 0
    assert model.request_timeout == 9.5


async def test_deepseek_answer_stream_uses_nonthinking_mode(model_transport):
    create_model, requests, _ = model_transport
    model = create_model(ModelRole.ANSWER)

    chunks = [chunk async for chunk in model.astream("Say OK.")]

    assert "".join(chunk.content for chunk in chunks) == "OK"
    assert requests[0]["thinking"] == {"type": "disabled"}
    assert requests[0]["stream"] is True


async def test_deepseek_sync_rag_calls_use_the_same_validated_protocol(model_transport):
    create_model, requests, _ = model_transport
    structured = create_model().with_structured_output(ProbeDecision)

    result = await asyncio.to_thread(structured.invoke, "Set status to ok.")

    assert result == ProbeDecision(status="ok")
    assert len(requests) == 1
    assert requests[0]["thinking"] == {"type": "disabled"}
    assert "response_format" not in requests[0]


async def test_deepseek_agent_can_bind_and_force_tools(model_transport):
    create_model, requests, _ = model_transport
    model = create_model(ModelRole.ANSWER).bind_tools(
        [ProbeDecision], tool_choice="ProbeDecision"
    )

    result = await model.ainvoke("Set status to ok.")

    assert result.tool_calls[0]["args"] == {"status": "ok"}
    assert requests[0]["thinking"] == {"type": "disabled"}
    assert requests[0]["tool_choice"]["function"]["name"] == "ProbeDecision"


async def test_deepseek_keeps_schema_validation_and_does_not_retry_bad_output(
    model_transport,
):
    create_model, requests, response_state = model_transport
    response_state["status"] = "invalid"

    with pytest.raises(ValidationError):
        await (
            create_model().with_structured_output(ProbeDecision).ainvoke("Set status.")
        )

    assert len(requests) == 1


async def test_missing_structured_tool_output_is_rejected_without_adapter_retry(
    model_transport,
):
    create_model, requests, response_state = model_transport
    response_state["empty_responses"] = 1
    with pytest.raises(ValidationError):
        await (
            create_model().with_structured_output(ProbeDecision).ainvoke("Set status.")
        )
    assert len(requests) == 1


@pytest.mark.parametrize("empty_responses", [1, 2])
async def test_missing_structured_output_uses_only_the_provider_retry_owner(
    model_transport, empty_responses
):
    create_model, requests, response_state = model_transport
    response_state["empty_responses"] = empty_responses
    structured = create_model().with_structured_output(ProbeDecision)
    executor = ProviderExecutor()
    context = ProviderCallContext(
        provider="deepseek", operation=ProviderOperation.MODEL
    )
    policy = ProviderPolicy(
        max_attempts=2, initial_backoff_seconds=0, max_backoff_seconds=0
    )
    if empty_responses == 1:
        result = await executor.acall(
            lambda: structured.ainvoke("Set status."), context=context, policy=policy
        )
        assert result == ProbeDecision(status="ok")
    else:
        with pytest.raises(ProviderError) as failure:
            await executor.acall(
                lambda: structured.ainvoke("Set status."),
                context=context,
                policy=policy,
            )
        assert failure.value.code == ProviderCode.MODEL_UNAVAILABLE
        assert failure.value.attempts == 2
    assert len(requests) == 2


async def test_deepseek_does_not_switch_protocol_or_retry_bad_requests(model_transport):
    create_model, requests, response_state = model_transport
    response_state["reject"] = True

    with pytest.raises(BadRequestError):
        await (
            create_model().with_structured_output(ProbeDecision).ainvoke("Set status.")
        )

    assert len(requests) == 1


@pytest.mark.parametrize(
    "base_url",
    [
        "https://models.example.test/v1",
        "https://api.deepseek.com.example.test/v1",
        "https://models.example.test/deepseek",
    ],
)
async def test_other_endpoints_keep_openai_structured_output(model_transport, base_url):
    create_model, requests, _ = model_transport

    result = (
        await create_model(base_url=base_url)
        .with_structured_output(ProbeDecision)
        .ainvoke("Set status.")
    )

    assert result == ProbeDecision(status="ok")
    assert requests[0]["response_format"]["type"] == "json_schema"
    assert "thinking" not in requests[0]


async def test_adapter_selection_uses_frozen_snapshot_endpoint(model_transport):
    create_model, requests, _ = model_transport
    snapshot = build_model_catalog_snapshot(
        {
            ModelRole.FAST: ModelRuntimeSpec(
                profile_id="model_" + "1" * 32,
                profile_version=1,
                display_name="DeepSeek",
                model_name="deepseek-flash",
                base_url="https://api.deepseek.com/v1",
                timeout_seconds=9.5,
                supports_stream=True,
                supports_structured_output=True,
            ),
        }
    )

    model = create_model(base_url="https://models.example.test/v1", snapshot=snapshot)
    result = await model.with_structured_output(ProbeDecision).ainvoke("Set status.")

    assert result == ProbeDecision(status="ok")
    assert requests[0]["thinking"] == {"type": "disabled"}
