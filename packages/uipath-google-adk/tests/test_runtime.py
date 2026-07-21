"""Tests for runtime/runtime.py — UiPathGoogleADKRuntime."""

import json
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
from google.adk.agents import BaseAgent, LlmAgent
from google.adk.events.event import Event
from google.adk.sessions.session import Session
from google.genai import types
from pydantic import BaseModel
from uipath.runtime import UiPathRuntimeResult, UiPathRuntimeStatus
from uipath.runtime.errors import UiPathErrorCode
from uipath.runtime.events import UiPathRuntimeStateEvent

from uipath_google_adk.runtime.errors import (
    UiPathGoogleADKErrorCode,
    UiPathGoogleADKRuntimeError,
)
from uipath_google_adk.runtime.runtime import UiPathGoogleADKRuntime


class OutModel(BaseModel):
    answer: str


def _make_runtime(
    agent: Optional[BaseAgent] = None,
    session_state: Optional[dict[str, Any]] = None,
) -> UiPathGoogleADKRuntime:
    agent = agent or LlmAgent(name="asst", model="gemini-2.0-flash")
    session = Session(
        id="sess-1",
        app_name=UiPathGoogleADKRuntime.APP_NAME,
        user_id=UiPathGoogleADKRuntime.USER_ID,
        state=session_state or {},
    )
    runner = MagicMock()
    session_service = MagicMock()
    session_service.get_session = AsyncMock(return_value=session)
    return UiPathGoogleADKRuntime(
        agent=agent,
        runner=runner,
        session=session,
        session_service=session_service,
        runtime_id="rt-1",
        entrypoint="asst",
    )


class TestPrepareUserMessage:
    def test_no_input_yields_empty_text(self):
        rt = _make_runtime()
        msg = rt._prepare_user_message(None)
        assert msg.parts is not None
        assert msg.role == "user" and msg.parts[0].text == ""

    def test_string_messages(self):
        rt = _make_runtime()
        msg = rt._prepare_user_message({"messages": "hello"})
        assert msg.parts is not None
        assert msg.parts[0].text == "hello"

    def test_typed_input_serialized_as_json(self):
        rt = _make_runtime()
        msg = rt._prepare_user_message({"query": "x", "limit": 5})
        assert msg.parts is not None
        assert json.loads(msg.parts[0].text or "") == {"query": "x", "limit": 5}


class TestExtractOutput:
    def test_output_key_from_session_state(self):
        agent = LlmAgent(name="a", model="gemini-2.0-flash", output_key="result")
        rt = _make_runtime(agent=agent, session_state={"result": {"answer": "42"}})
        assert rt._extract_output("ignored") == {"answer": "42"}

    def test_output_schema_parses_final_text(self):
        agent = LlmAgent(name="a", model="gemini-2.0-flash", output_schema=OutModel)
        rt = _make_runtime(agent=agent)
        assert rt._extract_output('{"answer": "hi"}') == {"answer": "hi"}

    def test_falls_back_to_raw_text(self):
        rt = _make_runtime()
        assert rt._extract_output("plain") == "plain"


class TestCreateSuccessResult:
    def test_dict_output_preserved(self):
        rt = _make_runtime()
        result = rt._create_success_result({"answer": "x"})
        assert result.status == UiPathRuntimeStatus.SUCCESSFUL
        assert result.output == {"answer": "x"}

    def test_non_dict_output_wrapped_in_messages(self):
        rt = _make_runtime()
        result = rt._create_success_result("plain text")
        assert isinstance(result.output, dict)
        assert result.output["messages"][0]["role"] == "assistant"


class TestCreateRuntimeError:
    def test_passthrough_existing_error(self):
        rt = _make_runtime()
        original = UiPathGoogleADKRuntimeError(
            UiPathGoogleADKErrorCode.AGENT_LOAD_ERROR, "t", "d"
        )
        assert rt._create_runtime_error(original) is original

    def test_json_decode_error_mapped(self):
        rt = _make_runtime()
        err = rt._create_runtime_error(json.JSONDecodeError("bad", "doc", 0))
        assert err.error_info.code.endswith(UiPathErrorCode.INPUT_INVALID_JSON.value)

    def test_timeout_error_mapped(self):
        rt = _make_runtime()
        err = rt._create_runtime_error(TimeoutError("slow"))
        assert err.error_info.code.endswith(
            UiPathGoogleADKErrorCode.TIMEOUT_ERROR.value
        )

    def test_generic_error_mapped(self):
        rt = _make_runtime()
        err = rt._create_runtime_error(RuntimeError("boom"))
        assert err.error_info.code.endswith(
            UiPathGoogleADKErrorCode.AGENT_EXECUTION_FAILURE.value
        )


class TestConvertEvent:
    def test_user_author_returns_empty(self):
        rt = _make_runtime()
        event = Event(author="user")
        assert rt._convert_event(event) == []

    def test_real_function_call_emits_agent_and_tools_nodes(self):
        rt = _make_runtime()
        part = types.Part.from_function_call(name="search", args={"q": "x"})
        event = Event(author="asst", content=types.Content(role="model", parts=[part]))
        events = rt._convert_event(event)
        node_names = {e.node_name for e in events}
        assert "asst" in node_names
        assert "asst_tools" in node_names

    def test_state_delta_event(self):
        rt = _make_runtime()
        event = Event(author="asst")
        event.actions.state_delta = {"key": "val"}
        events = rt._convert_event(event)
        metadata = events[0].metadata
        assert metadata is not None
        assert metadata["event_type"] == "state_delta"
        assert events[0].payload == {"key": "val"}


class TestExecute:
    @pytest.mark.asyncio
    async def test_execute_returns_final_text(self, monkeypatch):
        rt = _make_runtime()

        async def fake_run(**kwargs):
            yield Event(
                author="asst",
                content=types.Content(
                    role="model", parts=[types.Part(text="the answer")]
                ),
            )

        monkeypatch.setattr(rt._runner, "run_async", fake_run)
        result = await rt.execute({"messages": "q"})
        assert result.status == UiPathRuntimeStatus.SUCCESSFUL
        assert isinstance(result.output, dict)
        assert result.output["messages"][0]["contentParts"][0]["data"]["inline"] == (
            "the answer"
        )

    @pytest.mark.asyncio
    async def test_execute_wraps_errors(self, monkeypatch):
        rt = _make_runtime()

        async def failing_run(**kwargs):
            raise RuntimeError("kaboom")
            yield  # pragma: no cover

        monkeypatch.setattr(rt._runner, "run_async", failing_run)
        with pytest.raises(UiPathGoogleADKRuntimeError):
            await rt.execute({"messages": "q"})


class TestStream:
    @pytest.mark.asyncio
    async def test_stream_emits_lifecycle_and_success(self, monkeypatch):
        rt = _make_runtime()

        async def fake_run(**kwargs):
            yield Event(
                author="asst",
                content=types.Content(role="model", parts=[types.Part(text="done")]),
            )

        monkeypatch.setattr(rt._runner, "run_async", fake_run)
        events = [e async for e in rt.stream({"messages": "q"})]
        state_events = [e for e in events if isinstance(e, UiPathRuntimeStateEvent)]
        assert state_events  # graph lifecycle events emitted
        # last event is the success result
        result = events[-1]
        assert isinstance(result, UiPathRuntimeResult)
        assert result.status == UiPathRuntimeStatus.SUCCESSFUL


class TestGetSchema:
    @pytest.mark.asyncio
    async def test_schema_has_input_output_and_graph(self):
        rt = _make_runtime()
        schema = await rt.get_schema()
        assert schema.type == "agent"
        assert schema.file_path == "asst"
        assert schema.graph is not None
