"""UiPath OpenAI LLM for Google ADK agents.

Routes OpenAI-format requests through UiPath's LLM Gateway (AgentHub).
Implements BaseLlm so it can be used directly as an Agent's model.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import httpx
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from pydantic import ConfigDict
from uipath._utils._ssl_context import get_httpx_client_kwargs

from ._common import build_gateway_url, get_uipath_config, get_uipath_headers

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Finish-reason mapping (OpenAI → google.genai)
# ---------------------------------------------------------------------------
_FINISH_REASON_MAP: Dict[str, types.FinishReason] = {
    "stop": types.FinishReason.STOP,
    "length": types.FinishReason.MAX_TOKENS,
    "tool_calls": types.FinishReason.STOP,
    "function_call": types.FinishReason.STOP,
    "content_filter": types.FinishReason.SAFETY,
}

# ---------------------------------------------------------------------------
# ADK Content ↔ OpenAI message helpers
# ---------------------------------------------------------------------------


def _to_openai_role(role: Optional[str]) -> str:
    if role in ("model", "assistant"):
        return "assistant"
    return "user"


def _safe_json(obj: Any) -> str:
    if isinstance(obj, str):
        return obj
    try:
        return json.dumps(obj)
    except (TypeError, ValueError):
        return str(obj)


def _schema_to_dict(schema: Union[types.Schema, Dict[str, Any]]) -> Dict[str, Any]:
    """Recursively convert a google.genai Schema to a plain dict."""
    schema_dict = (
        schema.model_dump(exclude_none=True)
        if isinstance(schema, types.Schema)
        else dict(schema)
    )
    if "type" in schema_dict and schema_dict["type"] is not None:
        t = schema_dict["type"]
        schema_dict["type"] = (t.value if isinstance(t, types.Type) else str(t)).lower()
    if "items" in schema_dict:
        items = schema_dict["items"]
        if isinstance(items, (types.Schema, dict)):
            schema_dict["items"] = _schema_to_dict(items)
    if "properties" in schema_dict:
        schema_dict["properties"] = {
            k: _schema_to_dict(v) if isinstance(v, (types.Schema, dict)) else v
            for k, v in schema_dict["properties"].items()
        }
    return schema_dict


# ---- request conversion ---------------------------------------------------


def _content_to_messages(
    content: types.Content,
) -> List[Dict[str, Any]]:
    """Convert a single types.Content to one or more OpenAI message dicts."""
    tool_messages: List[Dict[str, Any]] = []
    non_tool_parts: List[types.Part] = []

    for part in content.parts or []:
        if part.function_response:
            resp = part.function_response.response
            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": part.function_response.id,
                    "content": _safe_json(resp),
                }
            )
        else:
            non_tool_parts.append(part)

    if tool_messages and not non_tool_parts:
        return tool_messages

    if tool_messages and non_tool_parts:
        follow = _content_to_messages(
            types.Content(role=content.role, parts=non_tool_parts)
        )
        return tool_messages + follow

    role = _to_openai_role(content.role)

    if role == "user":
        parts = [p for p in (content.parts or []) if not p.thought]
        text = _parts_to_content(parts)
        return [{"role": "user", "content": text}]

    # assistant
    tool_calls: List[Dict[str, Any]] = []
    text_parts: List[str] = []
    for part in content.parts or []:
        if part.function_call:
            tool_calls.append(
                {
                    "type": "function",
                    "id": part.function_call.id,
                    "function": {
                        "name": part.function_call.name,
                        "arguments": _safe_json(part.function_call.args),
                    },
                }
            )
        elif part.text and not part.thought:
            text_parts.append(part.text)

    msg: Dict[str, Any] = {
        "role": "assistant",
        "content": " ".join(text_parts) if text_parts else None,
    }
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return [msg]


def _parts_to_content(
    parts: List[types.Part],
) -> Union[str, List[Dict[str, Any]]]:
    """Convert parts list to OpenAI content (string or multi-part array)."""
    if len(parts) == 1 and parts[0].text:
        return parts[0].text

    content_objects: List[Dict[str, Any]] = []
    for part in parts:
        if part.text:
            content_objects.append({"type": "text", "text": part.text})
        elif part.inline_data and part.inline_data.data and part.inline_data.mime_type:
            import base64

            mime = part.inline_data.mime_type
            b64 = base64.b64encode(part.inline_data.data).decode("utf-8")
            data_uri = f"data:{mime};base64,{b64}"
            if mime.startswith("image"):
                content_objects.append(
                    {"type": "image_url", "image_url": {"url": data_uri}}
                )
            elif mime.startswith("audio"):
                content_objects.append(
                    {"type": "audio_url", "audio_url": {"url": data_uri}}
                )
            else:
                content_objects.append(
                    {"type": "file", "file": {"file_data": data_uri}}
                )
    return content_objects or ""  # type: ignore[return-value]


def _function_declaration_to_tool(
    decl: types.FunctionDeclaration,
) -> Dict[str, Any]:
    """Convert a FunctionDeclaration → OpenAI tool dict."""
    assert decl.name
    parameters: Dict[str, Any] = {"type": "object", "properties": {}}
    if decl.parameters and decl.parameters.properties:
        properties = {
            k: _schema_to_dict(v) for k, v in decl.parameters.properties.items()
        }
        parameters = {"type": "object", "properties": properties}
    elif decl.parameters_json_schema:
        parameters = decl.parameters_json_schema

    tool: Dict[str, Any] = {
        "type": "function",
        "function": {
            "name": decl.name,
            "description": decl.description or "",
            "parameters": parameters,
        },
    }
    required = getattr(decl.parameters, "required", None) if decl.parameters else None
    if required:
        tool["function"]["parameters"]["required"] = required
    return tool


def _to_response_format(
    response_schema: types.SchemaUnion,
) -> Optional[Dict[str, Any]]:
    """Convert ADK response_schema to OpenAI response_format."""
    from pydantic import BaseModel as PydanticBaseModel

    schema_name = "response"
    if isinstance(response_schema, dict):
        schema_dict = dict(response_schema)
        if "title" in schema_dict:
            schema_name = str(schema_dict["title"])
    elif isinstance(response_schema, type) and issubclass(
        response_schema, PydanticBaseModel
    ):
        schema_dict = response_schema.model_json_schema()
        schema_name = response_schema.__name__
    elif isinstance(response_schema, PydanticBaseModel):
        if isinstance(response_schema, types.Schema):
            schema_dict = response_schema.model_dump(exclude_none=True, mode="json")
            if "title" in schema_dict:
                schema_name = str(schema_dict["title"])
        else:
            schema_dict = response_schema.__class__.model_json_schema()
            schema_name = response_schema.__class__.__name__
    else:
        return None

    if (
        isinstance(schema_dict, dict)
        and schema_dict.get("type") == "object"
        and "additionalProperties" not in schema_dict
    ):
        schema_dict = dict(schema_dict)
        schema_dict["additionalProperties"] = False

    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name,
            "strict": True,
            "schema": schema_dict,
        },
    }


# ---- response conversion --------------------------------------------------


def _message_to_llm_response(
    message: Dict[str, Any],
    *,
    is_partial: bool = False,
    model_version: Optional[str] = None,
) -> LlmResponse:
    """Convert an OpenAI message dict to an LlmResponse."""
    parts: List[types.Part] = []

    # reasoning / thinking
    reasoning = message.get("reasoning_content")
    if reasoning and isinstance(reasoning, str):
        parts.append(types.Part(text=reasoning, thought=True))

    # text content
    text = message.get("content")
    if isinstance(text, str) and text:
        parts.append(types.Part.from_text(text=text))

    # tool calls
    for tc in message.get("tool_calls") or []:
        if tc.get("type") == "function":
            fn = tc["function"]
            part = types.Part.from_function_call(
                name=fn["name"],
                args=json.loads(fn.get("arguments") or "{}"),
            )
            if part.function_call is not None:
                part.function_call.id = tc.get("id")
            parts.append(part)

    return LlmResponse(
        content=types.Content(role="model", parts=parts),
        partial=is_partial,
        model_version=model_version,
    )


def _parse_complete_response(data: Dict[str, Any]) -> LlmResponse:
    """Parse a non-streaming OpenAI response dict → LlmResponse."""
    choices = data.get("choices") or []
    if not choices:
        raise ValueError("No choices in response")

    choice = choices[0]
    message = choice.get("message", {})
    finish_reason_str = choice.get("finish_reason")

    resp = _message_to_llm_response(message, model_version=data.get("model"))

    if finish_reason_str:
        resp.finish_reason = _FINISH_REASON_MAP.get(
            str(finish_reason_str).lower(), types.FinishReason.OTHER
        )

    usage = data.get("usage")
    if usage:
        resp.usage_metadata = types.GenerateContentResponseUsageMetadata(
            prompt_token_count=usage.get("prompt_tokens", 0),
            candidates_token_count=usage.get("completion_tokens", 0),
            total_token_count=usage.get("total_tokens", 0),
        )
    return resp


# ---------------------------------------------------------------------------
# UiPathOpenAI – the main class
# ---------------------------------------------------------------------------


class UiPathOpenAI(BaseLlm):
    """OpenAI-compatible LLM that routes through UiPath's LLM Gateway.

    Supports GPT / O-series models. Implements ``BaseLlm`` so it can be
    passed directly as an Agent's ``model``.

    Example::

        from uipath_google_adk.chat import UiPathOpenAI
        from google.adk.agents import Agent

        agent = Agent(
            name="assistant",
            model=UiPathOpenAI(model="gpt-4.1-2025-04-14"),
            instruction="You are a helpful assistant.",
        )
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: str = "gpt-4.1-2025-04-14"

    @classmethod
    def supported_models(cls) -> list[str]:
        return [r"gpt-.*", r"o\d+-.*", r"o\d+"]

    # ------------------------------------------------------------------
    # generate_content_async  (the BaseLlm contract)
    # ------------------------------------------------------------------
    async def generate_content_async(
        self,
        llm_request: LlmRequest,
        stream: bool = False,
    ) -> AsyncGenerator[LlmResponse, None]:
        self._maybe_append_user_content(llm_request)

        uipath_url, token = get_uipath_config()
        effective_model = llm_request.model or self.model
        gateway_url = build_gateway_url("openai", effective_model, uipath_url)
        headers = {
            **get_uipath_headers(token),
            "Content-Type": "application/json",
            "X-UiPath-LlmGateway-ApiFlavor": "OpenAiChatCompletions",
        }

        # Build the request body
        body = self._build_request_body(llm_request, effective_model, stream)

        client_kwargs = get_httpx_client_kwargs()

        if stream:
            async for resp in self._stream_request(
                gateway_url, headers, body, client_kwargs
            ):
                yield resp
        else:
            async with httpx.AsyncClient(**client_kwargs) as client:
                response = await client.post(
                    gateway_url,
                    headers=headers,
                    json=body,
                )
                response.raise_for_status()
                yield _parse_complete_response(response.json())

    # ------------------------------------------------------------------
    # request body builder
    # ------------------------------------------------------------------
    def _build_request_body(
        self,
        llm_request: LlmRequest,
        model: str,
        stream: bool,
    ) -> Dict[str, Any]:
        messages: List[Dict[str, Any]] = []

        # system instruction
        if llm_request.config and llm_request.config.system_instruction:
            messages.append(
                {
                    "role": "system",
                    "content": llm_request.config.system_instruction,
                }
            )

        # conversation contents
        for content in llm_request.contents or []:
            messages.extend(_content_to_messages(content))

        body: Dict[str, Any] = {
            "model": model,
            "messages": messages,
        }

        # tools
        if (
            llm_request.config
            and llm_request.config.tools
            and isinstance(llm_request.config.tools[0], types.Tool)
            and llm_request.config.tools[0].function_declarations
        ):
            body["tools"] = [
                _function_declaration_to_tool(d)
                for d in llm_request.config.tools[0].function_declarations
            ]

        # response format
        if llm_request.config and llm_request.config.response_schema:
            fmt = _to_response_format(llm_request.config.response_schema)
            if fmt:
                body["response_format"] = fmt

        # generation params
        if llm_request.config:
            cfg = llm_request.config.model_dump(exclude_none=True)
            param_map = {
                "max_output_tokens": "max_completion_tokens",
                "stop_sequences": "stop",
            }
            for key in (
                "temperature",
                "max_output_tokens",
                "top_p",
                "stop_sequences",
                "presence_penalty",
                "frequency_penalty",
            ):
                if key in cfg:
                    body[param_map.get(key, key)] = cfg[key]

        if stream:
            body["stream"] = True
            body["stream_options"] = {"include_usage": True}

        return body

    # ------------------------------------------------------------------
    # streaming SSE parser
    # ------------------------------------------------------------------
    async def _stream_request(
        self,
        url: str,
        headers: Dict[str, str],
        body: Dict[str, Any],
        client_kwargs: Dict[str, Any],
    ) -> AsyncGenerator[LlmResponse, None]:
        text = ""
        function_calls: Dict[int, Dict[str, Any]] = {}
        usage_metadata = None
        aggregated: Optional[LlmResponse] = None
        aggregated_with_tools: Optional[LlmResponse] = None
        model_version: Optional[str] = None
        fallback_index = 0

        async with httpx.AsyncClient(**client_kwargs) as client:
            async with client.stream(
                "POST",
                url,
                headers={**headers, "X-UiPath-Streaming-Enabled": "true"},
                json=body,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[len("data: ") :]
                    if payload.strip() == "[DONE]":
                        break

                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        continue

                    model_version = chunk.get("model", model_version)
                    choices = chunk.get("choices") or []
                    finish_reason = None

                    for choice in choices:
                        finish_reason = choice.get("finish_reason")
                        delta = choice.get("delta") or {}

                        # text content
                        delta_text = delta.get("content")
                        if delta_text:
                            text += delta_text
                            yield _message_to_llm_response(
                                {
                                    "role": "assistant",
                                    "content": delta_text,
                                },
                                is_partial=True,
                                model_version=model_version,
                            )

                        # tool calls
                        for tc in delta.get("tool_calls") or []:
                            idx = tc.get("index", fallback_index)
                            if idx not in function_calls:
                                function_calls[idx] = {
                                    "name": "",
                                    "args": "",
                                    "id": None,
                                }
                            fn = tc.get("function") or {}
                            if fn.get("name"):
                                function_calls[idx]["name"] += fn["name"]
                            if fn.get("arguments"):
                                function_calls[idx]["args"] += fn["arguments"]
                                try:
                                    json.loads(function_calls[idx]["args"])
                                    fallback_index += 1
                                except json.JSONDecodeError:
                                    pass
                            function_calls[idx]["id"] = (
                                tc.get("id") or function_calls[idx]["id"] or str(idx)
                            )

                    # Handle finish_reason
                    if finish_reason in ("tool_calls", "stop") and function_calls:
                        tool_calls = [
                            {
                                "type": "function",
                                "id": fd["id"],
                                "function": {
                                    "name": fd["name"],
                                    "arguments": fd["args"],
                                },
                            }
                            for fd in function_calls.values()
                            if fd["id"]
                        ]
                        aggregated_with_tools = _message_to_llm_response(
                            {
                                "role": "assistant",
                                "content": text or None,
                                "tool_calls": tool_calls,
                            },
                            model_version=model_version,
                        )
                        aggregated_with_tools.finish_reason = _FINISH_REASON_MAP.get(
                            finish_reason, types.FinishReason.OTHER
                        )
                        text = ""
                        function_calls.clear()
                    elif finish_reason == "stop" and text:
                        aggregated = _message_to_llm_response(
                            {"role": "assistant", "content": text},
                            model_version=model_version,
                        )
                        aggregated.finish_reason = _FINISH_REASON_MAP.get(
                            finish_reason, types.FinishReason.OTHER
                        )
                        text = ""

                    # usage chunk
                    chunk_usage = chunk.get("usage")
                    if chunk_usage:
                        usage_metadata = types.GenerateContentResponseUsageMetadata(
                            prompt_token_count=chunk_usage.get("prompt_tokens", 0),
                            candidates_token_count=chunk_usage.get(
                                "completion_tokens", 0
                            ),
                            total_token_count=chunk_usage.get("total_tokens", 0),
                        )

        # Yield aggregated responses after stream ends
        if aggregated:
            if usage_metadata:
                aggregated.usage_metadata = usage_metadata
                usage_metadata = None
            yield aggregated

        if aggregated_with_tools:
            if usage_metadata:
                aggregated_with_tools.usage_metadata = usage_metadata
            yield aggregated_with_tools
