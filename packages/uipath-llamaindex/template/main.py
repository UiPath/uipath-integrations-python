from datetime import datetime, timezone

from llama_index.core.llms import ChatMessage
from llama_index.core.tools import FunctionTool, ToolSelection
from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from pydantic import Field

from uipath_llamaindex.llms import BedrockModel, GeminiModel, OpenAIModel, UiPathOpenAI
from uipath_llamaindex.llms.bedrock import UiPathChatBedrockConverse
from uipath_llamaindex.llms.vertex import UiPathVertex

# Choose your LLM provider by uncommenting one of the following:
llm = UiPathChatBedrockConverse(model=BedrockModel.anthropic_claude_haiku_4_5)
# llm = UiPathOpenAI(model=OpenAIModel.GPT_4_1_MINI_2025_04_14.value)
# llm = UiPathVertex(model=GeminiModel.gemini_2_5_flash)

SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Answer the user's query using the available tools when needed. "
    "Be concise and informative."
)

MAX_RESPONSE_LENGTH = 5000

WEATHER_DATA = {
    "paris": "Weather in Paris, France: 18°C, wind 12 km/h, partly cloudy",
    "london": "Weather in London, UK: 14°C, wind 20 km/h, overcast",
    "new york": "Weather in New York, USA: 22°C, wind 8 km/h, clear sky",
    "tokyo": "Weather in Tokyo, Japan: 26°C, wind 5 km/h, sunny",
    "sydney": "Weather in Sydney, Australia: 19°C, wind 15 km/h, light rain",
}


def get_current_time() -> str:
    """Get the current UTC date and time."""
    return datetime.now(timezone.utc).isoformat()


def get_weather(city: str, utc_time: str) -> str:
    """Get the current weather for a city. Requires the current UTC time from get_current_time.

    Args:
        city: The city name, e.g. 'Paris' or 'Tokyo'.
        utc_time: The current UTC time.
    """
    weather = WEATHER_DATA.get(city.lower().strip())
    if weather:
        return f"{weather} (as of {utc_time})"
    return f"Weather data not available for {city}"


tools = [
    FunctionTool.from_defaults(fn=get_current_time),
    FunctionTool.from_defaults(fn=get_weather),
]
tools_by_name = {t.metadata.name: t for t in tools}


class QueryEvent(StartEvent):
    question: str = Field(
        description="Question for the assistant, e.g. 'What's the weather in Paris?'"
    )


class LLMInputEvent(Event):
    pass


class ToolCallEvent(Event):
    tool_calls: list[ToolSelection]


class AgentOutputEvent(Event):
    response: str


class ResponseEvent(StopEvent):
    response: str


class TemplateAgent(Workflow):
    @step
    async def prepare(self, ctx: Context, ev: QueryEvent) -> LLMInputEvent:
        await ctx.store.set("messages", [
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            ChatMessage(role="user", content=ev.question),
        ])
        return LLMInputEvent()

    @step
    async def react_agent(self, ctx: Context, ev: LLMInputEvent) -> ToolCallEvent | AgentOutputEvent:
        messages = await ctx.store.get("messages")
        response = await llm.achat_with_tools(tools, chat_history=messages)
        messages.append(response.message)
        await ctx.store.set("messages", messages)

        tool_calls = llm.get_tool_calls_from_response(response)
        if tool_calls:
            return ToolCallEvent(tool_calls=tool_calls)

        return AgentOutputEvent(response=response.message.content or "")

    @step
    async def tool_executor(self, ctx: Context, ev: ToolCallEvent) -> LLMInputEvent:
        messages = await ctx.store.get("messages")
        for call in ev.tool_calls:
            tool = tools_by_name[call.tool_name]
            result = tool.call(**call.tool_kwargs)
            messages.append(ChatMessage(
                role="tool",
                content=str(result),
                additional_kwargs={"tool_call_id": call.tool_id, "name": call.tool_name},
            ))
        await ctx.store.set("messages", messages)
        return LLMInputEvent()

    @step
    async def postprocess(self, ctx: Context, ev: AgentOutputEvent) -> ResponseEvent:
        response = ev.response
        if len(response) > MAX_RESPONSE_LENGTH:
            response = response[:MAX_RESPONSE_LENGTH] + "..."
        return ResponseEvent(response=response)


agent = TemplateAgent(timeout=120, verbose=False)
