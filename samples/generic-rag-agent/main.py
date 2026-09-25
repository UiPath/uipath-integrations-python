import os
import time
from typing import Optional

from llama_index.core import get_response_synthesizer
from llama_index.core.agent import ReActAgent
from llama_index.core.response_synthesizers.type import ResponseMode
from llama_index.core.tools import FunctionTool, QueryEngineTool, ToolMetadata
from llama_index.core.workflow import (
    Context,
    Event,
    HumanResponseEvent,
    InputRequiredEvent,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from llama_index.tools.mcp import McpToolSpec
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from uipath import UiPath

from uipath_llamaindex.llms import UiPathOpenAI
from uipath_llamaindex.query_engines import ContextGroundingQueryEngine

INDEX_CONFIGS = {
    "primary_knowledge_base": {
        "index_name": "primary_index",
        "folder_path": "sample_data/primary_data",
        "tool_name": "primary_knowledge_base",
        "tool_description": "Primary knowledge base containing company policies, guidelines, and reference information",
    },
    "secondary_knowledge_base": {
        "index_name": "secondary_index",
        "folder_path": "sample_data/secondary_data",
        "tool_name": "secondary_knowledge_base",
        "tool_description": "Secondary knowledge base containing user preferences and personalization data",
    },
}

index_folder_path = "Shared"

llm = UiPathOpenAI(model="gpt-4o-2024-11-20")
uipath = UiPath()


class CustomStartEvent(StartEvent):
    query: str = ""
    add_data_to_index: Optional[bool] = False
    include_hitl: Optional[bool] = False


class QueryEvent(Event):
    """Triggers the agent query process."""

    pass


class AddDataToIndexEvent(Event):
    """Triggers the data ingestion process."""

    pass


class WaitForIndexIngestion(Event):
    """Wait for all ingestion jobs to finish."""

    pass


class AgentAnswerEvent(Event):
    """Stores the final, raw answer from the ReAct Agent."""

    agent_answer: str


class FormattedAnswerEvent(Event):
    """Event containing the formatted answer awaiting human confirmation."""

    formatted_answer: str
    iteration_count: int


class OutputEvent(StopEvent):
    """Event representing the final, formatted output."""

    output: str


def tavily_web_search(query: str) -> str:
    """
    Performs a web search using Tavily API.
    """
    try:
        from tavily import TavilyClient

        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return "Error: TAVILY_API_KEY environment variable is not set. Please set it to use web search functionality."

        client = TavilyClient(api_key=api_key)
        response = client.search(query, max_results=3)

        results = []
        for idx, result in enumerate(response.get("results", []), 1):
            results.append(
                f"{idx}. {result.get('title', 'No title')}\n"
                f"    URL: {result.get('url', 'No URL')}\n"
                f"    Snippet: {result.get('content', 'No content')}\n"
            )

        if not results:
            return "No web search results found."

        return "Web Search Results:\n\n" + "\n".join(results)

    except Exception as e:
        return f"Error performing web search: {str(e)}"


def generate_tools(
    response_mode: ResponseMode,
) -> list[QueryEngineTool | FunctionTool]:
    """Generates the list of tools for the ReAct Agent (excluding MCP tools)."""
    response_synthesizer = get_response_synthesizer(
        response_mode=response_mode, llm=llm
    )

    tools = []

    for _, config in INDEX_CONFIGS.items():
        query_engine = ContextGroundingQueryEngine(
            index_name=config["index_name"],
            folder_path=index_folder_path,
            response_synthesizer=response_synthesizer,
        )

        tools.append(
            QueryEngineTool(
                query_engine=query_engine,
                metadata=ToolMetadata(
                    name=config["tool_name"],
                    description=config["tool_description"],
                ),
            )
        )

    tavily_tool = FunctionTool.from_defaults(
        fn=tavily_web_search,
        name="web_search",
        description="Search the web for current, real-time information, recent events, prices, availability, or any information not in the knowledge bases. Use this when you need up-to-date information from the internet.",
    )
    tools.append(tavily_tool)

    return tools


async def in_progress_ingestion(index_name: str) -> bool:
    """
    Checks if ingestion is in progress for a given index.
    """
    index = await uipath.context_grounding.retrieve_async(
        index_name, folder_path=index_folder_path
    )
    return index.in_progress_ingestion()


class AgentWorkflow(Workflow):
    @step
    async def workflow_entrypoint(
        self, ctx: Context, ev: CustomStartEvent
    ) -> QueryEvent | AddDataToIndexEvent:
        await ctx.store.set("original_query", ev.query)
        await ctx.store.set("iteration_count", 0)
        await ctx.store.set("feedback_history", [])
        await ctx.store.set("include_hitl", ev.include_hitl)

        if ev.add_data_to_index:
            return AddDataToIndexEvent()
        return QueryEvent()

    @step
    async def add_data_to_index(self, ev: AddDataToIndexEvent) -> WaitForIndexIngestion:
        async def add_file_to_index(file_path, index_name, ingest_data):
            await uipath.context_grounding.add_to_index_async(
                name=index_name,
                folder_path=index_folder_path,
                source_path=file_path,
                blob_file_path=os.path.basename(file_path),
                ingest_data=ingest_data,
            )

        try:
            for config_key, config in INDEX_CONFIGS.items():
                files_directory = config["folder_path"]
                index_name = config["index_name"]

                if not os.path.exists(files_directory):
                    print(
                        f"Warning: Directory {files_directory} does not exist. Skipping..."
                    )
                    continue

                files = os.listdir(files_directory)
                if not files:
                    print(f"Warning: No files found in {files_directory}. Skipping...")
                    continue

                print(
                    f"Indexing {len(files)} file(s) from {files_directory} to index '{index_name}'..."
                )

                for i, file_name in enumerate(files):
                    is_last_config = config_key == list(INDEX_CONFIGS.keys())[-1]
                    is_last_file = i == len(files) - 1
                    ingest_data = is_last_config and is_last_file

                    await add_file_to_index(
                        os.path.join(files_directory, file_name),
                        index_name,
                        ingest_data,
                    )

                print(f"Completed indexing for '{index_name}'")

            return WaitForIndexIngestion()

        except Exception as e:
            print(f"Error during indexing: {e}")
            raise

    @step
    async def wait_for_index_ingestion(
        self, ev: WaitForIndexIngestion
    ) -> QueryEvent | OutputEvent:
        """
        Polls the ingestion status to ensure all data is indexed before querying.
        """
        no_of_tries = 10
        wait_seconds = 10

        ingestion_status = {
            config["index_name"]: False for config in INDEX_CONFIGS.values()
        }

        while no_of_tries:
            should_continue = False

            for index_name in ingestion_status.keys():
                if not ingestion_status[index_name]:
                    should_continue = True
                    ingestion_status[index_name] = not await in_progress_ingestion(
                        index_name
                    )

            if not should_continue:
                break

            no_of_tries -= 1
            pending_indexes = [
                name for name, done in ingestion_status.items() if not done
            ]
            print(
                f"Waiting for index ingestion... Pending: {pending_indexes}. "
                f"Retrying {no_of_tries} more time(s)"
            )
            time.sleep(wait_seconds)

        if all(ingestion_status.values()):
            print("All indexes ingested successfully. Moving to query step.")
            return QueryEvent()

        failed_indexes = [name for name, done in ingestion_status.items() if not done]
        return OutputEvent(
            output=f"Cannot evaluate query. Index ingestion is taking too long for: {failed_indexes}"
        )

    @step
    async def process_query(self, ctx: Context, ev: QueryEvent) -> AgentAnswerEvent:
        """
        Uses a ReActAgent for planning, execution, and initial synthesis.
        """
        original_query = await ctx.store.get("original_query")

        query_engine_tools = generate_tools(response_mode=ResponseMode.SIMPLE_SUMMARIZE)

        mcp_url = os.getenv("UIPATH_MCP_URL")
        if mcp_url:
            try:
                async with streamablehttp_client(
                    url=mcp_url,
                    headers={
                        "Authorization": f"Bearer {os.getenv('UIPATH_ACCESS_TOKEN')}"
                    },
                    timeout=60,
                ) as (read, write, _get_session_id_callback):
                    async with ClientSession(read, write) as client_session:
                        await client_session.initialize()
                        mcp_tool_spec = McpToolSpec(client=client_session)
                        mcp_tools = await mcp_tool_spec.to_tool_list_async()

                        print(
                            f"Connected to MCP server, loaded {len(mcp_tools)} tool(s)"
                        )
                        for tool in mcp_tools:
                            print(f"  - {tool.metadata.name}")

                        all_tools = query_engine_tools + mcp_tools

                        react_agent = ReActAgent(
                            tools=all_tools,
                            llm=llm,
                            verbose=True,
                            system_prompt=(
                                "You are a specialized AI assistant. Your task is to comprehensively answer "
                                "the user's query by breaking it down, gathering all necessary information "
                                "from the available knowledge bases, web search, and any other available tools, "
                                "and synthesizing the findings into a single, detailed, and accurate response. "
                                "Use the tools strategically to ensure all parts of the query are addressed."
                            ),
                        )

                        print(f"Executing query with ReAct Agent: {original_query}")
                        response = await react_agent.run(user_msg=original_query)

                        print("ReAct Agent execution complete.")
                        return AgentAnswerEvent(agent_answer=str(response))

            except Exception as e:
                print(f"Warning: Failed to connect to MCP server: {e}")
                print("Continuing with non-MCP tools only...")

        react_agent = ReActAgent(
            tools=query_engine_tools,
            llm=llm,
            verbose=True,
            system_prompt=(
                "You are a specialized AI assistant. Your task is to comprehensively answer "
                "the user's query by breaking it down, gathering all necessary information "
                "from the available knowledge bases and web search, and synthesizing the "
                "findings into a single, detailed, and accurate response. Use the tools "
                "strategically to ensure all parts of the query are addressed."
            ),
        )

        print(f"Executing query with ReAct Agent: {original_query}")
        response = await react_agent.run(user_msg=original_query)

        print("ReAct Agent execution complete.")
        return AgentAnswerEvent(agent_answer=str(response))

    @step
    async def format_final_answer(
        self, ctx: Context, ev: AgentAnswerEvent
    ) -> FormattedAnswerEvent:
        """
        Takes the ReAct Agent's raw answer and formats it into the final structure.
        """
        original_query = await ctx.store.get("original_query")
        iteration_count = await ctx.store.get("iteration_count")
        feedback_history = await ctx.store.get("feedback_history", [])
        agent_answer = ev.agent_answer

        feedback_context = ""
        if feedback_history:
            feedback_context = "\n\nPrevious feedback from the user:\n"
            for i, feedback in enumerate(feedback_history, 1):
                feedback_context += f"{i}. {feedback}\n"
            feedback_context += (
                "\nPlease address the feedback above in your reformatted answer.\n"
            )

        prompt = f"""
            You are a final synthesis engine. You are given the user's original query and
            a comprehensive answer generated by an intelligent multi-tool agent.
            Your task is to reformat this answer into a professional, well-structured
            report that directly addresses the original query.

            Structure your final response using Markdown as follows:

            **Summary:**
            Briefly restate the purpose of the query and the key conclusion.

            **Key Findings:**
            Present the main information gathered from all sources (knowledge bases and web).
            Use bullet points for clarity.

            **Analysis:**
            Connect the key findings to address the user's specific question.
            Provide context and insights.

            **Recommendations or Next Steps:**
            If applicable, suggest actionable items or state relevant policies.

            Original query: {original_query}
            {feedback_context}

            Agent's Comprehensive Answer to be Formatted:
            ---
            {agent_answer}
            ---
        """

        print("Generating final formatted response...")
        response = llm.complete(prompt)

        print("Final formatting complete.")

        return FormattedAnswerEvent(
            formatted_answer=response.text, iteration_count=iteration_count
        )

    @step
    async def request_human_confirmation(
        self, ctx: Context, ev: FormattedAnswerEvent
    ) -> OutputEvent | QueryEvent:
        """
        Present the formatted answer to the user and request confirmation.
        If include_hitl is False, skip human confirmation and return the output directly.
        """
        include_hitl = await ctx.store.get("include_hitl", False)

        if not include_hitl:
            print("\n" + "=" * 80)
            print("FORMATTED ANSWER (HITL disabled):")
            print("=" * 80)
            print(ev.formatted_answer)
            print("=" * 80)
            return OutputEvent(output=ev.formatted_answer)

        print("\n" + "=" * 80)
        print("FORMATTED ANSWER:")
        print("=" * 80)
        print(ev.formatted_answer)
        print("=" * 80)

        prompt_message = f"{ev.formatted_answer}\n\n{'=' * 80}\n\nIs this answer satisfactory? (yes/no or provide feedback): "

        ctx.write_event_to_stream(InputRequiredEvent(prefix=prompt_message))

        response = await ctx.wait_for_event(HumanResponseEvent)
        feedback = response.response.strip().lower()

        print(f"Received response: {feedback}")

        if feedback == "yes":
            print("\nAnswer approved by user.")
            return OutputEvent(output=ev.formatted_answer)

        max_iterations = 3
        iteration_count = ev.iteration_count

        if iteration_count >= max_iterations:
            print(f"\nMaximum iterations ({max_iterations}) reached.")
            return OutputEvent(
                output=f"{ev.formatted_answer}\n\n---\nNote: Maximum iteration limit reached. Please refine your query or requirements."
            )

        feedback_history = await ctx.store.get("feedback_history", [])
        feedback_history.append(feedback)
        await ctx.store.set("feedback_history", feedback_history)
        await ctx.store.set("iteration_count", iteration_count + 1)

        print(
            f"\nFeedback received: '{feedback}'. Re-querying (iteration {iteration_count + 1})..."
        )

        return QueryEvent()


agent = AgentWorkflow(timeout=600, verbose=True)
