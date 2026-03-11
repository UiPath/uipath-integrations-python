import os

from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.llms.openai import OpenAI
from llama_index.tools.tavily_research import TavilyToolSpec

llm = OpenAI(model="gpt-4o-mini")
tavily_tool = TavilyToolSpec(api_key=os.environ["TAVILY_API_KEY"])

SYSTEM_PROMPT = (
    "You are an advanced AI assistant specializing in book research and literature analysis. "
    "Your primary functions are:\n\n"
    "1. Book Information Research: Gather comprehensive information about books, including plot summaries, "
    "themes, publishing details, sales performance, critical reception, and awards.\n"
    "2. Author Research: Provide detailed information about authors, translators, editors, and other "
    "publishing industry professionals.\n"
    "3. Book Recommendations: Suggest books based on user preferences, genres, themes, or similar books "
    "they have enjoyed.\n"
    "4. Publishing Industry Analysis: Analyze trends, bestseller data, genre popularity, and insights "
    "from the literary world.\n"
    "5. Book Trivia and Facts: Share interesting facts, behind-the-scenes stories, and trivia about "
    "books, authors, and the publishing industry.\n\n"
    "Use the search tool for recent or factual information. "
    "Remember previous messages and maintain context across the discussion."
)

agent = AgentWorkflow.from_tools_or_functions(
    tools_or_functions=tavily_tool.to_tool_list(), llm=llm, system_prompt=SYSTEM_PROMPT
)
