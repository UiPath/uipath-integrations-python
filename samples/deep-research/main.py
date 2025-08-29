"""
Main entry point for the Deep Research Agent workflow.

This module provides the main interface for running the deep research workflow
with UiPath context grounding integration.
"""

import asyncio
import json
import os
from typing import Any, Dict, Optional

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from llama_index.core.llms import LLM  # type: ignore  # noqa: E402
from llama_index.core.query_engine import BaseQueryEngine  # type: ignore  # noqa: E402

from agents.data_models import FinalReport  # noqa: E402
from deep_research_workflow import DeepResearchWorkflow  # noqa: E402
from uipath_integration import (  # noqa: E402
    create_uipath_query_engines,
    validate_uipath_config,
)


def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """Load configuration from JSON file."""
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)

    # Default configuration
    return {
        "llm": {
            "provider": "openai",
            "model": "gpt-4-turbo",
            "temperature": 0.1,
            "max_tokens": 2000,
        },
        "workflow": {"timeout": 300.0},
        "web_search": {
            "enabled": True,
            "max_results": 5,
            "tavily_api_key": os.getenv("TAVILY_API_KEY", ""),
        },
        "uipath": {
            "enabled": bool(os.getenv("UIPATH_ORCHESTRATOR_URL")),
            "orchestrator_url": os.getenv("UIPATH_ORCHESTRATOR_URL", ""),
            "tenant_name": os.getenv("UIPATH_TENANT_NAME", ""),
            "client_id": os.getenv("UIPATH_CLIENT_ID", ""),
            "client_secret": os.getenv("UIPATH_CLIENT_SECRET", ""),
        },
    }


def create_llm(config: Dict[str, Any]):
    """Create LLM instance based on configuration."""
    llm_config = config.get("llm", {})

    if llm_config.get("provider") == "openai":
        try:
            from llama_index.llms.openai import OpenAI  # type: ignore

            return OpenAI(
                model=llm_config.get("model", "gpt-4-turbo"),
                temperature=llm_config.get("temperature", 0.1),
                max_tokens=llm_config.get("max_tokens", 2000),
            )
        except ImportError:
            # Fallback to mock for testing
            from tests.fixtures.mock_llm import MockLLM  # type: ignore

            return MockLLM()
    else:
        raise ValueError(f"Unsupported LLM provider: {llm_config.get('provider')}")


def create_uipath_query_engines_from_config(
    config: Dict[str, Any], llm
) -> Optional[Dict[str, Any]]:
    """Create UiPath query engines if configuration is available."""
    uipath_config = config.get("uipath", {})

    if not uipath_config.get("enabled"):
        print("ℹ️  UiPath integration disabled. Using web search only.")
        return None

    # Validate UiPath configuration
    if not validate_uipath_config(config):
        print("⚠️  UiPath configuration is incomplete. Skipping UiPath integration.")
        return None

    try:
        # Create real UiPath context grounding query engines
        engines = create_uipath_query_engines(
            orchestrator_url=uipath_config["orchestrator_url"],
            tenant_name=uipath_config["tenant_name"],
            client_id=uipath_config["client_id"],
            client_secret=uipath_config["client_secret"],
            llm=llm,
        )

        print(f"✅ Created {len(engines)} UiPath context grounding query engines")
        return engines

    except Exception as e:
        print(f"❌ Failed to create UiPath query engines: {e}")
        print("🔄 Continuing with web search only.")
        return None


async def run_deep_research(
    topic: str, context: str = "", config: Optional[Dict[str, Any]] = None
) -> FinalReport:
    """Run the deep research workflow."""
    if config is None:
        config = load_config()

    # Create LLM
    llm = create_llm(config)

    # Create UiPath query engines (optional)
    query_engines = create_uipath_query_engines_from_config(config, llm)

    # Create workflow
    workflow = DeepResearchWorkflow(
        llm=llm,
        query_engines=query_engines,
        timeout=config.get("workflow", {}).get("timeout", 300.0),
    )

    # Run workflow
    result = await workflow.run(topic=topic, context=context)
    return result


def format_report_output(report: FinalReport) -> str:
    """Format the final report for output."""
    output = []
    output.append("=" * 80)
    output.append("DEEP RESEARCH REPORT")
    output.append("=" * 80)
    output.append(f"\nTOPIC: {report.topic}")
    output.append(f"GENERATED: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}")

    output.append("\nEXECUTIVE SUMMARY:")
    output.append("-" * 40)
    output.append(report.executive_summary)

    output.append("\nDETAILED SECTIONS:")
    output.append("-" * 40)
    for section_name, section_content in report.sections.items():
        output.append(f"\n{section_name.upper()}:")
        output.append(section_content)

    output.append(f"\nSOURCES ({len(report.sources)}):")
    output.append("-" * 40)
    for i, source in enumerate(report.sources, 1):
        output.append(f"{i}. {source}")

    return "\n".join(output)


async def main():
    """Run main function for CLI usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Deep Research Agent with UiPath Context Grounding"
    )
    parser.add_argument("topic", help="Research topic")
    parser.add_argument("--context", help="Additional context for research", default="")
    parser.add_argument(
        "--config", help="Path to configuration file", default="config.json"
    )
    parser.add_argument("--output", help="Output file for the report")

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    try:
        print(f"🔍 Starting deep research on: {args.topic}")
        if args.context:
            print(f"📝 Context: {args.context}")

        # Run research
        report = await run_deep_research(args.topic, args.context, config)

        # Format output
        formatted_report = format_report_output(report)

        # Save or print output
        if args.output:
            with open(args.output, "w") as f:
                f.write(formatted_report)
            print(f"📄 Report saved to: {args.output}")
        else:
            print(formatted_report)

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    return 0


# Entry point for UiPath integration
async def deep_research_agent(
    topic: str, context: str = "", add_data_to_index: bool = False
) -> Dict[str, Any]:
    """
    Run main entry point for UiPath integration.

    This function is referenced in llama_index.json and provides the interface
    for UiPath to call the deep research workflow.
    """
    # Note: add_data_to_index parameter is reserved for future functionality
    _ = add_data_to_index  # Acknowledge unused parameter

    try:
        # Run the research workflow
        report = await run_deep_research(topic, context)

        # Convert report to dictionary for UiPath
        return {
            "report": {
                "topic": report.topic,
                "executive_summary": report.executive_summary,
                "sections": report.sections,
                "sources": report.sources,
                "generated_at": report.generated_at.isoformat(),
            }
        }

    except Exception as e:
        return {"error": str(e), "report": None}


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
