"""Synthesis Agent for Deep Research Workflow."""

from datetime import datetime
from typing import List

from llama_index.core.llms import LLM

from .data_models import FinalReport, ResearchPlan, ResearchResult


class SynthesisAgent:
    """Synthesize research results into final reports."""

    def __init__(self, llm: LLM):
        self.llm = llm

    async def synthesize_report(
        self, plan: ResearchPlan, results: List[ResearchResult]
    ) -> FinalReport:
        """Synthesize research results into a comprehensive final report."""
        synthesis_prompt = f"""
        You are an expert research analyst. Synthesize the following \
research findings into a comprehensive report.

        TOPIC: {plan.topic}

        RESEARCH FINDINGS:
        {self._format_results_for_synthesis(results)}

        EXPECTED SECTIONS: {', '.join(plan.expected_sections)}

        Create a comprehensive report with:
        1. Executive Summary (2-3 paragraphs)
        2. Detailed sections as specified
        3. Clear conclusions and recommendations

        Format your response as:
        EXECUTIVE_SUMMARY:
        [summary content]

        SECTION: [section name]
        [section content]

        SECTION: [section name]
        [section content]

        [continue for all sections]
        """

        response = await self.llm.acomplete(synthesis_prompt)
        response_str = str(response)
        # Debug: uncomment to see what we're parsing
        # print(f"DEBUG: Response string: {repr(response_str)}")
        return self._parse_synthesis_response(plan, results, response_str)

    def _format_results_for_synthesis(self, results: List[ResearchResult]) -> str:
        """Format research results for synthesis prompt."""
        formatted = []
        for i, result in enumerate(results, 1):
            # Limit sources for brevity
            limited_sources = ", ".join(result.sources[:3])
            formatted.append(
                f"""
            RESEARCH QUESTION {i}: {result.question}
            FINDINGS: {result.findings}
            CONFIDENCE: {result.confidence_score:.2f}
            SOURCES: {limited_sources}
            """
            )
        return "\n".join(formatted)

    def _parse_synthesis_response(
        self, plan: ResearchPlan, results: List[ResearchResult], response: str
    ) -> FinalReport:
        """Parse synthesis response into FinalReport object."""
        lines = response.strip().split("\n")

        executive_summary = ""
        sections = {}
        current_section = None
        current_content: List[str] = []

        for line in lines:
            line = line.strip()
            if line.startswith("EXECUTIVE_SUMMARY:"):
                current_section = "executive_summary"
                current_content = []
            elif line.startswith("SECTION:"):
                if current_section == "executive_summary":
                    executive_summary = "\n".join(current_content).strip()
                elif current_section:
                    section_name = current_section.replace("SECTION:", "").strip()
                    sections[section_name] = "\n".join(current_content).strip()

                current_section = line
                current_content = []
            elif line and current_section:
                current_content.append(line)

        if current_section == "executive_summary":
            executive_summary = "\n".join(current_content).strip()
        elif current_section:
            section_name = current_section.replace("SECTION:", "").strip()
            sections[section_name] = "\n".join(current_content).strip()

        all_sources = []
        for result in results:
            all_sources.extend(result.sources)

        return FinalReport(
            topic=plan.topic,
            executive_summary=executive_summary,
            sections=sections,
            sources=list(set(all_sources)),
            generated_at=datetime.now(),
        )
