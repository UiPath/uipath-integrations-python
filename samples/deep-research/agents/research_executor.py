"""Research Executor Agent for Deep Research Workflow."""

from typing import Dict, List, Optional

from llama_index.core.llms import LLM
from llama_index.core.query_engine import BaseQueryEngine

from .data_models import ResearchResult
from .web_search import create_web_search_client


class ResearchExecutorAgent:
    """Execute research using web search and optional UiPath context grounding."""

    def __init__(
        self,
        query_engines: Optional[Dict[str, BaseQueryEngine]],
        llm: LLM,
        use_web_search: bool = True,
        max_web_results: int = 5,
    ):
        self.query_engines = query_engines or {}
        self.llm = llm
        self.use_web_search = use_web_search
        self.max_web_results = max_web_results

        # Initialize web search client
        if self.use_web_search:
            self.web_search = create_web_search_client(
                use_mock=not self._has_tavily_key()
            )
        else:
            self.web_search = None

    def _has_tavily_key(self) -> bool:
        """Check if Tavily API key is available."""
        import os

        return bool(os.getenv("TAVILY_API_KEY"))

    async def execute_research(self, questions: List[str]) -> List[ResearchResult]:
        """Execute research for questions using available query engines."""
        results = []

        for question in questions:
            result = await self._research_question(question)
            results.append(result)

        return results

    async def _research_question(self, question: str) -> ResearchResult:
        """Research a single question using web search and available query engines."""
        findings = []
        sources = []

        # 1. Perform web search first (primary source)
        if self.use_web_search and self.web_search:
            try:
                web_results = await self.web_search.search(
                    query=question, max_results=self.max_web_results
                )

                if web_results:
                    web_findings = []
                    for result in web_results:
                        web_findings.append(f"**{result.title}**\n{result.content}")
                        sources.append(result.url)

                    findings.append("WEB SEARCH RESULTS:\n" + "\n\n".join(web_findings))
                else:
                    findings.append("WEB SEARCH: No results found")

            except Exception as e:
                findings.append(f"WEB SEARCH ERROR: {str(e)}")

        # 2. Query UiPath context grounding engines (additional context)
        if self.query_engines:
            uipath_findings = []
            for engine_name, query_engine in self.query_engines.items():
                try:
                    response = await query_engine.aquery(question)
                    uipath_findings.append(
                        f"**{engine_name.title()}**: {getattr(response, 'response', str(response))}"
                    )

                    if hasattr(response, "source_nodes"):
                        for node in response.source_nodes:
                            if hasattr(node, "metadata") and "source" in node.metadata:
                                sources.append(node.metadata["source"])
                            elif hasattr(node, "text"):
                                sources.append(f"{engine_name} context")

                except Exception as e:
                    uipath_findings.append(f"**{engine_name.title()} Error**: {str(e)}")

            if uipath_findings:
                findings.append(
                    "UIPATH CONTEXT GROUNDING:\n" + "\n\n".join(uipath_findings)
                )

        # 3. Combine all findings
        if findings:
            combined_findings = "\n\n" + ("=" * 50) + "\n\n".join(findings)
        else:
            combined_findings = "No findings available"
        confidence_score = self._calculate_confidence_score(
            combined_findings, len(sources)
        )

        return ResearchResult(
            question=question,
            findings=combined_findings,
            sources=list(set(sources)),
            confidence_score=confidence_score,
        )

    def _calculate_confidence_score(self, findings: str, num_sources: int = 0) -> float:
        """Calculate confidence score based on findings quality and source diversity."""
        base_score = 0.5

        # Penalize for errors
        if "Error" in findings or "ERROR" in findings:
            base_score = 0.3
            return base_score

        # Reward for content length (more comprehensive)
        if len(findings) >= 1000:
            base_score = 0.9  # High quality - long detailed findings
        elif len(findings) >= 200:
            base_score = 0.7  # Medium quality - moderate length
        else:
            base_score = 0.5  # Base score for any length

        # Bonus for source diversity
        if num_sources > 5:
            base_score = min(1.0, base_score + 0.3)
        elif num_sources > 3:
            base_score = min(1.0, base_score + 0.1)

        # Bonus for having both web and UiPath sources
        has_web = "WEB SEARCH RESULTS:" in findings
        has_uipath = "UIPATH CONTEXT GROUNDING:" in findings
        if has_web and has_uipath:
            base_score = min(1.0, base_score + 0.25)
        elif has_web:
            base_score = min(1.0, base_score + 0.05)  # Small bonus for web results

        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, base_score))
