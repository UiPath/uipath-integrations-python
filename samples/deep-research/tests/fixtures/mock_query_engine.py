"""
Mock query engine implementations for testing
"""

from typing import List, Optional
from unittest.mock import Mock


class MockQueryEngine:
    """Mock query engine for testing"""

    def __init__(self, response_text: str, sources: Optional[List[str]] = None):
        self.response_text = response_text
        self.sources = sources or []

    async def aquery(self, query: str) -> Mock:
        """Mock query that returns structured response"""
        response = Mock()
        response.response = self.response_text

        # Mock source nodes
        source_nodes = []
        for source in self.sources:
            node = Mock()
            node.metadata = {"source": source}
            source_nodes.append(node)

        response.source_nodes = source_nodes
        return response


def create_mock_query_engines() -> dict[str, MockQueryEngine]:
    """Create a set of mock query engines for testing"""
    return {
        "company_policy": MockQueryEngine(
            "Company policy states that automation must follow compliance guidelines.",
            ["policy_doc_1.pdf", "compliance_guide.pdf"],
        ),
        "technical_docs": MockQueryEngine(
            "Technical documentation shows best practices for AI implementation.",
            ["tech_guide.md", "api_docs.html"],
        ),
        "knowledge_base": MockQueryEngine(
            "Knowledge base contains information about enterprise processes.",
            ["kb_article_1.html", "process_guide.docx"],
        ),
    }
