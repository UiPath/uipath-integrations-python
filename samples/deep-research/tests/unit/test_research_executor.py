"""
Unit tests for ResearchExecutorAgent
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from agents.data_models import ResearchResult
from agents.research_executor import ResearchExecutorAgent
from agents.web_search import WebSearchResult


class TestResearchExecutorAgent:

    @pytest.fixture
    def executor_agent(self, mock_query_engines, mock_llm):
        """Create a ResearchExecutorAgent instance for testing"""
        return ResearchExecutorAgent(mock_query_engines, mock_llm, use_web_search=False)

    @pytest.fixture
    def executor_with_web_search(self, mock_query_engines, mock_llm):
        """Create a ResearchExecutorAgent instance with web search enabled"""
        return ResearchExecutorAgent(mock_query_engines, mock_llm, use_web_search=True)

    async def test_execute_research_single_question(self, executor_agent):
        """Test research execution with a single question"""
        questions = ["What are the compliance requirements for AI automation?"]

        # Execute
        results = await executor_agent.execute_research(questions)

        # Assert
        assert len(results) == 1
        assert isinstance(results[0], ResearchResult)
        assert results[0].question == questions[0]
        assert "Company_Policy" in results[0].findings
        assert "Technical_Docs" in results[0].findings
        assert len(results[0].sources) > 0
        assert isinstance(results[0].confidence_score, float)
        assert 0.0 <= results[0].confidence_score <= 1.0

    async def test_execute_research_multiple_questions(self, executor_agent):
        """Test research execution with multiple questions"""
        questions = [
            "What are AI automation best practices?",
            "How to ensure compliance in automated processes?",
            "What are the technical requirements?",
        ]

        # Execute
        results = await executor_agent.execute_research(questions)

        # Assert
        assert len(results) == 3
        for i, result in enumerate(results):
            assert isinstance(result, ResearchResult)
            assert result.question == questions[i]
            assert len(result.findings) > 0

    async def test_research_question_with_error_handling(
        self, mock_llm, mock_query_engines
    ):
        """Test research with query engine that raises errors"""
        from tests.fixtures import MockQueryEngine

        # Create a query engine that raises an error
        error_engine = Mock()
        error_engine.aquery = AsyncMock(side_effect=Exception("Connection failed"))

        working_engine = MockQueryEngine("Working response", ["source1.pdf"])

        query_engines = {"error_engine": error_engine, "working_engine": working_engine}

        executor = ResearchExecutorAgent(query_engines, mock_llm)

        # Execute
        result = await executor._research_question("Test question")

        # Assert
        assert "Error_Engine Error" in result.findings
        assert "Working response" in result.findings
        assert result.confidence_score < 1.0  # Should be reduced due to error

    def test_calculate_confidence_score_high_quality(self, executor_agent):
        """Test confidence score calculation for high-quality findings"""
        findings = "A" * 1000  # Long, detailed findings

        score = executor_agent._calculate_confidence_score(findings)

        assert score == 0.9

    def test_calculate_confidence_score_medium_quality(self, executor_agent):
        """Test confidence score calculation for medium-quality findings"""
        findings = "A" * 300  # Medium length findings

        score = executor_agent._calculate_confidence_score(findings)

        assert score == 0.7

    def test_calculate_confidence_score_low_quality(self, executor_agent):
        """Test confidence score calculation for low-quality findings"""
        findings = "Short response"  # Short findings

        score = executor_agent._calculate_confidence_score(findings)

        assert score == 0.5

    def test_calculate_confidence_score_with_errors(self, executor_agent):
        """Test confidence score calculation when errors are present"""
        findings = "Error occurred during processing"

        score = executor_agent._calculate_confidence_score(findings)

        assert score == 0.3

    async def test_source_extraction_with_metadata(self, mock_llm):
        """Test proper extraction of sources from query engine responses"""
        from tests.fixtures import MockQueryEngine

        # Create query engine with detailed source metadata
        engine = MockQueryEngine(
            "Test response", ["document1.pdf", "document2.docx", "webpage.html"]
        )

        executor = ResearchExecutorAgent({"test_engine": engine}, mock_llm)

        # Execute
        result = await executor._research_question("Test question")

        # Assert
        assert "document1.pdf" in result.sources
        assert "document2.docx" in result.sources
        assert "webpage.html" in result.sources

    async def test_source_extraction_fallback(self, mock_llm):
        """Test fallback source extraction when metadata is not available"""
        # Create query engine without source metadata
        engine = Mock()
        response = Mock()
        response.response = "Test response"

        # Mock source nodes without metadata
        node = Mock()
        node.text = "Some text content"
        # No metadata attribute
        if hasattr(node, "metadata"):
            delattr(node, "metadata")

        response.source_nodes = [node]
        engine.aquery = AsyncMock(return_value=response)

        executor = ResearchExecutorAgent({"test_engine": engine}, mock_llm)

        # Execute
        result = await executor._research_question("Test question")

        # Assert
        assert "test_engine context" in result.sources

    async def test_empty_query_engines(self, mock_llm):
        """Test behavior with no query engines"""
        executor = ResearchExecutorAgent({}, mock_llm, use_web_search=False)

        # Execute
        result = await executor._research_question("Test question")

        # Assert
        assert result.question == "Test question"
        assert "No findings available" in result.findings
        assert len(result.sources) == 0

    async def test_web_search_integration(self, mock_llm):
        """Test web search integration"""
        with patch(
            "agents.research_executor.create_web_search_client"
        ) as mock_create_client:
            # Mock web search client
            mock_web_client = Mock()
            mock_web_results = [
                WebSearchResult(
                    title="Web Result 1",
                    content="Web content about AI automation",
                    url="https://example.com/1",
                    score=0.9,
                ),
                WebSearchResult(
                    title="Web Result 2",
                    content="More information about automation",
                    url="https://example.com/2",
                    score=0.8,
                ),
            ]
            mock_web_client.search = AsyncMock(return_value=mock_web_results)
            mock_create_client.return_value = mock_web_client

            # Create executor with web search
            executor = ResearchExecutorAgent({}, mock_llm, use_web_search=True)
            executor.web_search = mock_web_client

            # Execute
            result = await executor._research_question("AI automation trends")

            # Assert
            assert "WEB SEARCH RESULTS:" in result.findings
            assert "Web Result 1" in result.findings
            assert "Web Result 2" in result.findings
            assert "https://example.com/1" in result.sources
            assert "https://example.com/2" in result.sources
            assert result.confidence_score > 0.5

    async def test_combined_web_and_uipath_sources(self, mock_query_engines, mock_llm):
        """Test combining web search and UiPath query engines"""
        with patch(
            "agents.research_executor.create_web_search_client"
        ) as mock_create_client:
            # Mock web search
            mock_web_client = Mock()
            mock_web_results = [
                WebSearchResult(
                    title="Web Source",
                    content="Web information",
                    url="https://web-source.com",
                    score=0.9,
                )
            ]
            mock_web_client.search = AsyncMock(return_value=mock_web_results)
            mock_create_client.return_value = mock_web_client

            # Create executor with both web search and UiPath
            executor = ResearchExecutorAgent(
                mock_query_engines, mock_llm, use_web_search=True
            )
            executor.web_search = mock_web_client

            # Execute
            result = await executor._research_question("Test question")

            # Assert
            assert "WEB SEARCH RESULTS:" in result.findings
            assert "UIPATH CONTEXT GROUNDING:" in result.findings
            assert "https://web-source.com" in result.sources
            assert "policy_doc_1.pdf" in result.sources
            assert result.confidence_score > 0.7  # Bonus for multiple sources

    async def test_web_search_error_handling(self, mock_llm):
        """Test web search error handling"""
        with patch(
            "agents.research_executor.create_web_search_client"
        ) as mock_create_client:
            # Mock web search that raises error
            mock_web_client = Mock()
            mock_web_client.search = AsyncMock(side_effect=Exception("Network error"))
            mock_create_client.return_value = mock_web_client

            executor = ResearchExecutorAgent({}, mock_llm, use_web_search=True)
            executor.web_search = mock_web_client

            # Execute
            result = await executor._research_question("Test question")

            # Assert
            assert "WEB SEARCH ERROR:" in result.findings
            assert "Network error" in result.findings

    def test_confidence_score_with_multiple_sources(self, mock_llm):
        """Test confidence score calculation with source diversity"""
        executor = ResearchExecutorAgent({}, mock_llm, use_web_search=False)

        # Test with many sources
        high_sources_findings = "Comprehensive findings with detailed analysis"
        score = executor._calculate_confidence_score(
            high_sources_findings, num_sources=8
        )
        assert score > 0.7

        # Test with few sources
        low_sources_findings = "Brief findings"
        score = executor._calculate_confidence_score(
            low_sources_findings, num_sources=1
        )
        assert score < 0.7

        # Test with both web and UiPath sources
        combined_findings = (
            "WEB SEARCH RESULTS: data\n\nUIPATH CONTEXT GROUNDING: more data"
        )
        score = executor._calculate_confidence_score(combined_findings, num_sources=5)
        assert score > 0.8  # Bonus for having both types

    async def test_no_web_results(self, mock_llm):
        """Test handling when web search returns no results"""
        with patch(
            "agents.research_executor.create_web_search_client"
        ) as mock_create_client:
            mock_web_client = Mock()
            mock_web_client.search = AsyncMock(return_value=[])  # No results
            mock_create_client.return_value = mock_web_client

            executor = ResearchExecutorAgent({}, mock_llm, use_web_search=True)
            executor.web_search = mock_web_client

            # Execute
            result = await executor._research_question("Obscure topic")

            # Assert
            assert "WEB SEARCH: No results found" in result.findings
