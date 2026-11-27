"""
End-to-end tests for the complete deep research workflow
"""

from unittest.mock import Mock, patch

import pytest

from deep_research_workflow import DeepResearchWorkflow


class TestFullWorkflow:
    """Test the complete deep research workflow end-to-end"""

    @pytest.fixture
    def workflow_with_mocks(self, mock_llm, mock_query_engines):
        """Create workflow with mocked dependencies"""
        return DeepResearchWorkflow(llm=mock_llm, query_engines=mock_query_engines)

    @pytest.fixture
    def workflow_with_web_search(self, mock_llm):
        """Create workflow with web search enabled"""
        return DeepResearchWorkflow(llm=mock_llm, query_engines=None)

    async def test_complete_workflow_with_uipath(self, workflow_with_mocks):
        """Test complete workflow with UiPath context grounding"""
        topic = "AI automation in business processes"
        context = "Focus on enterprise applications and compliance"

        # Execute the full workflow
        result = await workflow_with_mocks.run(topic=topic, context=context)

        # Assert the final report is generated
        assert result is not None
        assert hasattr(result, "topic")
        assert hasattr(result, "executive_summary")
        assert hasattr(result, "sections")
        assert hasattr(result, "sources")
        assert result.topic == topic

    async def test_complete_workflow_web_only(self, workflow_with_web_search):
        """Test complete workflow with web search only"""
        with patch("agents.web_search.create_web_search_client") as mock_create_client:
            # Mock web search results
            mock_web_client = Mock()
            mock_web_results = []  # Empty results for simplicity
            mock_web_client.search.return_value = mock_web_results
            mock_create_client.return_value = mock_web_client

            topic = "Latest trends in artificial intelligence"
            context = "Focus on 2024 developments"

            # Execute the workflow
            result = await workflow_with_web_search.run(topic=topic, context=context)

            # Assert basic structure
            assert result is not None
            assert result.topic == topic

    async def test_workflow_error_handling(self, mock_llm):
        """Test workflow error handling"""
        # Create workflow with no data sources
        workflow = DeepResearchWorkflow(llm=mock_llm, query_engines=None)

        topic = "Test topic"
        context = "Test context"

        # Should handle gracefully when no data sources available
        result = await workflow.run(topic=topic, context=context)

        # Should still return a result, even if minimal
        assert result is not None
        assert result.topic == topic
