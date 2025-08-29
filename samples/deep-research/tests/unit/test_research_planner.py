"""
Unit tests for ResearchPlannerAgent
"""

from unittest.mock import Mock

import pytest

from agents.data_models import ResearchPlan
from agents.research_planner import ResearchPlannerAgent


class TestResearchPlannerAgent:

    @pytest.fixture
    def planner_agent(self, mock_llm):
        """Create a ResearchPlannerAgent instance for testing"""
        return ResearchPlannerAgent(mock_llm)

    @pytest.fixture
    def sample_llm_response(self):
        """Sample LLM response for testing parsing"""
        return """
        RESEARCH_QUESTIONS:
        1. What are the current trends in AI automation?
        2. How does AI impact business efficiency?
        3. What are the compliance considerations?

        METHODOLOGY:
        This research will use a systematic approach combining literature review and case studies.

        EXPECTED_SECTIONS:
        1. Executive Summary
        2. Current Trends Analysis
        3. Business Impact Assessment
        4. Compliance Framework

        PRIORITY_ORDER:
        2, 1, 3
        """

    async def test_create_plan_success(
        self, planner_agent, mock_llm, sample_llm_response
    ):
        """Test successful research plan creation"""
        # Setup
        topic = "AI automation in business processes"
        context = "Focus on enterprise applications"
        mock_llm.set_response(sample_llm_response)

        # Execute
        result = await planner_agent.create_plan(topic, context)

        # Assert
        assert isinstance(result, ResearchPlan)
        assert result.topic == topic
        assert len(result.research_questions) == 3
        assert "AI automation" in result.research_questions[0]
        assert "systematic approach" in result.methodology
        assert len(result.expected_sections) == 4
        assert result.priority_order == [2, 1, 3]

    def test_parse_plan_response_complete(self, planner_agent, sample_llm_response):
        """Test parsing of complete LLM response"""
        topic = "Test Topic"

        # Execute
        result = planner_agent._parse_plan_response(topic, sample_llm_response)

        # Assert
        assert result.topic == topic
        assert len(result.research_questions) == 3
        assert (
            result.research_questions[0]
            == "What are the current trends in AI automation?"
        )
        assert "systematic approach" in result.methodology
        assert len(result.expected_sections) == 4
        assert result.expected_sections[0] == "Executive Summary"
        assert result.priority_order == [2, 1, 3]

    def test_parse_plan_response_missing_priority(self, planner_agent):
        """Test parsing when priority order is missing"""
        incomplete_response = """
        RESEARCH_QUESTIONS:
        1. Question one
        2. Question two

        METHODOLOGY:
        Test methodology

        EXPECTED_SECTIONS:
        1. Section one
        """

        # Execute
        result = planner_agent._parse_plan_response("Test", incomplete_response)

        # Assert
        assert result.priority_order == [1, 2]  # Default order

    def test_parse_plan_response_invalid_priority(self, planner_agent):
        """Test parsing when priority order is invalid"""
        invalid_response = """
        RESEARCH_QUESTIONS:
        1. Question one
        2. Question two

        METHODOLOGY:
        Test methodology

        EXPECTED_SECTIONS:
        1. Section one

        PRIORITY_ORDER:
        invalid, format
        """

        # Execute
        result = planner_agent._parse_plan_response("Test", invalid_response)

        # Assert
        assert result.priority_order == [1, 2]  # Falls back to default
