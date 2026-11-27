"""
Unit tests for SynthesisAgent
"""

from datetime import datetime
from unittest.mock import Mock

import pytest

from agents.data_models import FinalReport, ResearchPlan, ResearchResult
from agents.synthesis_agent import SynthesisAgent


class TestSynthesisAgent:

    @pytest.fixture
    def synthesis_agent(self, mock_llm):
        """Create a SynthesisAgent instance for testing"""
        return SynthesisAgent(mock_llm)

    @pytest.fixture
    def sample_research_plan(self):
        """Create a sample research plan for testing"""
        return ResearchPlan(
            topic="AI Automation in Business",
            research_questions=[
                "What are current AI trends?",
                "How does AI impact efficiency?",
                "What are compliance requirements?",
            ],
            methodology="Systematic literature review",
            expected_sections=[
                "Executive Summary",
                "Trends Analysis",
                "Impact Assessment",
            ],
            priority_order=[1, 2, 3],
        )

    @pytest.fixture
    def sample_research_results(self):
        """Create sample research results for testing"""
        return [
            ResearchResult(
                question="What are current AI trends?",
                findings="AI is increasingly used in automation. Machine learning adoption is growing.",
                sources=["source1.pdf", "source2.html"],
                confidence_score=0.8,
            ),
            ResearchResult(
                question="How does AI impact efficiency?",
                findings="Studies show 30% efficiency improvement with AI automation.",
                sources=["source3.docx", "source4.pdf"],
                confidence_score=0.9,
            ),
            ResearchResult(
                question="What are compliance requirements?",
                findings="Regulatory frameworks require transparency and audit trails.",
                sources=["compliance_guide.pdf"],
                confidence_score=0.7,
            ),
        ]

    @pytest.fixture
    def sample_synthesis_response(self):
        """Sample LLM synthesis response for testing"""
        return """
        EXECUTIVE_SUMMARY:
        This research examines AI automation in business contexts. Key findings indicate significant efficiency gains and growing adoption rates. However, compliance requirements must be carefully considered in implementation.

        The analysis reveals that organizations adopting AI automation see measurable improvements in operational efficiency while facing new regulatory challenges.

        SECTION: Trends Analysis
        Current trends show rapid adoption of machine learning technologies across industries. Organizations are investing heavily in AI infrastructure and talent development.

        SECTION: Impact Assessment
        Quantitative analysis demonstrates 30% average efficiency improvements. Cost savings are significant, but implementation requires careful change management.

        SECTION: Compliance Framework
        Regulatory requirements emphasize transparency, audit trails, and ethical AI practices. Organizations must develop robust governance frameworks.
        """

    async def test_synthesize_report_success(
        self,
        synthesis_agent,
        mock_llm,
        sample_research_plan,
        sample_research_results,
        sample_synthesis_response,
    ):
        """Test successful report synthesis"""
        # Setup
        mock_llm.set_response(sample_synthesis_response)

        # Execute
        result = await synthesis_agent.synthesize_report(
            sample_research_plan, sample_research_results
        )

        # Assert
        assert isinstance(result, FinalReport)
        assert result.topic == sample_research_plan.topic
        assert len(result.executive_summary) > 0
        assert "efficiency gains" in result.executive_summary
        assert len(result.sections) == 3
        assert "Trends Analysis" in result.sections
        assert "Impact Assessment" in result.sections
        assert "Compliance Framework" in result.sections
        assert len(result.sources) > 0
        assert isinstance(result.generated_at, datetime)

    def test_format_results_for_synthesis(
        self, synthesis_agent, sample_research_results
    ):
        """Test formatting of research results for synthesis"""
        # Execute
        formatted = synthesis_agent._format_results_for_synthesis(
            sample_research_results
        )

        # Assert
        assert "RESEARCH QUESTION 1:" in formatted
        assert "What are current AI trends?" in formatted
        assert "FINDINGS:" in formatted
        assert "CONFIDENCE:" in formatted
        assert "SOURCES:" in formatted
        assert "0.80" in formatted  # Confidence score formatting

    def test_parse_synthesis_response_complete(
        self,
        synthesis_agent,
        sample_research_plan,
        sample_research_results,
        sample_synthesis_response,
    ):
        """Test parsing of complete synthesis response"""
        # Execute
        result = synthesis_agent._parse_synthesis_response(
            sample_research_plan, sample_research_results, sample_synthesis_response
        )

        # Assert
        assert result.topic == sample_research_plan.topic
        assert "efficiency gains" in result.executive_summary
        assert len(result.sections) == 3
        assert "Trends Analysis" in result.sections
        assert "rapid adoption" in result.sections["Trends Analysis"]
        assert "30% average efficiency" in result.sections["Impact Assessment"]
        assert "transparency" in result.sections["Compliance Framework"]

    def test_parse_synthesis_response_minimal(
        self, synthesis_agent, sample_research_plan, sample_research_results
    ):
        """Test parsing of minimal synthesis response"""
        minimal_response = """
        EXECUTIVE_SUMMARY:
        Brief summary of findings.

        SECTION: Single Section
        Single section content.
        """

        # Execute
        result = synthesis_agent._parse_synthesis_response(
            sample_research_plan, sample_research_results, minimal_response
        )

        # Assert
        assert result.executive_summary == "Brief summary of findings."
        assert len(result.sections) == 1
        assert "Single Section" in result.sections
        assert result.sections["Single Section"] == "Single section content."

    def test_parse_synthesis_response_no_sections(
        self, synthesis_agent, sample_research_plan, sample_research_results
    ):
        """Test parsing when no sections are present"""
        summary_only_response = """
        EXECUTIVE_SUMMARY:
        Only executive summary provided.
        No additional sections.
        """

        # Execute
        result = synthesis_agent._parse_synthesis_response(
            sample_research_plan, sample_research_results, summary_only_response
        )

        # Assert
        assert "Only executive summary" in result.executive_summary
        assert len(result.sections) == 0

    def test_source_aggregation(
        self,
        synthesis_agent,
        sample_research_plan,
        sample_research_results,
        sample_synthesis_response,
    ):
        """Test aggregation of sources from research results"""
        # Execute
        result = synthesis_agent._parse_synthesis_response(
            sample_research_plan, sample_research_results, sample_synthesis_response
        )

        # Assert
        expected_sources = {
            "source1.pdf",
            "source2.html",
            "source3.docx",
            "source4.pdf",
            "compliance_guide.pdf",
        }
        assert set(result.sources) == expected_sources

    def test_source_deduplication(
        self, synthesis_agent, sample_research_plan, sample_synthesis_response
    ):
        """Test deduplication of sources"""
        # Create results with duplicate sources
        results_with_duplicates = [
            ResearchResult("Q1", "F1", ["source1.pdf", "source2.pdf"], 0.8),
            ResearchResult(
                "Q2", "F2", ["source1.pdf", "source3.pdf"], 0.9
            ),  # Duplicate source1.pdf
        ]

        # Execute
        result = synthesis_agent._parse_synthesis_response(
            sample_research_plan, results_with_duplicates, sample_synthesis_response
        )

        # Assert
        assert len(result.sources) == 3  # Duplicates removed
        assert "source1.pdf" in result.sources
        assert "source2.pdf" in result.sources
        assert "source3.pdf" in result.sources

    async def test_llm_prompt_construction(
        self, synthesis_agent, mock_llm, sample_research_plan, sample_research_results
    ):
        """Test that the LLM prompt is constructed correctly"""
        mock_llm.acomplete.return_value = Mock(text="EXECUTIVE_SUMMARY:\nTest summary")

        # Execute
        await synthesis_agent.synthesize_report(
            sample_research_plan, sample_research_results
        )

        # Assert
        mock_llm.acomplete.assert_called_once()
        call_args = mock_llm.acomplete.call_args[0][0]

        assert sample_research_plan.topic in call_args
        assert "RESEARCH FINDINGS:" in call_args
        assert "What are current AI trends?" in call_args
        assert ", ".join(sample_research_plan.expected_sections) in call_args

    async def test_empty_results_handling(
        self, synthesis_agent, mock_llm, sample_research_plan
    ):
        """Test handling of empty research results"""
        mock_llm.acomplete.return_value = Mock(
            text="EXECUTIVE_SUMMARY:\nNo findings available"
        )

        # Execute
        result = await synthesis_agent.synthesize_report(sample_research_plan, [])

        # Assert
        assert isinstance(result, FinalReport)
        assert result.topic == sample_research_plan.topic
        assert len(result.sources) == 0
