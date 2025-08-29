"""
Unit tests for DeepResearchWorkflow
"""

from datetime import datetime
from unittest.mock import patch

import pytest

from agents.data_models import FinalReport, ResearchPlan, ResearchResult
from deep_research_workflow import (
    DeepResearchWorkflow,
    PlanningCompleteEvent,
    ResearchCompleteEvent,
)


class TestDeepResearchWorkflow:

    @pytest.fixture
    def workflow(self, mock_llm, mock_query_engines):
        """Create a DeepResearchWorkflow instance for testing"""
        return DeepResearchWorkflow(
            llm=mock_llm, query_engines=mock_query_engines, timeout=30.0
        )

    @pytest.fixture
    def sample_research_plan(self):
        """Create a sample research plan for testing"""
        return ResearchPlan(
            topic="Test Topic",
            research_questions=["Question 1", "Question 2"],
            methodology="Test methodology",
            expected_sections=["Section 1", "Section 2"],
            priority_order=[1, 2],
        )

    @pytest.fixture
    def sample_research_results(self):
        """Create sample research results for testing"""
        return [
            ResearchResult("Question 1", "Findings 1", ["source1"], 0.8),
            ResearchResult("Question 2", "Findings 2", ["source2"], 0.9),
        ]

    @pytest.fixture
    def sample_final_report(self):
        """Create a sample final report for testing"""
        return FinalReport(
            topic="Test Topic",
            executive_summary="Test summary",
            sections={"Section 1": "Content 1", "Section 2": "Content 2"},
            sources=["source1", "source2"],
            generated_at=datetime.now(),
        )

    async def test_planning_step_success(self, workflow):
        """Test successful planning step execution"""
        from llama_index.core.workflow import Context, StartEvent

        # Mock the planner
        with patch.object(workflow.planner, "create_plan") as mock_create_plan:
            sample_plan = ResearchPlan(
                topic="Test Topic",
                research_questions=["Q1", "Q2"],
                methodology="Test method",
                expected_sections=["S1", "S2"],
                priority_order=[1, 2],
            )
            mock_create_plan.return_value = sample_plan

            # Create context and event
            ctx = Context()
            event = StartEvent(topic="Test Topic", context="Test context")

            # Execute
            result = await workflow.planning_step(ctx, event)

            # Assert
            assert isinstance(result, PlanningCompleteEvent)
            assert result.plan == sample_plan
            # Check that store.set was called correctly
            assert await ctx.store.get("topic") == "Test Topic"
            assert await ctx.store.get("plan") == sample_plan
            mock_create_plan.assert_called_once_with("Test Topic", "Test context")

    async def test_research_step_success(self, workflow, sample_research_plan):
        """Test successful research step execution"""
        from llama_index.core.workflow import Context

        # Mock the executor
        with patch.object(workflow.executor, "execute_research") as mock_execute:
            sample_results = [
                ResearchResult("Q1", "F1", ["s1"], 0.8),
                ResearchResult("Q2", "F2", ["s2"], 0.9),
            ]
            mock_execute.return_value = sample_results

            # Create context and event
            ctx = Context()
            event = PlanningCompleteEvent(plan=sample_research_plan)

            # Execute
            result = await workflow.research_step(ctx, event)

            # Assert
            assert isinstance(result, ResearchCompleteEvent)
            assert result.results == sample_results
            # Check that store.set was called correctly
            assert await ctx.store.get("results") == sample_results
            mock_execute.assert_called_once_with(
                sample_research_plan.research_questions  # Should be ordered by priority
            )

    async def test_synthesis_step_success(
        self, workflow, sample_research_plan, sample_research_results
    ):
        """Test successful synthesis step execution"""
        from llama_index.core.workflow import Context, StopEvent

        # Mock the synthesizer
        with patch.object(workflow.synthesizer, "synthesize_report") as mock_synthesize:
            sample_report = FinalReport(
                topic="Test Topic",
                executive_summary="Summary",
                sections={"S1": "Content"},
                sources=["source1"],
                generated_at=datetime.now(),
            )
            mock_synthesize.return_value = sample_report

            # Create context and event
            ctx = Context()
            await ctx.store.set("plan", sample_research_plan)
            event = ResearchCompleteEvent(results=sample_research_results)

            # Execute
            result = await workflow.synthesis_step(ctx, event)

            # Assert
            assert isinstance(result, StopEvent)
            assert result.result == sample_report
            mock_synthesize.assert_called_once_with(
                sample_research_plan, sample_research_results
            )

    async def test_workflow_initialization(self, mock_llm, mock_query_engines):
        """Test workflow initialization with correct agents"""
        workflow = DeepResearchWorkflow(
            llm=mock_llm, query_engines=mock_query_engines, timeout=60.0
        )

        # Assert
        assert workflow.planner is not None
        assert workflow.executor is not None
        assert workflow.synthesizer is not None
        assert workflow.timeout == 60.0

    async def test_workflow_default_timeout(self, mock_llm, mock_query_engines):
        """Test workflow with default timeout"""
        workflow = DeepResearchWorkflow(llm=mock_llm, query_engines=mock_query_engines)

        # Assert
        assert workflow.timeout == 300.0  # Default timeout

    async def test_planning_step_with_empty_context(self, workflow):
        """Test planning step with no context provided"""
        from llama_index.core.workflow import Context, StartEvent

        with patch.object(workflow.planner, "create_plan") as mock_create_plan:
            sample_plan = ResearchPlan("Topic", ["Q1"], "Method", ["S1"], [1])
            mock_create_plan.return_value = sample_plan

            # Create event without context
            ctx = Context()
            event = StartEvent(topic="Test Topic")

            # Execute
            result = await workflow.planning_step(ctx, event)

            # Assert
            assert isinstance(result, PlanningCompleteEvent)
            mock_create_plan.assert_called_once_with("Test Topic", "")

    async def test_research_step_with_priority_ordering(self, workflow):
        """Test research step respects priority ordering"""
        from llama_index.core.workflow import Context

        # Create plan with specific priority order
        plan = ResearchPlan(
            topic="Test",
            research_questions=["Q1", "Q2", "Q3"],
            methodology="Method",
            expected_sections=["S1"],
            priority_order=[3, 1, 2],  # Should execute Q3, Q1, Q2
        )

        with patch.object(workflow.executor, "execute_research") as mock_execute:
            mock_execute.return_value = []

            ctx = Context()
            event = PlanningCompleteEvent(plan=plan)

            # Execute
            await workflow.research_step(ctx, event)

            # Assert - should be called with reordered questions
            expected_order = ["Q3", "Q1", "Q2"]
            mock_execute.assert_called_once_with(expected_order)

    async def test_research_step_with_invalid_priority(self, workflow):
        """Test research step handles invalid priority indices"""
        from llama_index.core.workflow import Context

        # Create plan with invalid priority order
        plan = ResearchPlan(
            topic="Test",
            research_questions=["Q1", "Q2"],
            methodology="Method",
            expected_sections=["S1"],
            priority_order=[1, 5],  # Index 5 is out of range
        )

        with patch.object(workflow.executor, "execute_research") as mock_execute:
            mock_execute.return_value = []

            ctx = Context()
            event = PlanningCompleteEvent(plan=plan)

            # Execute
            await workflow.research_step(ctx, event)

            # Assert - should skip invalid indices
            expected_order = ["Q1"]  # Only valid index
            mock_execute.assert_called_once_with(expected_order)

    async def test_agent_error_handling(self, workflow):
        """Test workflow handles agent errors gracefully"""
        from llama_index.core.workflow import Context, StartEvent

        # Mock planner to raise an exception
        with patch.object(workflow.planner, "create_plan") as mock_create_plan:
            mock_create_plan.side_effect = Exception("Planning failed")

            ctx = Context()
            event = StartEvent(topic="Test Topic")

            # Execute and expect exception to propagate
            with pytest.raises(Exception, match="Planning failed"):
                await workflow.planning_step(ctx, event)
