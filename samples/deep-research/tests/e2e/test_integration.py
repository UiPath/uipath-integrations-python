"""
Integration tests for agent interactions
"""

from agents.research_executor import ResearchExecutorAgent
from agents.research_planner import ResearchPlannerAgent
from agents.synthesis_agent import SynthesisAgent


class TestAgentIntegration:
    """Test how agents work together"""

    async def test_planner_to_executor_flow(self, mock_llm, mock_query_engines):
        """Test the flow from planner to executor"""
        # Create agents
        planner = ResearchPlannerAgent(mock_llm)
        executor = ResearchExecutorAgent(
            mock_query_engines, mock_llm, use_web_search=False
        )

        # Plan research
        topic = "AI automation in business"
        context = "Focus on efficiency and compliance"
        plan = await planner.create_plan(topic, context)

        # Execute research based on plan
        results = await executor.execute_research(plan.research_questions)

        # Assert results match plan
        assert len(results) == len(plan.research_questions)
        for i, result in enumerate(results):
            assert result.question == plan.research_questions[i]
            assert len(result.findings) > 0

    async def test_executor_to_synthesis_flow(self, mock_llm, mock_query_engines):
        """Test the flow from executor to synthesis"""
        # Create agents
        executor = ResearchExecutorAgent(
            mock_query_engines, mock_llm, use_web_search=False
        )
        synthesizer = SynthesisAgent(mock_llm)

        # Create a simple research plan
        from agents.data_models import ResearchPlan

        plan = ResearchPlan(
            topic="Test Topic",
            research_questions=["What are the trends?", "What are the impacts?"],
            methodology="Test methodology",
            expected_sections=["Summary", "Analysis"],
            priority_order=[1, 2],
        )

        # Execute research
        results = await executor.execute_research(plan.research_questions)

        # Synthesize results
        final_report = await synthesizer.synthesize_report(plan, results)

        # Assert synthesis worked
        assert final_report.topic == plan.topic
        assert len(final_report.executive_summary) > 0
        assert len(final_report.sources) > 0

    async def test_full_agent_pipeline(self, mock_llm, mock_query_engines):
        """Test the complete agent pipeline"""
        # Create all agents
        planner = ResearchPlannerAgent(mock_llm)
        executor = ResearchExecutorAgent(
            mock_query_engines, mock_llm, use_web_search=False
        )
        synthesizer = SynthesisAgent(mock_llm)

        # Full pipeline
        topic = "Digital transformation strategies"
        context = "Enterprise-focused analysis"

        # Step 1: Plan
        plan = await planner.create_plan(topic, context)
        assert plan.topic == topic

        # Step 2: Execute
        results = await executor.execute_research(plan.research_questions)
        assert len(results) > 0

        # Step 3: Synthesize
        final_report = await synthesizer.synthesize_report(plan, results)
        assert final_report.topic == topic
        assert len(final_report.executive_summary) > 0
