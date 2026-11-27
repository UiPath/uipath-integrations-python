"""Deep Research Agent using LlamaIndex Workflows with UiPath Context Grounding.

This implementation follows the deep research pattern with:
1. Research Planning Agent - Creates structured research plans
2. Research Executor Agent - Executes research tasks using UiPath grounding
3. Synthesis Agent - Synthesizes findings into comprehensive reports
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from llama_index.core.llms import LLM
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)

from agents import ResearchExecutorAgent, ResearchPlannerAgent, SynthesisAgent
from agents.data_models import FinalReport, ResearchPlan, ResearchResult


@dataclass
class PlanningCompleteEvent(Event):
    plan: ResearchPlan


@dataclass
class ResearchCompleteEvent(Event):
    results: List[ResearchResult]


@dataclass
class SynthesisCompleteEvent(Event):
    report: FinalReport


class DeepResearchWorkflow(Workflow):
    """Main workflow orchestrating the deep research process."""

    def __init__(
        self,
        llm: LLM,
        query_engines: Optional[Dict[str, BaseQueryEngine]] = None,
        timeout: float = 300.0,
    ):
        super().__init__(timeout=timeout)

        self.planner = ResearchPlannerAgent(llm)
        self.executor = ResearchExecutorAgent(query_engines, llm)
        self.synthesizer = SynthesisAgent(llm)

    @step
    async def planning_step(
        self, ctx: Context, ev: StartEvent
    ) -> PlanningCompleteEvent:
        """Step 1: Create research plan."""
        topic = ev.get("topic")
        context = ev.get("context", "")

        await ctx.store.set("topic", topic)
        print(f"🔍 Planning research for: {topic}")

        plan = await self.planner.create_plan(topic, context)
        await ctx.store.set("plan", plan)

        print(f"📋 Created plan with {len(plan.research_questions)} research questions")
        return PlanningCompleteEvent(plan=plan)

    @step
    async def research_step(
        self, ctx: Context, ev: PlanningCompleteEvent
    ) -> ResearchCompleteEvent:
        """Step 2: Execute research based on plan."""
        plan = ev.plan

        print(f"🔬 Executing research for {len(plan.research_questions)} questions...")

        ordered_questions = [
            plan.research_questions[i - 1]
            for i in plan.priority_order
            if i <= len(plan.research_questions)
        ]

        results = await self.executor.execute_research(ordered_questions)
        await ctx.store.set("results", results)

        print(f"✅ Completed research with {len(results)} results")
        return ResearchCompleteEvent(results=results)

    @step
    async def synthesis_step(
        self, ctx: Context, ev: ResearchCompleteEvent
    ) -> StopEvent:
        """Step 3: Synthesize results into final report."""
        plan = await ctx.store.get("plan")
        results = ev.results

        print("📝 Synthesizing final report...")

        report = await self.synthesizer.synthesize_report(plan, results)

        print(f"🎉 Generated final report with {len(report.sections)} sections")
        return StopEvent(result=report)

    async def run(self, topic: str, context: str = "") -> FinalReport:
        """Run the complete deep research workflow."""
        start_event = StartEvent(topic=topic, context=context)
        result = await super().run(start_event)
        return result
