"""Research Planning Agent for Deep Research Workflow."""

from llama_index.core.llms import LLM

from .data_models import ResearchPlan


class ResearchPlannerAgent:
    """Create structured research plans."""

    def __init__(self, llm: LLM):
        self.llm = llm

    async def create_plan(self, topic: str, context: str = "") -> ResearchPlan:
        """Create a structured research plan for the given topic."""
        planning_prompt = f"""
        You are a research planning expert. Create a comprehensive \
research plan for the topic: "{topic}"

        Additional context: {context}

        Generate a structured plan that includes:
        1. 5-7 specific research questions that will thoroughly explore the topic
        2. A clear methodology for conducting the research
        3. Expected sections for the final report
        4. Priority order for the research questions (1=highest priority)

        Format your response as:
        RESEARCH_QUESTIONS:
        1. [question]
        2. [question]
        ...

        METHODOLOGY:
        [methodology description]

        EXPECTED_SECTIONS:
        1. [section name]
        2. [section name]
        ...

        PRIORITY_ORDER:
        [comma-separated list of question numbers in priority order]
        """

        response = await self.llm.acomplete(planning_prompt)
        return self._parse_plan_response(topic, str(response))

    def _parse_plan_response(self, topic: str, response: str) -> ResearchPlan:
        """Parse the LLM response into a ResearchPlan object."""
        lines = response.strip().split("\n")

        research_questions = []
        methodology = ""
        expected_sections = []
        priority_order = []

        current_section = None

        for line in lines:
            line = line.strip()
            if line.startswith("RESEARCH_QUESTIONS:"):
                current_section = "questions"
            elif line.startswith("METHODOLOGY:"):
                current_section = "methodology"
            elif line.startswith("EXPECTED_SECTIONS:"):
                current_section = "sections"
            elif line.startswith("PRIORITY_ORDER:"):
                current_section = "priority"
            elif line and current_section:
                if current_section == "questions" and line[0].isdigit():
                    question = line.split(".", 1)[1].strip() if "." in line else line
                    research_questions.append(question)
                elif current_section == "methodology":
                    methodology += line + " "
                elif current_section == "sections" and line[0].isdigit():
                    section = line.split(".", 1)[1].strip() if "." in line else line
                    expected_sections.append(section)
                elif current_section == "priority":
                    try:
                        priority_order = [int(x.strip()) for x in line.split(",")]
                    except ValueError:
                        priority_order = list(range(1, len(research_questions) + 1))

        if not priority_order:
            priority_order = list(range(1, len(research_questions) + 1))

        return ResearchPlan(
            topic=topic,
            research_questions=research_questions,
            methodology=methodology.strip(),
            expected_sections=expected_sections,
            priority_order=priority_order,
        )
