"""
Mock LLM implementations for testing
"""


class MockLLM:
    """Simple mock LLM for testing"""

    def __init__(self, response_text: str = None):
        self.response_text = response_text or self._default_response()
        # Create a mock acomplete method that can be configured by tests
        from unittest.mock import AsyncMock

        self.acomplete = AsyncMock()
        # Set up default behavior but allow tests to override
        self._setup_default_behavior()

    def _setup_default_behavior(self):
        """Set up default acomplete behavior"""

        async def default_acomplete(prompt: str):
            class MockResponse:
                def __init__(self, text: str):
                    self.text = text

                def __str__(self):
                    return self.text

            # Return different responses based on prompt type
            if (
                "synthesis" in prompt.lower()
                or "comprehensive report" in prompt.lower()
            ):
                return MockResponse(self._synthesis_response())
            else:
                return MockResponse(self.response_text)

        self.acomplete.side_effect = default_acomplete

    def set_response(self, response_text: str):
        """Set a custom response for the next acomplete call"""
        self.response_text = response_text
        # Reset the side_effect to use the new response
        self._setup_default_behavior()

    def _default_response(self) -> str:
        """Default structured response for testing"""
        return """
        RESEARCH_QUESTIONS:
        1. What are the current trends?
        2. How does this impact business?
        3. What are the compliance considerations?

        METHODOLOGY:
        Systematic review of available literature and case studies

        EXPECTED_SECTIONS:
        1. Executive Summary
        2. Trends Analysis
        3. Impact Assessment

        PRIORITY_ORDER:
        1, 2, 3
        """

    def _synthesis_response(self) -> str:
        """Default synthesis response for testing"""
        return """
        EXECUTIVE_SUMMARY:
        This research examines AI automation in business contexts. Key findings indicate significant efficiency gains and growing adoption rates. However, compliance requirements must be carefully considered in implementation.

        SECTION: Trends Analysis
        The analysis reveals important insights about market trends and business implications. AI automation is increasingly adopted across industries with significant efficiency gains.

        SECTION: Impact Assessment
        The impact assessment shows substantial benefits including cost reduction and improved operational efficiency. Organizations report measurable improvements in productivity.

        SECTION: Compliance Framework
        Regulatory frameworks require transparency and audit trails. Compliance considerations must be integrated into automation strategies from the outset.
        """


class MockOpenAI(MockLLM):
    """Mock OpenAI LLM with proper interface"""

    def __init__(self, model="gpt-4", temperature=0.1, max_tokens=2000, **kwargs):
        super().__init__()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
