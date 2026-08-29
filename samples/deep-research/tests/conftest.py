"""
Pytest configuration and shared fixtures
"""

import sys
from unittest.mock import Mock

import pytest


# Mock llama_index modules to avoid import errors in tests
def setup_mock_modules():
    """Set up mock modules for llama_index"""
    # Create mock modules
    mock_modules = {
        "llama_index": Mock(),
        "llama_index.core": Mock(),
        "llama_index.core.llms": Mock(),
        "llama_index.core.query_engine": Mock(),
        "llama_index.core.retrievers": Mock(),
        "llama_index.core.schema": Mock(),
        "llama_index.core.workflow": Mock(),
        # Don't mock aiohttp globally - tests need to override it properly
    }

    # Add to sys.modules
    for module_name, mock_module in mock_modules.items():
        sys.modules[module_name] = mock_module

    # Set up specific mock classes
    from tests.fixtures import MockLLM, MockOpenAI, MockQueryEngine

    sys.modules["llama_index.core.llms"].LLM = MockLLM
    sys.modules["llama_index.core.llms"].OpenAI = MockOpenAI
    sys.modules["llama_index.core.llms"].MockLLM = MockLLM
    sys.modules["llama_index.core.query_engine"].BaseQueryEngine = MockQueryEngine
    sys.modules["llama_index.core.query_engine"].RetrieverQueryEngine = MockQueryEngine

    # Mock workflow classes
    class MockEvent:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

        def get(self, key, default=None):
            return getattr(self, key, default)

    class MockWorkflow:
        def __init__(self, timeout=300.0):
            self.timeout = timeout

        async def run(self, start_event):
            # Mock implementation that returns a FinalReport
            from datetime import datetime

            from agents.data_models import FinalReport

            topic = getattr(start_event, "topic", "Test Topic")
            return FinalReport(
                topic=topic,
                executive_summary="Mock executive summary for testing purposes.",
                sections={
                    "Introduction": "Mock introduction section",
                    "Analysis": "Mock analysis section",
                    "Conclusions": "Mock conclusions section",
                },
                sources=["mock_source_1.pdf", "mock_source_2.html"],
                generated_at=datetime.now(),
            )

    class MockContext:
        def __init__(self):
            self.data = {}
            self.store = MockStore()

    class MockStore:
        def __init__(self):
            self._data = {}

        async def set(self, key, value):
            self._data[key] = value

        async def get(self, key, default=None):
            return self._data.get(key, default)

    class MockStartEvent(MockEvent):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    class MockStopEvent(MockEvent):
        def __init__(self, result=None, **kwargs):
            super().__init__(**kwargs)
            self.result = result

    sys.modules["llama_index.core.workflow"].Event = MockEvent
    sys.modules["llama_index.core.workflow"].Workflow = MockWorkflow
    sys.modules["llama_index.core.workflow"].Context = MockContext
    sys.modules["llama_index.core.workflow"].StartEvent = MockStartEvent
    sys.modules["llama_index.core.workflow"].StopEvent = MockStopEvent
    sys.modules["llama_index.core.workflow"].step = lambda f: f

    # Mock aiohttp - Keep it minimal to allow test overrides
    class MockSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        def post(self, *args, **kwargs):
            class MockResponse:
                def __init__(self):
                    self.status = 200

                async def __aenter__(self):
                    return self

                async def __aexit__(self, *args):
                    pass

                async def json(self):
                    return {"results": []}

            return MockResponse()

    # Don't override aiohttp globally - let individual tests handle it
    # sys.modules["aiohttp"].ClientSession = MockSession


# Set up mocks before any tests run
setup_mock_modules()


@pytest.fixture
def mock_llm():
    """Fixture providing a mock LLM"""
    from tests.fixtures import MockLLM

    return MockLLM()


@pytest.fixture
def mock_query_engines():
    """Fixture providing mock query engines"""
    from tests.fixtures import create_mock_query_engines

    return create_mock_query_engines()
