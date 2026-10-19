"""Tests for UiPathAgentFrameworkRuntimeFactory."""

from uipath.runtime import UiPathRuntimeContext

from uipath_agent_framework.runtime.factory import UiPathAgentFrameworkRuntimeFactory


async def test_settings_report_the_framework():
    factory = UiPathAgentFrameworkRuntimeFactory(context=UiPathRuntimeContext())
    settings = await factory.get_settings()
    assert settings is not None
    assert settings.agent_type == "uipath_coded"
    assert settings.agent_framework == "agent-framework"
