"""Tests for UiPathLlamaIndexRuntimeFactory."""

from uipath.runtime import UiPathRuntimeContext

from uipath_llamaindex.runtime.factory import UiPathLlamaIndexRuntimeFactory


async def test_settings_report_the_framework():
    factory = UiPathLlamaIndexRuntimeFactory(context=UiPathRuntimeContext())
    settings = await factory.get_settings()
    assert settings is not None
    assert settings.agent_type == "uipath_coded"
    assert settings.agent_framework == "llamaindex"
