"""Tests for chat package lazy exports and the requires_approval decorator."""

import pytest
from agent_framework import FunctionTool

from uipath_agent_framework import chat
from uipath_agent_framework.chat.tools import requires_approval


def test_requires_approval_bare_decorator_sets_always_require():
    @requires_approval
    def transfer(amount: float) -> str:
        """Move money."""
        return str(amount)

    assert isinstance(transfer, FunctionTool)
    assert transfer.approval_mode == "always_require"


def test_requires_approval_called_form_sets_always_require():
    @requires_approval()
    def transfer(amount: float) -> str:
        """Move money."""
        return str(amount)

    assert isinstance(transfer, FunctionTool)
    assert transfer.approval_mode == "always_require"


def test_chat_getattr_lazily_resolves_openai_client():
    from uipath_agent_framework.chat.openai import UiPathOpenAIChatClient

    name = "UiPathOpenAIChatClient"
    assert getattr(chat, name) is UiPathOpenAIChatClient


def test_chat_getattr_unknown_name_raises_attribute_error():
    name = "DoesNotExist"
    with pytest.raises(AttributeError):
        getattr(chat, name)
