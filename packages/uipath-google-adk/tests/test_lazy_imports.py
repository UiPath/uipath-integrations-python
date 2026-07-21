"""Tests for lazy __getattr__ exports and middleware registration."""

import pytest

import uipath_google_adk
from uipath_google_adk import chat
from uipath_google_adk.chat import UiPathOpenAI


class TestChatLazyImports:
    def test_getattr_resolves_openai(self):
        from uipath_google_adk.chat.openai import UiPathOpenAI as Direct

        assert chat.UiPathOpenAI is Direct

    def test_unknown_attribute_raises(self):
        with pytest.raises(AttributeError):
            _ = chat.NotARealThing


class TestPackageLazyImports:
    def test_top_level_reexports_from_chat(self):
        assert uipath_google_adk.UiPathOpenAI is UiPathOpenAI

    def test_unknown_attribute_raises(self):
        with pytest.raises(AttributeError):
            _ = uipath_google_adk.Nonexistent


class TestMiddlewareRegistration:
    def test_register_middleware_registers_new_hook(self, monkeypatch):
        from uipath._cli.middlewares import Middlewares

        from uipath_google_adk.middlewares import register_middleware

        captured = {}
        monkeypatch.setattr(
            Middlewares,
            "register",
            classmethod(lambda cls, name, fn: captured.update({name: fn})),
        )
        register_middleware()
        assert "new" in captured
