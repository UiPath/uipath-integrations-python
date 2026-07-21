"""Tests for ContextGroundingQueryEngine."""

from unittest.mock import MagicMock, patch

import pytest

from uipath_llamaindex.query_engines import ContextGroundingQueryEngine


def _make_engine(uipath):
    synthesizer = MagicMock()
    synthesizer.synthesize.return_value = "synthesized answer"
    engine = ContextGroundingQueryEngine(
        response_synthesizer=synthesizer,
        index_name="idx",
        folder_path="Shared",
        uipath=uipath,
    )
    return engine, synthesizer


def test_custom_query_retrieves_then_synthesizes():
    uipath = MagicMock()
    uipath.context_grounding.search.return_value = []
    engine, synthesizer = _make_engine(uipath)

    with patch.object(engine._retriever, "retrieve", return_value=["node"]) as retrieve:
        result = engine.custom_query("question")

    retrieve.assert_called_once_with("question")
    synthesizer.synthesize.assert_called_once_with("question", ["node"])
    assert result == "synthesized answer"


@pytest.mark.asyncio
async def test_acustom_query_uses_async_retrieve():
    uipath = MagicMock()
    engine, synthesizer = _make_engine(uipath)

    async def fake_aretrieve(query_str):
        return ["async-node"]

    with patch.object(engine._retriever, "aretrieve", side_effect=fake_aretrieve):
        result = await engine.acustom_query("question")

    synthesizer.synthesize.assert_called_once_with("question", ["async-node"])
    assert result == "synthesized answer"
