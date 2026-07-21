"""Tests for ContextGroundingRetriever."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from llama_index.core.schema import QueryBundle

from uipath_llamaindex.retrievers import ContextGroundingRetriever


def _chunk(content: str, score: float):
    return SimpleNamespace(
        content=content,
        source_document_id="doc-1",
        source="source.pdf",
        page_number=3,
        score=score,
    )


def test_retrieve_maps_search_results_to_scored_nodes():
    uipath = MagicMock()
    uipath.context_grounding.search.return_value = [
        _chunk("first", 0.9),
        _chunk("second", 0.4),
    ]

    retriever = ContextGroundingRetriever(
        index_name="my-index",
        folder_path="Shared",
        uipath=uipath,
        number_of_results=5,
    )

    nodes = retriever._retrieve(QueryBundle(query_str="what is X?"))

    uipath.context_grounding.search.assert_called_once_with(
        "my-index", "what is X?", 5, folder_path="Shared", folder_key=None
    )
    assert [n.node.get_content() for n in nodes] == ["first", "second"]
    assert nodes[0].score == 0.9
    assert nodes[0].node.metadata["source_document_id"] == "doc-1"
    assert nodes[0].node.metadata["page_number"] == 3


@pytest.mark.asyncio
async def test_aretrieve_uses_async_search():
    uipath = MagicMock()
    uipath.context_grounding.search_async = AsyncMock(
        return_value=[_chunk("async chunk", 0.7)]
    )

    retriever = ContextGroundingRetriever(index_name="idx", uipath=uipath)

    nodes = await retriever._aretrieve(QueryBundle(query_str="q"))

    uipath.context_grounding.search_async.assert_awaited_once()
    assert nodes[0].node.get_content() == "async chunk"
    assert nodes[0].score == 0.7
