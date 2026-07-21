"""Tests for LlamaIndexWorkflowLoader."""

import textwrap

import pytest
from workflows import Workflow

from uipath_llamaindex.runtime.workflow import LlamaIndexWorkflowLoader

WORKFLOW_MODULE = textwrap.dedent(
    """
    from llama_index.core.workflow import StartEvent, StopEvent, Workflow, step


    class Flow(Workflow):
        @step
        async def run_step(self, ev: StartEvent) -> StopEvent:
            return StopEvent(result="done")


    workflow = Flow(timeout=60)
    not_a_workflow = object()
    """
)


def test_from_path_string_splits_file_and_variable():
    loader = LlamaIndexWorkflowLoader.from_path_string("agent", "main.py:workflow")
    assert loader.file_path == "main.py"
    assert loader.variable_name == "workflow"


def test_from_path_string_rejects_missing_separator():
    with pytest.raises(ValueError, match="Invalid path format"):
        LlamaIndexWorkflowLoader.from_path_string("agent", "main.py")


async def test_load_rejects_file_outside_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    loader = LlamaIndexWorkflowLoader("agent", "/etc/hosts", "workflow")
    with pytest.raises(ValueError, match="must be within current directory"):
        await loader.load()


async def test_load_raises_for_missing_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    loader = LlamaIndexWorkflowLoader("agent", "missing.py", "workflow")
    with pytest.raises(FileNotFoundError):
        await loader.load()


async def test_load_raises_when_variable_absent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "flow_a.py").write_text(WORKFLOW_MODULE)
    loader = LlamaIndexWorkflowLoader("agent", "flow_a.py", "nonexistent")
    with pytest.raises(AttributeError, match="not found"):
        await loader.load()


async def test_load_raises_type_error_for_non_workflow(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "flow_b.py").write_text(WORKFLOW_MODULE)
    loader = LlamaIndexWorkflowLoader("agent", "flow_b.py", "not_a_workflow")
    with pytest.raises(TypeError, match="Expected Workflow"):
        await loader.load()


async def test_load_returns_workflow_instance(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "flow_c.py").write_text(WORKFLOW_MODULE)
    loader = LlamaIndexWorkflowLoader("agent", "flow_c.py", "workflow")

    workflow = await loader.load()

    assert isinstance(workflow, Workflow)


async def test_resolve_workflow_handles_async_context_manager():
    loader = LlamaIndexWorkflowLoader("agent", "x.py", "workflow")
    entered = {"value": False}
    exited = {"value": False}

    class CtxManager:
        async def __aenter__(self):
            entered["value"] = True
            return "the-workflow"

        async def __aexit__(self, *args):
            exited["value"] = True

    resolved = await loader._resolve_workflow(CtxManager())
    assert resolved == "the-workflow"
    assert entered["value"] is True

    await loader.cleanup()
    assert exited["value"] is True
