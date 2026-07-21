"""Tests for the UiPath LlamaIndex event models."""

from workflows.events import InputRequiredEvent

from uipath_llamaindex.models import (
    CreateTaskEvent,
    InvokeProcessEvent,
    WaitJobEvent,
    WaitTaskEvent,
)


def test_event_models_are_input_required_events():
    # Each event mixes a UiPath platform action with InputRequiredEvent so the
    # workflow engine treats it as a human/external interaction point.
    for event_cls in (
        CreateTaskEvent,
        WaitTaskEvent,
        InvokeProcessEvent,
        WaitJobEvent,
    ):
        assert issubclass(event_cls, InputRequiredEvent)


def test_invoke_process_event_carries_process_fields():
    event = InvokeProcessEvent(name="MyProcess", input_arguments={"topic": "UiPath"})

    assert event.name == "MyProcess"
    assert event.input_arguments == {"topic": "UiPath"}
