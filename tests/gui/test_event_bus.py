from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.event_bus import (
    EventBus,
    GuiEvent,
    MdChangedPayload,
    MlChangedPayload,
)
from zcu_tools.meta_tool import MetaDict, ModuleLibrary


def test_event_bus_validates_payload_type() -> None:
    bus = EventBus()

    with pytest.raises(TypeError, match="md_changed expects MdChangedPayload"):
        bus.emit(
            GuiEvent.MD_CHANGED,
            cast(Any, MlChangedPayload(ml=ModuleLibrary())),
        )


def test_event_bus_propagates_subscriber_exceptions() -> None:
    bus = EventBus()
    bus.subscribe(
        GuiEvent.MD_CHANGED, MagicMock(side_effect=RuntimeError("subscriber failed"))
    )

    with pytest.raises(RuntimeError, match="subscriber failed"):
        bus.emit(GuiEvent.MD_CHANGED, MdChangedPayload(md=MetaDict()))


def test_event_bus_logs_all_subscriber_exceptions(
    caplog: pytest.LogCaptureFixture,
) -> None:
    bus = EventBus()
    first = MagicMock(side_effect=RuntimeError("first failed"))
    second = MagicMock(side_effect=ValueError("second failed"))
    third = MagicMock()
    bus.subscribe(GuiEvent.MD_CHANGED, first)
    bus.subscribe(GuiEvent.MD_CHANGED, second)
    bus.subscribe(GuiEvent.MD_CHANGED, third)

    with pytest.raises(RuntimeError, match="first failed"):
        bus.emit(GuiEvent.MD_CHANGED, MdChangedPayload(md=MetaDict()))

    assert first.called
    assert second.called
    assert third.called
    messages = [record.getMessage() for record in caplog.records]
    assert sum("EventBus subscriber failed" in message for message in messages) == 2
