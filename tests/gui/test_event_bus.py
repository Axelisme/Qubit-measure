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
