"""Shared type vocabulary for EventBus wire serialization."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from zcu_tools.gui.event_bus import BasePayload

WirePayload = Mapping[str, object] | None
EventSerializer = Callable[[BasePayload], WirePayload]


def wire_event_name(payload_type: type[BasePayload]) -> str:
    """Return the stable wire name declared by a payload type."""
    return payload_type.EVENT.value
