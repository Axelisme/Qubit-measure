"""Test adapters from typed completion facts to compact callback assertions."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from zcu_tools.gui.app.main.events.completion import AnalyzeFailedPayload
from zcu_tools.gui.app.main.events.tab import (
    TabInteractionChangedPayload,
    TabInteractionFact,
)
from zcu_tools.gui.session.events import (
    ConnectionFinishedPayload,
    DeviceOperationFinishedPayload,
    DeviceSetupFinishedPayload,
)


def on_connection_finished(service: Any, callback: Callable[[], object]) -> None:
    service._bus.subscribe(
        ConnectionFinishedPayload,
        lambda payload: callback() if payload.success else None,
    )


def on_connection_failed(service: Any, callback: Callable[[str], object]) -> None:
    service._bus.subscribe(
        ConnectionFinishedPayload,
        lambda payload: (
            callback(payload.error_message or "") if not payload.success else None
        ),
    )


def on_device_connected(service: Any, callback: Callable[[object], object]) -> None:
    service._bus.subscribe(
        DeviceOperationFinishedPayload,
        lambda payload: (
            callback(payload)
            if payload.success and payload.action == "connect"
            else None
        ),
    )


def on_device_disconnected(service: Any, callback: Callable[[object], object]) -> None:
    service._bus.subscribe(
        DeviceOperationFinishedPayload,
        lambda payload: (
            callback(payload)
            if payload.success and payload.action == "disconnect"
            else None
        ),
    )


def on_device_operation_failed(
    service: Any, callback: Callable[[str, str], object]
) -> None:
    service._bus.subscribe(
        DeviceOperationFinishedPayload,
        lambda payload: (
            callback(payload.name, payload.error_message or "")
            if not payload.success
            else None
        ),
    )


def _on_setup(
    service: Any,
    outcome: str,
    callback: Callable[[DeviceSetupFinishedPayload], object],
) -> None:
    service._bus.subscribe(
        DeviceSetupFinishedPayload,
        lambda payload: callback(payload) if payload.outcome == outcome else None,
    )


def on_setup_finished(service: Any, callback: Callable[[str], object]) -> None:
    _on_setup(service, "finished", lambda payload: callback(payload.name))


def on_setup_failed(service: Any, callback: Callable[[str, str], object]) -> None:
    _on_setup(
        service,
        "failed",
        lambda payload: callback(payload.name, payload.error_message or ""),
    )


def on_setup_cancelled(service: Any, callback: Callable[[str], object]) -> None:
    _on_setup(service, "cancelled", lambda payload: callback(payload.name))


def on_analyze_finished(
    service: Any, callback: Callable[[str, object], object]
) -> None:
    service._bus.subscribe(
        TabInteractionChangedPayload,
        lambda payload: (
            callback(
                payload.tab_id, service._state.get_tab(payload.tab_id).analyze_result
            )
            if payload.fact is TabInteractionFact.PRIMARY_ANALYZE_SUCCEEDED
            else None
        ),
    )


def on_post_analyze_finished(
    service: Any, callback: Callable[[str, object], object]
) -> None:
    service._bus.subscribe(
        TabInteractionChangedPayload,
        lambda payload: (
            callback(
                payload.tab_id,
                service._state.get_tab(payload.tab_id).post_analyze_result,
            )
            if payload.fact is TabInteractionFact.POST_ANALYZE_SUCCEEDED
            else None
        ),
    )


def on_analyze_failed(service: Any, callback: Callable[[str, str], object]) -> None:
    service._bus.subscribe(
        AnalyzeFailedPayload,
        lambda payload: (
            callback(payload.tab_id, payload.error_message)
            if payload.stage == "primary"
            else None
        ),
    )


def on_post_analyze_failed(
    service: Any, callback: Callable[[str, str], object]
) -> None:
    service._bus.subscribe(
        AnalyzeFailedPayload,
        lambda payload: (
            callback(payload.tab_id, payload.error_message)
            if payload.stage == "post"
            else None
        ),
    )
