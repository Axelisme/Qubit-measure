"""Main RemoteControlAdapter startup rollback on endpoint failures."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


def test_remote_control_adapter_start_rolls_back_bind_error(qapp) -> None:
    """RemoteControlServiceBase.start() rolls back, then propagates bind errors.

    The shared runtime catches this and prints a user-friendly message. This
    test confirms the error is not swallowed at the service layer and
    measure-gui's extra listeners are not leaked.
    """

    # Import the measure-gui adapter (representative of all four).
    from zcu_tools.gui.app.measure.remote import RemoteControlAdapter
    from zcu_tools.gui.app.measure.remote.events import EVENT_SERIALIZERS
    from zcu_tools.gui.event_bus import BaseEventBus
    from zcu_tools.gui.remote.rpc_endpoint import ControlOptions, NdjsonRpcEndpoint
    from zcu_tools.gui.session.adapters.qt_owner_scheduler import QtOwnerScheduler

    bus = BaseEventBus()
    ctrl_mock = MagicMock()
    ctrl_mock.get_bus.return_value = bus
    render_view_mock = MagicMock()

    opts = ControlOptions(port=0)  # valid opts; we'll patch the underlying endpoint

    adapter = RemoteControlAdapter(
        controller=ctrl_mock,
        opts=opts,
        owner_scheduler=QtOwnerScheduler(),
        render_view=render_view_mock,
    )

    bind_error = RuntimeError(
        "NdjsonRpcEndpoint bind 127.0.0.1:0 failed: [Errno 98] Address already in use"
    )

    with (
        patch.object(
            bus, "subscribe_with_meta", wraps=bus.subscribe_with_meta
        ) as meta_subscribe,
        patch.object(
            bus, "unsubscribe_with_meta", wraps=bus.unsubscribe_with_meta
        ) as meta_unsubscribe,
        patch.object(bus, "subscribe", wraps=bus.subscribe) as legacy_subscribe,
        patch.object(bus, "unsubscribe", wraps=bus.unsubscribe) as legacy_unsubscribe,
        patch.object(NdjsonRpcEndpoint, "start", side_effect=bind_error),
        patch.object(NdjsonRpcEndpoint, "stop") as endpoint_stop,
        pytest.raises(RuntimeError, match="bind"),
    ):
        adapter.start()

    endpoint_stop.assert_not_called()
    assert len(adapter._bus_subs) == 0
    assert meta_subscribe.call_count == len(EVENT_SERIALIZERS)
    assert {call.args[0] for call in meta_subscribe.call_args_list} == set(
        EVENT_SERIALIZERS
    )
    assert meta_unsubscribe.call_count == meta_subscribe.call_count
    for subscribe_call in meta_subscribe.call_args_list:
        assert meta_unsubscribe.call_args_list.count(
            subscribe_call
        ) == meta_subscribe.call_args_list.count(subscribe_call)
    assert legacy_unsubscribe.call_count == legacy_subscribe.call_count
    for subscribe_call in legacy_subscribe.call_args_list:
        assert legacy_unsubscribe.call_args_list.count(
            subscribe_call
        ) == legacy_subscribe.call_args_list.count(subscribe_call)
    ctrl_mock.add_diagnostic_sink.assert_called_once_with(adapter)
    ctrl_mock.remove_diagnostic_sink.assert_called_once_with(adapter)


def test_remote_control_adapter_start_fails_fast_and_rolls_back_event_subscription(
    qapp,
) -> None:
    from zcu_tools.gui.app.measure.remote import RemoteControlAdapter
    from zcu_tools.gui.app.measure.remote.events import EVENT_SERIALIZERS
    from zcu_tools.gui.event_bus import BaseEventBus, EventMeta
    from zcu_tools.gui.remote.rpc_endpoint import ControlOptions, NdjsonRpcEndpoint
    from zcu_tools.gui.session.adapters.qt_owner_scheduler import QtOwnerScheduler

    class FailingEventBus(BaseEventBus):
        def __init__(self) -> None:
            super().__init__()
            self.subscribe_count = 0

        def subscribe_with_meta(
            self,
            payload_type: type[Any],
            cb: Callable[[Any, EventMeta], None],
        ) -> Any:
            self.subscribe_count += 1
            if self.subscribe_count == 2:
                raise RuntimeError("subscribe failed")
            return super().subscribe_with_meta(payload_type, cb)

    bus = FailingEventBus()
    ctrl_mock = MagicMock()
    ctrl_mock.get_bus.return_value = bus
    adapter = RemoteControlAdapter(
        controller=ctrl_mock,
        opts=ControlOptions(port=0),
        owner_scheduler=QtOwnerScheduler(),
        render_view=MagicMock(),
    )

    with (
        patch.object(
            bus, "subscribe_with_meta", wraps=bus.subscribe_with_meta
        ) as meta_subscribe,
        patch.object(
            bus, "unsubscribe_with_meta", wraps=bus.unsubscribe_with_meta
        ) as meta_unsubscribe,
        patch.object(bus, "subscribe", wraps=bus.subscribe) as legacy_subscribe,
        patch.object(bus, "unsubscribe", wraps=bus.unsubscribe) as legacy_unsubscribe,
        patch.object(NdjsonRpcEndpoint, "start") as endpoint_start,
        pytest.raises(RuntimeError, match="subscribe failed"),
    ):
        adapter.start()

    endpoint_start.assert_not_called()
    assert len(adapter._bus_subs) == 0
    assert meta_subscribe.call_count == 2
    assert all(
        call.args[0] in EVENT_SERIALIZERS for call in meta_subscribe.call_args_list
    )
    # The second attempt raises before registration; only the first needs cleanup.
    assert meta_unsubscribe.call_args_list == [meta_subscribe.call_args_list[0]]
    assert legacy_unsubscribe.call_count == legacy_subscribe.call_count
    for subscribe_call in legacy_subscribe.call_args_list:
        assert legacy_unsubscribe.call_args_list.count(
            subscribe_call
        ) == legacy_subscribe.call_args_list.count(subscribe_call)


def test_remote_control_adapter_start_rolls_back_advertise_error(qapp) -> None:
    """RemoteControlServiceBase.start() also rolls back after advertise failure."""

    from zcu_tools.gui.app.measure.remote import RemoteControlAdapter
    from zcu_tools.gui.app.measure.remote.events import EVENT_SERIALIZERS
    from zcu_tools.gui.event_bus import BaseEventBus
    from zcu_tools.gui.remote.rpc_endpoint import ControlOptions, NdjsonRpcEndpoint
    from zcu_tools.gui.session.adapters.qt_owner_scheduler import QtOwnerScheduler

    bus = BaseEventBus()
    ctrl_mock = MagicMock()
    ctrl_mock.get_bus.return_value = bus
    adapter = RemoteControlAdapter(
        controller=ctrl_mock,
        opts=ControlOptions(port=0, app_slug="measure"),
        owner_scheduler=QtOwnerScheduler(),
        render_view=MagicMock(),
    )
    advertise_error = RuntimeError("discovery write failed")

    with (
        patch.object(
            bus, "subscribe_with_meta", wraps=bus.subscribe_with_meta
        ) as meta_subscribe,
        patch.object(
            bus, "unsubscribe_with_meta", wraps=bus.unsubscribe_with_meta
        ) as meta_unsubscribe,
        patch.object(bus, "subscribe", wraps=bus.subscribe) as legacy_subscribe,
        patch.object(bus, "unsubscribe", wraps=bus.unsubscribe) as legacy_unsubscribe,
        patch.object(NdjsonRpcEndpoint, "start", return_value=12345),
        patch.object(NdjsonRpcEndpoint, "stop") as endpoint_stop,
        patch(
            "zcu_tools.gui.remote.control_service.write_session",
            side_effect=advertise_error,
        ),
        pytest.raises(RuntimeError, match="discovery write failed"),
    ):
        adapter.start()

    endpoint_stop.assert_called_once_with()
    assert len(adapter._bus_subs) == 0
    assert meta_subscribe.call_count == len(EVENT_SERIALIZERS)
    assert {call.args[0] for call in meta_subscribe.call_args_list} == set(
        EVENT_SERIALIZERS
    )
    assert meta_unsubscribe.call_count == meta_subscribe.call_count
    for subscribe_call in meta_subscribe.call_args_list:
        assert meta_unsubscribe.call_args_list.count(
            subscribe_call
        ) == meta_subscribe.call_args_list.count(subscribe_call)
    assert legacy_unsubscribe.call_count == legacy_subscribe.call_count
    for subscribe_call in legacy_subscribe.call_args_list:
        assert legacy_unsubscribe.call_args_list.count(
            subscribe_call
        ) == legacy_subscribe.call_args_list.count(subscribe_call)
    ctrl_mock.add_diagnostic_sink.assert_called_once_with(adapter)
    ctrl_mock.remove_diagnostic_sink.assert_called_once_with(adapter)
