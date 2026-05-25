"""Tests for DeviceRefSpec, DeviceRefLiveField, and DeviceService.DEVICE_CHANGED."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from zcu_tools.gui.adapter import DeviceRefSpec, DirectValue
from zcu_tools.gui.event_bus import DeviceChangedPayload, EventBus, GuiEvent
from zcu_tools.gui.live_model import DeviceRefLiveField, LiveModelEnv


def _make_env() -> LiveModelEnv:
    ctrl = MagicMock()
    return LiveModelEnv(ctrl=ctrl)


def _make_field(initial_name: str = "") -> DeviceRefLiveField:
    spec = DeviceRefSpec(label="Flux Device")
    env = _make_env()
    initial = DirectValue(initial_name) if initial_name else None
    return DeviceRefLiveField(spec, env, initial)


# ---------------------------------------------------------------------------
# Validity
# ---------------------------------------------------------------------------


def test_device_ref_valid_when_device_exists():
    with patch(
        "zcu_tools.device.manager.GlobalDeviceManager.get_all_devices",
        return_value={"flux_yoko": MagicMock()},
    ):
        field = _make_field("flux_yoko")
        assert field.is_valid()
        assert field.get_chosen_name() == "flux_yoko"


def test_device_ref_invalid_when_device_missing():
    with patch(
        "zcu_tools.device.manager.GlobalDeviceManager.get_all_devices",
        return_value={},
    ):
        field = _make_field("flux_yoko")
        assert not field.is_valid()


def test_device_ref_invalid_when_empty():
    with patch(
        "zcu_tools.device.manager.GlobalDeviceManager.get_all_devices",
        return_value={"flux_yoko": MagicMock()},
    ):
        field = _make_field("")
        assert not field.is_valid()


# ---------------------------------------------------------------------------
# set_chosen_name / set_value
# ---------------------------------------------------------------------------


def test_set_chosen_name_emits_on_change():
    events: list = []
    with patch(
        "zcu_tools.device.manager.GlobalDeviceManager.get_all_devices",
        return_value={"dev_a": MagicMock(), "dev_b": MagicMock()},
    ):
        field = _make_field("dev_a")
        field.on_change.connect(lambda v: events.append(v))
        field.set_chosen_name("dev_b")
        assert len(events) == 1
        assert isinstance(events[0], DirectValue)
        assert events[0].value == "dev_b"


def test_set_value_direct_value():
    with patch(
        "zcu_tools.device.manager.GlobalDeviceManager.get_all_devices",
        return_value={"dev_a": MagicMock()},
    ):
        field = _make_field("")
        field.set_value(DirectValue("dev_a"))
        assert field.get_chosen_name() == "dev_a"


def test_set_value_invalid_type_raises():
    with patch(
        "zcu_tools.device.manager.GlobalDeviceManager.get_all_devices",
        return_value={},
    ):
        field = _make_field("")
        with pytest.raises(TypeError):
            field.set_value(42)


# ---------------------------------------------------------------------------
# refresh_external with DEVICE_CHANGED
# ---------------------------------------------------------------------------


def test_refresh_external_device_changed_updates_validity():
    validity_events: list[bool] = []
    with patch(
        "zcu_tools.device.manager.GlobalDeviceManager.get_all_devices",
        return_value={},
    ) as mock_devices:
        field = _make_field("flux_yoko")
        field.on_validity_changed.connect(lambda v: validity_events.append(v))
        assert not field.is_valid()

        # simulate device being registered
        mock_devices.return_value = {"flux_yoko": MagicMock()}
        field.refresh_external(GuiEvent.DEVICE_CHANGED)

        assert field.is_valid()
        assert True in validity_events


# ---------------------------------------------------------------------------
# GuiEvent.DEVICE_CHANGED emitted by DeviceService
# ---------------------------------------------------------------------------


def test_device_service_emits_device_changed_on_register():
    from unittest.mock import patch

    from zcu_tools.gui.services.device import DeviceService
    from zcu_tools.gui.state import ExpContext, State

    state = State(
        ExpContext(md=MagicMock(), ml=MagicMock(), soc=None, soccfg=None, result_dir="")
    )
    bus = EventBus()
    received: list = []
    bus.subscribe(GuiEvent.DEVICE_CHANGED, lambda p: received.append(p))

    svc = DeviceService(state, bus)
    with patch("zcu_tools.device.GlobalDeviceManager.register_device"):
        svc.register_device("dev1", MagicMock())

    assert len(received) == 1
    assert isinstance(received[0], DeviceChangedPayload)


def test_device_service_emits_device_changed_on_drop():
    from unittest.mock import patch

    from zcu_tools.gui.services.device import DeviceService
    from zcu_tools.gui.state import ExpContext, State

    state = State(
        ExpContext(md=MagicMock(), ml=MagicMock(), soc=None, soccfg=None, result_dir="")
    )
    bus = EventBus()
    received: list = []
    bus.subscribe(GuiEvent.DEVICE_CHANGED, lambda p: received.append(p))

    svc = DeviceService(state, bus)
    with patch("zcu_tools.device.GlobalDeviceManager.drop_device"):
        svc.drop_device("dev1")

    assert len(received) == 1
    assert isinstance(received[0], DeviceChangedPayload)
