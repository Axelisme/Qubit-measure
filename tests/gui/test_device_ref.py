"""Tests for DeviceRefSpec, DeviceRefLiveField, and DeviceService.DEVICE_CHANGED."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from zcu_tools.gui.adapter import DeviceRefSpec, DirectValue
from zcu_tools.gui.event_bus import DeviceChangedPayload, EventBus, GuiEvent
from zcu_tools.gui.live_model import DeviceRefLiveField, LiveModelEnv


def _make_env(device_names: list[str] | None = None) -> LiveModelEnv:
    ctrl = MagicMock()
    ctrl.list_device_names.return_value = list(device_names or [])
    return LiveModelEnv(ctrl=ctrl)


def _make_field(
    initial_name: str = "", device_names: list[str] | None = None
) -> DeviceRefLiveField:
    spec = DeviceRefSpec(label="Flux Device")
    env = _make_env(device_names)
    initial = DirectValue(initial_name) if initial_name else None
    return DeviceRefLiveField(spec, env, initial)


# ---------------------------------------------------------------------------
# Validity
# ---------------------------------------------------------------------------


def test_device_ref_valid_when_device_exists():
    field = _make_field("flux_yoko", device_names=["flux_yoko"])
    assert field.is_valid()
    assert field.get_chosen_name() == "flux_yoko"


def test_device_ref_invalid_when_device_missing():
    field = _make_field("flux_yoko", device_names=[])
    assert not field.is_valid()


def test_device_ref_invalid_when_empty():
    field = _make_field("", device_names=["flux_yoko"])
    assert not field.is_valid()


# ---------------------------------------------------------------------------
# set_chosen_name / set_value
# ---------------------------------------------------------------------------


def test_set_chosen_name_emits_on_change():
    events: list = []
    field = _make_field("dev_a", device_names=["dev_a", "dev_b"])
    field.on_change.connect(lambda v: events.append(v))
    field.set_chosen_name("dev_b")
    assert len(events) == 1
    assert isinstance(events[0], DirectValue)
    assert events[0].value == "dev_b"


def test_set_value_direct_value():
    field = _make_field("", device_names=["dev_a"])
    field.set_value(DirectValue("dev_a"))
    assert field.get_chosen_name() == "dev_a"


def test_set_value_invalid_type_raises():
    field = _make_field("", device_names=[])
    with pytest.raises(TypeError):
        field.set_value(42)


# ---------------------------------------------------------------------------
# refresh_external with DEVICE_CHANGED
# ---------------------------------------------------------------------------


def test_refresh_external_device_changed_updates_validity():
    validity_events: list[bool] = []
    field = _make_field("flux_yoko", device_names=[])
    field.on_validity_changed.connect(lambda v: validity_events.append(v))
    assert not field.is_valid()

    # Simulate device being registered by updating the env-side mock.
    field.env.ctrl.list_device_names.return_value = ["flux_yoko"]  # type: ignore[attr-defined]
    field.refresh_external(GuiEvent.DEVICE_CHANGED)

    assert field.is_valid()
    assert True in validity_events


# ---------------------------------------------------------------------------
# GuiEvent.DEVICE_CHANGED emitted by DeviceService
# ---------------------------------------------------------------------------


def test_device_service_emits_device_changed_on_register():

    from zcu_tools.gui.services.device import DeviceService, RegisterDeviceRequest
    from zcu_tools.gui.state import ExpContext, State

    state = State(
        ExpContext(md=MagicMock(), ml=MagicMock(), soc=None, soccfg=None, result_dir="")
    )
    bus = EventBus()
    received: list = []
    bus.subscribe(GuiEvent.DEVICE_CHANGED, lambda p: received.append(p))

    svc = DeviceService(
        state,
        bus,
        driver_factory=lambda type_name, address: MagicMock(),
    )
    with patch("zcu_tools.device.GlobalDeviceManager.register_device"):
        svc.register_device(
            RegisterDeviceRequest(type_name="FakeDevice", name="dev1", address="")
        )

    assert len(received) == 1
    assert isinstance(received[0], DeviceChangedPayload)


def test_device_service_emits_device_changed_on_drop():

    from zcu_tools.gui.services.device import DeviceService
    from zcu_tools.gui.state import ExpContext, State

    state = State(
        ExpContext(md=MagicMock(), ml=MagicMock(), soc=None, soccfg=None, result_dir="")
    )
    bus = EventBus()
    received: list = []
    bus.subscribe(GuiEvent.DEVICE_CHANGED, lambda p: received.append(p))

    from zcu_tools.device.fake import FakeDeviceInfo

    svc = DeviceService(state, bus)
    device = MagicMock()
    with (
        patch(
            "zcu_tools.device.manager.GlobalDeviceManager.get_info",
            return_value=FakeDeviceInfo(address=""),
        ),
        patch(
            "zcu_tools.device.manager.GlobalDeviceManager.get_device",
            return_value=device,
        ),
        patch("zcu_tools.device.GlobalDeviceManager.drop_device"),
    ):
        svc.drop_device("dev1")

    device.close.assert_called_once_with()
    assert len(received) == 1
    assert isinstance(received[0], DeviceChangedPayload)
