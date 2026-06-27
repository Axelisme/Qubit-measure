from __future__ import annotations

import dataclasses
from typing import cast
from unittest.mock import MagicMock

import pytest
from zcu_tools.device.fake import FakeDeviceInfo
from zcu_tools.device.yoko import YOKOGS200Info
from zcu_tools.gui.app.main.adapter import ContextReadiness
from zcu_tools.gui.app.main.state import ExpContext, State
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.session.events import (
    ContextSwitchedPayload,
    DeviceChangedPayload,
    PredictorChangedPayload,
)
from zcu_tools.gui.session.services.value_sources import ValueSourceBinder
from zcu_tools.gui.session.state import DeviceState, DeviceStatus
from zcu_tools.gui.session.value_lookup import MissingValue, ValueRegistry
from zcu_tools.meta_tool import MetaDict, ModuleLibrary


def _state() -> State:
    return State(
        ExpContext(
            md=MetaDict(),
            ml=ModuleLibrary(),
            soc=None,
            soccfg=None,
            chip_name="chip",
            qub_name="q1",
            res_name="r1",
            result_dir="/result",
            database_path="/db",
            active_label="flux_0.0_A",
            readiness=ContextReadiness.ACTIVE,
        )
    )


def _binder(state: State) -> tuple[ValueRegistry, BaseEventBus, ValueSourceBinder]:
    registry = ValueRegistry()
    bus = BaseEventBus()
    binder = ValueSourceBinder(state=state, bus=bus, registry=registry)
    return registry, bus, binder


def test_context_sources_read_live_context_values() -> None:
    state = _state()
    registry, bus, _ = _binder(state)

    assert registry.get_as("context.chip_name", str) == "chip"
    assert registry.get_as("context.qub_name", str) == "q1"
    assert registry.get_as("context.active_label", str) == "flux_0.0_A"
    assert registry.get_as("project.result_dir", str) == "/result"

    state.set_context(dataclasses.replace(state.exp_context, chip_name="chip2"))
    bus.emit(ContextSwitchedPayload(md=state.exp_context.md, ml=state.exp_context.ml))

    assert registry.get_as("context.chip_name", str) == "chip2"


def test_predictor_sources_follow_predictor_lifecycle() -> None:
    state = _state()
    registry, bus, _ = _binder(state)

    assert registry.get_as("predictor.loaded", bool) is False
    with pytest.raises(MissingValue):
        registry.get_as("predictor.EJ", float)

    predictor = MagicMock()
    predictor.params = (4.0, 1.0, 0.5)
    predictor.flux_half = 0.25
    predictor.flux_period = 0.5
    predictor.flux_bias = -0.01
    state.set_context(dataclasses.replace(state.exp_context, predictor=predictor))
    bus.emit(PredictorChangedPayload())

    assert registry.get_as("predictor.loaded", bool) is True
    assert registry.get_as("predictor.EJ", float) == pytest.approx(4.0)
    assert registry.get_as("predictor.flux_bias", float) == pytest.approx(-0.01)

    state.set_context(dataclasses.replace(state.exp_context, predictor=None))
    bus.emit(PredictorChangedPayload())

    assert registry.get_as("predictor.loaded", bool) is False
    with pytest.raises(MissingValue):
        registry.get_as("predictor.EJ", float)


def test_device_sources_read_cached_state_without_polling() -> None:
    state = _state()
    registry, bus, _ = _binder(state)
    state.put_device(
        DeviceState(
            name="flux",
            type_name="FakeDevice",
            address="none",
            status=DeviceStatus.CONNECTED,
            remember=True,
            info=FakeDeviceInfo(address="none", value=1.25, output="on"),
        )
    )
    bus.emit(DeviceChangedPayload(name="flux"))

    assert registry.get_as("device.flux.status", str) == "connected"
    assert registry.get_as("device.flux.type", str) == "FakeDevice"
    assert registry.get_as("device.flux.address", str) == "none"
    assert registry.get_as("device.flux.value", float) == pytest.approx(1.25)
    assert registry.get_as("device.flux.output", str) == "on"

    state.remove_device("flux")
    bus.emit(DeviceChangedPayload(name="flux"))

    with pytest.raises(MissingValue):
        registry.get_as("device.flux.status", str)


def test_active_flux_prefers_flux_yoko_and_falls_back_to_first_connected_flux() -> None:
    state = _state()
    registry, bus, _ = _binder(state)
    state.put_device(
        DeviceState(
            name="other_flux",
            type_name="FakeDevice",
            address="none",
            status=DeviceStatus.CONNECTED,
            remember=True,
            info=FakeDeviceInfo(address="none", value=0.5),
        )
    )
    state.put_device(
        DeviceState(
            name="flux_yoko",
            type_name="YOKOGS200",
            address="GPIB::1",
            status=DeviceStatus.CONNECTED,
            remember=True,
            info=YOKOGS200Info(
                address="GPIB::1",
                mode="voltage",
                value=0.003,
            ),
        )
    )
    bus.emit(DeviceChangedPayload(name=None))

    assert registry.get_as("device.active_flux.name", str) == "flux_yoko"
    assert registry.get_as("device.active_flux.value", float) == pytest.approx(0.003)
    assert registry.get_as("device.active_flux.unit", str) == "V"

    state.put_device(
        dataclasses.replace(
            cast(DeviceState, state.get_device("flux_yoko")),
            status=DeviceStatus.MEMORY_ONLY,
            info=None,
        )
    )
    bus.emit(DeviceChangedPayload(name="flux_yoko"))

    assert registry.get_as("device.active_flux.name", str) == "other_flux"
    assert registry.get_as("device.active_flux.value", float) == pytest.approx(0.5)
    assert registry.get_as("device.active_flux.unit", str) == "none"
