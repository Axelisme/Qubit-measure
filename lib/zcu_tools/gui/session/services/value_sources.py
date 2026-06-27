"""Session value-source registration.

``ValueSourceBinder`` projects selected live ``SessionState`` facts into the
read-only ``ValueLookup`` registry. It is deliberately a follower service: it
subscribes to context/device/predictor events, reads cached state only, and never
commands devices or predictors.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from zcu_tools.gui.session.events import (
    ContextSwitchedPayload,
    DeviceChangedPayload,
    PredictorChangedPayload,
)
from zcu_tools.gui.session.state import DeviceState, DeviceStatus
from zcu_tools.gui.session.value_lookup import (
    UnavailableValue,
    ValueKey,
    ValueProviderSpec,
    ValueRegistry,
)

if TYPE_CHECKING:
    from zcu_tools.gui.event_bus import BaseEventBus
    from zcu_tools.gui.session.state import SessionState

_CONTEXT_OWNER = "context"
_PREDICTOR_OWNER = "predictor"
_ACTIVE_FLUX_OWNER = "device:active_flux"
_FLUX_DEVICE_DEFAULT_UNITS = {
    "FakeDevice": "none",
    "YOKOGS200": "A",
}
_ACTIVE_FLUX_PREFERRED_NAMES = ("flux_yoko",)


class ValueSourceBinder:
    """Keep built-in value-source providers synchronized with session state."""

    def __init__(
        self,
        *,
        state: SessionState,
        bus: BaseEventBus,
        registry: ValueRegistry,
    ) -> None:
        self._state = state
        self._bus = bus
        self._registry = registry

        bus.subscribe(ContextSwitchedPayload, self._on_context_switched)
        bus.subscribe(PredictorChangedPayload, self._on_predictor_changed)
        bus.subscribe(DeviceChangedPayload, self._on_device_changed)

        self.refresh_all()

    def refresh_all(self) -> None:
        self._refresh_context()
        self._refresh_predictor()
        self._refresh_all_devices()
        self._refresh_active_flux()

    def _on_context_switched(self, payload: ContextSwitchedPayload) -> None:
        del payload
        self._refresh_context()

    def _on_predictor_changed(self, payload: PredictorChangedPayload) -> None:
        del payload
        self._refresh_predictor()

    def _on_device_changed(self, payload: DeviceChangedPayload) -> None:
        if payload.name is None:
            self._refresh_all_devices()
        else:
            self._refresh_device(payload.name)
        self._refresh_active_flux()

    def _refresh_context(self) -> None:
        self._registry.replace_owner(
            _CONTEXT_OWNER,
            [
                _str_source(
                    "context.chip_name",
                    _CONTEXT_OWNER,
                    lambda: self._context_string("chip_name"),
                    "active context chip name",
                ),
                _str_source(
                    "context.qub_name",
                    _CONTEXT_OWNER,
                    lambda: self._context_string("qub_name"),
                    "active context qubit name",
                ),
                _str_source(
                    "context.res_name",
                    _CONTEXT_OWNER,
                    lambda: self._context_string("res_name"),
                    "active context resonator name",
                ),
                _str_source(
                    "context.active_label",
                    _CONTEXT_OWNER,
                    lambda: self._context_string("active_label"),
                    "active context label",
                ),
                _str_source(
                    "project.result_dir",
                    _CONTEXT_OWNER,
                    lambda: self._context_string("result_dir"),
                    "active project result directory",
                ),
                _str_source(
                    "project.database_path",
                    _CONTEXT_OWNER,
                    lambda: self._context_string("database_path"),
                    "active project database path",
                ),
            ],
        )

    def _refresh_predictor(self) -> None:
        specs: list[ValueProviderSpec] = [
            ValueProviderSpec(
                ValueKey("predictor.loaded", bool),
                lambda: self._state.exp_context.predictor is not None,
                owner=_PREDICTOR_OWNER,
                description="whether a Fluxonium predictor is loaded",
            )
        ]
        if self._state.exp_context.predictor is not None:
            specs.extend(
                [
                    _float_source(
                        "predictor.EJ",
                        _PREDICTOR_OWNER,
                        lambda: self._predictor_param(0),
                        "loaded predictor EJ in GHz",
                    ),
                    _float_source(
                        "predictor.EC",
                        _PREDICTOR_OWNER,
                        lambda: self._predictor_param(1),
                        "loaded predictor EC in GHz",
                    ),
                    _float_source(
                        "predictor.EL",
                        _PREDICTOR_OWNER,
                        lambda: self._predictor_param(2),
                        "loaded predictor EL in GHz",
                    ),
                    _float_source(
                        "predictor.flux_half",
                        _PREDICTOR_OWNER,
                        lambda: self._predictor_float("flux_half"),
                        "predictor half-flux anchor in device-value units",
                    ),
                    _float_source(
                        "predictor.flux_period",
                        _PREDICTOR_OWNER,
                        lambda: self._predictor_float("flux_period"),
                        "predictor flux period in device-value units",
                    ),
                    _float_source(
                        "predictor.flux_bias",
                        _PREDICTOR_OWNER,
                        lambda: self._predictor_float("flux_bias"),
                        "predictor flux bias",
                    ),
                ]
            )
        self._registry.replace_owner(_PREDICTOR_OWNER, specs)

    def _refresh_all_devices(self) -> None:
        live_names = {dev.name for dev in self._state.list_devices()}
        for info in self._registry.describe():
            if info.owner.startswith("device:"):
                device_name = info.owner.removeprefix("device:")
                if device_name not in live_names and device_name != "active_flux":
                    self._registry.unregister_owner(info.owner)
        for dev in self._state.list_devices():
            self._refresh_device(dev.name)

    def _refresh_device(self, name: str) -> None:
        owner = _device_owner(name)
        dev = self._state.get_device(name)
        if dev is None:
            self._registry.unregister_owner(owner)
            return

        specs: list[ValueProviderSpec] = [
            _str_source(
                f"device.{name}.status",
                owner,
                lambda name=name: self._device_status(name),
                "cached device lifecycle status",
            ),
            _str_source(
                f"device.{name}.type",
                owner,
                lambda name=name: self._device_type(name),
                "registered device type name",
            ),
            _str_source(
                f"device.{name}.address",
                owner,
                lambda name=name: self._device_address(name),
                "registered device address",
            ),
        ]
        if dev.info is not None:
            if _has_numeric_attr(dev.info, "value"):
                specs.append(
                    _float_source(
                        f"device.{name}.value",
                        owner,
                        lambda name=name: self._device_float_attr(name, "value"),
                        "cached device value",
                    )
                )
            if _has_numeric_attr(dev.info, "rampstep"):
                specs.append(
                    _float_source(
                        f"device.{name}.rampstep",
                        owner,
                        lambda name=name: self._device_float_attr(name, "rampstep"),
                        "cached device ramp step",
                    )
                )
            for attr in ("mode", "output", "label"):
                if _has_string_attr(dev.info, attr):
                    specs.append(
                        _str_source(
                            f"device.{name}.{attr}",
                            owner,
                            lambda name=name, attr=attr: self._device_str_attr(
                                name, attr
                            ),
                            f"cached device {attr}",
                        )
                    )

        self._registry.replace_owner(owner, specs)

    def _refresh_active_flux(self) -> None:
        dev = self._select_active_flux_device()
        if dev is None:
            self._registry.unregister_owner(_ACTIVE_FLUX_OWNER)
            return
        self._registry.replace_owner(
            _ACTIVE_FLUX_OWNER,
            [
                _str_source(
                    "device.active_flux.name",
                    _ACTIVE_FLUX_OWNER,
                    lambda: self._active_flux_name(),
                    "selected flux device name",
                ),
                _float_source(
                    "device.active_flux.value",
                    _ACTIVE_FLUX_OWNER,
                    lambda: self._active_flux_value(),
                    "selected flux device cached value",
                ),
                _str_source(
                    "device.active_flux.unit",
                    _ACTIVE_FLUX_OWNER,
                    lambda: self._active_flux_unit(),
                    "selected flux device unit",
                ),
            ],
        )

    def _context_string(self, attr: str) -> str:
        ctx = self._state.exp_context
        if not ctx.has_context():
            raise UnavailableValue(f"context.{attr}", "No experiment context")
        return str(getattr(ctx, attr))

    def _predictor_param(self, index: int) -> float:
        predictor = self._state.exp_context.predictor
        if predictor is None:
            raise UnavailableValue("predictor.params", "No predictor loaded")
        return float(predictor.params[index])

    def _predictor_float(self, attr: str) -> float:
        predictor = self._state.exp_context.predictor
        if predictor is None:
            raise UnavailableValue(f"predictor.{attr}", "No predictor loaded")
        return float(getattr(predictor, attr))

    def _device_status(self, name: str) -> str:
        return self._require_device(name).status.value

    def _device_type(self, name: str) -> str:
        return self._require_device(name).type_name

    def _device_address(self, name: str) -> str:
        return self._require_device(name).address

    def _device_float_attr(self, name: str, attr: str) -> float:
        info = self._require_device_info(name, f"device.{name}.{attr}")
        value = getattr(info, attr, None)
        if type(value) not in (int, float):
            raise UnavailableValue(
                f"device.{name}.{attr}", f"Device {name!r} has no numeric {attr!r}"
            )
        return float(cast(int | float, value))

    def _device_str_attr(self, name: str, attr: str) -> str:
        info = self._require_device_info(name, f"device.{name}.{attr}")
        value = getattr(info, attr, None)
        if type(value) is not str:
            raise UnavailableValue(
                f"device.{name}.{attr}", f"Device {name!r} has no string {attr!r}"
            )
        return value

    def _active_flux_name(self) -> str:
        return self._require_active_flux().name

    def _active_flux_value(self) -> float:
        dev = self._require_active_flux()
        return self._device_float_attr(dev.name, "value")

    def _active_flux_unit(self) -> str:
        return _device_unit(self._require_active_flux())

    def _select_active_flux_device(self) -> DeviceState | None:
        candidates = [
            dev
            for dev in self._state.list_devices()
            if _is_flux_capable(dev) and dev.is_connected() and dev.info is not None
        ]
        if not candidates:
            return None
        for preferred in _ACTIVE_FLUX_PREFERRED_NAMES:
            for dev in candidates:
                if dev.name == preferred:
                    return dev
        return candidates[0]

    def _require_active_flux(self) -> DeviceState:
        dev = self._select_active_flux_device()
        if dev is None:
            raise UnavailableValue(
                "device.active_flux.name", "No connected flux-capable device"
            )
        return dev

    def _require_device(self, name: str) -> DeviceState:
        dev = self._state.get_device(name)
        if dev is None:
            raise UnavailableValue(f"device.{name}", f"Device {name!r} is not known")
        return dev

    def _require_device_info(self, name: str, key: str) -> Any:
        dev = self._require_device(name)
        if dev.info is None or dev.is_memory_only():
            raise UnavailableValue(key, f"Device {name!r} has no cached info")
        return dev.info


def _str_source(
    key: str,
    owner: str,
    provider: Callable[[], str],
    description: str,
) -> ValueProviderSpec[str]:
    return ValueProviderSpec(
        key=ValueKey(key, str),
        provider=provider,
        owner=owner,
        description=description,
    )


def _float_source(
    key: str,
    owner: str,
    provider: Callable[[], float],
    description: str,
) -> ValueProviderSpec[float]:
    return ValueProviderSpec(
        key=ValueKey(key, float),
        provider=provider,
        owner=owner,
        description=description,
    )


def _device_owner(name: str) -> str:
    return f"device:{name}"


def _has_numeric_attr(info: Any, attr: str) -> bool:
    value = getattr(info, attr, None)
    return type(value) in (int, float)


def _has_string_attr(info: Any, attr: str) -> bool:
    return type(getattr(info, attr, None)) is str


def _is_flux_capable(dev: DeviceState) -> bool:
    return dev.type_name in _FLUX_DEVICE_DEFAULT_UNITS and _has_numeric_attr(
        dev.info, "value"
    )


def _device_unit(dev: DeviceState) -> str:
    if dev.type_name == "YOKOGS200" and dev.info is not None:
        return "V" if getattr(dev.info, "mode", None) == "voltage" else "A"
    return _FLUX_DEVICE_DEFAULT_UNITS.get(dev.type_name, "none")
