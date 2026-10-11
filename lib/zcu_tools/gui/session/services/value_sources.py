"""Session value-source registration.

``ValueSourceBinder`` projects selected live ``SessionState`` facts into the
read-only ``ValueLookup`` registry. It is deliberately a follower service: it
subscribes to context/device/predictor events, reads cached state only, and never
commands devices or predictors.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from zcu_tools.gui.session.events import (
    ContextSwitchedPayload,
    DeviceChangedPayload,
)
from zcu_tools.gui.session.value_lookup import (
    ScalarValue,
    UnavailableValue,
    ValueKey,
    ValueProviderSpec,
    ValueRegistry,
)

if TYPE_CHECKING:
    from zcu_tools.gui.event_bus import BaseEventBus
    from zcu_tools.gui.session.state import DeviceState, SessionState

_CONTEXT_OWNER = "context"


@dataclass(frozen=True)
class ValueSourceSnapshot:
    revision: int
    values: dict[str, ScalarValue]


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
        self._capture_revision = 0
        self._captures: dict[str, dict[str, ScalarValue]] = {}

        bus.subscribe(ContextSwitchedPayload, self._on_context_switched)
        bus.subscribe(DeviceChangedPayload, self._on_device_changed)

        self.refresh_all()

    def snapshot(self) -> ValueSourceSnapshot:
        """Detach already published cache values; never call a provider here."""
        return ValueSourceSnapshot(
            self._capture_revision,
            deepcopy(
                {
                    key: value
                    for values in self._captures.values()
                    for key, value in values.items()
                }
            ),
        )

    def _publish_owner(
        self, owner: str, providers: Iterable[ValueProviderSpec[Any]]
    ) -> None:
        specs = tuple(providers)
        values: dict[str, ScalarValue] = {}
        for spec in specs:
            try:
                values[spec.key.path] = spec.provider()
            except UnavailableValue:
                continue
        self._registry.replace_owner(owner, specs)
        self._captures[owner] = values
        self._capture_revision += 1

    def _unregister_owner(self, owner: str) -> None:
        self._registry.unregister_owner(owner)
        self._captures.pop(owner, None)
        self._capture_revision += 1

    def refresh_all(self) -> None:
        self._refresh_context()
        self._refresh_all_devices()

    def _on_context_switched(self, payload: ContextSwitchedPayload) -> None:
        del payload
        self._refresh_context()

    def _on_device_changed(self, payload: DeviceChangedPayload) -> None:
        if payload.name is None:
            self._refresh_all_devices()
        else:
            self._refresh_device(payload.name)

    def _refresh_context(self) -> None:
        self._publish_owner(
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

    def _refresh_all_devices(self) -> None:
        live_names = {dev.name for dev in self._state.list_devices()}
        for info in self._registry.describe():
            if info.owner.startswith("device:"):
                device_name = info.owner.removeprefix("device:")
                if device_name not in live_names:
                    self._unregister_owner(info.owner)
        for dev in self._state.list_devices():
            self._refresh_device(dev.name)

    def _refresh_device(self, name: str) -> None:
        owner = _device_owner(name)
        dev = self._state.get_device(name)
        if dev is None:
            self._unregister_owner(owner)
            return

        specs: list[ValueProviderSpec[Any]] = [
            _str_source(
                f"device.{name}.name",
                owner,
                lambda name=name: name,
                "registered device name",
            ),
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

        self._publish_owner(owner, specs)

    def _context_string(self, attr: str) -> str:
        ctx = self._state.session_env
        if not ctx.has_context():
            raise UnavailableValue(f"context.{attr}", "No experiment context")
        return str(getattr(ctx, attr))

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
