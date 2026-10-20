"""Explicit instrument adapter catalog; unknown types have no factory or panel."""

from zcu_tools.gui.session.devices.base import BaseDeviceAdapter
from zcu_tools.gui.session.devices.fake import FakeDeviceAdapter
from zcu_tools.gui.session.devices.sgs100a import SGS100AAdapter
from zcu_tools.gui.session.devices.yoko import YOKOGS200Adapter

_ADAPTERS: tuple[BaseDeviceAdapter, ...] = (
    FakeDeviceAdapter(),
    YOKOGS200Adapter(),
    SGS100AAdapter(),
)


def device_adapters() -> tuple[BaseDeviceAdapter, ...]:
    """Return known adapters in stable UI order; this does not connect drivers."""
    return _ADAPTERS


def get_device_adapter(type_name: str) -> BaseDeviceAdapter | None:
    """Return the adapter for an exact driver class name, or None if unknown."""
    return next(
        (adapter for adapter in _ADAPTERS if adapter.driver_cls.__name__ == type_name),
        None,
    )
