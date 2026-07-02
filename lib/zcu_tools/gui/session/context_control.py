"""Narrow context-control facet for shared UI and remote driving adapters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from zcu_tools.gui.session.services.context import ContextService
    from zcu_tools.gui.session.services.device import DeviceService
    from zcu_tools.gui.session.value_lookup import ScalarValue, ValueInfo
    from zcu_tools.meta_tool import MetaDict, ModuleLibrary


class ContextControlPort(Protocol):
    """Context switching, md/ml, and value-source surface for shared consumers."""

    def has_project(self) -> bool: ...

    def use_context(self, label: str) -> None: ...
    def new_context(
        self,
        bind_device: str | None = None,
        clone_from: str | None = None,
    ) -> None: ...
    def get_context_labels(self) -> list[str]: ...
    def get_active_context_label(self) -> str | None: ...

    def list_value_sources(self) -> tuple[ValueInfo, ...]: ...
    def read_value_source(
        self, key: str, type_name: str | None = None
    ) -> tuple[ValueInfo, ScalarValue]: ...

    def get_current_md(self) -> MetaDict: ...
    def get_current_ml(self) -> ModuleLibrary: ...
    def coerce_md_value(self, key: str, text: str) -> Any: ...
    def set_md_attr(self, key: str, value: Any) -> None: ...
    def del_md_attr(self, key: str) -> None: ...
    def rename_ml_module(self, old: str, new: str) -> None: ...
    def rename_ml_waveform(self, old: str, new: str) -> None: ...
    def del_ml_module(self, name: str) -> None: ...
    def del_ml_waveform(self, name: str) -> None: ...


class ContextControlFacet:
    """Composite adapter over ContextService and device-backed context creation."""

    def __init__(self, *, context: ContextService, device: DeviceService) -> None:
        self._context = context
        self._device = device

    def has_project(self) -> bool:
        return self._context.has_project()

    def use_context(self, label: str) -> None:
        self._context.use_context(label)

    def new_context(
        self,
        bind_device: str | None = None,
        clone_from: str | None = None,
    ) -> None:
        if bind_device is not None:
            unit = self._device.get_device_unit_strict(bind_device)
            value = self._device.get_device_value_for_new_context(bind_device)
        else:
            unit, value = "none", None
        self._context.new_context(value=value, unit=unit, clone_from=clone_from)

    def get_context_labels(self) -> list[str]:
        return self._context.get_context_labels()

    def get_active_context_label(self) -> str | None:
        return self._context.get_active_context_label()

    def list_value_sources(self) -> tuple[ValueInfo, ...]:
        return self._context.list_value_sources()

    def read_value_source(
        self, key: str, type_name: str | None = None
    ) -> tuple[ValueInfo, ScalarValue]:
        return self._context.read_value_source(key, type_name)

    def get_current_md(self) -> MetaDict:
        return self._context.get_current_md()

    def get_current_ml(self) -> ModuleLibrary:
        return self._context.get_current_ml()

    def coerce_md_value(self, key: str, text: str) -> Any:
        return self._context.coerce_md_value(key, text)

    def set_md_attr(self, key: str, value: Any) -> None:
        self._context.set_md_attr(key, value)

    def del_md_attr(self, key: str) -> None:
        self._context.del_md_attr(key)

    def rename_ml_module(self, old: str, new: str) -> None:
        self._context.rename_ml_module(old, new)

    def rename_ml_waveform(self, old: str, new: str) -> None:
        self._context.rename_ml_waveform(old, new)

    def del_ml_module(self, name: str) -> None:
        self._context.del_ml_module(name)

    def del_ml_waveform(self, name: str) -> None:
        self._context.del_ml_waveform(name)
