from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Protocol, cast

from zcu_tools.gui.cfg import CfgSchema, CfgSectionSpec, CfgSectionValue
from zcu_tools.gui.cfg.binding import CfgDraft, ResolvedReference
from zcu_tools.gui.session.expression import evaluate_numeric_expr
from zcu_tools.meta_tool import MetaDict, ModuleLibrary

from .module_adapter import (
    module_cfg_shape_label,
    module_cfg_to_value,
    waveform_cfg_shape_label,
    waveform_cfg_to_value,
)

_DEVICES_SOURCE = "devices"
_MATERIALIZABLE_MODULE_LABELS = frozenset({"Pulse", "Pulse Readout"})

_ReferenceConverter = Callable[[object], tuple[CfgSectionSpec, CfgSectionValue]]


class AutofluxCfgBindingHost(Protocol):
    def get_current_md(self) -> MetaDict: ...

    def get_current_ml(self) -> ModuleLibrary: ...

    def list_device_names(self) -> list[str]: ...


class AutofluxCfgBindings:
    """Autoflux-local cfg expression, option, and reference policy."""

    def __init__(self, host: AutofluxCfgBindingHost) -> None:
        self._host = host

    def new_draft(self, schema: CfgSchema) -> CfgDraft:
        return CfgDraft(
            schema,
            evaluate_expression=self.evaluate_expression,
            provide_options=self.provide_options,
            references=self,
        )

    def evaluate_expression(self, expression: str) -> int | float:
        return evaluate_numeric_expr(expression, self._host.get_current_md())

    def provide_options(self, source_id: str) -> Sequence[object]:
        if source_id == _DEVICES_SOURCE:
            return self._host.list_device_names()
        raise RuntimeError(f"Unsupported autoflux cfg option source {source_id!r}")

    def keys(self, kind: str, allowed_labels: frozenset[str]) -> Sequence[str]:
        store, _ = self._store_and_converter(kind)
        shape_label = self._shape_label_resolver(kind)
        compatible: list[str] = []
        for key, value in store.items():
            label = shape_label(value)
            if label in allowed_labels and self._can_materialize(kind, label):
                compatible.append(key)
        return tuple(sorted(compatible))

    def resolve(self, kind: str, key: str) -> ResolvedReference | None:
        store, converter = self._store_and_converter(kind)
        if key not in store:
            return None
        value = store[key]
        label = self._shape_label_resolver(kind)(value)
        if not self._can_materialize(kind, label):
            return ResolvedReference(label=label, value=None)
        spec, section_value = converter(value)
        if spec.label != label:
            raise RuntimeError(
                f"Autoflux {kind} converter returned shape {spec.label!r}; "
                f"expected {label!r}"
            )
        return ResolvedReference(label=label, value=section_value)

    @staticmethod
    def _shape_label_resolver(kind: str) -> Callable[[object], str]:
        if kind == "module":
            return module_cfg_shape_label
        if kind == "waveform":
            return waveform_cfg_shape_label
        raise RuntimeError(f"Unsupported autoflux cfg reference kind {kind!r}")

    @staticmethod
    def _can_materialize(kind: str, label: str) -> bool:
        if kind == "module":
            return label in _MATERIALIZABLE_MODULE_LABELS
        if kind == "waveform":
            return True
        raise RuntimeError(f"Unsupported autoflux cfg reference kind {kind!r}")

    def _store_and_converter(
        self, kind: str
    ) -> tuple[Mapping[str, object], _ReferenceConverter]:
        ml = self._host.get_current_ml()
        if kind == "module":
            return ml.modules, cast(_ReferenceConverter, module_cfg_to_value)
        if kind == "waveform":
            return ml.waveforms, cast(_ReferenceConverter, waveform_cfg_to_value)
        raise RuntimeError(f"Unsupported autoflux cfg reference kind {kind!r}")
