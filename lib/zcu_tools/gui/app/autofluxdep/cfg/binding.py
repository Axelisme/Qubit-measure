from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Protocol, cast

from zcu_tools.gui.cfg import CfgSchema, CfgSectionSpec, CfgSectionValue
from zcu_tools.gui.cfg.binding import CfgDraft, ResolvedReference
from zcu_tools.gui.measure_cfg import (
    ProgramCfgKind,
    ProgramShape,
    program_shape_for_input,
)
from zcu_tools.gui.session.expression import evaluate_numeric_expr
from zcu_tools.meta_tool import MetaDict, ModuleLibrary

from .module_adapter import (
    AUTOFLUX_PROGRAM_MATERIALIZATION_POLICY,
    module_cfg_to_value,
    waveform_cfg_to_value,
)

_DEVICES_SOURCE = "devices"
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
        store, catalog_kind = self._store(kind)
        compatible: list[str] = []
        for key, value in store.items():
            shape = program_shape_for_input(catalog_kind, value)
            if shape.label in allowed_labels and self._can_materialize(shape):
                compatible.append(key)
        return tuple(sorted(compatible))

    def resolve(self, kind: str, key: str) -> ResolvedReference | None:
        store, catalog_kind = self._store(kind)
        if key not in store:
            return None
        value = store[key]
        shape = program_shape_for_input(catalog_kind, value)
        if not self._can_materialize(shape):
            return ResolvedReference(label=shape.label, value=None)
        converter = self._converter(kind)
        spec, section_value = converter(value)
        if spec.label != shape.label:
            raise RuntimeError(
                f"Autoflux {kind} converter returned shape {spec.label!r}; "
                f"expected {shape.label!r}"
            )
        return ResolvedReference(label=shape.label, value=section_value)

    @staticmethod
    def _can_materialize(shape: ProgramShape) -> bool:
        policy = AUTOFLUX_PROGRAM_MATERIALIZATION_POLICY
        if shape.kind == "module":
            return shape.discriminator in policy.allowed_module_discriminators
        return shape.discriminator in policy.allowed_waveform_styles

    def _store(self, kind: str) -> tuple[Mapping[str, object], ProgramCfgKind]:
        ml = self._host.get_current_ml()
        if kind == "module":
            return ml.modules, "module"
        if kind == "waveform":
            return ml.waveforms, "waveform"
        raise RuntimeError(f"Unsupported autoflux cfg reference kind {kind!r}")

    @staticmethod
    def _converter(kind: str) -> _ReferenceConverter:
        if kind == "module":
            return cast(_ReferenceConverter, module_cfg_to_value)
        if kind == "waveform":
            return cast(_ReferenceConverter, waveform_cfg_to_value)
        raise RuntimeError(f"Unsupported autoflux cfg reference kind {kind!r}")
