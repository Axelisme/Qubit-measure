from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Protocol, cast

from qtpy.QtWidgets import QLineEdit  # type: ignore[attr-defined]

from zcu_tools.gui.cfg import (
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
)
from zcu_tools.gui.cfg.binding import CfgDraft, ResolvedReference
from zcu_tools.gui.measure_cfg import ProgramCfgKind, program_shape_for_input
from zcu_tools.gui.session.expression import evaluate_numeric_expr
from zcu_tools.gui.session.ui.value_source_input import (
    SessionValueSourceInputHost,
    SessionValueSourcePort,
    ValueSourceInputController,
)
from zcu_tools.gui.session.value_lookup import (
    ScalarValue as LookupScalarValue,
)
from zcu_tools.gui.session.value_lookup import (
    ValueInfo,
    ValueRef,
    ValueTypeError,
    name_from_type,
)
from zcu_tools.meta_tool import MetaDict, ModuleLibrary

from .cfg_schemas import module_cfg_to_value, waveform_cfg_to_value

_DEVICES_SOURCE = "devices"
_ARB_WAVEFORMS_SOURCE = "arb_waveforms"

_ReferenceConverter = Callable[[object], tuple[CfgSectionSpec, CfgSectionValue]]
TextInputEnhancer = Callable[[QLineEdit], object | None]


class MeasureCfgBindingHost(Protocol):
    def get_current_md(self) -> MetaDict: ...

    def get_current_ml(self) -> ModuleLibrary: ...

    def list_device_names(self) -> list[str]: ...

    def list_arb_waveforms(self) -> list[str]: ...

    def read_value_source(
        self, key: str, type_name: str | None = None
    ) -> tuple[ValueInfo, LookupScalarValue]: ...


class MeasureCfgBindings:
    """Measure-app policy adapter for the shared mutable cfg binding."""

    def __init__(self, host: MeasureCfgBindingHost) -> None:
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
        if source_id == _ARB_WAVEFORMS_SOURCE:
            return self._host.list_arb_waveforms()
        raise RuntimeError(f"Unsupported measure cfg option source {source_id!r}")

    def keys(self, kind: str, allowed_labels: frozenset[str]) -> Sequence[str]:
        store, catalog_kind = self._store(kind)
        compatible: list[str] = []
        for key, value in store.items():
            shape = program_shape_for_input(catalog_kind, value)
            if shape.label in allowed_labels:
                compatible.append(key)
        return tuple(sorted(compatible))

    def resolve(self, kind: str, key: str) -> ResolvedReference | None:
        store, _ = self._store(kind)
        if key not in store:
            return None
        value = store[key]
        converter = self._converter(kind)
        spec, section_value = converter(value)
        return ResolvedReference(label=spec.label, value=section_value)

    def resolve_value_ref(self, ref: ValueRef, target_type: type) -> DirectValue:
        try:
            target_type_name = name_from_type(target_type)  # type: ignore[arg-type]
        except AssertionError as exc:
            raise ValueTypeError(
                ref.key,
                f"Value source {ref.key!r} cannot target unsupported scalar field type "
                f"{target_type.__name__!r}; only int, float, str, and bool fields are supported",
            ) from exc
        if ref.type_name is not None and ref.type_name != target_type_name:
            raise ValueTypeError(
                ref.key,
                f"Value source {ref.key!r} requested as {ref.type_name!r} but "
                f"target field expects {target_type_name!r}",
            )
        _, value = self._host.read_value_source(ref.key, target_type_name)
        return DirectValue(value)

    def _store(self, kind: str) -> tuple[Mapping[str, object], ProgramCfgKind]:
        ml = self._host.get_current_ml()
        if kind == "module":
            return ml.modules, "module"
        if kind == "waveform":
            return ml.waveforms, "waveform"
        raise RuntimeError(f"Unsupported measure cfg reference kind {kind!r}")

    @staticmethod
    def _converter(kind: str) -> _ReferenceConverter:
        if kind == "module":
            return cast(_ReferenceConverter, module_cfg_to_value)
        if kind == "waveform":
            return cast(_ReferenceConverter, waveform_cfg_to_value)
        raise RuntimeError(f"Unsupported measure cfg reference kind {kind!r}")


def make_value_source_input_enhancer(
    host: SessionValueSourcePort,
) -> TextInputEnhancer:
    source_host = SessionValueSourceInputHost(host)

    def enhance(line_edit: QLineEdit) -> object:
        controller = ValueSourceInputController(
            line_edit,
            source_host,
            parent=line_edit,
        )
        controller.resolve_failed.connect(line_edit.setToolTip)  # type: ignore[attr-defined]
        return controller

    return enhance
