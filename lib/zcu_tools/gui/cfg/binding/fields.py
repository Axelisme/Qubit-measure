from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import cast

from ..inheritance import make_default_value, select_ref_value_spec
from ..model import (
    CenteredSweepSpec,
    CenteredSweepValue,
    CfgNodeSpec,
    CfgNodeValue,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    EvalValue,
    LiteralSpec,
    ReferenceSpec,
    ReferenceValue,
    ScalarSpec,
    ScalarValue,
    SweepSpec,
    SweepValue,
)
from .ports import ExpressionEvaluator, OptionProvider, ReferenceCatalog
from .range import CenteredSweepEditor, SweepEditor

logger = logging.getLogger(__name__)

_CENTERED_SWEEP_LOCKED_CENTER_ABS_TOL = 1e-12


class CallbackList:
    """Small Qt-free callback collection used by binding fields and drafts."""

    def __init__(self) -> None:
        self._callbacks: list[Callable[..., None]] = []

    def connect(self, callback: Callable[..., None]) -> None:
        if callback not in self._callbacks:
            self._callbacks.append(callback)

    def disconnect(self, callback: Callable[..., None]) -> None:
        try:
            self._callbacks.remove(callback)
        except ValueError:
            pass

    def clear(self) -> None:
        self._callbacks.clear()

    def emit(self, *args: object, **kwargs: object) -> None:
        for callback in tuple(self._callbacks):
            callback(*args, **kwargs)


class CfgField(ABC):
    spec: CfgNodeSpec

    def __init__(self, spec: CfgNodeSpec) -> None:
        self.spec = spec
        self.on_change = CallbackList()
        self.on_validity_changed = CallbackList()
        self._valid = True
        self._closed = False

    @abstractmethod
    def get_value(self) -> object: ...

    @abstractmethod
    def set_value(self, value: object) -> None: ...

    def is_valid(self) -> bool:
        self._require_open()
        return self._valid

    def _set_valid(self, valid: bool) -> None:
        if valid == self._valid:
            return
        self._valid = valid
        self.on_validity_changed.emit(valid)

    def refresh_expressions(self) -> None:
        self._require_open()

    def refresh_options(self, source_id: str | None = None) -> None:
        self._require_open()
        del source_id

    def refresh_references(self, kind: str | None = None) -> None:
        self._require_open()
        del kind

    def teardown(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.on_change.clear()
        self.on_validity_changed.clear()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"{type(self).__name__} is closed")


class ScalarField(CfgField):
    spec: ScalarSpec

    def __init__(
        self,
        spec: ScalarSpec,
        evaluate_expression: ExpressionEvaluator,
        provide_options: OptionProvider | None = None,
        initial_val: object = None,
    ) -> None:
        super().__init__(spec)
        self._evaluate_expression = evaluate_expression
        self._provide_options = provide_options
        if isinstance(initial_val, (DirectValue, EvalValue)):
            self._value: ScalarValue = initial_val
        else:
            self._value = DirectValue(initial_val)
        if isinstance(self._value, DirectValue):
            self._validate_direct_value(self._value)

        self._dynamic_options: tuple[object, ...] | None = None
        if spec.choices_source:
            self._dynamic_options = self._load_dynamic_options()
        if isinstance(self._value, EvalValue):
            self._resolve_expression(emit_change=False)
        self._refresh_validity()

    def get_value(self) -> ScalarValue:
        self._require_open()
        return self._value

    def set_value(self, value: object) -> None:
        self._require_open()
        if isinstance(value, (DirectValue, EvalValue)):
            new_value: ScalarValue = value
        else:
            new_value = DirectValue(value)
        if isinstance(new_value, DirectValue):
            self._validate_direct_value(new_value)
        if isinstance(new_value, EvalValue):
            new_value = self._resolved_eval_value(new_value)

        if new_value != self._value:
            self._value = new_value
            self._refresh_validity()
            self.on_change.emit(self.get_value())
        else:
            self._refresh_validity()

    def _validate_direct_value(self, value: DirectValue) -> None:
        raw = value.value
        if raw is None:
            return
        if type(raw) is not self.spec.type:
            raise TypeError(
                f"ScalarField {self.spec.label!r} expects "
                f"{self.spec.type.__name__}, got {type(raw).__name__}"
            )

    def available_options(self) -> tuple[object, ...] | None:
        self._require_open()
        if self.spec.choices_source:
            assert self._dynamic_options is not None
            options = self._dynamic_options
            if self.spec.type is str and not self.spec.required and "" not in options:
                return ("", *options)
            return options
        if self.spec.choices is None:
            return None
        return tuple(self.spec.choices)

    def refresh_expressions(self) -> None:
        self._require_open()
        if isinstance(self._value, EvalValue):
            self._resolve_expression(emit_change=True)

    def refresh_options(self, source_id: str | None = None) -> None:
        self._require_open()
        own_source = self.spec.choices_source
        if not own_source or (source_id is not None and source_id != own_source):
            return
        new_options = self._load_dynamic_options()
        changed = new_options != self._dynamic_options
        self._dynamic_options = new_options
        self._refresh_validity()
        if changed:
            self.on_change.emit(self.get_value())

    def _load_dynamic_options(self) -> tuple[object, ...]:
        if self._provide_options is None:
            raise RuntimeError(
                f"ScalarSpec {self.spec.label!r} requires option source "
                f"{self.spec.choices_source!r}, but no OptionProvider was supplied"
            )
        options = self._provide_options(self.spec.choices_source)
        if isinstance(options, (str, bytes)):
            raise TypeError(
                f"OptionProvider {self.spec.choices_source!r} returned a string; "
                "expected a sequence of choices"
            )
        return tuple(options)

    def _resolved_eval_value(self, value: EvalValue) -> EvalValue:
        from dataclasses import replace

        try:
            raw = self._evaluate_expression(value.expr)
            resolved = _coerce_eval_result(raw, self.spec.type)
        except Exception as exc:
            return replace(value, resolved=None, error=str(exc))
        return replace(value, resolved=resolved, error=None)

    def _resolve_expression(self, *, emit_change: bool) -> None:
        assert isinstance(self._value, EvalValue)
        new_value = self._resolved_eval_value(self._value)
        if new_value != self._value:
            self._value = new_value
            self._refresh_validity()
            if emit_change:
                self.on_change.emit(self.get_value())
        else:
            self._refresh_validity()

    def _refresh_validity(self) -> None:
        if isinstance(self._value, DirectValue):
            raw = self._value.value
            valid = raw is not None or self.spec.optional
        else:
            raw = self._value.resolved
            valid = raw is not None
        if valid and self.spec.required and raw == "":
            valid = False
        options = self.available_options()
        if valid and raw is not None and options is not None and raw not in options:
            valid = False
        self._set_valid(valid)


def _coerce_eval_result(value: int | float, type_: type) -> int | float:
    if isinstance(value, bool):
        raise RuntimeError("Expression evaluator returned bool instead of a number")
    if type_ is float:
        return float(value)
    if type_ is int:
        if not float(value).is_integer():
            raise RuntimeError(f"Expression result {value!r} is not an integer")
        return int(value)
    raise RuntimeError(f"Eval mode only supports int or float, got {type_!r}")


class LiteralField(CfgField):
    spec: LiteralSpec

    def __init__(self, spec: LiteralSpec, initial_val: object = None) -> None:
        del initial_val
        super().__init__(spec)

    def get_value(self) -> ScalarValue:
        self._require_open()
        return DirectValue(self.spec.value)

    def set_value(self, value: object) -> None:
        self._require_open()
        del value


class SweepField(CfgField):
    spec: SweepSpec

    def __init__(
        self,
        spec: SweepSpec,
        evaluate_expression: ExpressionEvaluator,
        initial_val: object = None,
    ) -> None:
        super().__init__(spec)
        self._updating = False
        initial = (
            SweepEditor.canonicalize(initial_val)
            if isinstance(initial_val, SweepValue)
            else SweepValue(start=0.0, stop=1.0, expts=11, step=0.1)
        )
        self._expts = initial.expts
        self._step = initial.step
        edge_spec = ScalarSpec(
            label=spec.label,
            type=float,
            decimals=spec.decimals,
            editable=spec.editable,
        )
        self.start_field = ScalarField(
            edge_spec,
            evaluate_expression,
            initial_val=self._coerce_edge(initial.start),
        )
        self.stop_field = ScalarField(
            edge_spec,
            evaluate_expression,
            initial_val=self._coerce_edge(initial.stop),
        )
        self.start_field.on_change.connect(self._on_child_change)
        self.stop_field.on_change.connect(self._on_child_change)
        self.start_field.on_validity_changed.connect(self._on_child_validity_changed)
        self.stop_field.on_validity_changed.connect(self._on_child_validity_changed)
        self._refresh_validity()

    def get_value(self) -> SweepValue:
        self._require_open()
        return SweepValue(
            start=self._edge_value(self.start_field.get_value()),
            stop=self._edge_value(self.stop_field.get_value()),
            expts=self._expts,
            step=self._step,
        )

    def set_value(self, value: object) -> None:
        self._require_open()
        if not isinstance(value, SweepValue):
            raise TypeError(
                f"SweepField expects SweepValue, got {type(value).__name__}"
            )
        canonical = SweepEditor.canonicalize(value)
        self._updating = True
        try:
            self.start_field.set_value(self._coerce_edge(canonical.start))
            self.stop_field.set_value(self._coerce_edge(canonical.stop))
            self._expts = canonical.expts
            self._step = canonical.step
        finally:
            self._updating = False
        self._refresh_validity()
        self.on_change.emit(self.get_value())

    def update_expts(self, expts: int) -> None:
        self._require_open()
        self.set_value(SweepEditor.update_expts(self.get_value(), expts))

    def update_step(self, step: float) -> None:
        self._require_open()
        self.set_value(SweepEditor.update_step(self.get_value(), step))

    def refresh_expressions(self) -> None:
        self._require_open()
        self.start_field.refresh_expressions()
        self.stop_field.refresh_expressions()
        self._refresh_validity()

    def teardown(self) -> None:
        self.start_field.teardown()
        self.stop_field.teardown()
        super().teardown()

    @staticmethod
    def _coerce_edge(value: object) -> ScalarValue:
        if isinstance(value, EvalValue):
            return value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return DirectValue(float(value))
        raise TypeError(
            f"Sweep edge expects float or EvalValue, got {type(value).__name__}"
        )

    @staticmethod
    def _edge_value(value: ScalarValue) -> float | EvalValue:
        if isinstance(value, EvalValue):
            return value
        if value.value is None:
            raise TypeError("Sweep edge DirectValue is unset (None)")
        return float(value.value)

    def _on_child_change(self, *_: object) -> None:
        if self._updating:
            return
        canonical = SweepEditor.canonicalize(self.get_value())
        self._expts = canonical.expts
        self._step = canonical.step
        self._refresh_validity()
        self.on_change.emit(canonical)

    def _on_child_validity_changed(self, *_: object) -> None:
        self._refresh_validity()

    def _refresh_validity(self) -> None:
        self._set_valid(self.start_field.is_valid() and self.stop_field.is_valid())


class CenteredSweepField(CfgField):
    spec: CenteredSweepSpec

    def __init__(
        self,
        spec: CenteredSweepSpec,
        evaluate_expression: ExpressionEvaluator,
        initial_val: object = None,
    ) -> None:
        super().__init__(spec)
        self._updating = False
        initial = (
            CenteredSweepEditor.canonicalize(initial_val)
            if isinstance(initial_val, CenteredSweepValue)
            else CenteredSweepValue(center=0.5, span=1.0, expts=11, step=0.1)
        )
        self._span = initial.span
        self._expts = initial.expts
        self._step = initial.step
        center_spec = ScalarSpec(
            label=spec.label,
            type=float,
            decimals=spec.decimals,
            editable=spec.editable and spec.center_editable,
        )
        self.center_field = ScalarField(
            center_spec,
            evaluate_expression,
            initial_val=self._coerce_center(initial.center),
        )
        self.center_field.on_change.connect(self._on_child_change)
        self.center_field.on_validity_changed.connect(self._on_child_validity_changed)
        self._refresh_validity()

    def get_value(self) -> CenteredSweepValue:
        self._require_open()
        return CenteredSweepValue(
            center=self._center_value(self.center_field.get_value()),
            span=self._span,
            expts=self._expts,
            step=self._step,
        )

    def set_value(self, value: object) -> None:
        self._require_open()
        if not isinstance(value, CenteredSweepValue):
            raise TypeError(
                "CenteredSweepField expects CenteredSweepValue, "
                f"got {type(value).__name__}"
            )
        canonical = CenteredSweepEditor.canonicalize(value)
        self._validate_value(canonical)
        self._updating = True
        try:
            self.center_field.set_value(self._coerce_center(canonical.center))
            self._span = canonical.span
            self._expts = canonical.expts
            self._step = canonical.step
        finally:
            self._updating = False
        self._refresh_validity()
        self.on_change.emit(self.get_value())

    def update_span(self, span: float) -> None:
        self._require_open()
        self.set_value(CenteredSweepEditor.update_span(self.get_value(), span))

    def update_expts(self, expts: int) -> None:
        self._require_open()
        self.set_value(CenteredSweepEditor.update_expts(self.get_value(), expts))

    def update_step(self, step: float) -> None:
        self._require_open()
        self.set_value(CenteredSweepEditor.update_step(self.get_value(), step))

    def refresh_expressions(self) -> None:
        self._require_open()
        self.center_field.refresh_expressions()
        self._refresh_validity()

    def teardown(self) -> None:
        self.center_field.teardown()
        super().teardown()

    @staticmethod
    def _coerce_center(value: object) -> ScalarValue:
        if isinstance(value, EvalValue):
            return value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return DirectValue(float(value))
        raise TypeError(
            f"Centered sweep center expects float or EvalValue, got {type(value).__name__}"
        )

    @staticmethod
    def _center_value(value: ScalarValue) -> float | EvalValue:
        if isinstance(value, EvalValue):
            return value
        if value.value is None:
            raise TypeError("Centered sweep center DirectValue is unset (None)")
        return float(value.value)

    def _validate_value(self, value: CenteredSweepValue) -> None:
        if value.expts > 1 and value.span <= 0.0:
            raise ValueError("Centered sweep span must be > 0 when expts > 1")
        if self.spec.locked_center is None:
            return
        center = self._resolved_center(value.center)
        if center is None:
            raise ValueError(
                "Centered sweep center is locked to "
                f"{float(self.spec.locked_center)!r}; unresolved EvalValue is not allowed"
            )
        if not math.isclose(
            center,
            float(self.spec.locked_center),
            rel_tol=0.0,
            abs_tol=_CENTERED_SWEEP_LOCKED_CENTER_ABS_TOL,
        ):
            raise ValueError(
                "Centered sweep center is locked to "
                f"{float(self.spec.locked_center)!r}, got {center!r}"
            )

    @staticmethod
    def _resolved_center(value: float | EvalValue) -> float | None:
        raw: object = value.resolved if isinstance(value, EvalValue) else value
        if raw is None:
            return None
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError("Centered sweep center must be numeric")
        center = float(raw)
        if not math.isfinite(center):
            raise ValueError("Centered sweep center must be finite")
        return center

    def _on_child_change(self, *_: object) -> None:
        if self._updating:
            return
        canonical = CenteredSweepEditor.canonicalize(self.get_value())
        self._span = canonical.span
        self._expts = canonical.expts
        self._step = canonical.step
        self._refresh_validity()
        self.on_change.emit(canonical)

    def _on_child_validity_changed(self, *_: object) -> None:
        self._refresh_validity()

    def _refresh_validity(self) -> None:
        self._set_valid(self.center_field.is_valid())


class SectionField(CfgField):
    spec: CfgSectionSpec

    def __init__(
        self,
        spec: CfgSectionSpec,
        *,
        evaluate_expression: ExpressionEvaluator,
        provide_options: OptionProvider,
        references: ReferenceCatalog,
        initial_val: CfgSectionValue | None = None,
    ) -> None:
        super().__init__(spec)
        if initial_val is not None:
            _validate_section_keys(spec, initial_val)
        self._evaluate_expression = evaluate_expression
        self._provide_options = provide_options
        self._references = references
        self._updating = False
        self.fields: dict[str, CfgField] = {}
        default_value = make_default_value(spec)
        provided_value = initial_val if initial_val is not None else default_value
        initialized = False
        try:
            for key, node_spec in spec.fields.items():
                if key in provided_value.fields:
                    child_value = provided_value.fields[key]
                elif isinstance(node_spec, ReferenceSpec) and node_spec.optional:
                    child_value = None
                else:
                    child_value = default_value.fields.get(key)
                child = create_field(
                    node_spec,
                    evaluate_expression=evaluate_expression,
                    provide_options=provide_options,
                    references=references,
                    initial_val=child_value,
                )
                self.fields[key] = child
                child.on_change.connect(self._on_child_change)
                child.on_validity_changed.connect(self._on_child_validity_change)
            self._refresh_validity()
            initialized = True
        finally:
            if not initialized:
                self.teardown()

    def get_value(self) -> CfgSectionValue:
        self._require_open()
        fields: dict[str, CfgNodeValue | None] = {
            key: cast("CfgNodeValue | None", child.get_value())
            for key, child in self.fields.items()
        }
        return CfgSectionValue(fields)

    def set_value(self, value: object) -> None:
        self._require_open()
        if not isinstance(value, CfgSectionValue):
            raise TypeError(
                f"SectionField expects CfgSectionValue, got {type(value).__name__}"
            )
        _validate_section_keys(self.spec, value)
        self._preflight(value)
        previous_value = self.get_value()
        self._updating = True
        try:
            for key, child in self.fields.items():
                if key in value.fields:
                    child.set_value(value.fields[key])
        finally:
            self._updating = False
        self._refresh_validity()
        if self.get_value() != previous_value:
            self.on_change.emit()

    def _preflight(self, value: CfgSectionValue) -> None:
        """Validate the complete update against an isolated candidate tree."""
        candidate_value = self.get_value()
        candidate = SectionField(
            self.spec,
            evaluate_expression=self._evaluate_expression,
            provide_options=self._provide_options,
            references=self._references,
            initial_val=candidate_value,
        )
        try:
            for key, child in candidate.fields.items():
                if key in value.fields:
                    child.set_value(value.fields[key])
        finally:
            candidate.teardown()

    def refresh_expressions(self) -> None:
        self._require_open()
        for child in self.fields.values():
            child.refresh_expressions()
        self._refresh_validity()

    def refresh_options(self, source_id: str | None = None) -> None:
        self._require_open()
        for child in self.fields.values():
            child.refresh_options(source_id)
        self._refresh_validity()

    def refresh_references(self, kind: str | None = None) -> None:
        self._require_open()
        for child in self.fields.values():
            child.refresh_references(kind)
        self._refresh_validity()

    def teardown(self) -> None:
        for child in self.fields.values():
            child.on_change.disconnect(self._on_child_change)
            child.on_validity_changed.disconnect(self._on_child_validity_change)
            child.teardown()
        super().teardown()

    def _on_child_change(self, *_: object) -> None:
        if not self._updating:
            self.on_change.emit()

    def _on_child_validity_change(self, *_: object) -> None:
        if not self._updating:
            self._refresh_validity()

    def _refresh_validity(self) -> None:
        self._set_valid(all(child.is_valid() for child in self.fields.values()))


def _validate_section_keys(
    spec: CfgSectionSpec,
    value: CfgSectionValue,
    *,
    path: str = "",
) -> None:
    unknown = sorted(value.fields.keys() - spec.fields.keys())
    if unknown:
        location = path or spec.label or "<root>"
        joined = ", ".join(repr(key) for key in unknown)
        raise KeyError(f"Section {location!r} has unknown field(s): {joined}")

    for key, child_value in value.fields.items():
        child_spec = spec.fields[key]
        if isinstance(child_spec, CfgSectionSpec) and isinstance(
            child_value, CfgSectionValue
        ):
            child_path = f"{path}.{key}" if path else key
            _validate_section_keys(child_spec, child_value, path=child_path)
        elif isinstance(child_spec, ReferenceSpec) and isinstance(
            child_value, ReferenceValue
        ):
            child_path = f"{path}.{key}" if path else key
            chosen_spec = select_ref_value_spec(child_spec, child_value)
            _validate_section_keys(
                chosen_spec,
                child_value.value,
                path=child_path,
            )


def create_field(
    spec: CfgNodeSpec,
    *,
    evaluate_expression: ExpressionEvaluator,
    provide_options: OptionProvider,
    references: ReferenceCatalog,
    initial_val: object = None,
) -> CfgField:
    if isinstance(spec, ScalarSpec):
        return ScalarField(spec, evaluate_expression, provide_options, initial_val)
    if isinstance(spec, LiteralSpec):
        return LiteralField(spec, initial_val)
    if isinstance(spec, SweepSpec):
        return SweepField(spec, evaluate_expression, initial_val)
    if isinstance(spec, CenteredSweepSpec):
        return CenteredSweepField(spec, evaluate_expression, initial_val)
    if isinstance(spec, ReferenceSpec):
        from .reference import ReferenceField

        return ReferenceField(
            spec,
            evaluate_expression=evaluate_expression,
            provide_options=provide_options,
            references=references,
            initial_val=initial_val,
        )
    if isinstance(spec, CfgSectionSpec):
        return SectionField(
            spec,
            evaluate_expression=evaluate_expression,
            provide_options=provide_options,
            references=references,
            initial_val=(
                initial_val if isinstance(initial_val, CfgSectionValue) else None
            ),
        )
    raise TypeError(f"Unknown spec type: {type(spec).__name__}")
