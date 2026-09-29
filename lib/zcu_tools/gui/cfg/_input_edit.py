"""Edit detached input values without constructing an evaluating field tree."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace

from zcu_tools.gui.expected_error import InvalidInputError

from ._capture import ExpressionCapture
from ._complete_input import require_complete_input, select_input_shape
from .binding.range import CenteredSweepEditor, SweepEditor
from .edit_codec import decode_input, encode_input
from .inheritance import (
    align_locked_literals,
    inherit_from,
    make_default_value,
    select_ref_value_spec,
)
from .model import (
    CenteredSweepSpec,
    CenteredSweepValue,
    CfgNodeSpec,
    CfgNodeValue,
    CfgSchema,
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
from .reference_key import make_custom_reference_key, parse_custom_reference_key
from .resource import (
    CfgEdit,
    CfgInput,
    CfgInputError,
    CfgInputReason,
    CfgPath,
    CfgResolution,
)
from .scalar import coerce_scalar_result, parse_scalar_text, validate_direct_scalar


class InputEditor:
    """One command's detached candidate and fixed source view."""

    def __init__(self, schema: CfgSchema, source: CfgResolution) -> None:
        self.schema = deepcopy(schema)
        self.source = source
        self.capture = ExpressionCapture(
            source.read_capture, source.validate_expression
        )

    def apply(self, edit: CfgEdit) -> None:
        value = self._write(self.schema.spec, self.schema.value, edit.path, edit.value)
        assert isinstance(value, CfgSectionValue)
        self.schema.value = value

    def select_custom_reference(self, path: CfgPath, label: str) -> None:
        """Build a complete Custom choice from the published input snapshot."""
        if not path:
            raise CfgInputError(
                CfgInputReason.UNKNOWN_PATH, "path must name a reference"
            )
        if not isinstance(label, str) or not label:  # pyright: ignore[reportUnnecessaryIsInstance]
            raise CfgInputError(
                CfgInputReason.INVALID_VALUE, "Custom label is required"
            )
        value = self._choose_custom(self.schema.spec, self.schema.value, path, label)
        if not isinstance(value, CfgSectionValue):
            raise RuntimeError("Custom choice did not produce a root section")
        self.schema.value = value

    def _choose_custom(
        self,
        spec: CfgNodeSpec,
        current: CfgNodeValue | None,
        path: CfgPath,
        label: str,
    ) -> CfgNodeValue | None:
        if path:
            if isinstance(spec, CfgSectionSpec):
                if not isinstance(current, CfgSectionValue):
                    raise CfgInputError(
                        CfgInputReason.UNKNOWN_PATH, "section is absent"
                    )
                key, *rest = path
                if key not in spec.fields:
                    raise CfgInputError(
                        CfgInputReason.UNKNOWN_PATH, f"unknown field {key!r}"
                    )
                fields = dict(current.fields)
                fields[key] = self._choose_custom(
                    spec.fields[key], fields.get(key), tuple(rest), label
                )
                return CfgSectionValue(fields)
            if isinstance(spec, ReferenceSpec) and isinstance(current, ReferenceValue):
                shape = select_ref_value_spec(spec, current)
                value = self._choose_custom(shape, current.value, path, label)
                if not isinstance(value, CfgSectionValue):
                    raise RuntimeError(
                        "Custom choice did not produce reference contents"
                    )
                return replace(
                    current,
                    value=value,
                    is_overridden=True,
                    resolved_label=None,
                    error=None,
                )
            raise CfgInputError(
                CfgInputReason.UNKNOWN_PATH, "reference path has no active children"
            )

        if not isinstance(spec, ReferenceSpec):
            raise CfgInputError(
                CfgInputReason.UNSUPPORTED_MODE, "node is not a reference"
            )
        shape = next(
            (
                allowed
                for allowed in spec.allowed
                if (allowed.label or "Custom") == label
            ),
            None,
        )
        if shape is None:
            raise CfgInputError(
                CfgInputReason.INVALID_VALUE, f"unknown Custom label {label!r}"
            )
        if isinstance(current, ReferenceValue):
            old_shape = select_ref_value_spec(spec, current)
            contents = inherit_from(current.value, old_shape, shape)
        else:
            contents = make_default_value(shape)
        return ReferenceValue(make_custom_reference_key(label), contents)

    def _write(
        self,
        spec: CfgNodeSpec,
        current: CfgNodeValue | None,
        path: CfgPath,
        value: CfgInput,
    ) -> CfgNodeValue | None:
        if isinstance(spec, LiteralSpec) or (
            isinstance(spec, (ScalarSpec, SweepSpec, CenteredSweepSpec))
            and not spec.editable
        ):
            raise CfgInputError(CfgInputReason.READONLY, "node is readonly")
        if isinstance(spec, ReferenceSpec):
            return self._reference(spec, current, path, value)
        if isinstance(spec, CfgSectionSpec):
            if not isinstance(current, CfgSectionValue):
                current = make_default_value(spec)
            return self._section(spec, current, path, value)
        if isinstance(spec, (SweepSpec, CenteredSweepSpec)):
            return self._range(spec, current, path, value)
        if not path:
            return self._scalar(spec, value)
        raise CfgInputError(
            CfgInputReason.UNKNOWN_PATH, "path does not name a writable node"
        )

    def _section(
        self,
        spec: CfgSectionSpec,
        current: CfgSectionValue,
        path: CfgPath,
        value: CfgInput,
    ) -> CfgSectionValue:
        if not path:
            if not isinstance(value, Mapping):
                raise CfgInputError(
                    CfgInputReason.INVALID_VALUE, "section input must be a mapping"
                )
            for key, child in value.items():
                current = self._section(spec, current, (key,), child)
            return current
        key, *rest = path
        if key not in spec.fields:
            raise CfgInputError(CfgInputReason.UNKNOWN_PATH, f"unknown field {key!r}")
        fields = dict(current.fields)
        fields[key] = self._write(spec.fields[key], fields.get(key), tuple(rest), value)
        return CfgSectionValue(fields)

    def _scalar(self, spec: ScalarSpec, value: CfgInput) -> ScalarValue:
        clean = decode_input(encode_input(value))
        if isinstance(clean, EvalValue):
            if (
                spec.type not in (int, float, complex)
                or spec.choices is not None
                or spec.choices_source
            ):
                raise CfgInputError(
                    CfgInputReason.UNSUPPORTED_MODE, "expression mode is unsupported"
                )
            return EvalValue(self.capture.prepare(clean.expr))
        if isinstance(clean, DirectValue) and clean.raw is not None:
            if spec.type not in (int, float, complex, str):
                raise CfgInputError(
                    CfgInputReason.UNSUPPORTED_MODE, "text mode is unsupported"
                )
            return parse_scalar_text(spec, clean.raw)
        direct = clean if isinstance(clean, DirectValue) else DirectValue(clean)
        try:
            validate_direct_scalar(spec, direct)
        except (TypeError, ValueError) as exc:
            raise CfgInputError(CfgInputReason.INVALID_VALUE, str(exc)) from exc
        return direct

    def _linked(self, spec: ReferenceSpec, key: str) -> ReferenceValue:
        resolved = self.source.references.resolve(spec.kind, key)
        if resolved is None:
            raise CfgInputError(
                CfgInputReason.UNKNOWN_PATH, "reference has no available contents"
            )
        shape = next(
            (shape for shape in spec.allowed if shape.label == resolved.label), None
        )
        if shape is None or resolved.value is None:
            raise RuntimeError(f"Reference {key!r} cannot materialize an allowed shape")
        return ReferenceValue(
            key,
            align_locked_literals(shape, deepcopy(resolved.value)),
            resolved_label=shape.label,
        )

    def _relink(self, spec: ReferenceSpec, value: CfgInput) -> ReferenceValue | None:
        if value is None:
            if not spec.optional:
                raise CfgInputError(
                    CfgInputReason.INVALID_VALUE,
                    "required reference cannot be disabled",
                )
            return None
        if not isinstance(value, str):
            raise CfgInputError(
                CfgInputReason.INVALID_VALUE, "reference key must be a string or null"
            )
        labels = frozenset(shape.label for shape in spec.allowed)
        if value not in self.source.references.keys(spec.kind, labels):
            raise CfgInputError(
                CfgInputReason.INVALID_VALUE, "reference key is unavailable"
            )
        return self._linked(spec, value)

    def _contents(
        self, spec: ReferenceSpec, current: CfgNodeValue | None
    ) -> tuple[CfgSectionSpec, ReferenceValue]:
        if not isinstance(current, ReferenceValue):
            raise CfgInputError(
                CfgInputReason.UNKNOWN_PATH, "reference has no active children"
            )
        if (
            not current.is_overridden
            and parse_custom_reference_key(current.chosen_key) is None
        ):
            current = self._linked(spec, current.chosen_key)
        return select_ref_value_spec(spec, current), current

    def _reference(
        self,
        spec: ReferenceSpec,
        current: CfgNodeValue | None,
        path: CfgPath,
        value: CfgInput,
    ) -> ReferenceValue | None:
        if path == ("__ref",):
            return self._relink(spec, value)
        if path:
            shape, reference = self._contents(spec, current)
            contents = self._section(shape, reference.value, path, value)
            return replace(
                reference,
                value=contents,
                is_overridden=True,
                resolved_label=shape.label,
                error=None,
            )
        if not isinstance(value, Mapping):
            raise CfgInputError(
                CfgInputReason.INVALID_VALUE, "reference input must be a mapping"
            )
        if "__ref" in value:
            if value["__ref"] is None and len(value) != 1:
                raise CfgInputError(
                    CfgInputReason.INVALID_VALUE,
                    "disabled reference cannot have children",
                )
            current = self._relink(spec, value["__ref"])
            if len(value) == 1:
                return current
        return self._reference_contents(spec, current, value)

    def _reference_contents(
        self,
        spec: ReferenceSpec,
        current: CfgNodeValue | None,
        value: Mapping[str, CfgInput],
    ) -> ReferenceValue:
        contents = {
            key: child
            for key, child in value.items()
            if key not in ("__ref", spec.discriminator)
        }
        selected = None
        completeness_error = None
        if spec.discriminator in value or len(spec.allowed) == 1:
            selected = select_input_shape(spec, value)
            if "__ref" not in value:
                try:
                    require_complete_input(selected, contents)
                except CfgInputError as exc:
                    completeness_error = exc
                else:
                    fresh = self._section(
                        selected, make_default_value(selected), (), contents
                    )
                    return ReferenceValue(
                        make_custom_reference_key(selected.label or "Custom"),
                        fresh,
                        resolved_label=selected.label,
                    )
        shape, reference = self._contents(spec, current)
        if selected is not None and selected != shape:
            if completeness_error is not None:
                raise completeness_error
            raise CfgInputError(
                CfgInputReason.INVALID_VALUE,
                "discriminator does not match the selected reference",
            )
        edited = self._section(shape, reference.value, (), contents)
        return replace(
            reference,
            value=edited,
            is_overridden=True,
            resolved_label=shape.label,
            error=None,
        )

    def _range(
        self,
        spec: SweepSpec | CenteredSweepSpec,
        current: CfgNodeValue | None,
        path: CfgPath,
        value: CfgInput,
    ) -> SweepValue | CenteredSweepValue:
        if not isinstance(current, (SweepValue, CenteredSweepValue)):
            raise RuntimeError("Range definition has no input value")
        if (
            isinstance(spec, CenteredSweepSpec)
            and isinstance(current, CenteredSweepValue)
            and spec.locked_center is not None
        ):
            current = replace(current, center=spec.locked_center, auto_norm=False)
        if not path:
            if not isinstance(value, Mapping):
                raise CfgInputError(
                    CfgInputReason.INVALID_VALUE, "range input must be a mapping"
                )
            if "expts" in value and "step" in value:
                raise CfgInputError(
                    CfgInputReason.INVALID_VALUE,
                    "range cannot specify both expts and step",
                )
            for key in sorted(value, key=lambda key: key in {"expts", "step"}):
                current = self._range(spec, current, (key,), value[key])
            return current
        if len(path) != 1:
            raise CfgInputError(
                CfgInputReason.UNKNOWN_PATH, "range paths have one segment"
            )
        key = path[0]
        if isinstance(current, SweepValue) and key in {"start", "stop"}:
            scalar = self._scalar(ScalarSpec(key, float), value)
            return (
                SweepEditor.update_start(current, scalar)
                if key == "start"
                else SweepEditor.update_stop(current, scalar)
            )
        if isinstance(current, CenteredSweepValue) and key == "center":
            assert isinstance(spec, CenteredSweepSpec)
            if not spec.center_editable or spec.locked_center is not None:
                raise CfgInputError(CfgInputReason.READONLY, "center is readonly")
            return CenteredSweepEditor.update_center(
                current, self._scalar(ScalarSpec(key, float), value)
            )
        return self._range_control(current, key, value)

    def _range_control(
        self, current: SweepValue | CenteredSweepValue, key: str, value: CfgInput
    ) -> SweepValue | CenteredSweepValue:
        if key not in {"expts", "step"} and not (
            key == "span" and isinstance(current, CenteredSweepValue)
        ):
            raise CfgInputError(
                CfgInputReason.UNKNOWN_PATH, f"unknown range field {key!r}"
            )
        clean = decode_input(encode_input(value))
        text = isinstance(clean, DirectValue) and clean.raw is not None
        parsed = _range_control_input(key, clean)
        resolved = current
        if isinstance(current, SweepValue) and key == "step":
            resolved = replace(
                current,
                start=self._range_edge(current.start),
                stop=self._range_edge(current.stop),
                auto_norm=False,
            )
        try:
            if isinstance(current, SweepValue):
                if key == "expts":
                    assert isinstance(parsed, (int, DirectValue))
                    return SweepEditor.update_expts(current, parsed)
                assert isinstance(resolved, SweepValue)
                updated = SweepEditor.update_step(resolved, parsed)
                return replace(
                    updated, start=current.start, stop=current.stop, auto_norm=False
                )
            if key == "expts":
                assert isinstance(parsed, (int, DirectValue))
                return CenteredSweepEditor.update_expts(current, parsed)
            if key == "span":
                return CenteredSweepEditor.update_span(current, parsed)
            return CenteredSweepEditor.update_step(current, parsed)
        except (TypeError, ValueError, OverflowError) as exc:
            if text:
                assert isinstance(clean, DirectValue)
                invalid = DirectValue(None, raw=clean.raw, error=str(exc))
                return replace(current, **{key: invalid}, auto_norm=False)
            raise CfgInputError(CfgInputReason.INVALID_VALUE, str(exc)) from exc

    def _range_edge(self, value: float | ScalarValue) -> float | ScalarValue:
        if not isinstance(value, EvalValue):
            return value
        try:
            result = self.source.evaluate_expression(value.expr)
        except InvalidInputError as exc:
            return replace(value, resolved=None, error=str(exc))
        try:
            resolved = coerce_scalar_result(result, float)
        except (RuntimeError, TypeError, ValueError) as exc:
            return replace(value, resolved=None, error=str(exc))
        return replace(value, resolved=resolved, error=None)


def _range_control_input(key: str, value: CfgInput) -> int | float | DirectValue:
    if isinstance(value, DirectValue) and value.raw is not None:
        return parse_scalar_text(
            ScalarSpec(key, int if key == "expts" else float), value.raw
        )
    if key == "expts":
        if type(value) is not int:
            raise CfgInputError(
                CfgInputReason.INVALID_VALUE, "expts must be an integer or text"
            )
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CfgInputError(
            CfgInputReason.INVALID_VALUE, "range control must be numeric or text"
        )
    return float(value)
