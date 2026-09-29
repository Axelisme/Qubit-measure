"""Candidate-only node writes; never mutate a published draft."""

from collections.abc import Callable, Mapping

from ._complete_input import require_complete_input, select_input_shape
from .binding.fields import (
    CenteredSweepField,
    CfgField,
    LiteralField,
    ScalarField,
    SectionField,
    SweepField,
)
from .binding.reference import ReferenceField
from .edit_codec import decode_input, encode_input
from .model import DirectValue, EvalValue
from .resource import CfgInput, CfgInputError, CfgInputReason, CfgPath


def write_node(
    node: CfgField, path: CfgPath, value: CfgInput, prepare: Callable[[str], str]
) -> None:
    if isinstance(node, LiteralField) or (
        isinstance(node, (ScalarField, SweepField, CenteredSweepField))
        and not node.spec.editable
    ):
        raise CfgInputError(CfgInputReason.READONLY, "node is readonly")
    if isinstance(node, ReferenceField):
        _write_reference(node, path, value, prepare)
    elif isinstance(node, SectionField):
        _write_section(node, path, value, prepare)
    elif isinstance(node, (SweepField, CenteredSweepField)):
        _write_range(node, path, value, prepare)
    elif isinstance(node, ScalarField) and not path:
        _write_scalar(node, value, prepare)
    else:
        raise CfgInputError(
            CfgInputReason.UNKNOWN_PATH, "path does not name a writable node"
        )


def _write_section(
    node: SectionField, path: CfgPath, value: CfgInput, prepare: Callable[[str], str]
) -> None:
    if path:
        key, *rest = path
        if key not in node.fields:
            raise CfgInputError(CfgInputReason.UNKNOWN_PATH, f"unknown field {key!r}")
        write_node(node.fields[key], tuple(rest), value, prepare)
        return
    if not isinstance(value, Mapping):
        raise CfgInputError(
            CfgInputReason.INVALID_VALUE, "section input must be a mapping"
        )
    for key, child in value.items():
        _write_section(node, (key,), child, prepare)


def _write_scalar(
    node: ScalarField, value: CfgInput, prepare: Callable[[str], str]
) -> None:
    # Carrier caches and diagnostics must not bypass target parsing/validation.
    clean = decode_input(encode_input(value))
    if isinstance(clean, EvalValue):
        if (
            node.spec.type not in (int, float, complex)
            or node.available_options() is not None
        ):
            raise CfgInputError(
                CfgInputReason.UNSUPPORTED_MODE, "expression mode is unsupported"
            )
        node.set_value(EvalValue(prepare(clean.expr)))
    elif isinstance(clean, DirectValue) and clean.raw is not None:
        if node.spec.type not in (int, float, complex, str):
            raise CfgInputError(
                CfgInputReason.UNSUPPORTED_MODE, "text mode is unsupported"
            )
        node.set_text(clean.raw)
    else:
        try:
            node.set_value(clean)
        except (TypeError, ValueError) as exc:
            raise CfgInputError(CfgInputReason.INVALID_VALUE, str(exc)) from exc


def _write_reference(
    node: ReferenceField, path: CfgPath, value: CfgInput, prepare: Callable[[str], str]
) -> None:
    if path == ("__ref",):
        _relink(node, value)
        return
    if not path:
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
            _relink(node, value["__ref"])
            if len(value) == 1:
                return
        _write_reference_contents(node, value, prepare)
        return
    if not node.is_enabled or node.sub_field is None or node.has_missing_library_ref():
        raise CfgInputError(
            CfgInputReason.UNKNOWN_PATH, "reference has no active children"
        )
    write_node(node.sub_field, path, value, prepare)
    node.detach()


def _write_reference_contents(
    node: ReferenceField, value: Mapping[str, CfgInput], prepare: Callable[[str], str]
) -> None:
    discriminator = node.spec.discriminator
    current = node.sub_field.spec if node.sub_field is not None else None
    available = (
        node.is_enabled and current is not None and not node.has_missing_library_ref()
    )
    shape = (
        select_input_shape(node.spec, value)
        if discriminator in value or not available
        else current
    )
    assert shape is not None
    contents = {
        key: child
        for key, child in value.items()
        if key not in ("__ref", discriminator)
    }
    if shape != current or not available:
        if "__ref" in value:
            raise CfgInputError(
                CfgInputReason.INVALID_VALUE,
                "discriminator does not match the selected reference",
            )
        require_complete_input(shape, contents)
        node.replace_inline(
            shape, lambda section: write_node(section, (), contents, prepare)
        )
        return
    for key, child in contents.items():
        _write_reference(node, (key,), child, prepare)
    node.detach()


def _relink(node: ReferenceField, value: CfgInput) -> None:
    if value is None:
        if not node.spec.optional:
            raise CfgInputError(
                CfgInputReason.INVALID_VALUE, "required reference cannot be disabled"
            )
        node.set_enabled(False)
        return
    if not isinstance(value, str):
        raise CfgInputError(
            CfgInputReason.INVALID_VALUE, "reference key must be a string or null"
        )
    if value not in node.available_keys():
        raise CfgInputError(
            CfgInputReason.INVALID_VALUE, "reference key is unavailable"
        )
    node.set_chosen_key(value)
    node.set_enabled(True)


def _write_range(
    node: SweepField | CenteredSweepField,
    path: CfgPath,
    value: CfgInput,
    prepare: Callable[[str], str],
) -> None:
    if not path:
        if not isinstance(value, Mapping):
            raise CfgInputError(
                CfgInputReason.INVALID_VALUE, "range input must be a mapping"
            )
        if "expts" in value and "step" in value:
            raise CfgInputError(
                CfgInputReason.INVALID_VALUE, "range cannot specify both expts and step"
            )
        # Derive the point count from the new geometry, not mapping order.
        for key in sorted(value, key=lambda key: key in {"expts", "step"}):
            _write_range(node, (key,), value[key], prepare)
        return
    if len(path) != 1:
        raise CfgInputError(CfgInputReason.UNKNOWN_PATH, "range paths have one segment")
    key = path[0]
    if isinstance(node, SweepField) and key in {"start", "stop"}:
        write_node(
            node.start_field if key == "start" else node.stop_field, (), value, prepare
        )
    elif isinstance(node, CenteredSweepField) and key == "center":
        if not node.spec.center_editable or node.spec.locked_center is not None:
            raise CfgInputError(CfgInputReason.READONLY, "center is readonly")
        write_node(node.center_field, (), value, prepare)
    else:
        _write_range_control(node, key, value)


def _write_range_control(
    node: SweepField | CenteredSweepField, key: str, value: CfgInput
) -> None:
    clean = decode_input(encode_input(value))
    try:
        if isinstance(clean, DirectValue) and clean.raw is not None:
            _write_range_text(node, key, clean.raw)
        elif key == "expts":
            if type(clean) is not int:
                raise CfgInputError(
                    CfgInputReason.INVALID_VALUE, "expts must be an integer or text"
                )
            node.update_expts(clean)
        elif key == "step":
            node.update_step(_range_number(clean))
        elif key == "span" and isinstance(node, CenteredSweepField):
            node.update_span(_range_number(clean))
        else:
            raise CfgInputError(
                CfgInputReason.UNKNOWN_PATH, f"unknown range field {key!r}"
            )
    except (TypeError, ValueError) as exc:
        raise CfgInputError(CfgInputReason.INVALID_VALUE, str(exc)) from exc


def _write_range_text(
    node: SweepField | CenteredSweepField, key: str, text: str
) -> None:
    if key in ("expts", "step"):
        node.set_text("expts" if key == "expts" else "step", text)
    elif key == "span" and isinstance(node, CenteredSweepField):
        node.set_text("span", text)
    else:
        raise CfgInputError(CfgInputReason.UNKNOWN_PATH, f"unknown range field {key!r}")


def _range_number(value: CfgInput) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CfgInputError(
            CfgInputReason.INVALID_VALUE, "range control must be numeric or text"
        )
    return float(value)
