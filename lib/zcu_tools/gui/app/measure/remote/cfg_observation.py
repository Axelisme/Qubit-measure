"""Lossless JSON projection of cached cfg observations, not persistence data."""

from __future__ import annotations

import math

from zcu_tools.gui.cfg import (
    CenteredSweepSpec,
    CenteredSweepValue,
    CfgSectionSpec,
    DirectValue,
    EvalValue,
    LiteralSpec,
    ReferenceSpec,
    ReferenceValue,
    ScalarSpec,
    SweepSpec,
    SweepValue,
    encode_complex,
)
from zcu_tools.gui.cfg.binding import CfgDraft, CfgNodeObservation
from zcu_tools.gui.cfg.edit_codec import encode_input, encode_ref
from zcu_tools.gui.cfg.resource import (
    CfgInputError,
    CfgPreconditionError,
    CfgStaleError,
)
from zcu_tools.gui.cfg.resource import CfgObservation as ResourceObservation
from zcu_tools.gui.remote.errors import RemoteError, remote_error_from_expected

CFG_OBSERVATION_DESCRIPTION = (
    "Returns {tree}, a complete cached cfg observation including locked fields. "
    "Nodes carry kind/path/label/valid; section/reference children are named nodes. "
    "Scalar/literal input and sweep inputs carry mode/raw/resolved/error/validation_error. "
    "Complex values use {__complex__: [real, imag]}. Scalar choices and reference "
    "ref/resolved_label/error/choices describe the displayed model, without resolving sources. "
    "Edit scalar node.path, sweep node.path plus an edge name, or reference node.path+'.ref'; "
    "children/input/inputs are read-format keys, not setter path segments. "
    "Prefix returns the selected node with its full path; a ref or sweep control "
    "selects its parent node, and unknown prefixes return {}. Omit prefix to "
    "establish a full cfg observation; even an empty prefix does not refresh its guard."
)


def cfg_error_to_remote(exc: CfgInputError | CfgPreconditionError) -> RemoteError:
    """Keep cfg detail at its wire boundary, separate from other resource guards."""
    mapped = remote_error_from_expected(exc)
    data: dict[str, object] = {}
    if exc.path is not None:
        data["path"] = list(exc.path)
    if exc.edit_index is not None:
        data["edit_index"] = exc.edit_index
    if isinstance(exc, CfgStaleError):
        data["expected"] = encode_ref(exc.expected)
        data["actual"] = encode_ref(exc.actual)
    return RemoteError(
        mapped.code, mapped.message, reason=mapped.reason, data=data or None
    )


def build_resource_observation(observation: ResourceObservation) -> dict[str, object]:
    """Serialize a complete publication without querying a live provider."""
    return {
        "cfg_ref": encode_ref(observation.ref),
        "status": observation.status.value,
        "tree": _project(observation.tree, ()),
        "source_basis": [
            {"source_id": source.source_id, "revision": str(source.revision)}
            for source in observation.source_basis
        ],
        "diagnostics": [
            {"path": list(item.path), "reason": item.reason, "message": item.message}
            for item in observation.diagnostics
        ],
    }


def build_cfg_observation(
    draft: CfgDraft, prefix: str | None = None
) -> dict[str, object]:
    """Read cached model data. Prefixes select views, never mutation aliases."""
    node = draft.observe()
    path = ""
    parts = prefix.split(".") if prefix else []
    for index, part in enumerate(parts):
        if part in node.children:
            node = node.children[part]
            path = f"{path}.{part}" if path else part
            continue
        if index == len(parts) - 1 and part in _control_names(node):
            break
        return {}
    return _project(node, path)


def _editing_input(value: object) -> object:
    if value is None or isinstance(
        value, (DirectValue, EvalValue, str, int, float, complex)
    ):
        return encode_input(value)
    raise TypeError(f"Unexpected scalar observation input {type(value).__name__}")


def _control_names(node: CfgNodeObservation) -> tuple[str, ...]:
    if isinstance(node.spec, ReferenceSpec):
        return ("ref",)
    if isinstance(node.spec, SweepSpec):
        return ("start", "stop", "expts", "step")
    if isinstance(node.spec, CenteredSweepSpec):
        return ("center", "span", "expts", "step")
    return ()


def _project(
    node: CfgNodeObservation, path: str | tuple[str, ...]
) -> dict[str, object]:
    spec = node.spec
    result: dict[str, object] = {
        "path": list(path) if isinstance(path, tuple) else path,
        "label": spec.label,
        "valid": node.valid,
    }
    if isinstance(spec, (CfgSectionSpec, ReferenceSpec)):
        result["children"] = {
            key: _project(
                child,
                (*path, key)
                if isinstance(path, tuple)
                else (f"{path}.{key}" if path else key),
            )
            for key, child in node.children.items()
        }
    if isinstance(spec, CfgSectionSpec):
        return {**result, "kind": "section"}
    if isinstance(spec, ReferenceSpec):
        return {**result, **_reference(node, spec)}
    if isinstance(spec, ScalarSpec):
        return {
            **result,
            "kind": "scalar",
            "type": spec.type.__name__,
            "editable": spec.editable,
            "required": spec.required,
            "optional": spec.optional,
            "choices": _json_value(node.options),
            "input": _input(node.value),
            **(
                {"editing_input": _editing_input(node.value)}
                if isinstance(path, tuple)
                else {}
            ),
        }
    if isinstance(spec, LiteralSpec):
        return {
            **result,
            "kind": "literal",
            "editable": False,
            "input": _input(node.value),
        }
    return {**result, **_range(node, spec)}


def _reference(node: CfgNodeObservation, spec: ReferenceSpec) -> dict[str, object]:
    value = node.value
    if value is not None and not isinstance(value, ReferenceValue):
        raise TypeError("Reference observation requires a reference value")
    return {
        "kind": "reference",
        "optional": spec.optional,
        "choices": _json_value(node.options),
        "ref": value.chosen_key if value is not None else None,
        "resolved_label": value.resolved_label if value is not None else None,
        "error": value.error if value is not None else None,
        "is_overridden": value.is_overridden if value is not None else False,
    }


def _range(
    node: CfgNodeObservation, spec: SweepSpec | CenteredSweepSpec
) -> dict[str, object]:
    value = node.value
    if isinstance(spec, SweepSpec) and isinstance(value, SweepValue):
        inputs = {
            "start": value.start,
            "stop": value.stop,
            "expts": value.expts,
            "step": value.step,
        }
        return {
            "kind": "sweep",
            "editable": spec.editable,
            "inputs": {key: _input(item) for key, item in inputs.items()},
        }
    if isinstance(spec, CenteredSweepSpec) and isinstance(value, CenteredSweepValue):
        inputs = {
            "center": value.center,
            "span": value.span,
            "expts": value.expts,
            "step": value.step,
        }
        return {
            "kind": "centered_sweep",
            "editable": spec.editable,
            "center_editable": spec.center_editable,
            "locked_center": _json_value(spec.locked_center),
            "inputs": {key: _input(item) for key, item in inputs.items()},
        }
    raise TypeError("Range observation spec and value do not match")


def _input(value: object) -> dict[str, object]:
    if isinstance(value, EvalValue):
        return {
            "mode": "expression",
            "raw": value.expr,
            "resolved": _json_value(value.resolved),
            "error": value.error,
            "validation_error": value.validation_error,
        }
    if isinstance(value, DirectValue):
        return {
            "mode": "direct",
            "raw": value.raw,
            "resolved": _json_value(value.value),
            "error": value.error,
            "validation_error": value.validation_error,
        }
    return {
        "mode": "direct",
        "raw": None,
        "resolved": _json_value(value),
        "error": None,
        "validation_error": None,
    }


def _json_value(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Nonfinite cfg observation value")
        return value
    if isinstance(value, complex):
        return _json_value(encode_complex(value))
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Cfg observation object keys must be strings")
        return {key: _json_value(item) for key, item in value.items()}
    raise TypeError(f"Unsupported cfg observation value: {type(value).__name__}")
