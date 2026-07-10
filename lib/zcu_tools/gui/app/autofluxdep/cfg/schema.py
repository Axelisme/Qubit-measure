"""Node-knob schema — typed value tree with logical-key projection.

``NodeCfgSchema`` is the per-placement SSOT for an autofluxdep node's
user-editable knobs. The visible value tree is cfg-shaped, while the write/lower
contract stays keyed by stable logical names (``reps``, ``detune_sweep``,
``qub_gain``). Adapter-backed autofluxdep nodes pass the measure-gui adapter's
native value tree plus ``logical_paths`` directly into ``NodeCfgSchema`` so the
UI stays cfg-shaped while builder code keeps stable keys.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final, cast

import numpy as np
from numpy.typing import NDArray

from zcu_tools.gui.app.main.adapter.lowering import schema_to_raw_dict
from zcu_tools.gui.cfg import (
    CenteredSweepSpec,
    CenteredSweepValue,
    CfgNodeSpec,
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    EvalValue,
    ModuleRefSpec,
    ModuleRefValue,
    ScalarSpec,
    SessionCodecError,
    SweepSpec,
    SweepValue,
    WaveformRefSpec,
    WaveformRefValue,
    raw_to_schema,
    schema_to_raw,
)

if TYPE_CHECKING:
    from zcu_tools.meta_tool import MetaDict, ModuleLibrary


_TRUE_STRINGS = frozenset({"1", "true", "yes", "on"})
_FALSE_STRINGS = frozenset({"0", "false", "no", "off"})


def sweepcfg_to_axis(sweep: Any) -> NDArray[np.float64]:
    """The explicit linspace axis a lowered ``SweepCfg`` (start/stop/expts) samples.

    A ``SweepSpec`` lowers to a ``SweepCfg`` (the FPGA sweep); a Result stores its
    swept axis as an explicit array. Rebuilds it from the cfg's endpoints + count
    so the Result columns match the program sweep exactly — the typed counterpart
    of the prototype's ``parse_detune_sweep`` (now that the axis is sweep-defined,
    not free-text ``start,stop,step``).
    """
    return np.linspace(float(sweep.start), float(sweep.stop), int(sweep.expts))


def _coerce_bool_text(text: str) -> bool:
    lowered = text.strip().lower()
    if lowered in _TRUE_STRINGS:
        return True
    if lowered in _FALSE_STRINGS:
        return False
    raise ValueError(f"Expected bool, got {text!r}")


def _coerce_int_text(text: str) -> int:
    stripped = text.strip()
    try:
        return int(stripped)
    except ValueError as exc:
        raise ValueError(f"Expected int, got {text!r}") from exc


def _coerce_scalar(value: Any, type_: type) -> Any:
    """Coerce a (possibly text) scalar to its declared type, or None if blank/unset.

    Bridges a raw (possibly text) override into the typed value tree: a
    blank string / None means "unset" (leaf stays None → spec default or the
    node's Fast-Fail guard applies). A present value is coerced to the field's
    physical type (str passes through; int/float/bool parse). A malformed numeric
    fast-fails — typed knobs do NOT silently degrade to a default (the prototype's
    ``parse_*`` fallbacks are removed by this typing).
    """
    if value is None:
        return None
    if isinstance(value, str):
        if not value.strip():
            return None
        if type_ is str:
            return value
        if type_ is bool:
            return _coerce_bool_text(value)
        if type_ is int:
            return _coerce_int_text(value)
        # float: parse, fast-failing a malformed value
        return type_(value)
    if type_ is str:
        return str(value)
    if type_ is bool:
        return bool(value)
    return type_(value)


def _centered_sweep_center_for_assignment(
    key: str, value: CenteredSweepValue
) -> float | None:
    center = value.center
    if isinstance(center, EvalValue):
        if center.resolved is None:
            return None
        center = center.resolved
    if isinstance(center, bool) or not isinstance(center, (int, float)):
        raise ValueError(f"Param {key!r} centered sweep center must be numeric")
    numeric = float(center)
    if not math.isfinite(numeric):
        raise ValueError(f"Param {key!r} centered sweep center must be finite")
    return numeric


def _ensure_centered_sweep_assignment(
    key: str,
    spec: CenteredSweepSpec,
    value: CenteredSweepValue,
) -> None:
    if value.expts > 1 and value.span <= 0.0:
        raise ValueError(
            f"Param {key!r} centered sweep span must be greater than 0 when expts > 1"
        )
    if spec.locked_center is None:
        return
    center = _centered_sweep_center_for_assignment(key, value)
    if center is None:
        raise ValueError(
            f"Param {key!r} centered sweep center is locked to "
            f"{float(spec.locked_center)!r}; unresolved EvalValue is not allowed"
        )
    if not math.isclose(
        center,
        float(spec.locked_center),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            f"Param {key!r} centered sweep center is locked to "
            f"{float(spec.locked_center)!r}, got {center!r}"
        )


def str_choice_spec(
    label: str, choices: tuple[str, ...], *, tooltip: str = ""
) -> ScalarSpec:
    """A required string choice field rendered as the cfg editor's combo box."""
    if not choices:
        raise ValueError("str_choice_spec needs at least one choice")
    return ScalarSpec(label=label, type=str, choices=list(choices), tooltip=tooltip)


class NodeCfgPersistenceError(RuntimeError):
    """Invalid persisted node cfg payload."""


def empty_node_schema() -> NodeCfgSchema:
    """Build an empty node cfg schema for services with no editable knobs."""
    return NodeCfgSchema(
        CfgSchema(
            spec=CfgSectionSpec(fields={}),
            value=CfgSectionValue(fields={}),
        )
    )


def _validate_path_part(kind: str, value: str) -> None:
    if not value:
        raise ValueError(f"Node {kind} must not be empty")
    if "." in value:
        raise ValueError(f"Node {kind} must not contain '.': {value!r}")


def _ensure_unique(kind: str, values: Any) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    if duplicates:
        raise ValueError(f"Duplicate {kind}: {', '.join(sorted(duplicates))}")


@dataclass
class NodeCfgSchema:
    """A node's typed param SSOT plus stable logical-key projection.

    Owns the defaults (in the value tree) and the types (in the spec). The node's
    ``make_cfg`` lowers it to a flat dict; the set_node_params bridge writes leaves
    through it; a placement's stored param dict seeds it via ``with_overrides``.
    """

    schema: CfgSchema
    logical_paths: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        paths = (
            dict(self.logical_paths)
            if self.logical_paths
            else {key: key for key in self.schema.spec.fields}
        )
        _validate_logical_paths(self.schema.spec, paths)
        self.logical_paths = paths

    @property
    def keys(self) -> tuple[str, ...]:
        """The declared knob keys (the node's typed user knobs)."""
        return tuple(self.logical_paths)

    def set_field(self, key: str, value: Any) -> None:
        """Write one knob leaf into the value tree, fast-failing an unknown key.

        The single typed entry into the SSOT, used by two writers: the typed cfg
        form (which emits value-tree leaves — ``DirectValue`` / ``SweepValue``)
        and tests / ``with_overrides`` (which pass raw scalars). An unknown key is
        a real typo (only declared knobs are writable), so it fast-fails.

        - A ``SweepSpec`` knob accepts a ``SweepValue`` (stored verbatim).
        - A module/waveform ref knob accepts the corresponding ref value (or None
          for optional refs), preserving adapter-native selections.
        - A ``ScalarSpec`` knob accepts either a ``DirectValue`` / ``EvalValue``
          (stored verbatim — the form produced it) or a raw value (coerced to
          the field's declared type; a blank/None leaves the leaf unset so the
          spec default or a node's Fast-Fail guard applies).
        """
        path = self._require_logical_path(key)
        spec = _resolve_spec_at_path(self.schema.spec, path)
        if isinstance(spec, SweepSpec):
            if not isinstance(value, SweepValue):
                raise TypeError(
                    f"Param {key!r} is a sweep; expected a SweepValue, "
                    f"got {type(value).__name__}"
                )
            _assign_value_at_path(self.schema.value, path, value)
            return
        if isinstance(spec, CenteredSweepSpec):
            if not isinstance(value, CenteredSweepValue):
                raise TypeError(
                    f"Param {key!r} is a centered sweep; expected a "
                    f"CenteredSweepValue, got {type(value).__name__}"
                )
            _ensure_centered_sweep_assignment(key, spec, value)
            _assign_value_at_path(self.schema.value, path, value)
            return
        if isinstance(spec, ModuleRefSpec):
            if value is None and spec.optional:
                _assign_value_at_path(self.schema.value, path, None)
                return
            if not isinstance(value, ModuleRefValue):
                raise TypeError(
                    f"Param {key!r} is a module ref; expected a ModuleRefValue, "
                    f"got {type(value).__name__}"
                )
            _assign_value_at_path(self.schema.value, path, value)
            return
        if isinstance(spec, WaveformRefSpec):
            if value is None and spec.optional:
                _assign_value_at_path(self.schema.value, path, None)
                return
            if not isinstance(value, WaveformRefValue):
                raise TypeError(
                    f"Param {key!r} is a waveform ref; expected a WaveformRefValue, "
                    f"got {type(value).__name__}"
                )
            _assign_value_at_path(self.schema.value, path, value)
            return
        if not isinstance(spec, ScalarSpec):
            raise TypeError(
                f"Param {key!r} maps to unsupported spec {type(spec).__name__}"
            )
        if isinstance(value, (DirectValue, EvalValue)):
            _assign_value_at_path(self.schema.value, path, value)
            return
        _assign_value_at_path(
            self.schema.value,
            path,
            DirectValue(_coerce_scalar(value, spec.type)),
        )

    def with_overrides(self, params: Mapping[str, Any]) -> NodeCfgSchema:
        """Seed the value tree from a flat ``params`` dict, fast-failing unknown keys.

        Turns a ``PlacedNode``'s stored param dict into the typed schema: each
        present key overwrites its leaf (the rest keep their declared defaults).
        An unknown key fast-fails — a placement param with no declared knob is a
        contract violation, not a silent extra.
        """
        for key, value in params.items():
            self.set_field(key, value)
        return self

    def lower(
        self, ml: ModuleLibrary | None, md: MetaDict | None = None
    ) -> dict[str, Any]:
        """The lowered flat knob dict ``make_cfg`` / ``make_init_result`` read.

        Scalars lower to their value; a ``SweepSpec`` lowers to a ``SweepCfg``.
        Numeric scalar and sweep-edge ``EvalValue`` leaves resolve against
        ``md`` through the shared cfg lowering path. Each logical path lowers
        independently so preview/init code can read sweep/scalar generation knobs
        without resolving unrelated adapter module refs.
        """
        knobs: dict[str, Any] = {}
        for logical_key, path in self.logical_paths.items():
            spec = _resolve_spec_at_path(self.schema.spec, path)
            if ml is None and isinstance(spec, (ModuleRefSpec, WaveformRefSpec)):
                continue
            value = _lower_value_at_path(self.schema.value, spec, path, ml, md)
            if value is _MISSING:
                continue
            knobs[logical_key] = value
        return knobs

    def lower_raw(
        self, ml: ModuleLibrary | None, md: MetaDict | None = None
    ) -> dict[str, Any]:
        """Lower the full adapter-shaped cfg tree, omitting autofluxdep-only knobs."""
        raw = schema_to_raw_dict(self.schema, md=md, ml=ml)
        raw.pop("generation", None)
        return raw

    def read_knobs(self) -> dict[str, Any]:
        """The current *un-lowered* user knob values, as a JSON-friendly dict.

        Read-only projection of the value tree for an observer (the remote
        bridge's ``node.cfg``): keys are stable logical knob names, a scalar knob
        reads to its plain value (or an eval marker), and sweep-like knobs keep
        their user-facing shape (``{start, stop, expts}`` or
        ``{center, span, expts, step}``). Unlike ``lower``, this does NOT build a
        ``SweepCfg`` (whose internal axis object is not JSON-serialisable) and
        needs no ``ModuleLibrary`` — it reports exactly what the user set, not
        the lowered run cfg.
        """
        knobs: dict[str, Any] = {}
        for logical_key, path in self.logical_paths.items():
            knobs[logical_key] = _jsonify_value_node(
                _get_value_at_path(self.schema.value, path)
            )
        return knobs

    def to_persisted_raw(self) -> dict[str, object]:
        """Encode the value tree using stable logical generation keys.

        The visible UI value tree may group ``generation`` fields, but the workflow
        memento and run manifest keep generation overrides flat by logical key so
        grouping stays a presentation detail.
        """
        return _flatten_generation_persistence_raw(
            schema_to_raw(self.schema),
            self.logical_paths,
        )

    def restore_persisted_raw(self, raw: Mapping[str, object]) -> None:
        """Restore the value tree from the shared cfg persistence raw shape."""
        try:
            normalized_raw = _expand_generation_persistence_raw(
                dict(raw),
                self.logical_paths,
            )
            self.schema = raw_to_schema(self.schema, normalized_raw)
        except (SessionCodecError, TypeError, ValueError, RuntimeError) as exc:
            raise NodeCfgPersistenceError(str(exc)) from exc

    def replace_value_tree(self, value: CfgSectionValue) -> None:
        """Replace the complete value tree while keeping this node's spec/projection."""
        self.schema = CfgSchema(spec=self.schema.spec, value=value)

    def _require_logical_path(self, logical_key: str) -> str:
        try:
            return self.logical_paths[logical_key]
        except KeyError as exc:
            raise KeyError(
                f"Unknown node param {logical_key!r}; declared: {', '.join(self.keys)}"
            ) from exc


def _validate_logical_paths(spec: CfgSectionSpec, paths: Mapping[str, str]) -> None:
    _ensure_unique("node logical path", paths.values())
    for logical_key, path in paths.items():
        _validate_path_part("logical key", logical_key)
        leaf_spec = _resolve_spec_at_path(spec, path)
        if not isinstance(
            leaf_spec,
            (
                ScalarSpec,
                SweepSpec,
                CenteredSweepSpec,
                ModuleRefSpec,
                WaveformRefSpec,
            ),
        ):
            raise TypeError(
                f"Logical node param {logical_key!r} maps to unsupported spec "
                f"{type(leaf_spec).__name__}; only ScalarSpec, SweepSpec, "
                "CenteredSweepSpec, ModuleRefSpec, and WaveformRefSpec are supported"
            )


def _split_path(path: str) -> tuple[str, ...]:
    parts = tuple(part for part in path.split(".") if part)
    if not parts:
        raise RuntimeError("Node field path must not be empty")
    return parts


def _resolve_spec_at_path(spec: CfgSectionSpec, path: str) -> CfgNodeSpec:
    node_spec: CfgNodeSpec = spec
    parts = _split_path(path)
    for idx, part in enumerate(parts):
        if isinstance(node_spec, CfgSectionSpec):
            if part not in node_spec.fields:
                raise KeyError(
                    f"Node field path {path!r} segment {part!r} not found; "
                    f"available: {', '.join(node_spec.fields)}"
                )
            node_spec = node_spec.fields[part]
            continue
        if isinstance(node_spec, (ModuleRefSpec, WaveformRefSpec)):
            remaining = ".".join(parts[idx:])
            node_spec = _resolve_ref_spec_at_path(node_spec, remaining)
            break
        else:
            raise RuntimeError(
                f"Node field path {path!r} cannot descend into "
                f"{type(node_spec).__name__} at {part!r}"
            )
    return node_spec


def _resolve_ref_spec_at_path(
    spec: ModuleRefSpec | WaveformRefSpec, path: str
) -> CfgNodeSpec:
    matches: list[CfgNodeSpec] = []
    for allowed in spec.allowed:
        try:
            matches.append(_resolve_spec_at_path(allowed, path))
        except (KeyError, RuntimeError):
            continue
    if not matches:
        allowed = ", ".join(shape.label for shape in spec.allowed)
        raise KeyError(
            f"Node ref path segment {path!r} not found in any allowed shape "
            f"(allowed: {allowed})"
        )
    first = matches[0]
    if not all(type(match) is type(first) for match in matches):
        raise TypeError(
            f"Node ref path {path!r} resolves to inconsistent spec types: "
            + ", ".join(type(match).__name__ for match in matches)
        )
    return first


def _get_value_at_path(value: CfgSectionValue, path: str) -> Any:
    section = value
    parts = _split_path(path)
    for part in parts[:-1]:
        child = section.fields.get(part)
        if isinstance(child, (ModuleRefValue, WaveformRefValue)):
            section = child.value
        elif isinstance(child, CfgSectionValue):
            section = child
        else:
            raise RuntimeError(
                f"Node field path {path!r} cannot descend into "
                f"{type(child).__name__} at {part!r}"
            )
    if parts[-1] not in section.fields:
        raise KeyError(f"Node field path {path!r} leaf {parts[-1]!r} not found")
    return section.fields[parts[-1]]


def _assign_value_at_path(value: CfgSectionValue, path: str, leaf: Any) -> None:
    section = value
    parts = _split_path(path)
    for part in parts[:-1]:
        child = section.fields.get(part)
        if isinstance(child, (ModuleRefValue, WaveformRefValue)):
            section = child.value
        elif isinstance(child, CfgSectionValue):
            section = child
        else:
            raise RuntimeError(
                f"Node field path {path!r} cannot descend into "
                f"{type(child).__name__} at {part!r}"
            )
    if parts[-1] not in section.fields:
        raise KeyError(f"Node field path {path!r} leaf {parts[-1]!r} not found")
    section.fields[parts[-1]] = leaf


_MISSING: Final = object()


def _generation_logical_paths(
    logical_paths: Mapping[str, str],
) -> dict[str, tuple[str, ...]]:
    paths: dict[str, tuple[str, ...]] = {}
    for logical_key, path in logical_paths.items():
        parts = _split_path(path)
        if parts[0] == "generation":
            paths[logical_key] = parts
    return paths


def _flatten_generation_persistence_raw(
    raw: dict[str, object],
    logical_paths: Mapping[str, str],
) -> dict[str, object]:
    generation_paths = _generation_logical_paths(logical_paths)
    if not generation_paths:
        return raw

    generation: dict[str, object] = {}
    for logical_key, parts in generation_paths.items():
        generation[logical_key] = _raw_value_at_path(raw, parts)
    raw["generation"] = generation
    return raw


def _expand_generation_persistence_raw(
    raw: dict[str, object],
    logical_paths: Mapping[str, str],
) -> dict[str, object]:
    generation_paths = _generation_logical_paths(logical_paths)
    if not generation_paths:
        return dict(raw)

    expanded = deepcopy(raw)
    raw_generation = expanded.get("generation")
    if raw_generation is None:
        return expanded
    if not isinstance(raw_generation, dict):
        return expanded

    generation = cast(dict[str, object], raw_generation)
    group_keys = {parts[1] for parts in generation_paths.values() if len(parts) > 2}
    flat_keys = set(generation_paths)
    unknown = set(generation) - group_keys - flat_keys
    if unknown:
        raise RuntimeError(
            "Unknown persisted generation key(s): " + ", ".join(sorted(unknown))
        )

    for logical_key, parts in generation_paths.items():
        if len(parts) <= 2 or logical_key not in generation:
            continue
        flat_value = generation.pop(logical_key)
        _assign_raw_path_if_absent_or_same(
            generation,
            parts[1:],
            flat_value,
            subject=logical_key,
        )
    return expanded


def _raw_value_at_path(raw: Mapping[str, object], parts: tuple[str, ...]) -> object:
    cur: object = raw
    for part in parts:
        if not isinstance(cur, Mapping):
            raise RuntimeError(
                f"Persisted node cfg path {'.'.join(parts)!r} cannot descend "
                f"through {type(cur).__name__}"
            )
        cur_map = cast(Mapping[str, object], cur)
        if part not in cur_map:
            raise RuntimeError(
                f"Persisted node cfg path {'.'.join(parts)!r} is missing"
            )
        cur = cur_map[part]
    return cur


def _assign_raw_path_if_absent_or_same(
    root: dict[str, object],
    parts: tuple[str, ...],
    value: object,
    *,
    subject: str,
) -> None:
    cur = root
    for part in parts[:-1]:
        child = cur.get(part)
        if child is None:
            child = {}
            cur[part] = child
        if not isinstance(child, dict):
            raise RuntimeError(
                f"Persisted generation key {subject!r} cannot descend through "
                f"{part!r}: found {type(child).__name__}"
            )
        cur = cast(dict[str, object], child)

    leaf = parts[-1]
    existing = cur.get(leaf, _MISSING)
    if existing is not _MISSING and existing != value:
        raise RuntimeError(f"Conflicting persisted generation values for {subject!r}")
    cur[leaf] = value


def _lower_value_at_path(
    value_tree: CfgSectionValue,
    spec: CfgNodeSpec,
    path: str,
    ml: ModuleLibrary | None,
    md: MetaDict | None,
) -> Any:
    value = _get_value_at_path(value_tree, path)
    try:
        raw = schema_to_raw_dict(
            CfgSchema(
                spec=CfgSectionSpec(fields={"value": spec}),
                value=CfgSectionValue(fields={"value": value}),
            ),
            md=md,
            ml=ml,
        )
    except RuntimeError as exc:
        raise RuntimeError(_rewrite_single_value_lower_error(str(exc), path)) from exc
    return raw.get("value", _MISSING)


def _rewrite_single_value_lower_error(message: str, path: str) -> str:
    """Preserve the node cfg path when lowering an isolated logical leaf."""
    return message.replace("Config field 'value.", f"Config field '{path}.").replace(
        "Config field 'value'", f"Config field '{path}'"
    )


def _jsonify_value_node(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, CfgSectionValue):
        return _jsonify_value_tree(value)
    if isinstance(value, ModuleRefValue):
        return {
            "__kind": "module_ref",
            "chosen_key": value.chosen_key,
            "is_overridden": bool(value.is_overridden),
            "value": _jsonify_value_tree(value.value),
        }
    if isinstance(value, WaveformRefValue):
        return {
            "__kind": "waveform_ref",
            "chosen_key": value.chosen_key,
            "is_overridden": bool(value.is_overridden),
            "value": _jsonify_value_tree(value.value),
        }
    if isinstance(value, SweepValue):
        return {
            "start": _knob_scalar_value(value.start),
            "stop": _knob_scalar_value(value.stop),
            "expts": int(value.expts),
        }
    if isinstance(value, CenteredSweepValue):
        return {
            "center": _knob_scalar_value(value.center),
            "span": float(value.span),
            "expts": int(value.expts),
            "step": float(value.step),
        }
    if isinstance(value, DirectValue):
        return _knob_scalar_value(value.value)
    if isinstance(value, EvalValue):
        return _knob_eval_value(value)
    raise TypeError(
        f"Unexpected node cfg value-tree leaf {type(value).__name__}; "
        "expected CfgSectionValue, DirectValue, EvalValue, SweepValue, "
        "or CenteredSweepValue"
    )


def _jsonify_value_tree(value: CfgSectionValue) -> dict[str, Any]:
    return {
        key: _jsonify_value_node(child)
        for key, child in value.fields.items()
        if child is not None
    }


def _knob_scalar_value(value: object) -> object:
    if isinstance(value, EvalValue):
        return _knob_eval_value(value)
    if isinstance(value, np.generic):
        return cast(np.generic, value).item()
    return value


def _knob_eval_value(value: EvalValue) -> dict[str, object]:
    data: dict[str, object] = {"__kind": "eval", "expr": value.expr}
    if value.resolved is not None:
        data["resolved"] = value.resolved
    if value.error is not None:
        data["error"] = value.error
    return data
