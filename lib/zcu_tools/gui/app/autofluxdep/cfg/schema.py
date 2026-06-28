"""Node-knob schema — typed value tree with logical-key projection.

``NodeCfgSchema`` is the per-placement SSOT for an autofluxdep node's
user-editable knobs. The visible value tree may be flat or sectioned, while the
write/lower contract stays keyed by stable logical names (``reps``,
``detune_sweep``, ``qub_gain``). That lets the UI present measure-like sections
without forcing every Builder to read nested dicts.

``flat_node_schema`` keeps the current flat Builder contract. New sectioned node
schemas should be built through ``sectioned_node_schema`` so the logical mapping
is explicit and owned by this seam.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final, cast

import numpy as np
from numpy.typing import NDArray

from zcu_tools.gui.app.main.adapter import (
    CfgNodeSpec,
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    EvalValue,
    ScalarSpec,
    SweepSpec,
    SweepValue,
)

if TYPE_CHECKING:
    from zcu_tools.meta_tool import MetaDict, ModuleLibrary


def sweepcfg_to_axis(sweep: Any) -> NDArray[np.float64]:
    """The explicit linspace axis a lowered ``SweepCfg`` (start/stop/expts) samples.

    A ``SweepSpec`` lowers to a ``SweepCfg`` (the FPGA sweep); a Result stores its
    swept axis as an explicit array. Rebuilds it from the cfg's endpoints + count
    so the Result columns match the program sweep exactly — the typed counterpart
    of the prototype's ``parse_detune_sweep`` (now that the axis is sweep-defined,
    not free-text ``start,stop,step``).
    """
    return np.linspace(float(sweep.start), float(sweep.stop), int(sweep.expts))


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
            return value.strip().lower() in ("1", "true", "yes", "on")
        # int/float: parse, fast-failing a malformed value
        return type_(float(value)) if type_ is int else type_(value)
    if type_ is str:
        return str(value)
    if type_ is bool:
        return bool(value)
    return type_(value)


def str_scalar_spec(
    label: str, *, required: bool = False, optional: bool = False
) -> ScalarSpec:
    """A string scalar field (e.g. a by-name waveform reference).

    The framework's ``IntSpec`` / ``FloatSpec`` sugar covers numerics; a string
    knob (the waveform name a node lowers via ``ml.get_waveform``) needs the bare
    ``ScalarSpec(type=str)``, so this is its explicit counterpart. ``optional``
    lets the knob lower to an omitted key when unset (a node then Fast-Fails on
    the missing waveform name with its own message).
    """
    return ScalarSpec(label=label, type=str, required=required, optional=optional)


# One field declaration: its spec + its default value. The default is the
# hardcoded value the old ``make_cfg`` carried (now the single source of truth).
NodeField = tuple[str, CfgNodeSpec, Any]


@dataclass(frozen=True)
class NodeFieldSpec:
    """One logical node knob mounted as a leaf inside a UI section."""

    logical_key: str
    section_key: str
    field_key: str
    spec: CfgNodeSpec
    default: Any

    @property
    def path(self) -> str:
        return f"{self.section_key}.{self.field_key}"


@dataclass(frozen=True)
class NodeSectionSpec:
    """A UI section grouping one-or-more logical node knobs."""

    key: str
    label: str
    fields: tuple[NodeFieldSpec, ...]


def _default_value_for(spec: CfgNodeSpec, default: Any) -> Any:
    """Build the value-tree leaf for ``spec`` from its declared ``default``."""
    if isinstance(spec, SweepSpec):
        if not isinstance(default, SweepValue):
            raise TypeError(
                f"SweepSpec default must be a SweepValue, got {type(default).__name__}"
            )
        return default
    if isinstance(spec, ScalarSpec):
        return DirectValue(default)
    raise TypeError(f"Unsupported node field spec: {type(spec).__name__}")


def flat_node_schema(fields: tuple[NodeField, ...]) -> CfgSchema:
    """A ``CfgSchema`` for a flat list of ``(key, spec, default)`` node knobs."""
    _ensure_unique("node field key", (key for key, _, _ in fields))
    spec = CfgSectionSpec(fields={key: node_spec for key, node_spec, _ in fields})
    value = CfgSectionValue(
        fields={
            key: _default_value_for(node_spec, default)
            for key, node_spec, default in fields
        }
    )
    return CfgSchema(spec=spec, value=value)


def sectioned_node_schema(sections: tuple[NodeSectionSpec, ...]) -> NodeCfgSchema:
    """Build a sectioned node schema with an explicit logical-key projection."""
    _ensure_unique("node section key", (section.key for section in sections))

    root_spec_fields: dict[str, CfgNodeSpec] = {}
    root_value_fields: dict[str, Any] = {}
    logical_paths: dict[str, str] = {}

    for section in sections:
        _validate_path_part("section key", section.key)
        _ensure_unique(
            f"field key in section {section.key!r}",
            (field_spec.field_key for field_spec in section.fields),
        )
        section_spec_fields: dict[str, CfgNodeSpec] = {}
        section_value_fields: dict[str, Any] = {}

        for field_spec in section.fields:
            _validate_node_field_spec(section.key, field_spec)
            if field_spec.logical_key in logical_paths:
                raise ValueError(
                    f"Duplicate node logical key {field_spec.logical_key!r}"
                )
            section_spec_fields[field_spec.field_key] = field_spec.spec
            section_value_fields[field_spec.field_key] = _default_value_for(
                field_spec.spec, field_spec.default
            )
            logical_paths[field_spec.logical_key] = field_spec.path

        root_spec_fields[section.key] = CfgSectionSpec(
            label=section.label,
            fields=section_spec_fields,
        )
        root_value_fields[section.key] = CfgSectionValue(fields=section_value_fields)

    return NodeCfgSchema(
        CfgSchema(
            spec=CfgSectionSpec(fields=root_spec_fields),
            value=CfgSectionValue(fields=root_value_fields),
        ),
        logical_paths=logical_paths,
    )


def _validate_node_field_spec(section_key: str, field_spec: NodeFieldSpec) -> None:
    _validate_path_part("logical key", field_spec.logical_key)
    _validate_path_part("field key", field_spec.field_key)
    if field_spec.section_key != section_key:
        raise ValueError(
            f"Node field {field_spec.logical_key!r} declares section "
            f"{field_spec.section_key!r}, but is mounted under {section_key!r}"
        )
    if not isinstance(field_spec.spec, (ScalarSpec, SweepSpec)):
        raise TypeError(
            f"Unsupported node field spec for {field_spec.logical_key!r}: "
            f"{type(field_spec.spec).__name__}; only ScalarSpec and SweepSpec "
            "are supported"
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

    def path_for(self, logical_key: str) -> str:
        """Return the dotted value-tree path for a stable logical knob key."""
        return self._require_logical_path(logical_key)

    def set_field(self, key: str, value: Any) -> None:
        """Write one knob leaf into the value tree, fast-failing an unknown key.

        The single typed entry into the SSOT, used by two writers: the typed cfg
        form (which emits value-tree leaves — ``DirectValue`` / ``SweepValue``)
        and tests / ``with_overrides`` (which pass raw scalars). An unknown key is
        a real typo (only declared knobs are writable), so it fast-fails.

        - A ``SweepSpec`` knob accepts a ``SweepValue`` (stored verbatim).
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
        ``md`` through the shared cfg lowering path.
        """
        raw = self.schema.to_raw_dict(md=md, ml=ml)
        knobs: dict[str, Any] = {}
        for logical_key, path in self.logical_paths.items():
            value = _get_raw_at_path(raw, path)
            if value is _MISSING:
                continue
            knobs[logical_key] = value
        return knobs

    def read_knobs(self) -> dict[str, Any]:
        """The current *un-lowered* user knob values, as a JSON-friendly dict.

        Read-only projection of the value tree for an observer (the remote
        bridge's ``node.cfg``): keys are stable logical knob names, a scalar knob
        reads to its plain value (or an eval marker), and a sweep knob reads to
        ``{start, stop, expts}``. Unlike ``lower``, this does NOT build a
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

    def read_value_tree(self) -> dict[str, Any]:
        """The sectioned value tree, JSON-friendly, for UI/debug assertions."""
        return _jsonify_value_tree(self.schema.value)

    def logical_updates_from(self, value: CfgSectionValue) -> dict[str, Any]:
        """Project a full UI draft value tree back into logical-key updates."""
        updates: dict[str, Any] = {}
        for logical_key, path in self.logical_paths.items():
            updates[logical_key] = _get_value_at_path(value, path)
        return updates

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
        if not isinstance(leaf_spec, (ScalarSpec, SweepSpec)):
            raise TypeError(
                f"Logical node param {logical_key!r} maps to unsupported spec "
                f"{type(leaf_spec).__name__}; only ScalarSpec and SweepSpec "
                "are supported"
            )


def _split_path(path: str) -> tuple[str, ...]:
    parts = tuple(part for part in path.split(".") if part)
    if not parts:
        raise RuntimeError("Node field path must not be empty")
    return parts


def _resolve_spec_at_path(spec: CfgSectionSpec, path: str) -> CfgNodeSpec:
    node_spec: CfgNodeSpec = spec
    for part in _split_path(path):
        if not isinstance(node_spec, CfgSectionSpec):
            raise RuntimeError(
                f"Node field path {path!r} cannot descend into "
                f"{type(node_spec).__name__} at {part!r}"
            )
        if part not in node_spec.fields:
            raise KeyError(
                f"Node field path {path!r} segment {part!r} not found; "
                f"available: {', '.join(node_spec.fields)}"
            )
        node_spec = node_spec.fields[part]
    return node_spec


def _get_value_at_path(value: CfgSectionValue, path: str) -> Any:
    section = value
    parts = _split_path(path)
    for part in parts[:-1]:
        child = section.fields.get(part)
        if not isinstance(child, CfgSectionValue):
            raise RuntimeError(
                f"Node field path {path!r} cannot descend into "
                f"{type(child).__name__} at {part!r}"
            )
        section = child
    if parts[-1] not in section.fields:
        raise KeyError(f"Node field path {path!r} leaf {parts[-1]!r} not found")
    return section.fields[parts[-1]]


def _assign_value_at_path(value: CfgSectionValue, path: str, leaf: Any) -> None:
    section = value
    parts = _split_path(path)
    for part in parts[:-1]:
        child = section.fields.get(part)
        if not isinstance(child, CfgSectionValue):
            raise RuntimeError(
                f"Node field path {path!r} cannot descend into "
                f"{type(child).__name__} at {part!r}"
            )
        section = child
    if parts[-1] not in section.fields:
        raise KeyError(f"Node field path {path!r} leaf {parts[-1]!r} not found")
    section.fields[parts[-1]] = leaf


_MISSING: Final = object()


def _get_raw_at_path(raw: Mapping[str, Any], path: str) -> Any:
    node: Any = raw
    for part in _split_path(path):
        if not isinstance(node, Mapping):
            raise RuntimeError(
                f"Lowered node field path {path!r} cannot descend into "
                f"{type(node).__name__} at {part!r}"
            )
        if part not in node:
            return _MISSING
        node = node[part]
    return node


def _jsonify_value_node(value: Any) -> Any:
    if isinstance(value, CfgSectionValue):
        return _jsonify_value_tree(value)
    if isinstance(value, SweepValue):
        return {
            "start": _knob_scalar_value(value.start),
            "stop": _knob_scalar_value(value.stop),
            "expts": int(value.expts),
        }
    if isinstance(value, DirectValue):
        return _knob_scalar_value(value.value)
    if isinstance(value, EvalValue):
        return _knob_eval_value(value)
    raise TypeError(
        f"Unexpected node cfg value-tree leaf {type(value).__name__}; "
        "expected CfgSectionValue, DirectValue, EvalValue, or SweepValue"
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
