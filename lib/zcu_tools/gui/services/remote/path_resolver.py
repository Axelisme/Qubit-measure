"""Dotted-path resolver for remote ``cfg.set_field``.

Walks a live ``SectionLiveField`` tree and mutates a single leaf using the
field's existing public mutation surface, so a remote edit behaves exactly
like a user edit (auto-commit via the form's ``on_change`` -> ``schema_changed``
-> ``Controller.update_tab_cfg`` chain).

Path grammar (segments split on ``.``):

  - Scalar leaf:        ``section.sub.field``           -> ``ScalarLiveField.set_value(value)``
  - Sweep sub-field:    ``...path.sweep.start|stop|expts|step``
  - ModuleRef key:      ``...path.ref``                  -> ``set_chosen_key(value)``
  - ModuleRef sub:      ``...path.value.<sub>...``       -> recurse into ``.sub_field``
  - DeviceRef:          ``...path.device``               -> ``set_chosen_name(value)``
  - Literal:            rejected (immutable)

Unknown paths, type mismatches, and immutable targets raise
``RemoteError(INVALID_PARAMS, ...)`` with a discriminating message.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from zcu_tools.gui.adapter import (
    CfgSectionSpec,
    DeviceRefSpec,
    DirectValue,
    EvalValue,
    LiteralSpec,
    ModuleRefSpec,
    ScalarSpec,
    SweepSpec,
    WaveformRefSpec,
)
from zcu_tools.gui.live_model import (
    DeviceRefLiveField,
    LiteralLiveField,
    ModuleRefLiveField,
    ScalarLiveField,
    SectionLiveField,
    SweepLiveField,
)
from zcu_tools.gui.sweep_model import SweepEditor

from .errors import ErrorCode, RemoteError

if TYPE_CHECKING:
    from zcu_tools.gui.adapter import CfgNodeSpec
    from zcu_tools.gui.live_model import LiveField

_SWEEP_EDGES = {"start", "stop", "expts", "step"}


def resolve_and_set(root: SectionLiveField, path: str, value: object) -> None:
    """Resolve ``path`` against ``root`` and set the leaf to ``value``.

    ``root`` must be the live tab ``SectionLiveField``; mutations propagate
    through its existing ``on_change`` bubbling.
    """
    if not path:
        raise RemoteError(ErrorCode.INVALID_PARAMS, "empty path")
    segments = path.split(".")
    _set_recursive(root, segments, path, value)


def _set_recursive(
    field: "LiveField", segments: list[str], full_path: str, value: object
) -> None:
    head = segments[0]
    rest = segments[1:]

    if isinstance(field, SectionLiveField):
        child = field.fields.get(head)
        if child is None:
            raise RemoteError(
                ErrorCode.INVALID_PARAMS,
                f"unknown field {head!r} in path {full_path!r}",
            )
        if not rest:
            _set_leaf(child, full_path, value)
            return
        _set_recursive(child, rest, full_path, value)
        return

    # Reached a non-section field but still have segments to consume.
    if isinstance(field, SweepLiveField):
        _set_sweep_edge(field, head, rest, full_path, value)
        return

    if isinstance(field, ModuleRefLiveField):
        _set_moduleref(field, head, rest, full_path, value)
        return

    raise RemoteError(
        ErrorCode.INVALID_PARAMS,
        f"path {full_path!r} descends into non-container {type(field).__name__}",
    )


def _set_leaf(field: "LiveField", full_path: str, value: object) -> None:
    if isinstance(field, LiteralLiveField):
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            f"path {full_path!r} targets an immutable literal field",
        )
    if isinstance(field, ScalarLiveField):
        field.set_value(value)
        return
    if isinstance(field, DeviceRefLiveField):
        if not isinstance(value, str):
            raise RemoteError(
                ErrorCode.INVALID_PARAMS,
                f"device ref at {full_path!r} expects a string name",
            )
        field.set_chosen_name(value)
        return
    if isinstance(field, ModuleRefLiveField):
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            f"path {full_path!r} targets a module ref; set "
            f"'{full_path}.ref' (key) or '{full_path}.value.<sub>' instead",
        )
    if isinstance(field, SweepLiveField):
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            f"path {full_path!r} targets a sweep; set "
            f"'{full_path}.start|stop|expts|step' instead",
        )
    if isinstance(field, SectionLiveField):
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            f"path {full_path!r} targets a section; descend to a leaf field",
        )
    raise RemoteError(
        ErrorCode.INVALID_PARAMS,
        f"path {full_path!r} targets unsupported field {type(field).__name__}",
    )


def _set_sweep_edge(
    sweep: SweepLiveField,
    edge: str,
    rest: list[str],
    full_path: str,
    value: object,
) -> None:
    # Allow an optional ``sweep`` qualifier: ``path.sweep.start`` and
    # ``path.start`` are both accepted for ergonomics.
    if edge == "sweep" and rest:
        edge = rest[0]
        rest = rest[1:]
    if rest:
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            f"path {full_path!r} has trailing segments after sweep edge",
        )
    if edge not in _SWEEP_EDGES:
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            f"sweep edge must be one of {sorted(_SWEEP_EDGES)}, got {edge!r}",
        )
    current = sweep.get_value()
    if edge == "expts":
        if not isinstance(value, int) or isinstance(value, bool):
            raise RemoteError(
                ErrorCode.INVALID_PARAMS, "sweep 'expts' must be an integer"
            )
        sweep.set_value(SweepEditor.update_expts(current, value))
        return
    if edge == "step":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RemoteError(ErrorCode.INVALID_PARAMS, "sweep 'step' must be a number")
        sweep.set_value(SweepEditor.update_step(current, float(value)))
        return
    # start / stop accept a number (EvalValue editing is out of scope here).
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"sweep '{edge}' must be a number")
    if edge == "start":
        sweep.set_value(SweepEditor.update_start(current, float(value)))
    else:
        sweep.set_value(SweepEditor.update_stop(current, float(value)))


def _set_moduleref(
    ref: ModuleRefLiveField,
    head: str,
    rest: list[str],
    full_path: str,
    value: object,
) -> None:
    if head == "ref":
        if rest:
            raise RemoteError(
                ErrorCode.INVALID_PARAMS,
                f"path {full_path!r} has trailing segments after 'ref'",
            )
        if not isinstance(value, str):
            raise RemoteError(
                ErrorCode.INVALID_PARAMS,
                f"module ref key at {full_path!r} expects a string",
            )
        ref.set_chosen_key(value)
        return
    if head == "value":
        sub = ref.sub_field
        if sub is None:
            raise RemoteError(
                ErrorCode.PRECONDITION_FAILED,
                f"module ref at {full_path!r} has no editable sub-fields for "
                "the current key",
            )
        if not rest:
            raise RemoteError(
                ErrorCode.INVALID_PARAMS,
                f"path {full_path!r} must name a sub-field after 'value'",
            )
        _set_recursive(sub, rest, full_path, value)
        return
    raise RemoteError(
        ErrorCode.INVALID_PARAMS,
        f"module ref segment must be 'ref' or 'value', got {head!r} in {full_path!r}",
    )


# ---------------------------------------------------------------------------
# Path discovery — the inverse of resolve_and_set
# ---------------------------------------------------------------------------


def _scalar_wire_value(field: ScalarLiveField) -> object:
    """Return a JSON-safe current value for a scalar field.

    DirectValue → its value (or None when unset); EvalValue → its expression
    string (so the agent sees what to overwrite).
    """
    val = field.get_value()
    if isinstance(val, EvalValue):
        return val.expr
    if isinstance(val, DirectValue):
        return None if val.is_unset else val.value
    return None


def _scalar_entry(path: str, field: ScalarLiveField) -> dict[str, object]:
    entry: dict[str, object] = {
        "path": path,
        "kind": "scalar",
        "value": _scalar_wire_value(field),
        "type": field.spec.type.__name__,
    }
    if field.spec.choices is not None:
        entry["choices"] = list(field.spec.choices)
    return entry


def _sweep_entries(path: str, field: SweepLiveField) -> list[dict[str, object]]:
    sweep = field.get_value()
    edges = {
        "start": sweep.start,
        "stop": sweep.stop,
        "expts": sweep.expts,
        "step": sweep.step,
    }
    out: list[dict[str, object]] = []
    for edge, raw in edges.items():
        value: object = raw.expr if isinstance(raw, EvalValue) else raw
        out.append(
            {
                "path": f"{path}.{edge}",
                "kind": "sweep_edge",
                "value": value,
                "type": "integer" if edge == "expts" else "number",
            }
        )
    return out


def _list_field(path: str, field: "LiveField") -> list[dict[str, object]]:
    """Recurse one LiveField, returning the settable leaves beneath it."""
    if isinstance(field, LiteralLiveField):
        return []  # immutable — resolve_and_set rejects it
    if isinstance(field, ScalarLiveField):
        return [_scalar_entry(path, field)]
    if isinstance(field, SweepLiveField):
        return _sweep_entries(path, field)
    if isinstance(field, DeviceRefLiveField):
        return [
            {
                "path": f"{path}.device",
                "kind": "deviceref",
                "value": field.get_chosen_name(),
                "type": "string",
            }
        ]
    if isinstance(field, ModuleRefLiveField):
        out = [
            {
                "path": f"{path}.ref",
                "kind": "moduleref_key",
                "value": field.get_chosen_key(),
                "type": "string",
                "choices": [s.label for s in field.spec.allowed],
            }
        ]
        sub = field.sub_field
        if sub is not None:
            for key, child in sub.fields.items():
                out.extend(_list_field(f"{path}.value.{key}", child))
        return out
    if isinstance(field, SectionLiveField):
        out = []
        for key, child in field.fields.items():
            out.extend(_list_field(f"{path}.{key}" if path else key, child))
        return out
    return []


def list_settable_paths(root: SectionLiveField) -> list[dict[str, object]]:
    """Enumerate every dotted path that ``resolve_and_set`` can mutate.

    Each entry is ``{path, kind, value, type[, choices]}``. The path grammar is
    identical to resolve_and_set's, so every listed path round-trips through
    ``cfg.set_field``. Literal (immutable) leaves are omitted.
    """
    return _list_field("", root)


# ---------------------------------------------------------------------------
# Static spec-tree path discovery — no context, no values (for adapter.cfg_spec)
# ---------------------------------------------------------------------------


def _sweep_spec_entries(path: str) -> list[dict[str, object]]:
    return [
        {
            "path": f"{path}.{edge}",
            "kind": "sweep_edge",
            "type": "integer" if edge == "expts" else "number",
        }
        for edge in ("start", "stop", "expts", "step")
    ]


def _list_spec_field(path: str, node: "CfgNodeSpec") -> list[dict[str, object]]:
    """Recurse one spec node, returning settable leaves (no current value).

    The static analogue of ``_list_field``: it walks the spec tree instead of a
    live LiveModel, so an adapter's shape can be listed without a tab/context.
    ModuleRef/WaveformRef expose *every* allowed option's sub-fields under
    ``value.<label>`` (the live version only shows the chosen one).
    """
    if isinstance(node, LiteralSpec):
        return []  # immutable
    if isinstance(node, ScalarSpec):
        entry: dict[str, object] = {
            "path": path,
            "kind": "scalar",
            "type": node.type.__name__,
            "label": node.label,
        }
        if node.choices is not None:
            entry["choices"] = list(node.choices)
        return [entry]
    if isinstance(node, SweepSpec):
        return _sweep_spec_entries(path)
    if isinstance(node, DeviceRefSpec):
        return [{"path": f"{path}.device", "kind": "deviceref", "type": "string"}]
    if isinstance(node, (ModuleRefSpec, WaveformRefSpec)):
        kind = "moduleref_key" if isinstance(node, ModuleRefSpec) else "waveformref_key"
        out = [
            {
                "path": f"{path}.ref",
                "kind": kind,
                "type": "string",
                "choices": [s.label for s in node.allowed],
            }
        ]
        for section in node.allowed:
            for key, child in section.fields.items():
                out.extend(
                    _list_spec_field(f"{path}.value.{section.label}.{key}", child)
                )
        return out
    if isinstance(node, CfgSectionSpec):
        out = []
        for key, child in node.fields.items():
            out.extend(_list_spec_field(f"{path}.{key}" if path else key, child))
        return out
    return []


def list_spec_paths(spec: CfgSectionSpec) -> list[dict[str, object]]:
    """Enumerate an adapter's settable cfg paths from its static spec tree.

    Like ``list_settable_paths`` but over a pure ``CfgSectionSpec`` (no values,
    no live model), so it works without building a tab. Use for adapter
    introspection; use ``list_settable_paths`` for a live tab's current values.
    """
    return _list_spec_field("", spec)


# ---------------------------------------------------------------------------
# Sub-tree discovery — list the leaves beneath the field a path points at
# ---------------------------------------------------------------------------


def _navigate(root: SectionLiveField, segments: list[str]) -> tuple["LiveField", str]:
    """Walk ``segments`` from ``root`` to the addressed field and its root path.

    Returns ``(field, base_path)`` where ``base_path`` is the dotted path that
    ``_list_field`` should prepend so the re-listed sub-tree round-trips through
    ``set_field``. For a ModuleRef, a trailing ``ref`` (the key) resolves back to
    the ref *field itself* (base path without ``.ref``), because switching the
    key rebuilds the whole ref sub-tree — that is what callers want re-listed.
    A ``value`` segment descends into the bound sub-section.
    """
    field: "LiveField" = root
    consumed: list[str] = []
    i = 0
    while i < len(segments):
        head = segments[i]
        if isinstance(field, SectionLiveField):
            child = field.fields.get(head)
            if child is None:
                raise RemoteError(
                    ErrorCode.INVALID_PARAMS,
                    f"unknown field {head!r} in path {'.'.join(segments)!r}",
                )
            field = child
            consumed.append(head)
            i += 1
            continue
        if isinstance(field, ModuleRefLiveField):
            if head == "ref":
                # The key segment maps back to the ref field itself; its
                # sub-tree is re-listed at the ref's own base path.
                i += 1
                continue
            if head == "value":
                sub = field.sub_field
                if sub is None:
                    raise RemoteError(
                        ErrorCode.PRECONDITION_FAILED,
                        f"module ref at {'.'.join(consumed)!r} has no sub-fields",
                    )
                field = sub
                consumed.append("value")
                i += 1
                continue
            raise RemoteError(
                ErrorCode.INVALID_PARAMS,
                f"module ref segment must be 'ref' or 'value', got {head!r}",
            )
        # Sweep / multi-sweep edges are leaves — stop here; the caller lists
        # the sweep node itself.
        break
    return field, ".".join(consumed)


def list_subtree_paths(root: SectionLiveField, path: str) -> list[dict[str, object]]:
    """Return the settable leaves beneath the field addressed by ``path``.

    After a ``resolve_and_set`` mutation (especially a ModuleRef key switch that
    rebuilds the sub-tree), this re-lists only what changed. A scalar path
    returns its own single entry; a ref/section path returns the freshly-bound
    sub-tree. Paths are rooted so they round-trip through ``set_field``.
    """
    if not path:
        return list_settable_paths(root)
    field, base_path = _navigate(root, path.split("."))
    return _list_field(base_path, field)
