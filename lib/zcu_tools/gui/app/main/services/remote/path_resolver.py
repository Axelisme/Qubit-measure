"""Dotted-path resolver for remote ``cfg.set_field``.

Walks a live ``SectionLiveField`` tree and mutates a single leaf using the
field's existing public mutation surface, so a remote edit behaves exactly
like a user edit (auto-commit via the form's ``on_change`` -> ``schema_changed``
-> ``Controller.update_tab_cfg`` chain).

Path grammar (segments split on ``.``):

  - Scalar leaf:        ``section.sub.field``           -> ``ScalarLiveField.set_value(value)``
  - Sweep sub-field:    ``...path.sweep.start|stop|expts|step``
  - ModuleRef key:      ``...path.ref``                  -> ``set_chosen_key(value)``
  - ModuleRef sub:      ``...path.<sub>...``             -> recurse into ``.sub_field``
  - DeviceRef:          ``...path.device``               -> ``set_chosen_name(value)``
  - Literal:            rejected (immutable)

ModuleRef sub-fields descend directly (no ``value`` wrapper segment); a path
that still carries a ``value`` segment is rejected with a migration message.

Unknown paths, type mismatches, and immutable targets raise
``RemoteError(INVALID_PARAMS, ...)`` with a discriminating message.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from zcu_tools.gui.app.main.adapter import (
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
from zcu_tools.gui.app.main.live_model import (
    DeviceRefLiveField,
    LiteralLiveField,
    ModuleRefLiveField,
    ScalarLiveField,
    SectionLiveField,
    SweepLiveField,
)
from zcu_tools.gui.app.main.sweep_model import SweepEditor
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.adapter import CfgNodeSpec
    from zcu_tools.gui.app.main.live_model import LiveField

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
            f"'{full_path}.ref' (key) or '{full_path}.<sub>' instead",
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
        # list_paths advertises a ModuleRef's options as bare variant labels in
        # 'choices' (e.g. "Pulse Readout"), but a chosen_key for a built-in
        # variant is the tagged form "<Custom:Pulse Readout>". An agent naturally
        # echoes the bare label back; normalize it so set_chosen_key stores a
        # valid key. A bare label is treated as LINKED (library entry name)
        # otherwise, which silently rebuilds an empty sub-field and later fails
        # lowering with "Unknown module reference". Only a value that exactly
        # matches an allowed variant label is wrapped; anything else (already
        # tagged, or a real library entry name) passes through unchanged.
        key = value
        if not (key.startswith("<Custom:") and key.endswith(">")):
            allowed_labels = {s.label for s in ref.spec.allowed}
            if key in allowed_labels:
                key = f"<Custom:{key}>"
        ref.set_chosen_key(key)
        return
    if head == "value":
        # The old grammar wrapped ref sub-fields under a 'value' segment; it is
        # gone (paths now descend straight into the sub-field). Fail fast so a
        # stale path is obvious rather than silently mis-resolving.
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            f"path {full_path!r} uses the removed 'value' segment; drop it "
            f"(e.g. '<ref>.{'.'.join(rest)}' instead of '<ref>.value.{'.'.join(rest)}')",
        )
    # Anything other than 'ref' is a sub-field of the currently-bound section.
    sub = ref.sub_field
    if sub is None:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            f"module ref at {full_path!r} has no editable sub-fields for "
            "the current key",
        )
    _set_recursive(sub, [head, *rest], full_path, value)


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
        return val.value  # None means unset (ADR-0010)
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
                out.extend(_list_field(f"{path}.{key}", child))
        return out
    if isinstance(field, SectionLiveField):
        out = []
        for key, child in field.fields.items():
            out.extend(_list_field(f"{path}.{key}" if path else key, child))
        return out
    return []


# Fields dropped at each verbosity level (lower = less token noise). 'full'
# keeps everything; 'compact' drops the current value + python type (re-readable
# elsewhere) but KEEPS kind + choices (Fast-Fail semantics: agent needs the enum
# choices to validate before commit, and 'kind' to know a ref-switch point);
# 'paths' is a bare list[str].
_VERBOSITY_DROP = {"compact": ("value", "type")}


def _project(
    entries: list[dict[str, object]], verbosity: str
) -> "list[dict[str, object]] | list[str]":
    """Project full path entries down to the requested verbosity."""
    if verbosity == "full":
        return entries
    if verbosity == "paths":
        return [str(e["path"]) for e in entries]
    if verbosity == "compact":
        drop = _VERBOSITY_DROP["compact"]
        return [{k: v for k, v in e.items() if k not in drop} for e in entries]
    raise RemoteError(
        ErrorCode.INVALID_PARAMS,
        f"unknown verbosity {verbosity!r}; expected one of full/compact/paths",
    )


def list_settable_paths(
    root: SectionLiveField,
    under: "str | None" = None,
    verbosity: str = "full",
) -> "list[dict[str, object]] | list[str]":
    """Enumerate the dotted paths that ``resolve_and_set`` can mutate.

    The path grammar is identical to resolve_and_set's, so every listed path
    round-trips through ``cfg.set_field``. Literal (immutable) leaves are
    omitted.

    ``under`` restricts the listing to the sub-tree rooted at that dotted path
    (same navigation as a ModuleRef key switch's re-list); omit it for the whole
    draft. ``verbosity`` controls the per-entry shape: ``full`` (default, the
    mechanism layer's full fidelity) = ``{path, kind, value, type[, choices]}``;
    ``compact`` drops ``value``/``type`` but keeps ``kind``/``choices``;
    ``paths`` = bare ``list[str]``. The agent-facing default (compact) is chosen
    by the mcp/RPC layer, not here.
    """
    if under:
        field, base_path = _navigate(root, under.split("."))
        entries = _list_field(base_path, field)
    else:
        entries = _list_field("", root)
    return _project(entries, verbosity)


def list_settable_paths_full(
    root: SectionLiveField, under: "str | None" = None
) -> list[dict[str, object]]:
    """``list_settable_paths`` at full verbosity, typed as the dict-entry list.

    Internal callers (diffing, sub-tree re-list) need the dict form and a
    non-union return type; this thin wrapper gives them that without casts.
    """
    result = list_settable_paths(root, under=under, verbosity="full")
    return cast("list[dict[str, object]]", result)


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

    ModuleRef/WaveformRef nodes do NOT descend into any variant's sub-fields —
    they emit only the ``.ref`` selector plus its allowed ``choices``. Which
    variant is the live default is a value-layer decision (the adapter's
    ``make_default_value(ctx)``, context-dependent) that a static, context-free
    spec walk cannot know; and a variant's fields only become concrete once a
    tab is built. So the agent reads the shape skeleton + ref options here, then
    picks a ref and reads that variant's live fields via ``tab.list_paths``.
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
        return [
            {
                "path": f"{path}.ref",
                "kind": kind,
                "type": "string",
                "choices": [s.label for s in node.allowed],
            }
        ]
    if isinstance(node, CfgSectionSpec):
        out = []
        for key, child in node.fields.items():
            seg = f"{path}.{key}" if path else key
            out.extend(_list_spec_field(seg, child))
        return out
    return []


def list_spec_paths(spec: CfgSectionSpec) -> list[dict[str, object]]:
    """Enumerate an adapter's settable cfg paths from its static spec tree.

    Like ``list_settable_paths`` but over a pure ``CfgSectionSpec`` (no values,
    no live model), so it works without building a tab. ModuleRef/WaveformRef
    nodes list only their ``.ref`` selector + allowed choices, not any variant's
    inner fields (see ``_list_spec_field``). Use for adapter introspection; use
    ``list_settable_paths`` on a live tab to read a chosen variant's fields.
    """
    return _list_spec_field("", spec)


# ---------------------------------------------------------------------------
# Path navigation — walk a dotted path to the field it addresses (used by the
# ``under`` sub-tree scoping in list_settable_paths).
# ---------------------------------------------------------------------------


def _navigate(root: SectionLiveField, segments: list[str]) -> tuple["LiveField", str]:
    """Walk ``segments`` from ``root`` to the addressed field and its root path.

    Returns ``(field, base_path)`` where ``base_path`` is the dotted path that
    ``_list_field`` should prepend so the re-listed sub-tree round-trips through
    ``set_field``. For a ModuleRef, a trailing ``ref`` (the key) resolves back to
    the ref *field itself* (base path without ``.ref``), because switching the
    key rebuilds the whole ref sub-tree — that is what callers want re-listed.
    Other segments descend into the bound sub-section directly (no ``value``).
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
            # Any other segment is a sub-field of the bound section (no 'value'
            # wrapper). Descend into sub_field but keep the SAME base path, so
            # re-listed leaves match the wire path the agent uses.
            sub = field.sub_field
            if sub is None:
                raise RemoteError(
                    ErrorCode.PRECONDITION_FAILED,
                    f"module ref at {'.'.join(consumed)!r} has no sub-fields",
                )
            field = sub
            continue
        # Sweep / multi-sweep edges are leaves — stop here; the caller lists
        # the sweep node itself.
        break
    return field, ".".join(consumed)
