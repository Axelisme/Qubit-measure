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
    field: LiveField, segments: list[str], full_path: str, value: object
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

    if isinstance(field, DeviceRefLiveField):
        _set_deviceref(field, head, rest, full_path, value)
        return

    raise RemoteError(
        ErrorCode.INVALID_PARAMS,
        f"path {full_path!r} descends into non-container {type(field).__name__}",
    )


def _set_leaf(field: LiveField, full_path: str, value: object) -> None:
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


def _set_deviceref(
    ref: DeviceRefLiveField,
    head: str,
    rest: list[str],
    full_path: str,
    value: object,
) -> None:
    # list_settable_paths advertises a DeviceRef as '<path>.device' (mirroring a
    # ModuleRef's '.ref' selector), so the advertised path must resolve here —
    # 'device' is the only valid trailing segment, and the value is the device
    # name. (The bare-leaf form '<path>' is also accepted via _set_leaf.)
    if head != "device" or rest:
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            f"device ref at {full_path!r} takes only a trailing '.device' "
            "segment set to a device name",
        )
    if not isinstance(value, str):
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            f"device ref at {full_path!r} expects a string name",
        )
    ref.set_chosen_name(value)


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


def _list_field(path: str, field: LiveField) -> list[dict[str, object]]:
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


def _filter_by_prefix(
    entries: list[dict[str, object]], prefix: str
) -> list[dict[str, object]]:
    """Keep entries whose dotted path equals ``prefix`` or is below it.

    Dotted-path prefix semantics (NOT fnmatch glob): ``modules.readout`` matches
    ``modules.readout`` itself and every ``modules.readout.*`` descendant, but
    NOT a sibling like ``modules.readout_extra`` (the boundary is a dotted
    segment, hence the ``prefix + "."`` test). Paths are returned as full dotted
    strings (no re-rooting). A prefix matching nothing yields an empty list —
    not an error — so an over-narrow prefix is a graceful no-result, not a
    fast-fail.
    """
    return [
        e
        for e in entries
        if (p := str(e["path"])) == prefix or p.startswith(prefix + ".")
    ]


def list_settable_paths(
    root: SectionLiveField,
    under: str | None = None,
    prefix: str | None = None,
) -> list[dict[str, object]]:
    """Enumerate the dotted paths that ``resolve_and_set`` can mutate.

    DIFF-ONLY / INTERNAL: this flat lister is NOT an agent-facing view (agents
    read the nested ``build_settable_tree`` via editor.get / tab.list_paths). Its
    only caller is ``set_field``'s before/after diff (which paths a ref switch
    added/removed), reached via the ``list_settable_paths_full`` wrapper — it
    needs the flat ``{path, kind, value, type[, choices]}`` entry list.

    The path grammar is identical to resolve_and_set's, so every listed path
    round-trips through ``cfg.set_field``. Literal (immutable) leaves are
    omitted. ``under`` restricts the listing to the sub-tree rooted at that
    dotted path and validates the path (unknown field → INVALID_PARAMS; ModuleRef
    ref receives special handling); omit it for the whole draft. ``prefix`` is a
    pure flat string-prefix filter applied after full-path listing: it keeps only
    the paths equal to or below that dotted prefix (dotted-segment boundary, not
    glob), does not validate the prefix, and returns an empty list on no match.
    Both ``under`` and ``prefix`` output full dotted path strings.
    """
    if under:
        field, base_path = _navigate(root, under.split("."))
        entries = _list_field(base_path, field)
    else:
        entries = _list_field("", root)
    if prefix:
        entries = _filter_by_prefix(entries, prefix)
    return entries


def list_settable_paths_full(
    root: SectionLiveField, under: str | None = None
) -> list[dict[str, object]]:
    """``list_settable_paths`` typed as the dict-entry list.

    Internal callers (diffing, sub-tree re-list) want a non-union return type;
    this thin wrapper preserves that call-site stability.
    """
    return list_settable_paths(root, under=under)


# ---------------------------------------------------------------------------
# Nested-tree path discovery — a value tree shaped like the cfg, where each
# leaf carries its current live value. The structural inverse of the flat
# ``list_settable_paths`` (which stays for set_field's before/after diff only):
# this one is the read-only view tab.list_paths returns.
#
# Reserved ``$``-prefixed keys distinguish a leaf's metadata from a sub-tree:
#   - a dict carrying ``$value`` (enum scalar) or ``$ref`` (ref node) is a
#     SPECIAL node, not a sub-tree;
#   - any other dict is a sub-tree (its keys are child field names);
#   - a non-dict is a bare scalar current value (None = unset, ADR-0010).
# Only the currently-chosen ref variant is expanded; ``options`` lists names.
# Values come straight off the live tree (the same ``_scalar_wire_value`` /
# sweep-edge logic the flat lister uses), NOT via the cfg_summary codec.
# ---------------------------------------------------------------------------


def _tree_sweep(field: SweepLiveField) -> dict[str, object]:
    """A sweep node as a sub-tree of bare scalar edges (start/stop/expts/step)."""
    sweep = field.get_value()
    edges = {
        "start": sweep.start,
        "stop": sweep.stop,
        "expts": sweep.expts,
        "step": sweep.step,
    }
    return {
        edge: (raw.expr if isinstance(raw, EvalValue) else raw)
        for edge, raw in edges.items()
    }


# Sentinel: a field that has no settable tree node (a literal / unsupported
# field). Section/ref recursion drops such children, mirroring the flat lister
# which omits immutable leaves entirely — so the tree shows only what set_field
# can actually mutate (a literal must NOT read as a settable ``null`` scalar).
_NO_NODE = object()


def _tree_field(field: LiveField) -> object:
    """Build the nested value tree for one LiveField (see module section header).

    Returns the node that represents ``field``: a bare scalar value, an enum
    ``{$value, $choices}`` leaf, a sweep sub-tree, a ``$ref`` node (with the
    chosen variant's sub-tree merged in), or a plain section sub-tree. Returns
    the ``_NO_NODE`` sentinel for an immutable/unsupported field so the parent
    drops it.
    """
    if isinstance(field, LiteralLiveField):
        return _NO_NODE  # immutable — not a settable path (flat lister drops it)
    if isinstance(field, ScalarLiveField):
        value = _scalar_wire_value(field)
        if field.spec.choices is not None:
            return {"$value": value, "$choices": list(field.spec.choices)}
        return value
    if isinstance(field, SweepLiveField):
        return _tree_sweep(field)
    if isinstance(field, DeviceRefLiveField):
        # A device ref is a leaf selector (no settable sub-tree); ``options`` is
        # the live registered-device list, not a static spec list.
        return {
            "$ref": {
                "current": field.get_chosen_name(),
                "options": list(field.env.ctrl.list_device_names()),
            }
        }
    if isinstance(field, ModuleRefLiveField):
        # Only the currently-bound variant is expanded; ``options`` lists the
        # allowed variant labels (bare), while ``current`` is the chosen key (a
        # built-in variant reads as the tagged ``<Custom:label>`` form, mirroring
        # the flat lister's value/choices split).
        node: dict[str, object] = {
            "$ref": {
                "current": field.get_chosen_key(),
                "options": [s.label for s in field.spec.allowed],
            }
        }
        sub = field.sub_field
        if sub is not None:
            node.update(_tree_section_children(sub))
        return node
    if isinstance(field, SectionLiveField):
        return _tree_section_children(field)
    return _NO_NODE


def _tree_section_children(section: SectionLiveField) -> dict[str, object]:
    """Recurse a section's fields into a sub-tree, dropping ``_NO_NODE`` ones."""
    out: dict[str, object] = {}
    for key, child in section.fields.items():
        node = _tree_field(child)
        if node is not _NO_NODE:
            out[key] = node
    return out


def build_settable_tree(
    root: SectionLiveField, prefix: str | None = None
) -> dict[str, object]:
    """Build the nested current-value tree for the live cfg draft.

    Without ``prefix`` returns the whole draft as a nested dict. With ``prefix``
    (a dotted path) returns the sub-tree rooted at that node — reusing the same
    ``_navigate`` walk as the flat lister's ``under`` scoping (ModuleRef ref
    navigation included). A prefix that resolves to a leaf/special node is
    wrapped so the result is always a dict; a prefix matching nothing returns
    ``{}`` (graceful, not a fast-fail).
    """
    if not prefix:
        tree = _tree_field(root)
        # ``root`` is a SectionLiveField, so _tree_field always returns a dict.
        return cast("dict[str, object]", tree)
    try:
        field, _ = _navigate(root, prefix.split("."))
    except RemoteError:
        # An unresolvable prefix yields an empty sub-tree, matching the flat
        # lister's "no match → empty" contract rather than raising.
        return {}
    node = _tree_field(field)
    if node is _NO_NODE:
        # The prefix addresses an immutable/unsupported field — no settable node.
        return {}
    if isinstance(node, dict):
        return node
    # A scalar / enum prefix node is not itself a dict; wrap it under its leaf
    # segment so the reply is always a dict (the caller indexes by name).
    leaf = prefix.split(".")[-1]
    return {leaf: node}


# ---------------------------------------------------------------------------
# Static spec-tree path discovery — no context, no values (for adapter.cfg_spec).
# The structural skeleton analogue of build_settable_tree: same nested $-keyed
# shape, but a leaf carries its *type* ($type) instead of its live value, since
# a static CfgSectionSpec has no context/tab to read a value from. Recursion is
# kept independent of the live ``_tree_field`` family — the node types differ
# (CfgNodeSpec vs LiveField) so a shared walk would just be two unioned branches.
#
# Node shapes:
#   - scalar (no choices) → {"$type": "<python type name>"}
#   - scalar (choices)    → {"$type": ..., "$choices": [...]}
#   - sweep               → {start/stop/step: {"$type":"number"},
#                            expts: {"$type":"integer"}}  (edges as $type leaves)
#   - module/waveform ref → {"$ref": {"options": [<allowed labels>]}}  — NO
#                           variant sub-tree (static can't pick a default
#                           variant) and NO ``current`` (no live chosen key)
#   - device ref          → {"$ref": {"options": []}}  (no live device list)
#   - literal             → dropped (the _NO_NODE sentinel, as in the live tree)
#   - section             → a plain sub-tree of its child nodes
# ---------------------------------------------------------------------------


def _tree_spec_sweep() -> dict[str, object]:
    """A static sweep node: four typed edges (expts integer, others number)."""
    return {
        edge: {"$type": "integer" if edge == "expts" else "number"}
        for edge in ("start", "stop", "expts", "step")
    }


def _tree_spec_field(node: CfgNodeSpec) -> object:
    """Build the static spec skeleton node for one spec node.

    Returns the node representing ``node``: a ``{$type[, $choices]}`` scalar
    leaf, a sweep sub-tree of typed edges, a ``{$ref:{options}}`` ref node (no
    variant sub-tree — see module section header), or a plain section sub-tree.
    Returns ``_NO_NODE`` for a literal (immutable) field so the parent drops it.
    """
    if isinstance(node, LiteralSpec):
        return _NO_NODE  # immutable — not a settable path (mirrors live tree)
    if isinstance(node, ScalarSpec):
        leaf: dict[str, object] = {"$type": node.type.__name__}
        if node.choices is not None:
            leaf["$choices"] = list(node.choices)
        return leaf
    if isinstance(node, SweepSpec):
        return _tree_spec_sweep()
    if isinstance(node, DeviceRefSpec):
        # No live device registry in a static walk — options is empty.
        return {"$ref": {"options": []}}
    if isinstance(node, (ModuleRefSpec, WaveformRefSpec)):
        # Only the allowed variant labels are advertised; the variant sub-tree is
        # NOT expanded (a static spec has no default variant — that is the
        # context-dependent value layer's call). No ``current`` for the same
        # reason. Read a chosen variant's fields via a live ``tab.list_paths``.
        return {"$ref": {"options": [s.label for s in node.allowed]}}
    if isinstance(node, CfgSectionSpec):
        return _tree_spec_section_children(node)
    return _NO_NODE


def _tree_spec_section_children(section: CfgSectionSpec) -> dict[str, object]:
    """Recurse a spec section's fields into a sub-tree, dropping ``_NO_NODE``."""
    out: dict[str, object] = {}
    for key, child in section.fields.items():
        node = _tree_spec_field(child)
        if node is not _NO_NODE:
            out[key] = node
    return out


def _navigate_spec(spec: CfgSectionSpec, segments: list[str]) -> CfgNodeSpec | None:
    """Walk a dotted path through a static spec to the node it addresses.

    The static analogue of ``_navigate`` (over CfgNodeSpec, no live model):
    descend ``CfgSectionSpec.fields``; for a ModuleRef/WaveformRef, a trailing
    ``ref`` (no rest) resolves to the ref node itself, while any other segment
    duck-types down its ``allowed`` shapes (returning the first match, mirroring
    ``_path_exists`` in ``adapter/types.py``). Returns ``None`` when the path
    does not resolve — the caller turns that into an empty sub-tree.
    """
    node: CfgNodeSpec = spec
    i = 0
    while i < len(segments):
        head = segments[i]
        rest = segments[i + 1 :]
        if isinstance(node, CfgSectionSpec):
            child = node.fields.get(head)
            if child is None:
                return None
            node = child
            i += 1
            continue
        if isinstance(node, (ModuleRefSpec, WaveformRefSpec)):
            if head == "ref" and not rest:
                # The key segment maps back to the ref node itself.
                return node
            # Any other segment descends into a variant shape; pick the first
            # allowed shape that contains the remaining path (duck-type descent).
            for shape in node.allowed:
                resolved = _navigate_spec(shape, segments[i:])
                if resolved is not None:
                    return resolved
            return None
        # A scalar / sweep / device ref leaf cannot be descended further.
        return None
    return node


def build_spec_tree(
    spec: CfgSectionSpec, prefix: str | None = None
) -> dict[str, object]:
    """Build the static cfg skeleton tree for an adapter spec (no live values).

    Without ``prefix`` returns the whole spec as a nested dict (the structural
    twin of ``build_settable_tree`` but with ``$type`` leaves instead of live
    values). With ``prefix`` (a dotted path) returns the sub-tree rooted at that
    node, navigating via ``_navigate_spec`` (ModuleRef/WaveformRef ``ref`` +
    duck-typed variant descent included). A prefix matching nothing — or one
    addressing a literal (no settable node) — returns ``{}`` (graceful, not a
    fast-fail), matching ``build_settable_tree``'s contract.
    """
    if not prefix:
        return _tree_spec_section_children(spec)
    node = _navigate_spec(spec, prefix.split("."))
    if node is None:
        return {}
    built = _tree_spec_field(node)
    if built is _NO_NODE:
        # The prefix addresses an immutable/unsupported field — no spec node.
        return {}
    if isinstance(built, dict):
        return built
    # A scalar / enum prefix node is not itself a dict; wrap it under its leaf
    # segment so the reply is always a dict (the caller indexes by name).
    leaf = prefix.split(".")[-1]
    return {leaf: built}


# ---------------------------------------------------------------------------
# Path navigation — walk a dotted path to the field it addresses (used by the
# ``under`` sub-tree scoping in list_settable_paths).
# ---------------------------------------------------------------------------


def _navigate(root: SectionLiveField, segments: list[str]) -> tuple[LiveField, str]:
    """Walk ``segments`` from ``root`` to the addressed field and its root path.

    Returns ``(field, base_path)`` where ``base_path`` is the dotted path that
    ``_list_field`` should prepend so the re-listed sub-tree round-trips through
    ``set_field``. For a ModuleRef, a trailing ``ref`` (the key) resolves back to
    the ref *field itself* (base path without ``.ref``), because switching the
    key rebuilds the whole ref sub-tree — that is what callers want re-listed.
    Other segments descend into the bound sub-section directly (no ``value``).
    """
    field: LiveField = root
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
