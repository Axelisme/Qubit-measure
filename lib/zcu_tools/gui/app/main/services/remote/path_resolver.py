"""Dotted-path resolver for remote ``cfg.set_field``.

Walks a ``CfgDraft`` field tree and mutates a single leaf using the
field's existing public mutation surface, so a remote edit behaves exactly
like a user edit (auto-commit via the form's ``on_change`` -> ``schema_changed``
-> ``Controller.update_tab_cfg`` chain).

Path grammar (segments split on ``.``):

  - Scalar leaf:        ``section.sub.field``           -> ``ScalarField.set_value(value)``
  - Sweep sub-field:    ``...path.sweep.start|stop|expts|step``
  - ModuleRef key:      ``...path.ref``                  -> ``set_chosen_key(value)``
  - ModuleRef sub:      ``...path.<sub>...``             -> recurse into ``.sub_field``
  - Literal:            rejected (immutable)

ModuleRef sub-fields descend directly (no ``value`` wrapper segment); a path
that still carries a ``value`` segment is rejected with a migration message.

Unknown paths, type mismatches, and immutable targets raise
``RemoteError(INVALID_PARAMS, ...)`` with a discriminating message.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from zcu_tools.gui.app.main.adapter import (
    DirectValue,
    EvalValue,
)
from zcu_tools.gui.cfg.binding import (
    CenteredSweepEditor,
    CenteredSweepField,
    CfgDraft,
    CfgField,
    LiteralField,
    ReferenceField,
    ScalarField,
    SectionField,
    SweepEditor,
    SweepField,
)
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError
from zcu_tools.gui.session.value_lookup import ValueRef

_SWEEP_EDGES = {"start", "stop", "expts", "step"}
_CENTERED_SWEEP_EDGES = {"center", "span", "expts", "step"}
_MAX_UNKNOWN_FIELD_SUGGESTIONS = 3


ValueRefResolver = Callable[[ValueRef, type], DirectValue]


def resolve_and_set(
    draft: CfgDraft,
    path: str,
    value: object,
    *,
    resolve_value_ref: ValueRefResolver,
) -> None:
    """Resolve ``path`` against ``root`` and set the leaf to ``value``.

    Mutations propagate through the draft's existing ``on_change`` bubbling.
    """
    if not path:
        raise RemoteError(ErrorCode.INVALID_PARAMS, "empty path")
    segments = path.split(".")
    _set_recursive(draft.root, segments, path, value, resolve_value_ref)


def _set_recursive(
    field: CfgField,
    segments: list[str],
    full_path: str,
    value: object,
    resolve_value_ref: ValueRefResolver,
) -> None:
    head = segments[0]
    rest = segments[1:]

    if isinstance(field, SectionField):
        child = field.fields.get(head)
        if child is None:
            raise RemoteError(
                ErrorCode.INVALID_PARAMS,
                _unknown_field_message(field, head, segments, full_path),
            )
        if not rest:
            _set_leaf(child, full_path, value, resolve_value_ref)
            return
        _set_recursive(child, rest, full_path, value, resolve_value_ref)
        return

    # Reached a non-section field but still have segments to consume.
    if isinstance(field, SweepField):
        _set_sweep_edge(field, head, rest, full_path, value)
        return

    if isinstance(field, CenteredSweepField):
        _set_centered_sweep_edge(field, head, rest, full_path, value)
        return

    if isinstance(field, ReferenceField):
        _set_moduleref(field, head, rest, full_path, value, resolve_value_ref)
        return

    raise RemoteError(
        ErrorCode.INVALID_PARAMS,
        f"path {full_path!r} descends into non-container {type(field).__name__}",
    )


def _unknown_field_message(
    section: SectionField, head: str, segments: list[str], full_path: str
) -> str:
    msg = f"unknown field {head!r} in path {full_path!r}"
    suggestions = _same_subtree_leaf_suggestions(section, head, segments, full_path)
    if suggestions:
        paths = ", ".join(repr(path) for path in suggestions)
        msg = f"{msg}; did you mean {paths}?"
    return msg


def _same_subtree_leaf_suggestions(
    section: SectionField, leaf_name: str, segments: list[str], full_path: str
) -> list[str]:
    """Suggest legal descendant paths when a skipped section is unambiguous."""
    full_segments = full_path.split(".")
    prefix_len = max(len(full_segments) - len(segments), 0)
    prefix = ".".join(full_segments[:prefix_len])
    candidates: list[str] = []
    for entry in _list_field(prefix, section):
        path = str(entry["path"])
        if path.rsplit(".", 1)[-1] == leaf_name:
            candidates.append(path)
    unique = sorted(set(candidates))
    if len(unique) > _MAX_UNKNOWN_FIELD_SUGGESTIONS:
        return []
    return unique


def _set_leaf(
    field: CfgField,
    full_path: str,
    value: object,
    resolve_value_ref: ValueRefResolver,
) -> None:
    if isinstance(field, LiteralField):
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            f"path {full_path!r} targets an immutable literal field",
        )
    if isinstance(field, ScalarField):
        if isinstance(value, ValueRef):
            value = resolve_value_ref(value, field.spec.type)
        elif not isinstance(
            value, (DirectValue, EvalValue)
        ) and not _matches_scalar_type(value, field.spec.type):
            raise RemoteError(
                ErrorCode.INVALID_PARAMS,
                f"scalar at {full_path!r} expects {field.spec.type.__name__}",
            )
        field.set_value(value)
        return
    if isinstance(field, ReferenceField):
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            f"path {full_path!r} targets a module ref; set "
            f"'{full_path}.ref' (key) or '{full_path}.<sub>' instead",
        )
    if isinstance(field, SweepField):
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            f"path {full_path!r} targets a sweep; set "
            f"'{full_path}.start|stop|expts|step' instead",
        )
    if isinstance(field, CenteredSweepField):
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            f"path {full_path!r} targets a centered sweep; set "
            f"'{full_path}.center|span|expts|step' instead",
        )
    if isinstance(field, SectionField):
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            f"path {full_path!r} targets a section; descend to a leaf field",
        )
    raise RemoteError(
        ErrorCode.INVALID_PARAMS,
        f"path {full_path!r} targets unsupported field {type(field).__name__}",
    )


def _matches_scalar_type(value: object, type_: type) -> bool:
    if value is None:
        return True
    if type_ is bool:
        return isinstance(value, bool)
    if type_ is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if type_ is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return isinstance(value, type_)


def _set_sweep_edge(
    sweep: SweepField,
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
        try:
            updated = SweepEditor.update_expts(current, value)
            sweep.set_value(updated)
        except ValueError as exc:
            raise RemoteError(ErrorCode.INVALID_PARAMS, str(exc)) from exc
        return
    if edge == "step":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RemoteError(ErrorCode.INVALID_PARAMS, "sweep 'step' must be a number")
        try:
            updated = SweepEditor.update_step(current, float(value))
            sweep.set_value(updated)
        except ValueError as exc:
            raise RemoteError(ErrorCode.INVALID_PARAMS, str(exc)) from exc
        return
    # start / stop accept a number (EvalValue editing is out of scope here).
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"sweep '{edge}' must be a number")
    try:
        updated = (
            SweepEditor.update_start(current, float(value))
            if edge == "start"
            else SweepEditor.update_stop(current, float(value))
        )
        sweep.set_value(updated)
    except ValueError as exc:
        raise RemoteError(ErrorCode.INVALID_PARAMS, str(exc)) from exc


def _set_centered_sweep_edge(
    sweep: CenteredSweepField,
    edge: str,
    rest: list[str],
    full_path: str,
    value: object,
) -> None:
    if edge == "sweep" and rest:
        edge = rest[0]
        rest = rest[1:]
    if rest:
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            f"path {full_path!r} has trailing segments after centered sweep edge",
        )
    if edge not in _CENTERED_SWEEP_EDGES:
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            "centered sweep edge must be one of "
            f"{sorted(_CENTERED_SWEEP_EDGES)}, got {edge!r}",
        )
    current = sweep.get_value()
    if edge == "expts":
        if not isinstance(value, int) or isinstance(value, bool):
            raise RemoteError(
                ErrorCode.INVALID_PARAMS,
                "centered sweep 'expts' must be an integer",
            )
        try:
            updated = CenteredSweepEditor.update_expts(current, value)
            sweep.set_value(updated)
        except ValueError as exc:
            raise RemoteError(ErrorCode.INVALID_PARAMS, str(exc)) from exc
        return
    if edge == "step":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RemoteError(
                ErrorCode.INVALID_PARAMS,
                "centered sweep 'step' must be a number",
            )
        try:
            updated = CenteredSweepEditor.update_step(current, float(value))
            sweep.set_value(updated)
        except ValueError as exc:
            raise RemoteError(ErrorCode.INVALID_PARAMS, str(exc)) from exc
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RemoteError(
            ErrorCode.INVALID_PARAMS, f"centered sweep '{edge}' must be a number"
        )
    try:
        updated = (
            CenteredSweepEditor.update_center(current, float(value))
            if edge == "center"
            else CenteredSweepEditor.update_span(current, float(value))
        )
        sweep.set_value(updated)
    except ValueError as exc:
        raise RemoteError(ErrorCode.INVALID_PARAMS, str(exc)) from exc


def _set_moduleref(
    ref: ReferenceField,
    head: str,
    rest: list[str],
    full_path: str,
    value: object,
    resolve_value_ref: ValueRefResolver,
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
    _set_recursive(sub, [head, *rest], full_path, value, resolve_value_ref)


# ---------------------------------------------------------------------------
# Path discovery — the inverse of resolve_and_set
# ---------------------------------------------------------------------------


def _scalar_wire_value(field: ScalarField) -> object:
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


def _scalar_entry(path: str, field: ScalarField) -> dict[str, object]:
    entry: dict[str, object] = {
        "path": path,
        "kind": "scalar",
        "value": _scalar_wire_value(field),
        "type": field.spec.type.__name__,
    }
    options = field.available_options()
    if options is not None:
        entry["choices"] = list(options)
    return entry


def _sweep_entries(path: str, field: SweepField) -> list[dict[str, object]]:
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


def _centered_sweep_entries(
    path: str, field: CenteredSweepField
) -> list[dict[str, object]]:
    sweep = field.get_value()
    edges = {
        "center": sweep.center,
        "span": sweep.span,
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


def _list_field(path: str, field: CfgField) -> list[dict[str, object]]:
    """Recurse one CfgField, returning the settable leaves beneath it."""
    if isinstance(field, LiteralField):
        return []  # immutable — resolve_and_set rejects it
    if isinstance(field, ScalarField):
        return [_scalar_entry(path, field)]
    if isinstance(field, SweepField):
        return _sweep_entries(path, field)
    if isinstance(field, CenteredSweepField):
        return _centered_sweep_entries(path, field)
    if isinstance(field, ReferenceField):
        out = [
            {
                "path": f"{path}.ref",
                "kind": "moduleref_key",
                "value": field.get_chosen_key(),
                "type": "string",
                "choices": [
                    *(spec.label for spec in field.spec.allowed),
                    *field.available_keys(),
                ],
            }
        ]
        sub = field.sub_field
        if sub is not None:
            for key, child in sub.fields.items():
                out.extend(_list_field(f"{path}.{key}", child))
        return out
    if isinstance(field, SectionField):
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
    draft: CfgDraft,
    under: str | None = None,
    prefix: str | None = None,
) -> list[dict[str, object]]:
    """Enumerate the dotted paths that ``resolve_and_set`` can mutate.

    DIFF-ONLY / INTERNAL: this flat lister is NOT an agent-facing view (agents
    read the nested ``build_settable_tree`` via editor.get / tab.get_cfg). Its
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
        field, base_path = _navigate(draft.root, under.split("."))
        entries = _list_field(base_path, field)
    else:
        entries = _list_field("", draft.root)
    if prefix:
        entries = _filter_by_prefix(entries, prefix)
    return entries


def list_settable_paths_full(
    draft: CfgDraft, under: str | None = None
) -> list[dict[str, object]]:
    """``list_settable_paths`` typed as the dict-entry list.

    Internal callers (diffing, sub-tree re-list) want a non-union return type;
    this thin wrapper preserves that call-site stability.
    """
    return list_settable_paths(draft, under=under)


# ---------------------------------------------------------------------------
# Nested-tree path discovery — a value tree shaped like the cfg, where each
# leaf carries its current live value. The structural inverse of the flat
# ``list_settable_paths`` (which stays for set_field's before/after diff only):
# this one is the read-only view tab.get_cfg returns.
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


def _tree_sweep(field: SweepField) -> dict[str, object]:
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


def _tree_centered_sweep(field: CenteredSweepField) -> dict[str, object]:
    """A centered sweep node as bare scalar edges (center/span/expts/step)."""
    sweep = field.get_value()
    edges = {
        "center": sweep.center,
        "span": sweep.span,
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


def _tree_field(field: CfgField) -> object:
    """Build the nested value tree for one CfgField (see module section header).

    Returns the node that represents ``field``: a bare scalar value, an enum
    ``{$value, $choices}`` leaf, a sweep sub-tree, a ``$ref`` node (with the
    chosen variant's sub-tree merged in), or a plain section sub-tree. Returns
    the ``_NO_NODE`` sentinel for an immutable/unsupported field so the parent
    drops it.
    """
    if isinstance(field, LiteralField):
        return _NO_NODE  # immutable — not a settable path (flat lister drops it)
    if isinstance(field, ScalarField):
        value = _scalar_wire_value(field)
        options = field.available_options()
        if options is not None:
            return {"$value": value, "$choices": list(options)}
        return value
    if isinstance(field, SweepField):
        return _tree_sweep(field)
    if isinstance(field, CenteredSweepField):
        return _tree_centered_sweep(field)
    if isinstance(field, ReferenceField):
        # Only the currently-bound variant is expanded; ``options`` lists the
        # allowed variant labels (bare), while ``current`` is the chosen key (a
        # built-in variant reads as the tagged ``<Custom:label>`` form, mirroring
        # the flat lister's value/choices split).
        node: dict[str, object] = {
            "$ref": {
                "current": field.get_chosen_key(),
                "options": [
                    *(spec.label for spec in field.spec.allowed),
                    *field.available_keys(),
                ],
            }
        }
        sub = field.sub_field
        if sub is not None:
            node.update(_tree_section_children(sub))
        return node
    if isinstance(field, SectionField):
        return _tree_section_children(field)
    return _NO_NODE


def _tree_section_children(section: SectionField) -> dict[str, object]:
    """Recurse a section's fields into a sub-tree, dropping ``_NO_NODE`` ones."""
    out: dict[str, object] = {}
    for key, child in section.fields.items():
        node = _tree_field(child)
        if node is not _NO_NODE:
            out[key] = node
    return out


def build_settable_tree(
    draft: CfgDraft, prefix: str | None = None
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
        tree = _tree_field(draft.root)
        # ``root`` is a SectionField, so _tree_field always returns a dict.
        return cast("dict[str, object]", tree)
    try:
        field, _ = _navigate(draft.root, prefix.split("."))
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
# Path navigation — walk a dotted path to the field it addresses (used by the
# ``under`` sub-tree scoping in list_settable_paths).
# ---------------------------------------------------------------------------


def _navigate(root: SectionField, segments: list[str]) -> tuple[CfgField, str]:
    """Walk ``segments`` from ``root`` to the addressed field and its root path.

    Returns ``(field, base_path)`` where ``base_path`` is the dotted path that
    ``_list_field`` should prepend so the re-listed sub-tree round-trips through
    ``set_field``. For a ModuleRef, a trailing ``ref`` (the key) resolves back to
    the ref *field itself* (base path without ``.ref``), because switching the
    key rebuilds the whole ref sub-tree — that is what callers want re-listed.
    Other segments descend into the bound sub-section directly (no ``value``).
    """
    field: CfgField = root
    consumed: list[str] = []
    i = 0
    while i < len(segments):
        head = segments[i]
        if isinstance(field, SectionField):
            child = field.fields.get(head)
            if child is None:
                raise RemoteError(
                    ErrorCode.INVALID_PARAMS,
                    _unknown_field_message(
                        field, head, segments[i:], ".".join(segments)
                    ),
                )
            field = child
            consumed.append(head)
            i += 1
            continue
        if isinstance(field, ReferenceField):
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
