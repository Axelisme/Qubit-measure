"""Measure MCP tools-cfg override tools."""

from __future__ import annotations

from typing import Any

from zcu_tools.mcp.measure.tool_context import (
    MeasureToolContext,
    _coerce_pairs,
    bind_context,
    send_gui_rpc,
)


def _resolve_editor_id(arguments: dict[str, Any]) -> str:
    """Return the ``editor_id`` from arguments (required, fail-fast).

    Tab cfg editing goes through gui_tab_set_cfg / gui_tab_get_cfg. The editor
    tools (gui_editor_get_cfg / gui_editor_set) operate on non-tab editors
    (e.g. gui_editor_open on an ml entry) and require an explicit editor_id —
    tab_id is not accepted here.
    """
    editor_id = arguments.get("editor_id")
    if editor_id is None:
        raise ValueError(
            "supply 'editor_id' (tab cfg editing uses gui_tab_set_cfg / "
            "gui_tab_get_cfg; editor tools require an explicit editor_id)"
        )
    return str(editor_id)


def _fold_tab_editing_context(tab_id: str, reply: dict[str, Any]) -> dict[str, Any]:
    """Fold a fresh tab's editing context into ``reply``, in place.

    After tab.new the agent always reads tab.snapshot (for the editor_id) and
    tab.get_cfg (the settable cfg tree) before it can edit cfg. Folding those
    reads collapses the calls into one. Pure mcp-side fan-out over EXISTING wire
    reads. Reused by gui_tab_open (step 1). Adds {editor_id, tree}; the
    caller owns ``tab_id`` and ``adapter`` in ``reply``. tab.get_cfg returns a
    nested current-value tree, so the settable paths and their current values
    arrive in one ``tree``.
    """
    # tab.snapshot always returns {tabs: [...]}; a single tab_id yields a
    # one-element list (no shape-switch).
    snap = send_gui_rpc("tab.snapshot", {"tab_id": tab_id})["tabs"][0]
    reply["editor_id"] = snap.get("editor_id")
    reply["tree"] = send_gui_rpc("tab.get_cfg", {"tab_id": tab_id}).get("tree")
    return reply


def tool_gui_editor_open(arguments: dict[str, Any]) -> dict[str, Any]:
    """Open a stateful editing session over an EXISTING ml entry, addressed later
    by ``editor_id``.

    Thin override over the editor.new RPC: folds the wire ``tree`` key to ``cfg``
    so every cfg view the agent reads (gui_tab_get_cfg, gui_editor_get_cfg, this
    open reply) uses the same ``cfg`` key. Returns ``{editor_id, cfg}`` — cfg is
    the nested current-value tree of the freshly-opened draft.
    """
    opened = send_gui_rpc(
        "editor.new",
        {
            "item_kind": str(arguments["item_kind"]),
            "from_name": str(arguments["from_name"]),
        },
    )
    return {"editor_id": opened["editor_id"], "cfg": opened.get("tree")}


def tool_gui_editor_get_cfg(arguments: dict[str, Any]) -> dict[str, Any]:
    """Read an editing session's settable cfg as a nested current-value tree.

    Thin override over the editor.get RPC: folds the wire ``tree`` key to ``cfg``
    (the same key gui_tab_get_cfg uses). Returns ``{cfg}``.
    """
    editor_id = _resolve_editor_id(arguments)
    params: dict[str, Any] = {"editor_id": editor_id}
    prefix = arguments.get("prefix")
    if prefix is not None:
        params["prefix"] = str(prefix)
    got = send_gui_rpc("editor.get", params)
    return {"cfg": got.get("tree")}


def tool_gui_editor_set(arguments: dict[str, Any]) -> dict[str, Any]:
    """Batch-set fields on ONE cfg-editor session, fail-fast in order.

    The editor is addressed by ``editor_id`` (from gui_editor_open). For tab cfg
    editing use gui_tab_set_cfg instead. Batch-only fan-out (a for-loop over the
    single-field editor.set_field RPC) — there is no atomicity: edits before the
    failing one stay applied and are NOT rolled back. On the first error this
    raises, reporting how many succeeded and which path failed so the agent can
    reconcile. On success returns ``{applied, valid}`` — the count applied and
    whether the resulting draft is valid. It does NOT echo cfg content (that would
    force a lowering pass which eagerly evaluates EvalValue); read the cfg with
    gui_editor_get_cfg if needed.
    """
    editor_id = _resolve_editor_id(arguments)
    edits = _coerce_pairs(arguments.get("edits"), field="edits", keys=("path", "value"))
    valid = True
    for i, edit in enumerate(edits):
        try:
            res = send_gui_rpc(
                "editor.set_field",
                {
                    "editor_id": editor_id,
                    "path": str(edit["path"]),
                    "value": edit["value"],
                },
            )
        except Exception as exc:
            raise RuntimeError(
                f"batch set failed at edits[{i}] (path={edit['path']!r}); "
                f"{i} edit(s) already applied and NOT rolled back: {exc}"
            ) from exc
        valid = bool(res.get("valid", True))
    return {"applied": len(edits), "valid": valid}


def tool_gui_tab_set_cfg(arguments: dict[str, Any]) -> dict[str, Any]:
    return send_gui_rpc(
        "tab.set_cfg",
        {
            "tab_id": str(arguments["tab_id"]),
            "edits": _coerce_pairs(
                arguments.get("edits"), field="edits", keys=("path", "value")
            ),
        },
    )


NON_GENERATED_METHODS = frozenset(
    {
        "editor.new",
        "editor.get",
        "editor.set_field",
        "tab.set_cfg",
    }
)


OVERRIDE_TOOLS: dict[str, dict[str, Any]] = {
    "gui_editor_open": {
        "handler": tool_gui_editor_open,
        "description": (
            "Open a stateful editing session over an EXISTING ModuleLibrary "
            "module/waveform (by 'from_name'). To create a new blank/shaped entry, "
            "use gui_context_ml_create_from_role then gui_editor_open(from_name=...) "
            "to edit it. item_kind is 'module' or 'waveform'. Returns "
            "{editor_id, cfg} — cfg is the nested current-value tree (same shape "
            "as gui_editor_get_cfg / gui_tab_get_cfg). Address later edits with the "
            "returned editor_id via gui_editor_set; persist with gui_editor_save."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "item_kind": {
                    "type": "string",
                    "description": "'module' or 'waveform'",
                },
                "from_name": {
                    "type": "string",
                    "description": "Existing ml entry name to load for editing",
                },
            },
            "required": ["item_kind", "from_name"],
        },
    },
    "gui_editor_get_cfg": {
        "handler": tool_gui_editor_get_cfg,
        "description": (
            "Read an editing session's settable cfg as a NESTED tree of current "
            "values, addressed by 'editor_id' (from gui_editor_open). Returns "
            "{cfg} — the same tree shape and '$'-prefixed leaf encoding as "
            "gui_tab_get_cfg (SCALAR / ENUM '$value'+'$choices' / SWEEP edges / "
            "REF '$ref'). Edit a leaf with gui_editor_set using its dotted path. "
            "'prefix' (optional, dotted) returns just the sub-tree rooted there; "
            "a prefix matching nothing returns {}."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "editor_id": {
                    "type": "string",
                    "description": "Editor session id (from gui_editor_open)",
                },
                "prefix": {
                    "type": "string",
                    "description": (
                        "Return only the sub-tree rooted at this dotted path "
                        "(e.g. 'modules.readout'); omit for the whole draft. "
                        "No match → {}"
                    ),
                },
            },
            "required": ["editor_id"],
        },
    },
    "gui_editor_set": {
        "handler": tool_gui_editor_set,
        "description": (
            "Batch-set fields on ONE cfg-editor session in order (fail-fast, "
            "non-atomic), addressed by 'editor_id' (from gui_editor_open). For tab "
            "cfg editing use gui_tab_set_cfg instead. 'edits' is an ORDERED list of "
            "{path, value}: 'path' is dotted (see gui_editor_get_cfg); 'value' is a "
            "JSON scalar, an md-ref {__kind:eval, expr}, or a resolve-once value "
            "source {__kind:value_ref, key, type?} (eval/value_ref forms are "
            "accepted only on a scalar leaf, never a sweep edge). Discover value "
            "source keys with gui_value_list / gui_value_read. Apply ref-switch edits "
            "before dependent inner-path edits (a ref switch removes child paths). "
            "Stops at the first failure and edits applied before it are NOT rolled "
            "back; the error names the failing path and how many already applied. On "
            "success returns {applied, valid} — the count applied and whether the "
            "resulting draft is valid. It does NOT echo cfg content (reading it "
            "would force a lowering pass that eagerly evaluates EvalValue); read the "
            "cfg with gui_editor_get_cfg if needed."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "editor_id": {
                    "type": "string",
                    "description": "Editor session id (from gui_editor_open)",
                },
                "edits": {
                    "type": "array",
                    "description": "Edits applied in order; each {path, value}",
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Dotted field path",
                            },
                            "value": {
                                # Untyped schema = "any JSON value", matching the
                                # generator's JsonType.JSON rendering. A 'type' union
                                # listing "string" would let the client coerce a
                                # number (0.2) to "0.2" and fail the float check.
                                "description": (
                                    "JSON scalar, {__kind:eval, expr}, or "
                                    "{__kind:value_ref, key, type?}"
                                )
                            },
                        },
                        "required": ["path", "value"],
                    },
                },
            },
            "required": ["editor_id", "edits"],
        },
    },
    "gui_tab_set_cfg": {
        "handler": tool_gui_tab_set_cfg,
        "description": (
            "Batch-set cfg fields on a tab in order (non-atomic batch). Apply "
            "ref-switch edits BEFORE dependent inner-path edits — a ref switch "
            "removes child paths and a stale inner-path edit after it will fail. "
            "On the first failing edit the call RAISES (same contract as "
            "gui_context_md_write): edits applied before it stay applied and are "
            "NOT rolled back. On success returns {valid, removed, added} "
            "aggregated across the batch. A running tab is rejected (cancel the "
            "run first). Each value is a JSON scalar, {__kind:eval, expr}, or "
            "{__kind:value_ref, key, type?}; value_ref resolves immediately and "
            "source keys come from gui_value_list / gui_value_read. Read the current "
            "tree with gui_tab_get_cfg."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "tab_id": {
                    "type": "string",
                    "description": "Tab to edit",
                },
                "edits": {
                    "type": "array",
                    "description": "Ordered list of edits; each {path, value}",
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Dotted field path (from gui_tab_get_cfg)",
                            },
                            "value": {
                                # Untyped: numbers must not be coerced to strings.
                                "description": (
                                    "JSON scalar, {__kind:eval, expr}, or "
                                    "{__kind:value_ref, key, type?}"
                                ),
                            },
                        },
                        "required": ["path", "value"],
                    },
                },
            },
            "required": ["tab_id", "edits"],
        },
    },
}


def build_override_tools(ctx: MeasureToolContext) -> dict[str, dict[str, Any]]:
    bind_context(ctx)
    return OVERRIDE_TOOLS
