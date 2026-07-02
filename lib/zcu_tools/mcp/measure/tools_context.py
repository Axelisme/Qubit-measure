"""Measure MCP tools-context override tools."""

from __future__ import annotations

from typing import Any

from zcu_tools.mcp.measure.tool_context import (
    MeasureToolContext,
    _coerce_pairs,
    bind_context,
    send_gui_rpc,
)


def tool_gui_context_md_write(arguments: dict[str, Any]) -> dict[str, Any]:
    """Batch-write MetaDict attributes, fail-fast in order (E5).

    Batch-only fan-out over context.md_set_attr — there is no atomicity: attrs
    before the failing one stay set and are NOT rolled back. On the first error
    this raises a message carrying ``applied_count`` (how many succeeded) and
    ``failed_index`` (the 0-based position of the failing attr) so the agent can
    reconcile (this surface never returns a structured partial result — across
    bridges, a failed write always raises). Complex md scalars round-trip via the
    {"__complex__": [re, im]} tag on the value. On success returns ``{applied}``.
    """
    attrs = _coerce_pairs(arguments.get("attrs"), field="attrs", keys=("key", "value"))
    for i, attr in enumerate(attrs):
        try:
            send_gui_rpc(
                "context.md_set_attr",
                {"key": str(attr["key"]), "value": attr["value"]},
            )
        except Exception as exc:
            raise RuntimeError(
                f"batch md write failed at attrs[{i}] (key={attr['key']!r}); "
                f"applied_count={i}, failed_index={i} — attrs before it stay set "
                f"and are NOT rolled back: {exc}"
            ) from exc
    return {"applied": len(attrs)}


def tool_gui_context_md_read(arguments: dict[str, Any]) -> dict[str, Any]:
    """Read MetaDict attributes — the whole tree, or a named subset.

    Omit ``keys`` to read every attribute (fans out context.md_get for the key
    list, then context.md_get_attr for each); pass ``keys`` to read only that
    subset. Returns ``{values: {key: value}}`` — a map keyed by attribute name the
    agent indexes straight into. Reads are side-effect-free, so there is no
    partial-state concern; an unknown key in an explicit ``keys`` list fails fast
    (the underlying RPC raises invalid_params), never silently skipped. Complex md
    scalars arrive as the {"__complex__": [re, im]} tag (symmetric with write).
    """
    raw_keys = arguments.get("keys")
    if raw_keys is None:
        keys = [str(k) for k in send_gui_rpc("context.md_get", {}).get("keys", [])]
    elif isinstance(raw_keys, list):
        keys = [str(k) for k in raw_keys]
    else:
        raise ValueError("'keys' must be a list (or omitted to read the whole tree)")
    values: dict[str, Any] = {}
    for key in keys:
        res = send_gui_rpc("context.md_get_attr", {"key": key})
        values[key] = res.get("value")
    return {"values": values}


def tool_gui_context_md_delete(arguments: dict[str, Any]) -> dict[str, Any]:
    """Batch-delete MetaDict attributes, fail-fast in order.

    Batch-only fan-out over context.md_del_attr. Idempotent per key: deleting a
    key that does not exist is a no-op (not an error), matching delete semantics.
    No atomicity — keys before a (non-idempotent) failure stay deleted and are NOT
    rolled back; on the first such error this raises a message carrying
    ``applied_count`` and ``failed_index``. On success returns ``{deleted: [key]}``.
    """
    raw_keys = arguments.get("keys")
    if not isinstance(raw_keys, list) or not raw_keys:
        raise ValueError("'keys' must be a non-empty list")
    keys = [str(k) for k in raw_keys]
    for i, key in enumerate(keys):
        try:
            send_gui_rpc("context.md_del_attr", {"key": key})
        except Exception as exc:
            raise RuntimeError(
                f"batch md delete failed at keys[{i}] (key={key!r}); "
                f"applied_count={i}, failed_index={i} — keys before it stay deleted "
                f"and are NOT rolled back: {exc}"
            ) from exc
    return {"deleted": keys}


def tool_gui_context_list(arguments: dict[str, Any]) -> dict[str, Any]:
    """List context labels plus the active one (the orientation read for contexts).

    Folds context.active + context.labels into one reply
    ``{active, has_active_context, labels: [str]}``. ``labels`` are plain strings:
    per-label unit/value are NOT available — unit/value are transient creation
    metadata (consumed by the device + the auto-label), never persisted (FC2). Only
    the active context's unit could be inferred, and even that is out of scope here.
    """
    del arguments
    active = send_gui_rpc("context.active", {}).get("label")
    labels = list(send_gui_rpc("context.labels", {}).get("labels", []))
    return {
        "active": active,
        "has_active_context": active is not None,
        "labels": labels,
    }


def tool_gui_context_ml_inspect(arguments: dict[str, Any]) -> dict[str, Any]:
    """Read one ModuleLibrary entry's full cfg WITHOUT opening a tab.

    Opens a headless, gc-reclaimable cfg-editor draft on the existing ml entry
    (editor.new), reads its settable tree, then discards the draft (editor.discard).
    This is a pure read: opening/discarding a draft bumps no agent-visible resource
    version (only an *edit* would bump the editor version, and only editor.commit
    bumps context) — so it never disturbs concurrency guards. Returns ``{cfg}`` (the
    nested current-value tree, same shape as gui_tab_get_cfg). The draft is always
    discarded, even if the read raises.
    """
    item_kind = str(arguments["item_kind"])
    name = str(arguments["name"])
    opened = send_gui_rpc("editor.new", {"item_kind": item_kind, "from_name": name})
    editor_id = opened["editor_id"]
    try:
        return {"cfg": opened.get("tree")}
    finally:
        send_gui_rpc("editor.discard", {"editor_id": editor_id})


NON_GENERATED_METHODS = frozenset(
    {
        "context.active",
        "context.labels",
        "context.md_get",
        "context.md_get_attr",
        "context.md_set_attr",
        "context.md_del_attr",
    }
)


OVERRIDE_TOOLS: dict[str, dict[str, Any]] = {
    "gui_context_list": {
        "handler": tool_gui_context_list,
        "description": (
            "List the context labels plus which one is active — the orientation "
            "read for contexts. Returns {active: str|null, has_active_context: bool, "
            "labels: [str]}. 'labels' are plain strings: per-label unit/value are "
            "NOT available (unit/value are transient creation metadata, never "
            "persisted). Switch with gui_context_switch; create with "
            "gui_context_create."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    "gui_context_md_read": {
        "handler": tool_gui_context_md_read,
        "description": (
            "Read MetaDict attributes — omit 'keys' to read the WHOLE tree, or pass "
            "'keys' to read only that subset. Returns {values: {key: value}} keyed "
            "by attribute name. An unknown key in an explicit 'keys' list fails fast "
            "(invalid_params) — keys are never silently skipped. Reads are "
            'side-effect-free. Complex scalars arrive as {"__complex__": [re, im]} '
            "(symmetric with gui_context_md_write)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "keys": {
                    "type": "array",
                    "description": "MetaDict keys to read; omit to read every attribute",
                    "items": {"type": "string"},
                },
            },
        },
    },
    "gui_context_md_write": {
        "handler": tool_gui_context_md_write,
        "description": (
            "Batch-write MetaDict attributes in order. NOT atomic: stops at the "
            "first failure (fail-fast), attrs set before it are NOT rolled back, and "
            "the error message carries applied_count + failed_index. Returns "
            "{applied} on success. Complex scalars round-trip via "
            '{"__complex__": [re, im]} on the value.'
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "attrs": {
                    "type": "array",
                    "description": "Attributes set in order; each {key, value}",
                    "items": {
                        "type": "object",
                        "properties": {
                            "key": {"type": "string", "description": "MetaDict key"},
                            "value": {"description": "JSON-safe value"},
                        },
                        "required": ["key", "value"],
                    },
                },
            },
            "required": ["attrs"],
        },
    },
    "gui_context_md_delete": {
        "handler": tool_gui_context_md_delete,
        "description": (
            "Batch-delete MetaDict attributes by key. Idempotent per key: deleting a "
            "missing key is a no-op (not an error). NOT atomic across keys: on a "
            "(non-idempotent) failure the error carries applied_count + failed_index "
            "and keys before it stay deleted. Returns {deleted: [key]} on success."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "keys": {
                    "type": "array",
                    "description": "MetaDict keys to delete",
                    "items": {"type": "string"},
                },
            },
            "required": ["keys"],
        },
    },
    "gui_context_ml_inspect": {
        "handler": tool_gui_context_ml_inspect,
        "description": (
            "Read one ModuleLibrary entry's full cfg WITHOUT opening a tab or a "
            "lasting editing session. Returns {cfg} — the nested current-value tree "
            "(same shape as gui_tab_get_cfg). A pure read: it opens a headless draft "
            "on the entry and discards it, bumping no agent-visible resource "
            "version. Use gui_context_ml_list first to find names + kinds."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "item_kind": {
                    "type": "string",
                    "description": "'module' or 'waveform'",
                },
                "name": {"type": "string", "description": "ml entry name"},
            },
            "required": ["item_kind", "name"],
        },
    },
}


def build_override_tools(ctx: MeasureToolContext) -> dict[str, dict[str, Any]]:
    bind_context(ctx)
    return OVERRIDE_TOOLS
