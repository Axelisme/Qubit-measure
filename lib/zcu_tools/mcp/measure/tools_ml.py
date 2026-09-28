"""Fixed ModuleLibrary tool contracts over the GUI-owned context/editor."""

from __future__ import annotations

from functools import partial
from typing import Any

from zcu_tools.mcp.measure.session import GuiRpcError
from zcu_tools.mcp.measure.tool_context import MeasureToolContext

_NAME = {"type": "string", "minLength": 1}
_KIND = {"type": "string", "enum": ["module", "waveform"]}
_EDITS = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {"path": _NAME, "value": {}},
        "required": ["path", "value"],
    },
}


def ml_get(ctx: MeasureToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    """Read the library index or one stored cfg without changing the library."""
    return ctx.send_gui_rpc("context.ml_get", arguments)


def ml_roles(
    ctx: MeasureToolContext, arguments: dict[str, Any]
) -> list[dict[str, Any]]:
    """List live GUI role templates for ModuleLibrary creation."""
    del arguments
    roles = ctx.send_gui_rpc("context.ml_list_roles", {})["roles"]
    return [
        {
            "role_id": role["role_id"],
            "label": role["label"],
            "kind": role["item_kind"],
            "default_name": role["default_name"],
        }
        for role in roles
    ]


def ml_create(ctx: MeasureToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    """Create from a role's md-backed defaults and return the stored cfg."""
    role_id = arguments["role_id"]
    roles = ml_roles(ctx, {})
    role = next((entry for entry in roles if entry["role_id"] == role_id), None)
    if role is None:
        raise ValueError(
            f"unknown role_id {role_id!r}; available: "
            f"{[entry['role_id'] for entry in roles]}"
        )
    name = arguments.get("name", role["default_name"])
    if not isinstance(name, str) or not name:
        raise ValueError(f"role {role_id!r} has no default name; supply name")
    ctx.send_gui_rpc("context.ml_create_from_role", {"role_id": role_id, "name": name})
    return ml_get(ctx, {"name": name, "kind": role["kind"]})


def _entry_kind(index: dict[str, Any], name: str, requested: Any) -> str:
    matches = [
        kind
        for kind, key in (("module", "modules"), ("waveform", "waveforms"))
        if any(entry["name"] == name for entry in index[key])
    ]
    if not matches:
        raise ValueError(
            f"unknown library name {name!r}; available modules: "
            f"{[entry['name'] for entry in index['modules']]}, waveforms: "
            f"{[entry['name'] for entry in index['waveforms']]}"
        )
    if requested is None and len(matches) > 1:
        raise ValueError(
            f"ambiguous library name {name!r}; supply kind='module' or 'waveform'"
        )
    if requested is not None and requested not in matches:
        raise ValueError(f"no {requested!r} named {name!r}; available kinds: {matches}")
    return requested or matches[0]


def ml_edit(ctx: MeasureToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    """Edit a disposable draft; commit only after every edit succeeds."""
    name = arguments["name"]
    index = ml_get(ctx, {})
    kind = _entry_kind(index, name, arguments.get("kind"))
    destination = arguments.get("save_as", name)
    if "save_as" in arguments:
        collection = index["modules" if kind == "module" else "waveforms"]
        if any(entry["name"] == destination for entry in collection):
            raise ValueError(
                f"save_as name {destination!r} already exists as {kind}; "
                "choose a new name"
            )
    # editor.commit depends on a full context observation. A summary index is
    # not a guard baseline; the snapshot also refuses incomplete context data.
    ctx.session.read_internal("context.snapshot", {})
    editor_id = ctx.send_gui_rpc("editor.new", {"item_kind": kind, "from_name": name})[
        "editor_id"
    ]
    commit_attempted = False
    try:
        result = ctx.send_gui_rpc(
            "editor.set_fields", {"editor_id": editor_id, "edits": arguments["edits"]}
        )
        if not result["valid"]:
            raise ValueError("edited library draft is invalid; no changes committed")
        # The batch bumps the draft version; a full read reveals that version
        # for the guarded commit without weakening the GUI's stale check.
        ctx.session.read_internal("editor.get", {"editor_id": editor_id})
        commit_attempted = True
        ctx.send_gui_rpc("editor.commit", {"editor_id": editor_id, "name": destination})
    except Exception as error:
        # A transport failure at commit is ambiguous: it may have written the
        # entry. Do not issue another mutating RPC in that state.
        if (
            commit_attempted
            and isinstance(error, GuiRpcError)
            and error.reason in ("gui_transport_timeout", "gui_handler_timeout")
        ):
            raise
        try:
            ctx.send_gui_rpc("editor.discard", {"editor_id": editor_id})
        except Exception as cleanup_error:
            raise GuiRpcError(
                f"library edit failed: {error}; discarding draft "
                f"{editor_id!r} also failed: {cleanup_error}",
                reason="cleanup_failed",
            ) from cleanup_error
        raise
    return {
        "name": destination,
        "cfg": ml_get(ctx, {"name": destination, "kind": kind})["cfg"],
    }


def ml_rename(ctx: MeasureToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    """Rename only the library entry; leave linked reference names unchanged."""
    name = arguments["name"]
    kind = _entry_kind(ml_get(ctx, {}), name, arguments.get("kind"))
    result = ctx.send_gui_rpc(
        f"context.ml_rename_{kind}", {"old": name, "new": arguments["new_name"]}
    )
    return {
        **result,
        "warning": "LINKED references still store the old name, are not converted "
        "to inline values, and may become invalid; MODIFIED inline values are preserved.",
    }


def ml_delete(ctx: MeasureToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    """Delete only the library entry; preserve modified inline reference values."""
    name = arguments["name"]
    kind = _entry_kind(ml_get(ctx, {}), name, arguments.get("kind"))
    result = ctx.send_gui_rpc(f"context.ml_del_{kind}", {"name": name})
    return {
        **result,
        "warning": "LINKED references still store the deleted name, are not converted "
        "to inline values, and may become invalid; MODIFIED inline values are preserved.",
    }


ML_TOOLS: dict[str, dict[str, Any]] = {
    "ml_rename": {
        "handler": ml_rename,
        "description": "Rename a library entry and return {renamed, warning}. kind disambiguates module/waveform names. Name clashes fail. LINKED references keep the old name, are not converted to inline, and may become invalid; MODIFIED inline values are preserved. Read context.snapshot explicitly before mutation.",
        "inputSchema": {
            "type": "object",
            "properties": {"name": _NAME, "new_name": _NAME, "kind": _KIND},
            "required": ["name", "new_name"],
        },
    },
    "ml_delete": {
        "handler": ml_delete,
        "description": "Delete a library entry and return {deleted, warning}. kind disambiguates module/waveform names. LINKED references keep the deleted name, are not converted to inline, and may become invalid; MODIFIED inline values are preserved. Read context.snapshot explicitly before mutation.",
        "inputSchema": {
            "type": "object",
            "properties": {"name": _NAME, "kind": _KIND},
            "required": ["name"],
        },
    },
    "ml_get": {
        "handler": ml_get,
        "description": "Without name list modules/waveforms by name, kind/style and description. With name return {name, kind, cfg}; provide kind=module|waveform if both collections contain the name. A read does not open an editing draft.",
        "inputSchema": {
            "type": "object",
            "properties": {"name": _NAME, "kind": _KIND},
        },
    },
    "ml_roles": {
        "handler": ml_roles,
        "description": "List the current GUI role templates [{role_id, label, kind, default_name}].",
        "inputSchema": {"type": "object"},
    },
    "ml_create": {
        "handler": ml_create,
        "description": "Create a module/waveform from role_id with md-backed defaults; use the role default_name if name omitted. Return {name, kind, cfg}; if no default_name exists, supply name.",
        "inputSchema": {
            "type": "object",
            "properties": {"role_id": _NAME, "name": _NAME},
            "required": ["role_id"],
        },
    },
    "ml_edit": {
        "handler": ml_edit,
        "description": "Edit an existing named ModuleLibrary item using ordered canonical {path,value} agent edits, then commit only on full success. save_as writes a new item and leaves the source unchanged; an error discards the draft without writing to the library. Eval inputs lower to numeric values at commit. Return {name,cfg}; kind disambiguates shared names.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": _NAME,
                "kind": _KIND,
                "edits": _EDITS,
                "save_as": _NAME,
            },
            "required": ["name", "edits"],
        },
    },
}


def build_ml_tools(ctx: MeasureToolContext) -> dict[str, dict[str, Any]]:
    return {
        name: {**entry, "handler": partial(entry["handler"], ctx)}
        for name, entry in ML_TOOLS.items()
    }
