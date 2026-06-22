"""Method dispatcher for RemoteControlAdapter.

Every handler is a pure synchronous function ``(adapter, params) -> dict`` that
runs on the Qt main thread. The adapter layer is responsible for marshalling —
handlers must not touch threading or Qt directly.

Adding a method:
  1. Implement ``def _h_<dotted_name>(adapter, params): ...`` (returns wire
     dict). Reach the façade via ``adapter.ctrl.<m>``; View surfaces (render /
     snapshot) via the adapter's own methods.
  2. Register it in ``METHOD_REGISTRY`` below.
  3. Document the wire shape in ``services/remote/README.md``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, cast

from zcu_tools.gui.app.main.adapter import (
    MetaDictWriteback,
    ModuleWriteback,
    WaveformWriteback,
)

if TYPE_CHECKING:
    # Type-only: runtime import would cycle (controller.py imports
    # remote.dialogs). Handlers receive the RemoteControlAdapter (the driving
    # adapter that hosts them); they reach the command face via ``adapter.ctrl``
    # and the canvas View's pure-read surface via ``adapter.render_view`` (see
    # _render_view). String annotations let pyright check those call sites.
    from zcu_tools.gui.app.main.controller import RenderView

    from .service import RemoteControlAdapter
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError
from zcu_tools.gui.remote.method_spec import BoundMethod, build_method_registry
from zcu_tools.gui.remote.wire import optional_bool, require_int, require_str
from zcu_tools.gui.session.services.connection import (
    ConnectMockRequest,
    ConnectRemoteRequest,
    ConnectRequest,
)
from zcu_tools.gui.session.services.context import MlEntryValidationError
from zcu_tools.gui.session.services.device import (
    ConnectDeviceRequest,
    DisconnectDeviceRequest,
    SetupDeviceRequest,
)

from .method_specs import METHOD_SPECS

logger = logging.getLogger(__name__)

# Precise per-app handler alias (assignable to the shared, unconstrained
# ``method_spec.Handler``): every handler takes this app's RemoteControlAdapter.
Handler = Callable[["RemoteControlAdapter", Mapping[str, object]], Mapping[str, object]]


def _render_view(adapter: RemoteControlAdapter) -> RenderView:
    """The canvas-bearing View's pure-read surface (screenshot / snapshot /
    dialog). None in a headless process — render queries fail-fast there."""
    rv = adapter.render_view
    if rv is None:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            "no render view attached (headless process)",
        )
    return rv


# ---------------------------------------------------------------------------
# Tab handlers
# ---------------------------------------------------------------------------


def _h_tab_new(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    name = str(params["adapter_name"])
    if name not in adapter.ctrl.get_adapter_names():
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown adapter: {name!r}")
    tab_id = adapter.ctrl.new_tab(name)
    return {"tab_id": tab_id}


def _h_tab_close(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    tab_id = str(params["tab_id"])
    if not adapter.ctrl.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    adapter.ctrl.close_tab(tab_id)
    return {"ok": True}


def _h_tab_set_active(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    tab_id = str(params["tab_id"])
    if not adapter.ctrl.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    adapter.ctrl.set_active_tab(tab_id)
    return {"ok": True}


def _h_tab_list_all(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    running_tab_id = adapter.ctrl.get_running_tab_id()
    tabs = [
        {
            "tab_id": tid,
            "adapter_name": adapter.ctrl.get_tab_adapter_name(tid),
            "is_running": tid == running_tab_id,
        }
        for tid in adapter.ctrl.list_tab_ids()
    ]
    # active_tab_id is a view projection (which tab the user is focused on),
    # sourced from the same RenderView snapshot _assemble_overview reads.
    active_tab_id = _render_view(adapter).get_view_snapshot().get("active_tab_id")
    return {
        "tabs": tabs,
        "active_tab_id": active_tab_id,
        "running_tab_id": running_tab_id,
    }


def _tab_snapshot_wire(adapter: RemoteControlAdapter, tab_id: str) -> dict[str, object]:
    snap = adapter.ctrl.get_tab_snapshot(tab_id)
    interaction = snap.interaction
    # Render snapshot always fills the live fields (persist/restore form is the
    # only one that leaves them None, and it never hits the wire).
    assert interaction is not None
    return {
        "tab_id": tab_id,
        "adapter_name": adapter.ctrl.get_tab_adapter_name(tab_id),
        # Shared cfg-editor session id for this tab (None until the tab's form
        # is populated). Address it with the editor.* methods to edit cfg with
        # the GUI reflecting every change. (A tab uses its tab_id as owner key.)
        "editor_id": adapter.ctrl.editor_id_for_owner(tab_id),
        "interaction": {
            "global_run_active": bool(interaction.global_run_active),
            "is_running": bool(interaction.is_running),
            "is_analyzing": bool(interaction.is_analyzing),
            "is_saving_data": bool(interaction.is_saving_data),
            "has_context": bool(interaction.has_context),
            "has_active_context": bool(interaction.has_active_context),
            "has_soc": bool(interaction.has_soc),
            "has_run_result": bool(interaction.has_run_result),
            "has_analyze_result": bool(interaction.has_analyze_result),
            "has_figure": bool(interaction.has_figure),
        },
        "save_paths": _save_paths_wire(snap.save_paths),
    }


def _h_tab_snapshot(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    # Always returns {tabs: [...]} (a single tab_id yields a one-element list);
    # no shape-switch, so callers index reply["tabs"] uniformly.
    tab_id_raw = params.get("tab_id")
    if tab_id_raw is None:
        tab_ids = adapter.ctrl.list_tab_ids()
    else:
        tab_id = str(tab_id_raw)
        if not adapter.ctrl.has_tab(tab_id):
            raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
        tab_ids = [tab_id]
    return {"tabs": [_tab_snapshot_wire(adapter, tid) for tid in tab_ids]}


def _save_paths_wire(paths) -> dict[str, str] | None:
    if paths is None:
        return None
    return {"data_path": paths.data_path, "image_path": paths.image_path}


def _h_tab_get_cfg(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    from .path_resolver import build_settable_tree

    tab_id = str(params["tab_id"])
    if not adapter.ctrl.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    # A tab's cfg draft is a CfgEditorService session keyed by its tab_id (the
    # same draft the open form attaches to). Build the settable tree off that
    # session's live root — the one tab.set_cfg/editor.set_field mutates — so
    # the tree mirrors exactly what can be edited and agent+user share one model
    # (ADR-0013 F11). Leaf values come straight off the live tree
    # (ADR-0010: None = unset).
    editor_id = adapter.ctrl.editor_id_for_owner(tab_id)
    if editor_id is None:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            f"tab {tab_id!r} cfg form has no live model yet",
        )
    raw_prefix = params.get("prefix")
    prefix = str(raw_prefix) if raw_prefix else None
    root = adapter.ctrl.get_cfg_editor_root(editor_id)
    return {"tree": build_settable_tree(root, prefix=prefix)}


def _h_tab_set_cfg(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    from zcu_tools.gui.app.main.services.cfg_editor import CfgEditorError

    tab_id = str(params["tab_id"])
    if not adapter.ctrl.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    # Block edits while the tab is running — same guard the human gets via the
    # disabled form (ADR-0013 F11).
    if adapter.ctrl.get_running_tab_id() == tab_id:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            f"tab {tab_id!r} is currently running; cancel the run before editing cfg",
        )
    editor_id = adapter.ctrl.editor_id_for_owner(tab_id)
    if editor_id is None:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            f"tab {tab_id!r} cfg form has no live model yet",
        )
    raw_edits = params.get("edits")
    if not isinstance(raw_edits, list):
        raise RemoteError(ErrorCode.INVALID_PARAMS, "'edits' must be a list")
    # Apply edits sequentially (fail-fast, non-atomic); caller orders ref-switch
    # edits before dependent inner-path edits. Delegate to cfg_editor_set_field
    # — the same path the editor.set_field handler uses — to avoid duplicating
    # path resolution and validation logic.
    all_removed: list[str] = []
    all_added: list[str] = []
    valid = True
    for i, edit in enumerate(raw_edits):
        if not isinstance(edit, dict) or "path" not in edit or "value" not in edit:
            raise RemoteError(
                ErrorCode.INVALID_PARAMS,
                f"edits[{i}] must be an object with 'path' and 'value'",
            )
        path = str(edit["path"])
        value = edit["value"]
        try:
            result = adapter.ctrl.cfg_editor_set_field(editor_id, path, value)
        except CfgEditorError as exc:
            raise RemoteError(ErrorCode.INVALID_PARAMS, str(exc)) from exc
        except RemoteError:
            raise
        except (KeyError, RuntimeError) as exc:
            raise RemoteError(
                ErrorCode.INVALID_PARAMS,
                str(exc),
                reason=getattr(exc, "reason_code", ""),
            ) from exc
        valid = bool(result.get("valid", True))
        removed = result.get("removed", [])
        added = result.get("added", [])
        if isinstance(removed, list):
            all_removed.extend(str(p) for p in removed)
        if isinstance(added, list):
            all_added.extend(str(p) for p in added)
    return {"valid": valid, "removed": all_removed, "added": all_added}


# ---------------------------------------------------------------------------
# Run / Save handlers
# ---------------------------------------------------------------------------


def _h_tab_run_start(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    tab_id = str(params["tab_id"])
    if not adapter.ctrl.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    try:
        operation_id = adapter.ctrl.start_run(tab_id)
    except RuntimeError as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason=getattr(exc, "reason_code", ""),
        ) from exc
    return {"operation_id": operation_id}


def _h_tab_run_cancel(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    # cancelled is best-effort: True when a live run was signalled, False on a
    # no-op. The worker's true terminal is observed via the run handle (ADR-0026
    # §8) — cancel only requests, it does not wait for the stop.
    cancelled = adapter.ctrl.cancel_run()
    return {"ok": True, "cancelled": cancelled}


def _h_analyze_cancel(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    tab_id = str(params["tab_id"])
    if not adapter.ctrl.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    # Graceful by contract (no interactive analyze in flight is not an error): the
    # cancelled flag tells the agent whether anything was actually settled.
    cancelled = adapter.ctrl.cancel_analyze(tab_id)
    return {"ok": True, "cancelled": cancelled}


def _h_run_running_tab(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    return {"tab_id": adapter.ctrl.get_running_tab_id()}


def _h_tab_save_data(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    tab_id = str(params["tab_id"])
    data_path = params["data_path"]
    comment = str(params["comment"])
    try:
        written = adapter.ctrl.save_data(
            tab_id, str(data_path) if data_path is not None else None, comment=comment
        )
    except RuntimeError as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason=getattr(exc, "reason_code", ""),
        ) from exc
    # The save runs async, but the resolved path (.hdf5 + uniqueness suffix) is
    # known synchronously — return it so the caller need not recover it from a
    # later diagnostic / snapshot.
    return {"data_path": written}


def _h_tab_save_image(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    tab_id = str(params["tab_id"])
    image_path = params["image_path"]
    try:
        written = adapter.ctrl.save_image(
            tab_id, str(image_path) if image_path is not None else None
        )
    except RuntimeError as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason=getattr(exc, "reason_code", ""),
        ) from exc
    return {"image_path": written}


def _h_tab_save_post_image(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    tab_id = str(params["tab_id"])
    image_path = params["image_path"]
    try:
        written = adapter.ctrl.save_post_image(
            tab_id, str(image_path) if image_path is not None else None
        )
    except RuntimeError as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason=getattr(exc, "reason_code", ""),
        ) from exc
    return {"image_path": written}


def _h_tab_save_result(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    tab_id = str(params["tab_id"])
    data_path = params["data_path"]
    image_path = params["image_path"]
    comment = str(params["comment"])
    try:
        written_data, written_image = adapter.ctrl.save_result(
            tab_id,
            str(data_path) if data_path is not None else None,
            str(image_path) if image_path is not None else None,
            comment=comment,
        )
    except RuntimeError as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason=getattr(exc, "reason_code", ""),
        ) from exc
    # The data save runs async, but both resolved paths (the data path's .hdf5 +
    # uniqueness suffix included) are known synchronously — return them so the
    # caller need not recover them from a later diagnostic.
    return {"data_path": written_data, "image_path": written_image}


def _h_tab_save_set_paths(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    tab_id = str(params["tab_id"])
    if not adapter.ctrl.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    data_path = str(params["data_path"])
    image_path = str(params["image_path"])
    adapter.ctrl.update_tab_save_paths(tab_id, data_path, image_path)
    return {"data_path": data_path, "image_path": image_path}


# ---------------------------------------------------------------------------
# Context / state / session handlers
# ---------------------------------------------------------------------------


def _h_context_use(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    # A context lives under a project; without one there are no labels to switch
    # to. Map that precondition to agent language rather than leaking a controller
    # error (mirror _h_context_new).
    if not adapter.ctrl.has_project():
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            "No project applied yet; apply a project first (gui_project_apply).",
            reason="no_project",
        )
    label = str(params["label"])
    available = list(adapter.ctrl.get_context_labels())
    if label not in available:
        # Fast-fail an unknown label with the valid choices so the agent can
        # correct without a separate gui_context_list round-trip.
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            f"unknown context label: {label!r}; available: {available}",
        )
    adapter.ctrl.use_context(label)
    return {
        "label": adapter.ctrl.get_active_context_label(),
        "has_active_context": adapter.ctrl.get_active_context_label() is not None,
    }


def _h_context_new(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    # A context lives under a project's experiment dir; without a project the
    # IOManager has no dir to create it in. Translate that precondition into
    # agent language here rather than leaking the internal "IOManager not set
    # up" RuntimeError as a controller_error.
    if not adapter.ctrl.has_project():
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            "No project applied yet; apply a project first (gui_project_apply).",
            reason="no_project",
        )
    bind_device = params["bind_device"]
    clone_from = params["clone_from"]
    adapter.ctrl.new_context(
        bind_device=str(bind_device) if bind_device is not None else None,
        clone_from=str(clone_from) if clone_from is not None else None,
    )
    # new_context makes the new context active — return its label so the agent
    # knows what was created without a follow-up read.
    label = adapter.ctrl.get_active_context_label()
    return {"label": label, "has_active_context": label is not None}


def _h_context_labels(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    return {"labels": list(adapter.ctrl.get_context_labels())}


def _h_context_active(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    return {"label": adapter.ctrl.get_active_context_label()}


# Wire tag for a Python ``complex`` scalar. JSON has no complex type, so a
# complex md value (e.g. a single-shot IQ centre) is carried as a self-describing
# structured tag the agent can both read and round-trip — ``_coerce_wire_value``
# turns it back into ``complex`` on the set/apply input side. Lossless, unlike the
# old ``{"__repr__"}`` fallback (which stringified complex one-way).
_COMPLEX_TAG = "__complex__"


def _complex_tag(value: complex) -> dict[str, list[float]]:
    return {_COMPLEX_TAG: [value.real, value.imag]}


def _is_complex_tag(value: object) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == {_COMPLEX_TAG}
        and isinstance(value[_COMPLEX_TAG], (list, tuple))
        and len(value[_COMPLEX_TAG]) == 2
        and all(isinstance(p, (int, float)) for p in value[_COMPLEX_TAG])
    )


def _json_safe(value: object) -> object:
    """Make ``value`` JSON-safe without loss where the type has a wire encoding.

    ``complex`` → ``{"__complex__": [re, im]}`` (round-trips via
    ``_coerce_wire_value``). Otherwise return ``value`` if it round-trips through
    JSON as-is, else its ``repr`` (lossy last resort for opaque objects).
    """
    import json

    if isinstance(value, complex):
        return _complex_tag(value)
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return {"__repr__": repr(value)}


def _coerce_wire_value(value: object) -> object:
    """Inverse of :func:`_json_safe`'s structured tags for inbound wire values.

    A ``{"__complex__": [re, im]}`` tag becomes a Python ``complex``; every other
    value passes through untouched. Used on the writeback ``set`` input so an
    agent-supplied complex proposed_value applies as a real ``complex`` (the
    in-process md apply + MetaDict persistence both speak ``complex``)."""
    if _is_complex_tag(value):
        re, im = value[_COMPLEX_TAG]  # type: ignore[index]
        return complex(re, im)
    return value


def _h_context_md_get(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    md = adapter.ctrl.get_current_md()
    return {"keys": sorted(str(k) for k in md.keys())}


def _h_context_md_get_attr(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    key = str(params["key"])
    md = adapter.ctrl.get_current_md()
    sentinel = object()
    value = md.get(key, sentinel)
    if value is sentinel:
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown md key: {key!r}")
    return {"key": key, "value": _json_safe(value)}


def _h_context_ml_get(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    ml = adapter.ctrl.get_current_ml()
    # Each stored cfg is a pydantic discriminated-union value: modules tag on
    # 'type' (e.g. 'pulse', 'reset/bath'), waveforms on 'style' (e.g. 'gauss').
    # Surface the discriminator so the agent can tell entry kinds apart without
    # opening each one (gui_context_ml_inspect).
    return {
        "modules": [
            {"name": name, "kind": getattr(ml.modules[name], "type")}
            for name in sorted(ml.modules.keys())
        ],
        "waveforms": [
            {"name": name, "style": getattr(ml.waveforms[name], "style")}
            for name in sorted(ml.waveforms.keys())
        ],
    }


def _h_context_ml_list_roles(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    """List the experiment-role templates available for create_from_role."""
    del params
    try:
        catalog = adapter.ctrl.get_role_catalog()
    except RuntimeError as exc:
        raise RemoteError(ErrorCode.PRECONDITION_FAILED, str(exc)) from exc
    return {"roles": list(catalog.list_meta())}


def _h_context_ml_create_from_role(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    """Create a blank ml module/waveform from a named role and register it.

    One-shot: seeds md-linked defaults (lowered against the live md), writes ml.
    Edit afterwards via editor.new(from_name=...).
    """
    role_id = str(params["role_id"])
    name = str(params["name"])
    # The item kind is a property of the role, not an independent agent input —
    # derive it from role_id so the agent cannot pass a mismatching pair. An
    # unknown role_id fails fast as invalid_params; a missing catalog (no project)
    # surfaces as precondition_failed (mirror _h_context_ml_list_roles).
    try:
        item_kind = adapter.ctrl.get_role_catalog().get(role_id).item_kind
    except KeyError as exc:
        raise RemoteError(ErrorCode.INVALID_PARAMS, str(exc)) from exc
    except RuntimeError as exc:
        raise RemoteError(ErrorCode.PRECONDITION_FAILED, str(exc)) from exc
    try:
        adapter.ctrl.create_from_role(item_kind, role_id, name)
    except KeyError as exc:
        raise RemoteError(ErrorCode.INVALID_PARAMS, str(exc)) from exc
    except MlEntryValidationError as exc:
        raise RemoteError(ErrorCode.INVALID_PARAMS, str(exc)) from exc
    except RuntimeError as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason=getattr(exc, "reason_code", ""),
        ) from exc
    return {"created": name}


def _h_context_md_set_attr(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    key = str(params["key"])
    value = params["value"]
    try:
        adapter.ctrl.set_md_attr(key, value)
    except RuntimeError as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason=getattr(exc, "reason_code", ""),
        ) from exc
    return {}


def _h_context_md_del_attr(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    key = str(params["key"])
    try:
        adapter.ctrl.del_md_attr(key)
    except (AttributeError, RuntimeError) as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason=getattr(exc, "reason_code", ""),
        ) from exc
    return {}


def _h_context_ml_del_module(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    name = str(params["name"])
    try:
        adapter.ctrl.del_ml_module(name)
    except (KeyError, RuntimeError) as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason=getattr(exc, "reason_code", ""),
        ) from exc
    return {"deleted": name}


def _h_context_ml_rename_module(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    old = str(params["old"])
    new = str(params["new"])
    try:
        adapter.ctrl.rename_ml_module(old, new)
    except RuntimeError as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason=getattr(exc, "reason_code", ""),
        ) from exc
    return {"renamed": new}


def _h_context_ml_rename_waveform(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    old = str(params["old"])
    new = str(params["new"])
    try:
        adapter.ctrl.rename_ml_waveform(old, new)
    except RuntimeError as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason=getattr(exc, "reason_code", ""),
        ) from exc
    return {"renamed": new}


def _h_context_ml_del_waveform(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    name = str(params["name"])
    try:
        adapter.ctrl.del_ml_waveform(name)
    except (KeyError, RuntimeError) as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason=getattr(exc, "reason_code", ""),
        ) from exc
    return {"deleted": name}


def _h_state_has_project(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    return {"value": bool(adapter.ctrl.has_project())}


def _h_state_has_context(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    return {"value": bool(adapter.ctrl.has_context())}


def _h_state_has_active_context(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    return {"value": bool(adapter.ctrl.has_active_context())}


def _h_state_has_soc(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    return {"value": bool(adapter.ctrl.has_soc())}


def _h_soc_info(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    include_cfg = bool(params["include_cfg"])  # ParamSpec(_bool_default)-validated
    try:
        return adapter.ctrl.get_soc_info(include_cfg=include_cfg)
    except RuntimeError as exc:
        raise RemoteError(ErrorCode.PRECONDITION_FAILED, str(exc)) from exc


def _h_project_info(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    # Project identity (the chip / qubit / resonator names + their output roots).
    # It lives only on the in-process ExpContext, so this is the sole wire query
    # that exposes it — _assemble_overview folds {chip, qub, res} from here. The
    # res_name field is measure-specific (the other GUIs' shared project.info
    # carries only chip/qub/result_dir/database_path).
    del params
    if not adapter.ctrl.has_project():
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            "No project applied yet; apply a project first (gui_project_apply).",
            reason="no_project",
        )
    ctx = adapter.ctrl.get_exp_context()
    return {
        "chip_name": ctx.chip_name,
        "qub_name": ctx.qub_name,
        "res_name": ctx.res_name,
        "result_dir": ctx.result_dir,
        "database_path": ctx.database_path,
    }


def _h_resources_versions(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    return {"versions": adapter.ctrl.resources_versions()}


# ---------------------------------------------------------------------------
# Connection / startup / device handlers (typed-request coercion)
# ---------------------------------------------------------------------------
#
# The coercion helpers below turn a raw wire ``params`` mapping into a typed
# domain request. They live here (beside their only callers) rather than in
# ``wire.py`` so the wire layer stays a pure transport primitive — it knows the
# field-level ``_require_*``/``_optional_*`` rules but not the connection/device
# domain shapes those rules assemble into.


def coerce_connect_request(params: Mapping[str, object]) -> ConnectRequest:
    """Coerce ``{kind: 'mock'}`` or ``{kind: 'remote', ip, port}``."""
    kind = require_str(params, "kind")
    if kind == "mock":
        return ConnectMockRequest()
    if kind == "remote":
        return ConnectRemoteRequest(
            ip=require_str(params, "ip"),
            port=require_int(params, "port"),
        )
    raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown connect kind: {kind!r}")


def coerce_connect_device_request(
    params: Mapping[str, object],
) -> ConnectDeviceRequest:
    return ConnectDeviceRequest(
        type_name=require_str(params, "type_name"),
        name=require_str(params, "name"),
        address=require_str(params, "address"),
        remember=optional_bool(params, "remember", True),
    )


def coerce_disconnect_device_request(
    params: Mapping[str, object],
) -> DisconnectDeviceRequest:
    return DisconnectDeviceRequest(
        name=require_str(params, "name"),
        remember=optional_bool(params, "remember", True),
    )


def _h_soc_connect(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    # Synchronous connect: runs on the Qt main thread (the IO worker blocks on the
    # _dispatch_on_main marshal), so the connect work + ALL post-connect side
    # effects (State write, soc version bump, SocChangedPayload → FLUX-AWARE-MOCK
    # provisioning) complete before this returns. A connect failure raises a typed
    # RemoteError. The worst-case main-thread block is bounded by make_soc_proxy's
    # 1s COMMTIMEOUT for a remote board (mock is instant). Connect is no longer an
    # async operation handle — run / analyze / device keep theirs.
    req = coerce_connect_request(params)
    try:
        adapter.ctrl.connect_sync(req)
    except Exception as exc:
        raise RemoteError(
            ErrorCode.CONTROLLER_ERROR,
            f"SoC connect failed: {exc}",
        ) from exc
    # Return the SoC summary directly — the same {description, is_mock} the old
    # finished short-wait reply folded in. The structured cfg is fetched on demand
    # via soc.info (it is ~2 KB and rarely needed at connect time).
    info = adapter.ctrl.get_soc_info()
    return {"soc": {"description": info["description"], "is_mock": info["is_mock"]}}


def _h_startup_apply(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    # result_dir / database_path are optional. When omitted, the RPC fills the
    # default per-qubit roots via derive_project_paths(chip, qub, cwd) — the same
    # paths the setup dialog pre-fills when a user types the chip/qub names — so
    # an agent gets a runnable project without having to know the chip/qub path
    # layout. (The setup dialog keeps its own empty-result_dir = DRAFT path for
    # interactive use; this agent-facing entry intentionally defaults to runnable
    # rather than DRAFT.)
    from zcu_tools.gui.session.services.startup import (
        StartupProjectRequest,
        derive_project_paths,
    )

    chip = str(params["chip_name"])
    qub = str(params["qub_name"])
    result_dir = str(params["result_dir"] or "")
    database_path = str(params["database_path"] or "")
    if not result_dir or not database_path:
        default_result, default_db = derive_project_paths(
            chip, qub, adapter.ctrl.get_project_root()
        )
        result_dir = result_dir or default_result
        database_path = database_path or default_db

    req = StartupProjectRequest(
        chip_name=chip,
        qub_name=qub,
        res_name=str(params["res_name"]),
        result_dir=result_dir,
        database_path=database_path,
    )
    # Echo the resolved project (apply always mutates and either succeeds or
    # raises — there is no no-op outcome, so no {applied:false} branch).
    return adapter.ctrl.apply_startup_project(req)


def _h_device_connect(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    req = coerce_connect_device_request(params)
    try:
        operation_id = adapter.ctrl.start_connect_device(req)
    except RuntimeError as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason=getattr(exc, "reason_code", ""),
        ) from exc
    return {"operation_id": operation_id}


def _h_device_disconnect(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    req = coerce_disconnect_device_request(params)
    try:
        operation_id = adapter.ctrl.start_disconnect_device(req)
    except RuntimeError as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason=getattr(exc, "reason_code", ""),
        ) from exc
    return {"operation_id": operation_id}


def _h_device_reconnect(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    name = str(params["name"])
    try:
        operation_id = adapter.ctrl.start_reconnect_device(name)
    except RuntimeError as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason=getattr(exc, "reason_code", ""),
        ) from exc
    # Reconnect runs asynchronously like connect/disconnect/setup; expose the
    # operation_id so the MCP short-wait/handle path can track it (FC1).
    return {"operation_id": operation_id}


def _h_device_forget(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    name = str(params["name"])
    try:
        adapter.ctrl.forget_device(name)
    except RuntimeError as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason=getattr(exc, "reason_code", ""),
        ) from exc
    # Synchronous sync mutation: echo the forgotten name so the reply is
    # self-verifying (no follow-up read needed to confirm what was dropped).
    return {"forgotten": name}


def _h_device_setup(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    name = str(params["name"])
    updates = cast(dict, params["updates"])  # ParamSpec(_obj)-validated
    try:
        info = adapter.ctrl.get_device_info(name)
    except RuntimeError as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason=getattr(exc, "reason_code", ""),
        ) from exc
    if info is None:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            f"Device {name!r} has no live info to update",
        )
    try:
        updated = info.with_updates(**dict(updates))
    except ValueError as exc:
        raise RemoteError(ErrorCode.INVALID_PARAMS, str(exc)) from exc
    try:
        operation_id = adapter.ctrl.start_setup_device(
            SetupDeviceRequest(name=name, info=updated)
        )
    except RuntimeError as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason=getattr(exc, "reason_code", ""),
        ) from exc
    return {"operation_id": operation_id}


# device.setup field-schema discovery: BaseDeviceInfo is a pydantic model whose
# model_fields carry the settable fields' types + Literal choices. Protected
# fields (type/address; cf. BaseDeviceInfo.with_updates) are reported
# settable=false. Pure read of pydantic metadata (same scope as snapshot's
# info.to_dict()); no device-side change.
_DEVICE_SETUP_PROTECTED = frozenset({"type", "address"})


def _field_type_and_choices(annotation: object) -> tuple[str, list | None]:
    """Wire (type, choices) for a BaseDeviceInfo field annotation.

    Literal[...] → ('enum', [members]). Optional[X] unwraps to X.
    Otherwise map the scalar python type to a JSON type name.

    Pydantic model fields use typing.Literal; get_origin returns the Literal
    sentinel from typing (typing_extensions.Literal is an alias post-3.11).
    """
    import types
    import typing

    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    # Literal: get_origin returns typing.Literal for both typing.Literal and
    # typing_extensions.Literal on Python 3.11+ (they are the same object).
    if origin is typing.Literal:
        return "enum", list(args)
    # Optional[X] / Union[X, None] → unwrap to the non-None member.
    # Accept both typing.Union (Optional[T]) and types.UnionType (PEP 604,
    # X | None): get_origin returns types.UnionType for the latter.
    if origin is typing.Union or origin is types.UnionType:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _field_type_and_choices(non_none[0])
    _SCALAR = {float: "float", int: "int", str: "str", bool: "bool"}
    return _SCALAR.get(annotation, "str"), None  # type: ignore[arg-type]


def _h_device_setup_spec(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    name = str(params["name"])
    try:
        info = adapter.ctrl.get_device_info(name)
    except RuntimeError as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason=getattr(exc, "reason_code", ""),
        ) from exc
    if info is None:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            f"Device {name!r} has no live info (connect it first)",
        )
    fields: list[dict[str, object]] = []
    for fname, finfo in type(info).model_fields.items():
        ftype, choices = _field_type_and_choices(finfo.annotation)
        entry: dict[str, object] = {
            "name": fname,
            "type": ftype,
            "current": getattr(info, fname, None),
            "settable": fname not in _DEVICE_SETUP_PROTECTED,
        }
        if choices is not None:
            entry["choices"] = choices
        fields.append(entry)
    return {"fields": fields}


def _h_device_cancel_operation(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    name = str(params["name"])
    try:
        adapter.ctrl.cancel_device_operation(name)
    except RuntimeError as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason=getattr(exc, "reason_code", ""),
        ) from exc
    # Self-verifying echo: cancel succeeded (a non-cancellable / absent op raised
    # above). The terminal outcome is observed via the operation handle.
    return {"ok": True, "cancelled": True}


def _h_adapter_list(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    return {"adapters": list(adapter.ctrl.get_adapter_names())}


def _h_adapter_guide(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    name = str(params["adapter_name"])
    if name not in adapter.ctrl.get_adapter_names():
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown adapter: {name!r}")
    return {"guide": adapter.ctrl.get_adapter_guide(name)}


def _h_device_list(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    # Project the fine-grained status (DeviceEntry.status), consistent with the
    # snapshot/active_operations projections (single-status SSOT, FC7).
    devices = [
        {
            "name": e.name,
            "type_name": e.type_name,
            "status": e.status,
        }
        for e in adapter.ctrl.list_devices()
    ]
    return {"devices": devices}


def _h_device_snapshot(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    name = str(params["name"])
    snap = adapter.ctrl.get_device_snapshot(name)
    if snap is None:
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown device: {name!r}")
    # ``info`` is a ``BaseDeviceInfo`` (a pydantic ``ConfigBase``); ``to_dict()``
    # yields JSON-safe scalars (address/type/label + driver fields like the
    # source ``value``), so the agent can read the device's live parameters.
    return {
        "snapshot": {
            "name": snap.name,
            "type_name": snap.type_name,
            "address": snap.address,
            "status": snap.status.value,
            "error": snap.error,
            "info": snap.info.to_dict() if snap.info is not None else None,
        }
    }


def _progress_bars_wire(bars) -> Mapping[str, object]:
    """Shared run/device progress projection from live (token, ProgressBarModel)
    pairs — derived fields computed live at this read (the SSOT is the model)."""
    if not bars:
        return {"active": False, "bars": []}
    return {
        "active": True,
        "bars": [
            {
                "token": token,
                "format": m.format(),
                "maximum": m.qt_maximum(),
                "value": m.qt_value(),
                "percent": m.percent(),
                "n": m.n,
                "total": m.total,
            }
            for token, m in bars
        ],
    }


def _h_device_active_operations(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    # Phase C concurrency: enumerate *every* in-flight device operation (sorted
    # by name), each tagged with its kind (connect / disconnect / setup) and its
    # operation 'handle' so the agent can drive gui_op_poll / gui_op_wait per op.
    # device_name is the SSOT key; the duplicate snapshot.name field is dropped.
    return {
        "operations": [
            {
                "handle": op.token,
                "device_name": op.device_name,
                "kind": op.kind.value,
                "type_name": op.snapshot.type_name,
                "address": op.snapshot.address,
                "status": op.snapshot.status.value,
                "error": op.snapshot.error,
            }
            for op in adapter.ctrl.get_active_device_operations()
        ]
    }


def _h_operation_await(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    # off_main_thread handler: blocks the IO worker thread on the handle's
    # thread-safe registry (never touches main-thread-owned state). Returns a
    # structured payload with reason in {'completed', 'user_feedback', 'timeout'}
    # (ADR-0025). 'cancelled' is returned as structured data (status='cancelled',
    # optional feedback from the Stop reason); 'failed' is still raised as
    # PRECONDITION_FAILED so the agent sees it as an error.
    operation_id = int(params["operation_id"])  # type: ignore[arg-type]
    timeout = float(params["timeout"])  # type: ignore[arg-type]
    result = adapter.ctrl.await_operation(operation_id, timeout)
    if result is None:
        # Should not happen with the new API, but guard for forward-compat.
        raise RemoteError(
            ErrorCode.TIMEOUT,
            f"operation {operation_id} did not complete within {timeout}s",
        )
    if result.reason == "timeout":
        raise RemoteError(
            ErrorCode.TIMEOUT,
            f"operation {operation_id} did not complete within {timeout}s",
        )
    if result.reason == "user_feedback":
        # Non-terminal: operation still running; feedback delivered to the agent.
        return {
            "reason": "user_feedback",
            "feedback": result.feedback,
        }
    # reason == 'completed'
    outcome = result.outcome
    assert outcome is not None  # invariant: completed always has outcome
    if outcome.status == "cancelled":
        # Structured cancellation: return status + optional Stop reason so the
        # agent gets the full picture in one reply (ADR-0025 §cancelled-wire).
        # The feedback field is only present when a Stop reason was latched
        # (i.e. "Send & Stop" was used); a plain cancel has no feedback.
        payload: dict[str, object] = {"reason": "completed", "status": "cancelled"}
        if result.feedback:
            payload["feedback"] = result.feedback
        return payload
    if outcome.status == "failed":
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            outcome.error or "operation failed",
            reason="failed",
        )
    return {"reason": "completed", "status": outcome.status}


# ---------------------------------------------------------------------------
# Dialog / view-query handlers
# ---------------------------------------------------------------------------


def _h_app_shutdown(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    # Graceful close: trigger the window's normal close path (persist session,
    # tear down remote, cleanup) on the main thread. request_shutdown defers the
    # actual close to the next event-loop turn so this reply is sent before the
    # remote service tears down. No kill / OS signal — that path is the agent's
    # cross-platform-safe way to stop the GUI.
    del params
    _render_view(adapter).request_shutdown()
    return {"shutting_down": True}


def _h_view_snapshot(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    snap = _render_view(adapter).get_view_snapshot()
    if not isinstance(snap, dict):
        raise RemoteError(
            ErrorCode.INTERNAL,
            f"view snapshot returned non-dict {type(snap).__name__}",
        )
    return snap


def _h_dialog_screenshot(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    import base64

    from .dialogs import parse_dialog_name

    name_str = str(params["name"])
    try:
        dialog_name = parse_dialog_name(name_str)
        png = _render_view(adapter).take_dialog_screenshot(dialog_name)
    except (ValueError, RuntimeError) as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason=getattr(exc, "reason_code", ""),
        ) from exc
    if not isinstance(png, (bytes, bytearray)):
        raise RemoteError(
            ErrorCode.INTERNAL,
            f"screenshot returned non-bytes {type(png).__name__}",
        )
    payload = base64.b64encode(bytes(png)).decode("ascii")
    return {"png_b64": payload, "bytes": len(png)}


def _h_view_screenshot(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    import base64

    del params
    # Not off_main_thread → MainWindow.grab() is auto-marshalled to the Qt main
    # thread, the same path as dialog.screenshot. The whole window always exists
    # (headless is already fast-failed by _render_view), so there is no
    # PRECONDITION branch like the per-dialog grab.
    png = _render_view(adapter).take_window_screenshot()
    if not isinstance(png, (bytes, bytearray)):
        raise RemoteError(
            ErrorCode.INTERNAL,
            f"window screenshot returned non-bytes {type(png).__name__}",
        )
    payload = base64.b64encode(bytes(png)).decode("ascii")
    return {"png_b64": payload, "bytes": len(png)}


# ---------------------------------------------------------------------------
# Tab analyze result + analyze start + cfg summary handlers
# ---------------------------------------------------------------------------


def _h_tab_get_analyze_result(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    tab_id = str(params["tab_id"])
    if not adapter.ctrl.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    result = adapter.ctrl.get_tab_analyze_result(tab_id)
    if result is None:
        return {"summary": None}
    to_summary = getattr(result, "to_summary_dict", None)
    if not callable(to_summary):
        raise RemoteError(
            ErrorCode.INTERNAL,
            "analyze result does not implement to_summary_dict()",
        )
    return {"summary": to_summary()}


def _h_tab_get_analyze_params(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    import dataclasses

    tab_id = str(params["tab_id"])
    if not adapter.ctrl.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    snap = adapter.ctrl.get_tab_snapshot(tab_id)
    if snap.analyze_params is None:
        return {"analyze_params": None}
    ap = snap.analyze_params
    if not dataclasses.is_dataclass(ap) or isinstance(ap, type):
        return {"analyze_params": {}}
    return {"analyze_params": dataclasses.asdict(ap)}


def _h_tab_analyze(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    import dataclasses

    tab_id = str(params["tab_id"])
    if not adapter.ctrl.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    snap = adapter.ctrl.get_tab_snapshot(tab_id)
    # Order the checks by the true cause: analyze params only exist once a run
    # produced a result (they are built from it). A run-in-flight / failed /
    # cancelled tab has no result, so report that — not the downstream "no
    # analyze params", which reads as a config gap rather than "nothing to
    # analyze yet".
    interaction = snap.interaction
    if interaction is not None and not interaction.has_run_result:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            "No run result available to analyze.",
            reason="no_run_result",
        )
    if snap.analyze_params is None:
        raise RemoteError(ErrorCode.PRECONDITION_FAILED, "no analyze params available")
    raw_updates = cast(dict, params["updates"])  # ParamSpec(_obj)-validated
    ap = snap.analyze_params
    if not dataclasses.is_dataclass(ap) or isinstance(ap, type):
        raise RemoteError(
            ErrorCode.INTERNAL, "analyze_params is not a dataclass instance"
        )
    try:
        updated = dataclasses.replace(ap, **raw_updates)
    except (TypeError, ValueError) as exc:
        raise RemoteError(ErrorCode.INVALID_PARAMS, str(exc)) from exc
    try:
        operation_id = adapter.ctrl.analyze(tab_id, updated)
    except RuntimeError as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason=getattr(exc, "reason_code", ""),
        ) from exc
    return {"operation_id": operation_id}


# ---------------------------------------------------------------------------
# Post-analysis (second layer) result + params + start handlers. These mirror
# the analyze trio above, but every entry gates on the PRIMARY analyze result
# (post-analysis is built on top of it) instead of the run result.
# ---------------------------------------------------------------------------


def _h_tab_get_post_analyze_result(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    tab_id = str(params["tab_id"])
    if not adapter.ctrl.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    result = adapter.ctrl.get_post_analyze_result(tab_id)
    if result is None:
        return {"summary": None}
    to_summary = getattr(result, "to_summary_dict", None)
    if not callable(to_summary):
        raise RemoteError(
            ErrorCode.INTERNAL,
            "post-analysis result does not implement to_summary_dict()",
        )
    return {"summary": to_summary()}


def _h_tab_get_post_analyze_params(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    import dataclasses

    tab_id = str(params["tab_id"])
    if not adapter.ctrl.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    snap = adapter.ctrl.get_tab_snapshot(tab_id)
    if snap.post_analyze_params is None:
        return {"post_analyze_params": None}
    pp = snap.post_analyze_params
    if not dataclasses.is_dataclass(pp) or isinstance(pp, type):
        return {"post_analyze_params": {}}
    return {"post_analyze_params": dataclasses.asdict(pp)}


def _h_tab_post_analyze(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    import dataclasses

    tab_id = str(params["tab_id"])
    if not adapter.ctrl.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    snap = adapter.ctrl.get_tab_snapshot(tab_id)
    # Order the checks by the true cause: post params only exist once a primary
    # analyze produced a result (they are built from it). Report the missing
    # primary result first — it reads as "nothing to post-analyze yet" rather
    # than the downstream "no post params", which looks like a config gap.
    interaction = snap.interaction
    if interaction is not None and not interaction.has_analyze_result:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            "No primary analyze result available to post-analyze.",
            reason="no_analyze_result",
        )
    if snap.post_analyze_params is None:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED, "no post-analysis params available"
        )
    raw_updates = cast(dict, params["updates"])  # ParamSpec(_obj)-validated
    pp = snap.post_analyze_params
    if not dataclasses.is_dataclass(pp) or isinstance(pp, type):
        raise RemoteError(
            ErrorCode.INTERNAL, "post_analyze_params is not a dataclass instance"
        )
    try:
        updated = dataclasses.replace(pp, **raw_updates)
    except (TypeError, ValueError) as exc:
        raise RemoteError(ErrorCode.INVALID_PARAMS, str(exc)) from exc
    try:
        operation_id = adapter.ctrl.start_post_analyze(tab_id, updated)
    except RuntimeError as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason=getattr(exc, "reason_code", ""),
        ) from exc
    return {"operation_id": operation_id}


# ---------------------------------------------------------------------------
# Writeback preview / apply handlers
# ---------------------------------------------------------------------------


def _writeback_item_wire(item) -> dict[str, object]:
    base: dict[str, object] = {
        "id": item.session_id,
        "target_name": item.target_name,
        "description": item.description,
        "selected": bool(item.selected),
    }
    if isinstance(item, MetaDictWriteback):
        base["kind"] = "metadict"
        base["proposed_value"] = _json_safe(item.proposed_value)
    elif isinstance(item, (ModuleWriteback, WaveformWriteback)):
        is_module = isinstance(item, ModuleWriteback)
        base["kind"] = "module" if is_module else "waveform"
        base["editor_id"] = item.editor_id
        base["has_edit_schema"] = item.editor_id is not None
    else:
        base["kind"] = "unknown"
    return base


def _h_tab_writeback_preview(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    """Pure read of the tab's persistent writeback draft (not a dry-run): lists
    the items computed once at analyze time. ``has_draft`` is false before any
    analyze has produced a draft (empty item list)."""
    tab_id = str(params["tab_id"])
    if not adapter.ctrl.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    items = adapter.ctrl.get_tab_writeback_items(tab_id)
    return {
        "has_draft": bool(items),
        "items": [_writeback_item_wire(it) for it in items],
    }


def _h_tab_writeback_set(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    """Edit a persistent writeback item by id — the single writeback editing
    surface (ADR-0008). ``selected`` / ``target_name`` apply to any item;
    ``proposed_value`` is the metadict-only facet; ``edits`` is the
    module/waveform-only facet (cfg edits applied through the item's editor
    session internally — the agent never handles its editor_id). ``proposed_value``
    and ``edits`` are mutually exclusive (they target different item kinds);
    None disambiguates which facet is supplied. Echoes the edited ``item``; an
    ``edits`` batch also folds the aggregated ``{valid, removed, added}``."""
    tab_id = str(params["tab_id"])
    if not adapter.ctrl.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    session_id = str(params["id"])
    # The wire collapses "omitted optional" and "explicit JSON null" to the same
    # thing (a null-valued key), so a null here means "not provided" — never a
    # value to write. ``selected``/``target_name`` can never legitimately be null.
    # ``proposed_value`` is only forwarded when present *and* non-null; a metadict
    # item that genuinely needs a null value is out of scope for this surface.
    changes: dict[str, object] = {}
    if params.get("selected") is not None:
        changes["selected"] = bool(params["selected"])
    if params.get("target_name") is not None:
        name = params["target_name"]
        if not isinstance(name, str) or not name:
            raise RemoteError(
                ErrorCode.INVALID_PARAMS, "target_name must be a non-empty string"
            )
        changes["target_name"] = name
    has_proposed = params.get("proposed_value") is not None
    has_edits = params.get("edits") is not None
    if has_proposed and has_edits:
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            "'proposed_value' (metadict) and 'edits' (module/waveform) are "
            "mutually exclusive",
        )
    if has_proposed:
        # Structured tags (e.g. {"__complex__": [re, im]}) coerce back to their
        # Python type so the applied md value matches what preview serialized.
        changes["proposed_value"] = _coerce_wire_value(params["proposed_value"])
    if has_edits:
        raw_edits = params["edits"]
        if not isinstance(raw_edits, list):
            raise RemoteError(ErrorCode.INVALID_PARAMS, "'edits' must be a list")
        edits: list[dict[str, object]] = []
        for i, edit in enumerate(raw_edits):
            if not isinstance(edit, dict) or "path" not in edit or "value" not in edit:
                raise RemoteError(
                    ErrorCode.INVALID_PARAMS,
                    f"edits[{i}] must be an object with 'path' and 'value'",
                )
            edits.append({"path": str(edit["path"]), "value": edit["value"]})
        changes["edits"] = edits
    try:
        agg = adapter.ctrl.set_writeback_item(tab_id, session_id, **changes)
    except RuntimeError as exc:
        raise RemoteError(ErrorCode.INVALID_PARAMS, str(exc)) from exc
    # Echo the edited item so the agent sees the post-edit state in one round-trip.
    item = _find_writeback_item(adapter, tab_id, session_id)
    reply: dict[str, object] = {"item": _writeback_item_wire(item)}
    if has_edits:
        reply.update(agg)
    return reply


def _find_writeback_item(adapter: RemoteControlAdapter, tab_id: str, session_id: str):
    for item in adapter.ctrl.get_tab_writeback_items(tab_id):
        if item.session_id == session_id:
            return item
    raise RemoteError(
        ErrorCode.INVALID_PARAMS, f"unknown writeback item id: {session_id!r}"
    )


def _h_tab_writeback_apply(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    """Apply the tab's persistent writeback draft as-is (edit it first via
    gui_tab_writeback_set_item). Echoes what was written: applied_ids, the
    destination names actually pushed (``written`` by kind), and the post-apply
    ``context`` resource version (apply bumps it once, ADR-0006)."""
    tab_id = str(params["tab_id"])
    if not adapter.ctrl.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    try:
        result = adapter.ctrl.apply_writeback(tab_id)
    except RuntimeError as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason=getattr(exc, "reason_code", ""),
        ) from exc
    # Read the context version AFTER apply so the agent sees the bumped value it
    # can pass back as an expected_versions guard on a follow-up write.
    context_version = adapter.ctrl.resources_versions().get("context", 0)
    return {
        "applied_ids": list(result["applied_ids"]),
        "written": result["written"],
        "context_version": context_version,
    }


# ---------------------------------------------------------------------------
# CfgEditor session handlers (headless ml editing)
# ---------------------------------------------------------------------------


def _h_editor_new(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    from zcu_tools.gui.app.main.services.cfg_editor import CfgEditorError

    from .path_resolver import build_settable_tree

    item_kind = str(params["item_kind"])
    from_name = str(params["from_name"])
    # editor.new is modify-only: it edits an existing ml entry. Creating a blank
    # entry goes through context.ml_create_from_role (role_id='<disc>:blank').
    try:
        editor_id, _ = adapter.ctrl.open_cfg_editor(
            item_kind, discriminator=None, from_name=from_name
        )
    except CfgEditorError as exc:
        raise RemoteError(ErrorCode.INVALID_PARAMS, str(exc)) from exc
    # The agent reads every cfg view as a nested tree (same shape as
    # tab.get_cfg / editor.get), so the open reply carries the freshly-opened
    # draft as {tree} rather than the flat current_paths the session also tracks
    # internally for change-push / set_field diffing.
    root = adapter.ctrl.get_cfg_editor_root(editor_id)
    return {"editor_id": editor_id, "tree": build_settable_tree(root)}


def _h_editor_set_field(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    from zcu_tools.gui.app.main.services.cfg_editor import CfgEditorError

    editor_id = str(params["editor_id"])
    path = str(params["path"])
    value = params["value"]
    # A tab cfg draft is a session owned by the tab_id; editing it while that
    # tab runs is blocked — same guard the human gets via the disabled form
    # (ADR-0013 F11). owner-less / ml-entry sessions are unaffected.
    owner = adapter.ctrl.owner_of_editor(editor_id)
    if owner is not None and adapter.ctrl.get_running_tab_id() == owner:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            f"tab {owner!r} is currently running; cancel the run before editing cfg",
        )
    try:
        return adapter.ctrl.cfg_editor_set_field(editor_id, path, value)
    except CfgEditorError as exc:
        raise RemoteError(ErrorCode.INVALID_PARAMS, str(exc)) from exc
    except RemoteError:
        raise
    except (KeyError, RuntimeError) as exc:
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            str(exc),
            reason=getattr(exc, "reason_code", ""),
        ) from exc


def _h_editor_get(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    from zcu_tools.gui.app.main.services.cfg_editor import CfgEditorError

    from .path_resolver import build_settable_tree

    editor_id = str(params["editor_id"])
    raw_prefix = params.get("prefix")
    prefix = str(raw_prefix) if raw_prefix else None
    # Build the nested current-value tree off the session's live root — the same
    # tree shape tab.get_cfg returns, so the agent reads every cfg view as a tree
    # and edits leaves via editor.set_field (dotted paths). An unknown
    # editor_id raises CfgEditorError from get_cfg_editor_root → INVALID_PARAMS.
    try:
        root = adapter.ctrl.get_cfg_editor_root(editor_id)
    except CfgEditorError as exc:
        raise RemoteError(ErrorCode.INVALID_PARAMS, str(exc)) from exc
    return {"tree": build_settable_tree(root, prefix=prefix)}


def _h_editor_commit(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    from zcu_tools.gui.app.main.services.cfg_editor import CfgEditorError

    editor_id = str(params["editor_id"])
    name = str(params["name"])
    try:
        adapter.ctrl.commit_cfg_editor(editor_id, name)
    except CfgEditorError as exc:
        raise RemoteError(ErrorCode.INVALID_PARAMS, str(exc)) from exc
    except MlEntryValidationError as exc:
        raise RemoteError(ErrorCode.INVALID_PARAMS, str(exc)) from exc
    except RuntimeError as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason=getattr(exc, "reason_code", ""),
        ) from exc
    return {}


def _h_editor_discard(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    from zcu_tools.gui.app.main.services.cfg_editor import CfgEditorError

    editor_id = str(params["editor_id"])
    try:
        adapter.ctrl.discard_cfg_editor(editor_id)
    except CfgEditorError as exc:
        raise RemoteError(ErrorCode.INVALID_PARAMS, str(exc)) from exc
    return {}


# ---------------------------------------------------------------------------
# Notify prompt handlers (ADR-0025 two-RPC design)
#
# notify.open runs on the main thread: it mints a token inside NotifyHandles
# and opens a non-modal dialog via the controller's open_notify_prompt façade,
# which reaches the RenderHost (MainWindow) through _render_host. Returns
# {token} immediately so the off-main notify.await knows which channel to wait on.
#
# notify.await is off_main_thread=True: it blocks the IO worker until the dialog
# fires reply/dismiss/timeout, then folds the result into {reason, reply?}.
# Neither method is in _OP_KEY_OF (not a start-op; no operation_id to capture).
# ---------------------------------------------------------------------------


def _h_notify_open(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    message = str(params["message"])
    timeout = float(params["timeout"])  # type: ignore[arg-type]
    token = adapter.ctrl.open_notify_prompt(message, timeout)
    return {"token": token}


def _h_notify_await(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    # off_main_thread handler: blocks the IO worker on the thread-safe
    # NotifyChannel.consume(). Never touches main-thread-owned state.
    token = int(params["token"])  # type: ignore[arg-type]
    timeout = float(params["timeout"])  # type: ignore[arg-type]
    result = adapter.ctrl.await_notify(token, timeout)
    wire: dict[str, object] = {"reason": result.reason}
    if result.reply is not None:
        wire["reply"] = result.reply
    return wire


# ---------------------------------------------------------------------------
# Run progress handler
# ---------------------------------------------------------------------------


def _h_operation_progress(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    # Live (token, ProgressBarModel) pairs for one operation (run or device
    # setup alike, keyed by operation_id — the SSOT); _progress_bars_wire reads
    # their methods at this point. The mcp poll folds this into its reply.
    operation_id = int(params["operation_id"])  # type: ignore[arg-type]
    return _progress_bars_wire(adapter.ctrl.get_operation_progress(operation_id))


# ---------------------------------------------------------------------------
# Tab current-figure handler (the figure currently shown: a run's 2D map, or an
# analysis fit once analyzed — whichever is on top of the tab's plot stack)
# ---------------------------------------------------------------------------


def _h_tab_get_current_figure(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    import base64
    from pathlib import Path

    tab_id = str(params["tab_id"])
    out_path_raw = params.get("out_path")
    out_path: str | None = str(out_path_raw) if out_path_raw is not None else None
    try:
        png = _render_view(adapter).take_figure_screenshot(tab_id)
    except RuntimeError as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason=getattr(exc, "reason_code", ""),
        ) from exc
    if not isinstance(png, (bytes, bytearray)):
        raise RemoteError(
            ErrorCode.INTERNAL,
            f"figure screenshot returned non-bytes {type(png).__name__}",
        )
    if out_path:
        Path(out_path).write_bytes(png)
        return {"bytes": len(png), "saved_to": out_path}
    return {"png_b64": base64.b64encode(bytes(png)).decode("ascii"), "bytes": len(png)}


# ---------------------------------------------------------------------------
# Predictor handlers
# ---------------------------------------------------------------------------


def _h_predictor_load(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    from zcu_tools.gui.session.services.predictor import (
        LoadPredictorRequest,
        PredictorLoadError,
    )

    path = str(params["path"])
    flux_bias = float(params["flux_bias"])  # type: ignore[arg-type]
    try:
        adapter.ctrl.load_predictor(
            LoadPredictorRequest(path=path, flux_bias=flux_bias)
        )
    except PredictorLoadError as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason=getattr(exc, "reason_code", ""),
        ) from exc
    # Echo the installed model so the agent verifies the load without a follow-up
    # read; get_predictor_info() is non-None right after a successful install (a
    # None here is a broken invariant, so raise rather than echo a half-shape).
    info = adapter.ctrl.get_predictor_info()
    if info is None:
        raise RuntimeError("predictor missing immediately after a successful load")
    return {"loaded": True, **info}


def _h_predictor_set_model_params(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    from zcu_tools.gui.session.services.predictor import (
        PredictorLoadError,
        SetModelParamsRequest,
    )

    req = SetModelParamsRequest(
        EJ=float(params["EJ"]),  # type: ignore[arg-type]
        EC=float(params["EC"]),  # type: ignore[arg-type]
        EL=float(params["EL"]),  # type: ignore[arg-type]
        flux_half=float(params["flux_half"]),  # type: ignore[arg-type]
        flux_period=float(params["flux_period"]),  # type: ignore[arg-type]
        flux_bias=float(params["flux_bias"]),  # type: ignore[arg-type]
    )
    try:
        adapter.ctrl.set_predictor_model_params(req)
    except PredictorLoadError as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason=getattr(exc, "reason_code", ""),
        ) from exc
    # Echo the installed model (path is null — in-memory install has no file); a
    # None right after a successful install is a broken invariant, so raise.
    info = adapter.ctrl.get_predictor_info()
    if info is None:
        raise RuntimeError("predictor missing immediately after a successful install")
    return {"loaded": True, **info}


def _h_predictor_clear(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    adapter.ctrl.clear_predictor()
    return {"loaded": False}


def _h_predictor_predict(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    from zcu_tools.gui.session.services.predictor import (
        PredictFreqRequest,
        PredictorNotLoaded,
    )

    device_value = float(params["device_value"])  # type: ignore[arg-type]
    from_level = int(params["from_level"])  # type: ignore[arg-type]
    to_level = int(params["to_level"])  # type: ignore[arg-type]
    try:
        freq = adapter.ctrl.predict_freq(
            PredictFreqRequest(value=device_value, transition=(from_level, to_level))
        )
    except PredictorNotLoaded as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason=getattr(exc, "reason_code", ""),
        ) from exc
    return {"freq_mhz": freq}


def _h_predictor_info(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    # Flatten the model fields to the top level; the `loaded` flag replaces a null
    # payload so the agent never has to distinguish {info: null} from a real read.
    info = adapter.ctrl.get_predictor_info()
    if info is None:
        return {"loaded": False}
    return {"loaded": True, **info}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


# Each wire method maps to a synchronous handler; the parameter contract,
# timeout and description live in the Qt-free METHOD_SPECS table.
_HANDLERS: dict[str, Handler] = {
    "tab.new": _h_tab_new,
    "tab.close": _h_tab_close,
    "tab.set_active": _h_tab_set_active,
    "tab.list_all": _h_tab_list_all,
    "tab.snapshot": _h_tab_snapshot,
    "tab.get_cfg": _h_tab_get_cfg,
    "tab.set_cfg": _h_tab_set_cfg,
    "tab.run_start": _h_tab_run_start,
    "tab.run_cancel": _h_tab_run_cancel,
    "analyze.cancel": _h_analyze_cancel,
    "run.running_tab": _h_run_running_tab,
    "tab.save_data": _h_tab_save_data,
    "tab.save_image": _h_tab_save_image,
    "tab.save_post_image": _h_tab_save_post_image,
    "tab.save_result": _h_tab_save_result,
    "tab.save_set_paths": _h_tab_save_set_paths,
    "context.use": _h_context_use,
    "context.new": _h_context_new,
    "context.labels": _h_context_labels,
    "context.active": _h_context_active,
    "context.md_get": _h_context_md_get,
    "context.md_get_attr": _h_context_md_get_attr,
    "context.ml_get": _h_context_ml_get,
    "context.md_set_attr": _h_context_md_set_attr,
    "context.md_del_attr": _h_context_md_del_attr,
    "context.ml_del_module": _h_context_ml_del_module,
    "context.ml_del_waveform": _h_context_ml_del_waveform,
    "context.ml_rename_module": _h_context_ml_rename_module,
    "context.ml_rename_waveform": _h_context_ml_rename_waveform,
    "context.ml_list_roles": _h_context_ml_list_roles,
    "context.ml_create_from_role": _h_context_ml_create_from_role,
    "state.has_project": _h_state_has_project,
    "state.has_context": _h_state_has_context,
    "state.has_active_context": _h_state_has_active_context,
    "state.has_soc": _h_state_has_soc,
    "soc.info": _h_soc_info,
    "project.info": _h_project_info,
    "resources.versions": _h_resources_versions,
    "soc.connect": _h_soc_connect,
    "startup.apply": _h_startup_apply,
    "device.connect": _h_device_connect,
    "device.disconnect": _h_device_disconnect,
    "device.reconnect": _h_device_reconnect,
    "device.forget": _h_device_forget,
    "device.setup": _h_device_setup,
    "device.setup_spec": _h_device_setup_spec,
    "device.cancel_operation": _h_device_cancel_operation,
    "device.active_operations": _h_device_active_operations,
    "operation.await": _h_operation_await,
    "operation.progress": _h_operation_progress,
    "device.list": _h_device_list,
    "device.snapshot": _h_device_snapshot,
    "adapter.list": _h_adapter_list,
    "adapter.guide": _h_adapter_guide,
    "app.shutdown": _h_app_shutdown,
    "dialog.screenshot": _h_dialog_screenshot,
    "view.snapshot": _h_view_snapshot,
    "view.screenshot": _h_view_screenshot,
    "tab.get_current_figure": _h_tab_get_current_figure,
    "predictor.load": _h_predictor_load,
    "predictor.set_model_params": _h_predictor_set_model_params,
    "predictor.clear": _h_predictor_clear,
    "predictor.predict": _h_predictor_predict,
    "predictor.info": _h_predictor_info,
    "tab.get_analyze_result": _h_tab_get_analyze_result,
    "tab.get_analyze_params": _h_tab_get_analyze_params,
    "tab.analyze": _h_tab_analyze,
    "tab.get_post_analyze_result": _h_tab_get_post_analyze_result,
    "tab.get_post_analyze_params": _h_tab_get_post_analyze_params,
    "tab.post_analyze": _h_tab_post_analyze,
    "tab.writeback_preview": _h_tab_writeback_preview,
    "tab.writeback_set": _h_tab_writeback_set,
    "tab.writeback_apply": _h_tab_writeback_apply,
    "editor.new": _h_editor_new,
    "editor.set_field": _h_editor_set_field,
    "editor.get": _h_editor_get,
    "editor.commit": _h_editor_commit,
    "editor.discard": _h_editor_discard,
    "notify.open": _h_notify_open,
    "notify.await": _h_notify_await,
}

METHOD_REGISTRY: dict[str, BoundMethod] = build_method_registry(_HANDLERS, METHOD_SPECS)
