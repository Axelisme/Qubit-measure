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
    CfgSchema,
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
from zcu_tools.gui.app.main.services.session_codec import raw_to_schema, schema_to_raw
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
    return {}


def _h_tab_set_active(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    tab_id = str(params["tab_id"])
    if not adapter.ctrl.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    adapter.ctrl.set_active_tab(tab_id)
    return {}


def _h_tab_list(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    tabs = [
        {"tab_id": tid, "adapter_name": adapter.ctrl.get_tab_adapter_name(tid)}
        for tid in adapter.ctrl.list_tab_ids()
    ]
    return {"tabs": tabs}


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
    tab_id_raw = params.get("tab_id")
    if tab_id_raw is None:
        # batch: return all tabs
        return {
            "tabs": [
                _tab_snapshot_wire(adapter, tid) for tid in adapter.ctrl.list_tab_ids()
            ]
        }
    tab_id = str(tab_id_raw)
    if not adapter.ctrl.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    return _tab_snapshot_wire(adapter, tab_id)


def _save_paths_wire(paths) -> dict[str, str] | None:
    if paths is None:
        return None
    return {"data_path": paths.data_path, "image_path": paths.image_path}


def _h_tab_get_cfg(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    tab_id = str(params["tab_id"])
    if not adapter.ctrl.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    schema = adapter.ctrl.get_tab_cfg_schema(tab_id)
    raw = schema_to_raw(schema)
    return {"raw": raw}


def _h_tab_list_paths(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    tab_id = str(params["tab_id"])
    if not adapter.ctrl.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    # A tab's cfg draft is a CfgEditorService session keyed by its tab_id (the
    # same draft the open form attaches to). List its settable paths from that
    # session — the one ``editor.set_field`` mutates — so listed paths are
    # guaranteed settable and agent+user share one model (ADR-0013 F11).
    editor_id = adapter.ctrl.editor_id_for_owner(tab_id)
    if editor_id is None:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            f"tab {tab_id!r} cfg form has no live model yet",
        )
    under, verbosity = _path_view_args(params)
    return {
        "paths": adapter.ctrl.cfg_editor_get(
            editor_id, under=under, verbosity=verbosity
        )
    }


def _h_tab_update_cfg(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    tab_id = str(params["tab_id"])
    if not adapter.ctrl.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    if adapter.ctrl.get_running_tab_id() == tab_id:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            f"tab {tab_id!r} is currently running; cancel the run before editing cfg",
        )
    # ParamSpec(_obj) already validated this is a dict at the wire boundary; cast
    # to narrow for the type checker without a redundant runtime re-check.
    raw = cast(dict, params["raw"])
    base = adapter.ctrl.get_tab_cfg_schema(tab_id)
    try:
        schema: CfgSchema = raw_to_schema(base, dict(raw))
    except Exception as exc:
        raise RemoteError(
            ErrorCode.INVALID_PARAMS, f"invalid cfg payload: {exc}"
        ) from exc
    adapter.ctrl.update_tab_cfg(tab_id, schema)
    return {}


# ---------------------------------------------------------------------------
# Run / Save handlers
# ---------------------------------------------------------------------------


def _h_run_start(
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


def _h_run_cancel(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    adapter.ctrl.cancel_run()
    return {"ok": True}


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


def _h_save_data(
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


def _h_save_image(
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


def _h_save_post_image(
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


def _h_save_result(
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


def _h_save_set_paths(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    tab_id = str(params["tab_id"])
    if not adapter.ctrl.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    data_path = str(params["data_path"])
    image_path = str(params["image_path"])
    adapter.ctrl.update_tab_save_paths(tab_id, data_path, image_path)
    return {}


# ---------------------------------------------------------------------------
# Context / state / session handlers
# ---------------------------------------------------------------------------


def _h_context_use(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    adapter.ctrl.use_context(str(params["label"]))
    return {}


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
            "No project applied yet; apply a project first (gui_startup_apply).",
            reason="no_project",
        )
    bind_device = params["bind_device"]
    clone_from = params["clone_from"]
    adapter.ctrl.new_context(
        bind_device=str(bind_device) if bind_device is not None else None,
        clone_from=str(clone_from) if clone_from is not None else None,
    )
    # new_context makes the new context active — return its label so the agent
    # knows what was created without a follow-up gui_context_active.
    return {"label": adapter.ctrl.get_active_context_label()}


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


def _h_context_get_md(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    md = adapter.ctrl.get_current_md()
    return {"keys": sorted(str(k) for k in md.keys())}


def _h_context_get_md_attr(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    key = str(params["key"])
    md = adapter.ctrl.get_current_md()
    sentinel = object()
    value = md.get(key, sentinel)
    if value is sentinel:
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown md key: {key!r}")
    return {"key": key, "value": _json_safe(value)}


def _h_context_get_ml(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    ml = adapter.ctrl.get_current_ml()
    return {
        "modules": sorted(ml.modules.keys()),
        "waveforms": sorted(ml.waveforms.keys()),
    }


def _h_ml_list_roles(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    """List the experiment-role templates available for create_from_role."""
    del params
    try:
        catalog = adapter.ctrl.get_role_catalog()
    except RuntimeError as exc:
        raise RemoteError(ErrorCode.PRECONDITION_FAILED, str(exc)) from exc
    return {"roles": list(catalog.list_meta())}


def _h_ml_create_from_role(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    """Create a blank ml module/waveform from a named role and register it.

    One-shot: seeds md-linked defaults (lowered against the live md), writes ml.
    Edit afterwards via editor.open(from_name=...).
    """
    item_kind = str(params["item_kind"])
    role_id = str(params["role_id"])
    name = str(params["name"])
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
    return {}


def _h_context_set_md_attr(
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


def _h_context_del_md_attr(
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


def _h_context_del_ml_module(
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
    return {}


def _h_context_rename_ml_module(
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
    return {}


def _h_context_rename_ml_waveform(
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
    return {}


def _h_context_del_ml_waveform(
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
    return {}


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
    del params
    try:
        return adapter.ctrl.get_soc_info()
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
            "No project applied yet; apply a project first (gui_startup_apply).",
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


def _h_connect_start(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    req = coerce_connect_request(params)
    operation_id = adapter.ctrl.start_connect(req)
    return {"operation_id": operation_id}


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
    ok = adapter.ctrl.apply_startup_project(req)
    return {"ok": bool(ok)}


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
        adapter.ctrl.start_reconnect_device(name)
    except RuntimeError as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason=getattr(exc, "reason_code", ""),
        ) from exc
    return {}


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
    return {}


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
    return {}


def _h_adapter_list(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    return {"adapters": list(adapter.ctrl.get_adapter_names())}


def _h_adapter_cfg_spec(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    from .path_resolver import list_spec_paths

    name = str(params["adapter_name"])
    if name not in adapter.ctrl.get_adapter_names():
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown adapter: {name!r}")
    spec = adapter.ctrl.get_adapter_cfg_spec(name)
    return {"paths": list_spec_paths(spec)}


def _h_adapter_analyze_spec(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    name = str(params["adapter_name"])
    if name not in adapter.ctrl.get_adapter_names():
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown adapter: {name!r}")
    return {"params": adapter.ctrl.get_adapter_analyze_params(name)}


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
    devices = [
        {
            "name": e.name,
            "type_name": e.type_name,
            "is_connected": bool(e.is_connected),
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
        return {"snapshot": None}
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
    # by name), each tagged with its kind (connect / disconnect / setup) so the
    # agent knows which device and which operation is live.
    return {
        "active_operations": [
            {
                "device_name": op.device_name,
                "kind": op.kind.value,
                "name": op.snapshot.name,
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
    # (ADR-0025). failed/cancelled are still raised as PRECONDITION_FAILED so
    # existing callers see an error; other reasons are returned as wire data.
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
    if outcome.status in ("failed", "cancelled"):
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            outcome.error or f"operation {outcome.status}",
            reason=outcome.status,
        )
    return {"reason": "completed", "status": outcome.status}


# ---------------------------------------------------------------------------
# Dialog / view-query handlers (Phase 81a)
# ---------------------------------------------------------------------------


def _h_dialog_open(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    from .dialogs import DialogName, parse_dialog_name

    name_raw = str(params["name"])
    try:
        name: DialogName = parse_dialog_name(name_raw)
    except ValueError as exc:
        raise RemoteError(ErrorCode.INVALID_PARAMS, str(exc)) from exc
    _render_view(adapter).open_dialog(name)
    return {"opened": name.value}


def _h_dialog_close(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    from .dialogs import DialogName, parse_dialog_name

    name_raw = str(params["name"])
    try:
        name: DialogName = parse_dialog_name(name_raw)
    except ValueError as exc:
        raise RemoteError(ErrorCode.INVALID_PARAMS, str(exc)) from exc
    _render_view(adapter).close_dialog(name)
    return {"closed": name.value}


def _h_dialog_list_open(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    open_names = [n.value for n in _render_view(adapter).list_open_dialogs()]
    return {"open": open_names}


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


def _h_analyze_start(
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


def _h_post_analyze_start(
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


def _strip_cfg_tags(raw: object) -> object:
    if isinstance(raw, dict):
        kind = raw.get("__kind")
        if kind == "direct":
            return raw.get("value")  # None means unset (ADR-0010)
        elif kind == "disabled":
            return None  # disabled optional ref (ADR-0010)
        elif kind == "eval":
            return raw.get("expr")
        elif kind in ("module_ref", "waveform_ref"):
            return {
                "chosen": raw.get("chosen_key"),
                "value": _strip_cfg_tags(raw.get("value", {})),
            }
        else:
            return {k: _strip_cfg_tags(v) for k, v in raw.items()}
    return raw


def _is_number(value: object) -> bool:
    """A real (non-bool) numeric scalar. In a stripped summary, a sweep edge is
    either a number (resolved) or an expr string (an unresolved EvalValue, whose
    ``{"__kind": "eval"}`` tag _strip_cfg_tags collapsed to the bare expr)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _null_step_on_unresolved_sweeps(summary: object) -> None:
    """Blank out ``step`` on any sweep node whose start/stop is unresolved.

    ``step`` is the derived span/expts value the codec wrote against the lowered
    numeric edges. When an edge is shown as an expr string (an unresolved
    EvalValue in this read-only summary view), that step is stale relative to the
    expression the user actually sees — so it would contradict the start/stop
    span. Report ``None`` there rather than a misleading number; a numeric-edge
    sweep keeps its correct derived step. Mutates the stripped summary in place."""
    if isinstance(summary, dict):
        is_sweep = set(summary) == {"start", "stop", "expts", "step"}
        if is_sweep and not (
            _is_number(summary["start"]) and _is_number(summary["stop"])
        ):
            summary["step"] = None
        for child in summary.values():
            _null_step_on_unresolved_sweeps(child)
    elif isinstance(summary, list):
        for child in summary:
            _null_step_on_unresolved_sweeps(child)


def _h_tab_get_cfg_summary(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    tab_id = str(params["tab_id"])
    if not adapter.ctrl.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    schema = adapter.ctrl.get_tab_cfg_schema(tab_id)
    raw = schema_to_raw(schema)
    summary = _strip_cfg_tags(raw)
    # Projection-layer post-pass (does not touch session_codec, the persistence
    # SSOT): a sweep node with an unresolved (expr-string) edge gets step=None.
    _null_step_on_unresolved_sweeps(summary)
    return {"summary": summary}


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


def _h_writeback_preview(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    tab_id = str(params["tab_id"])
    if not adapter.ctrl.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    items = adapter.ctrl.get_tab_writeback_items(tab_id)
    return {"items": [_writeback_item_wire(it) for it in items]}


def _h_writeback_set(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    """Edit a persistent writeback item by id (selected / target_name /
    metadict proposed_value). Module/waveform cfg edits go through editor.* on
    the item's editor_id, not here."""
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
    if params.get("proposed_value") is not None:
        # Structured tags (e.g. {"__complex__": [re, im]}) coerce back to their
        # Python type so the applied md value matches what preview serialized.
        changes["proposed_value"] = _coerce_wire_value(params["proposed_value"])
    try:
        adapter.ctrl.set_writeback_item(tab_id, session_id, **changes)
    except RuntimeError as exc:
        raise RemoteError(ErrorCode.INVALID_PARAMS, str(exc)) from exc
    return {}


def _h_writeback_apply(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    """Apply the tab's persistent writeback draft as-is (edit it first via
    writeback.set / editor.*)."""
    tab_id = str(params["tab_id"])
    if not adapter.ctrl.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    try:
        applied = adapter.ctrl.apply_writeback(tab_id)
    except RuntimeError as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason=getattr(exc, "reason_code", ""),
        ) from exc
    return {"applied_ids": list(applied)}


# ---------------------------------------------------------------------------
# CfgEditor session handlers (headless ml editing)
# ---------------------------------------------------------------------------


def _h_editor_open(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    from zcu_tools.gui.app.main.services.cfg_editor import CfgEditorError

    item_kind = str(params["item_kind"])
    from_name = str(params["from_name"])
    # editor.open is modify-only: it edits an existing ml entry. Creating a blank
    # entry goes through ml.create_from_role (role_id='<disc>:blank').
    try:
        editor_id, paths = adapter.ctrl.open_cfg_editor(
            item_kind, discriminator=None, from_name=from_name
        )
    except CfgEditorError as exc:
        raise RemoteError(ErrorCode.INVALID_PARAMS, str(exc)) from exc
    return {"editor_id": editor_id, "paths": paths}


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


def _path_view_args(params: Mapping[str, object]) -> tuple[str | None, str]:
    """Extract optional ``under`` (sub-tree root) + ``verbosity`` from params.

    ``verbosity`` defaults to ``full`` at the wire layer (mechanism fidelity);
    the agent-facing compact default is applied by the mcp tool.
    """
    raw_under = params.get("under")
    under = str(raw_under) if raw_under else None
    verbosity = str(params.get("verbosity") or "full")
    return under, verbosity


def _h_editor_get(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    from zcu_tools.gui.app.main.services.cfg_editor import CfgEditorError

    editor_id = str(params["editor_id"])
    under, verbosity = _path_view_args(params)
    try:
        return {
            "paths": adapter.ctrl.cfg_editor_get(
                editor_id, under=under, verbosity=verbosity
            )
        }
    except CfgEditorError as exc:
        raise RemoteError(ErrorCode.INVALID_PARAMS, str(exc)) from exc


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
# Notify prompt handlers (Stage 4b, ADR-0025 two-RPC design)
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
    return {}


def _h_predictor_clear(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    adapter.ctrl.clear_predictor()
    return {}


def _h_predictor_predict(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    from zcu_tools.gui.session.services.predictor import (
        PredictFreqRequest,
        PredictorNotLoaded,
    )

    value = float(params["value"])  # type: ignore[arg-type]
    from_lvl = int(params["from_lvl"])  # type: ignore[arg-type]
    to_lvl = int(params["to_lvl"])  # type: ignore[arg-type]
    try:
        freq = adapter.ctrl.predict_freq(
            PredictFreqRequest(value=value, transition=(from_lvl, to_lvl))
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
    return {"info": adapter.ctrl.get_predictor_info()}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


# Each wire method maps to a synchronous handler; the parameter contract,
# timeout and description live in the Qt-free METHOD_SPECS table.
_HANDLERS: dict[str, Handler] = {
    "tab.new": _h_tab_new,
    "tab.close": _h_tab_close,
    "tab.set_active": _h_tab_set_active,
    "tab.list": _h_tab_list,
    "tab.snapshot": _h_tab_snapshot,
    "tab.get_cfg": _h_tab_get_cfg,
    "tab.list_paths": _h_tab_list_paths,
    "tab.update_cfg": _h_tab_update_cfg,
    "run.start": _h_run_start,
    "run.cancel": _h_run_cancel,
    "analyze.cancel": _h_analyze_cancel,
    "run.running_tab": _h_run_running_tab,
    "save.data": _h_save_data,
    "save.image": _h_save_image,
    "save.post_image": _h_save_post_image,
    "save.result": _h_save_result,
    "save.set_paths": _h_save_set_paths,
    "context.use": _h_context_use,
    "context.new": _h_context_new,
    "context.labels": _h_context_labels,
    "context.active": _h_context_active,
    "context.get_md": _h_context_get_md,
    "context.get_md_attr": _h_context_get_md_attr,
    "context.get_ml": _h_context_get_ml,
    "context.set_md_attr": _h_context_set_md_attr,
    "context.del_md_attr": _h_context_del_md_attr,
    "context.del_ml_module": _h_context_del_ml_module,
    "context.del_ml_waveform": _h_context_del_ml_waveform,
    "context.rename_ml_module": _h_context_rename_ml_module,
    "context.rename_ml_waveform": _h_context_rename_ml_waveform,
    "ml.list_roles": _h_ml_list_roles,
    "ml.create_from_role": _h_ml_create_from_role,
    "state.has_project": _h_state_has_project,
    "state.has_context": _h_state_has_context,
    "state.has_active_context": _h_state_has_active_context,
    "state.has_soc": _h_state_has_soc,
    "soc.info": _h_soc_info,
    "project.info": _h_project_info,
    "resources.versions": _h_resources_versions,
    "connect.start": _h_connect_start,
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
    "adapter.cfg_spec": _h_adapter_cfg_spec,
    "adapter.analyze_spec": _h_adapter_analyze_spec,
    "adapter.guide": _h_adapter_guide,
    "dialog.open": _h_dialog_open,
    "dialog.close": _h_dialog_close,
    "app.shutdown": _h_app_shutdown,
    "dialog.list_open": _h_dialog_list_open,
    "dialog.screenshot": _h_dialog_screenshot,
    "view.snapshot": _h_view_snapshot,
    "tab.get_current_figure": _h_tab_get_current_figure,
    "predictor.load": _h_predictor_load,
    "predictor.clear": _h_predictor_clear,
    "predictor.predict": _h_predictor_predict,
    "predictor.info": _h_predictor_info,
    "tab.get_analyze_result": _h_tab_get_analyze_result,
    "tab.get_analyze_params": _h_tab_get_analyze_params,
    "analyze.start": _h_analyze_start,
    "tab.get_post_analyze_result": _h_tab_get_post_analyze_result,
    "tab.get_post_analyze_params": _h_tab_get_post_analyze_params,
    "post_analyze.start": _h_post_analyze_start,
    "tab.get_cfg_summary": _h_tab_get_cfg_summary,
    "writeback.preview": _h_writeback_preview,
    "writeback.set": _h_writeback_set,
    "writeback.apply": _h_writeback_apply,
    "editor.open": _h_editor_open,
    "editor.set_field": _h_editor_set_field,
    "editor.get": _h_editor_get,
    "editor.commit": _h_editor_commit,
    "editor.discard": _h_editor_discard,
    "notify.open": _h_notify_open,
    "notify.await": _h_notify_await,
}

METHOD_REGISTRY: dict[str, BoundMethod] = build_method_registry(_HANDLERS, METHOD_SPECS)
