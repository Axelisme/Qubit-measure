"""Method dispatcher for RemoteControlService.

Every handler is a pure synchronous function ``(controller, params) -> dict``
that runs on the Qt main thread. The service layer is responsible for
marshalling — handlers must not touch threading or Qt directly.

Adding a method:
  1. Implement ``def _h_<dotted_name>(ctrl, params): ...`` (returns wire dict).
  2. Register it in ``METHOD_REGISTRY`` below.
  3. Document the wire shape in ``AI_NOTE.md``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Mapping, Optional

from zcu_tools.gui.adapter import CfgSchema
from zcu_tools.gui.services.session_persistence import SessionPersistenceService

from .errors import ErrorCode, RemoteError
from .wire import (
    _require_int,
    _require_str,
    coerce_connect_device_request,
    coerce_connect_request,
    coerce_startup_project_request,
    require_object,
)

logger = logging.getLogger(__name__)

Handler = Callable[["object", Mapping[str, object]], Mapping[str, object]]

# Shared serializer instance: we only use its pure schema_to_raw / raw_to_schema
# methods, never its persistence side effects.
_SCHEMA_CODEC = SessionPersistenceService()


# ---------------------------------------------------------------------------
# Method classification (used by the service for per-method timeouts)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MethodSpec:
    handler: Handler
    timeout_seconds: float
    description: str


# ---------------------------------------------------------------------------
# Tab handlers
# ---------------------------------------------------------------------------


def _h_tab_new(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    name = _require_str(params, "adapter_name")
    if name not in ctrl.get_adapter_names():
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown adapter: {name!r}")
    tab_id = ctrl.new_tab(name)
    return {"tab_id": tab_id}


def _h_tab_close(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    tab_id = _require_str(params, "tab_id")
    if not ctrl.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    ctrl.close_tab(tab_id)
    return {}


def _h_tab_set_active(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    tab_id = _require_str(params, "tab_id")
    if not ctrl.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    ctrl.set_active_tab(tab_id)
    return {}


def _h_tab_list(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    del params
    tabs = [
        {"tab_id": tid, "adapter_name": ctrl.get_tab_adapter_name(tid)}
        for tid in ctrl.list_tab_ids()
    ]
    return {"tabs": tabs}


def _h_tab_snapshot(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    tab_id = _require_str(params, "tab_id")
    if not ctrl.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    snap = ctrl.get_tab_snapshot(tab_id)
    interaction = snap.interaction
    return {
        "tab_id": tab_id,
        "adapter_name": ctrl.get_tab_adapter_name(tab_id),
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


def _save_paths_wire(paths) -> Optional[dict[str, str]]:
    if paths is None:
        return None
    return {"data_path": paths.data_path, "image_path": paths.image_path}


def _h_tab_get_cfg(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    tab_id = _require_str(params, "tab_id")
    if not ctrl.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    schema = ctrl.get_tab_cfg_schema(tab_id)
    raw = _SCHEMA_CODEC.schema_to_raw(schema, ml=None)
    return {"raw": raw}


def _h_tab_update_cfg(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    tab_id = _require_str(params, "tab_id")
    if not ctrl.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    raw = require_object(params, "raw")
    base = ctrl.get_tab_cfg_schema(tab_id)
    try:
        schema: CfgSchema = _SCHEMA_CODEC.raw_to_schema(base, dict(raw))
    except Exception as exc:
        raise RemoteError(
            ErrorCode.INVALID_PARAMS, f"invalid cfg payload: {exc}"
        ) from exc
    ctrl.update_tab_cfg(tab_id, schema)
    return {}


# ---------------------------------------------------------------------------
# Run / Save handlers
# ---------------------------------------------------------------------------


def _h_run_start(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    tab_id = _require_str(params, "tab_id")
    if not ctrl.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    try:
        ctrl.start_run(tab_id)
    except RuntimeError as exc:
        raise RemoteError(ErrorCode.PRECONDITION_FAILED, str(exc)) from exc
    return {}


def _h_run_cancel(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    del params
    ctrl.cancel_run()
    return {}


def _h_run_running_tab(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    del params
    return {"tab_id": ctrl.get_running_tab_id()}


def _h_save_data(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    tab_id = _require_str(params, "tab_id")
    data_path = _require_str(params, "data_path")
    try:
        ctrl.save_data(tab_id, data_path)
    except RuntimeError as exc:
        raise RemoteError(ErrorCode.PRECONDITION_FAILED, str(exc)) from exc
    return {}


def _h_save_image(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    tab_id = _require_str(params, "tab_id")
    image_path = _require_str(params, "image_path")
    try:
        ctrl.save_image(tab_id, image_path)
    except RuntimeError as exc:
        raise RemoteError(ErrorCode.PRECONDITION_FAILED, str(exc)) from exc
    return {}


def _h_save_both(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    tab_id = _require_str(params, "tab_id")
    data_path = _require_str(params, "data_path")
    image_path = _require_str(params, "image_path")
    try:
        ctrl.save_both(tab_id, data_path, image_path)
    except RuntimeError as exc:
        raise RemoteError(ErrorCode.PRECONDITION_FAILED, str(exc)) from exc
    return {}


# ---------------------------------------------------------------------------
# Context / state / session handlers
# ---------------------------------------------------------------------------


def _h_context_use(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    ctrl.use_context(_require_str(params, "label"))
    return {}


def _h_context_new(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    value = params.get("value")
    if value is not None and not isinstance(value, (int, float)):
        raise RemoteError(
            ErrorCode.INVALID_PARAMS, "'value' must be a number if present"
        )
    unit = params.get("unit", "A")
    if not isinstance(unit, str):
        raise RemoteError(ErrorCode.INVALID_PARAMS, "'unit' must be a string")
    clone = params.get("clone_from_current", False)
    if not isinstance(clone, bool):
        raise RemoteError(
            ErrorCode.INVALID_PARAMS, "'clone_from_current' must be boolean"
        )
    ctrl.new_context(
        value=float(value) if value is not None else None,
        unit=unit,
        clone_from_current=clone,
    )
    return {}


def _h_context_labels(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    del params
    return {"labels": list(ctrl.get_context_labels())}


def _h_context_active(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    del params
    return {"label": ctrl.get_active_context_label()}


def _h_state_has_project(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    del params
    return {"value": bool(ctrl.has_project())}


def _h_state_has_context(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    del params
    return {"value": bool(ctrl.has_context())}


def _h_state_has_active_context(
    ctrl, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    return {"value": bool(ctrl.has_active_context())}


def _h_state_has_soc(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    del params
    return {"value": bool(ctrl.has_soc())}


def _h_session_persist(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    del params
    ctrl.persist_tabs_session()
    return {}


def _h_session_restore(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    del params
    ctrl.restore_tabs_from_session()
    return {}


# ---------------------------------------------------------------------------
# Connection / startup / device handlers (typed-request coercion)
# ---------------------------------------------------------------------------


def _h_connect_start(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    req = coerce_connect_request(params)
    ctrl.start_connect(req)
    return {}


def _h_startup_apply(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    req = coerce_startup_project_request(params)
    ok = ctrl.apply_startup_project(req)
    return {"ok": bool(ok)}


def _h_device_connect(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    req = coerce_connect_device_request(params)
    ctrl.start_connect_device(req)
    return {}


def _h_adapter_list(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    del params
    return {"adapters": list(ctrl.get_adapter_names())}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


# `auth` is a sentinel handled by the service before the registry — left out here.
METHOD_REGISTRY: dict[str, MethodSpec] = {
    "tab.new": MethodSpec(_h_tab_new, 10.0, "Create a new tab"),
    "tab.close": MethodSpec(_h_tab_close, 5.0, "Close a tab"),
    "tab.set_active": MethodSpec(_h_tab_set_active, 5.0, "Activate a tab"),
    "tab.list": MethodSpec(_h_tab_list, 5.0, "List tabs"),
    "tab.snapshot": MethodSpec(_h_tab_snapshot, 5.0, "Tab summary"),
    "tab.get_cfg": MethodSpec(_h_tab_get_cfg, 5.0, "Read tab cfg raw"),
    "tab.update_cfg": MethodSpec(_h_tab_update_cfg, 10.0, "Replace tab cfg raw"),
    "run.start": MethodSpec(_h_run_start, 5.0, "Start a run (fire-and-forget)"),
    "run.cancel": MethodSpec(_h_run_cancel, 5.0, "Cancel current run"),
    "run.running_tab": MethodSpec(_h_run_running_tab, 5.0, "Current running tab"),
    "save.data": MethodSpec(_h_save_data, 30.0, "Save data file"),
    "save.image": MethodSpec(_h_save_image, 30.0, "Save image file"),
    "save.both": MethodSpec(_h_save_both, 30.0, "Save data and image"),
    "context.use": MethodSpec(_h_context_use, 5.0, "Switch context"),
    "context.new": MethodSpec(_h_context_new, 10.0, "Create new context"),
    "context.labels": MethodSpec(_h_context_labels, 5.0, "List context labels"),
    "context.active": MethodSpec(_h_context_active, 5.0, "Active context label"),
    "state.has_project": MethodSpec(_h_state_has_project, 5.0, ""),
    "state.has_context": MethodSpec(_h_state_has_context, 5.0, ""),
    "state.has_active_context": MethodSpec(_h_state_has_active_context, 5.0, ""),
    "state.has_soc": MethodSpec(_h_state_has_soc, 5.0, ""),
    "session.persist": MethodSpec(_h_session_persist, 10.0, "Persist tab session"),
    "session.restore": MethodSpec(_h_session_restore, 10.0, "Restore tab session"),
    "connect.start": MethodSpec(_h_connect_start, 30.0, "Connect to SoC"),
    "startup.apply": MethodSpec(_h_startup_apply, 30.0, "Apply startup project"),
    "device.connect": MethodSpec(_h_device_connect, 30.0, "Connect device"),
    "adapter.list": MethodSpec(_h_adapter_list, 5.0, "List available adapters"),
}


# Silence "unused" hint for helpers re-exported by Phase 81 dispatch additions.
_ = _require_int  # noqa: PLW0123
