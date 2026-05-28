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
from zcu_tools.gui.services.context import MlEntryValidationError
from zcu_tools.gui.services.device import SetupDeviceRequest
from zcu_tools.gui.services.session_persistence import SessionPersistenceService

from .errors import ErrorCode, RemoteError
from .wire import (
    _require_int,
    _require_str,
    coerce_connect_device_request,
    coerce_connect_request,
    coerce_disconnect_device_request,
    coerce_set_device_value_request,
    coerce_startup_project_request,
    require_json_safe,
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


def _tab_snapshot_wire(ctrl, tab_id: str) -> dict[str, object]:
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


def _h_tab_snapshot(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    tab_id_raw = params.get("tab_id")
    if tab_id_raw is None:
        # batch: return all tabs
        return {"tabs": [_tab_snapshot_wire(ctrl, tid) for tid in ctrl.list_tab_ids()]}
    if not isinstance(tab_id_raw, str):
        raise RemoteError(ErrorCode.INVALID_PARAMS, "'tab_id' must be a string")
    if not ctrl.has_tab(tab_id_raw):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id_raw!r}")
    return _tab_snapshot_wire(ctrl, tab_id_raw)


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
    if ctrl.get_running_tab_id() == tab_id:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            f"tab {tab_id!r} is currently running; cancel the run before editing cfg",
        )
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


def _h_cfg_set_field(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    tab_id = _require_str(params, "tab_id")
    if not ctrl.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    if ctrl.get_running_tab_id() == tab_id:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            f"tab {tab_id!r} is currently running; cancel the run before editing cfg",
        )
    path = _require_str(params, "path")
    if "value" not in params:
        raise RemoteError(ErrorCode.INVALID_PARAMS, "missing 'value'")
    value = params["value"]
    try:
        ctrl.set_tab_field(tab_id, path, value)
    except (KeyError, RuntimeError) as exc:
        raise RemoteError(ErrorCode.PRECONDITION_FAILED, str(exc)) from exc
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
    data_path: Optional[str] = None
    if "data_path" in params and params["data_path"] is not None:
        data_path = _require_str(params, "data_path")
    comment = str(params.get("comment", ""))
    try:
        ctrl.save_data(tab_id, data_path, comment=comment)
    except RuntimeError as exc:
        raise RemoteError(ErrorCode.PRECONDITION_FAILED, str(exc)) from exc
    return {}


def _h_save_image(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    tab_id = _require_str(params, "tab_id")
    image_path: Optional[str] = None
    if "image_path" in params and params["image_path"] is not None:
        image_path = _require_str(params, "image_path")
    try:
        ctrl.save_image(tab_id, image_path)
    except RuntimeError as exc:
        raise RemoteError(ErrorCode.PRECONDITION_FAILED, str(exc)) from exc
    return {}


def _h_save_both(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    tab_id = _require_str(params, "tab_id")
    data_path: Optional[str] = None
    image_path: Optional[str] = None
    if "data_path" in params and params["data_path"] is not None:
        data_path = _require_str(params, "data_path")
    if "image_path" in params and params["image_path"] is not None:
        image_path = _require_str(params, "image_path")
    comment = str(params.get("comment", ""))
    try:
        ctrl.save_both(tab_id, data_path, image_path, comment=comment)
    except RuntimeError as exc:
        raise RemoteError(ErrorCode.PRECONDITION_FAILED, str(exc)) from exc
    return {}


def _h_save_set_paths(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    tab_id = _require_str(params, "tab_id")
    if not ctrl.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    data_path = _require_str(params, "data_path")
    image_path = _require_str(params, "image_path")
    ctrl.update_tab_save_paths(tab_id, data_path, image_path)
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


def _json_safe(value: object) -> object:
    """Return ``value`` if it round-trips through JSON, else its ``repr``."""
    import json

    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return {"__repr__": repr(value)}


def _h_context_get_md(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    del params
    md = ctrl.get_current_md()
    return {"keys": sorted(str(k) for k in md.keys())}


def _h_context_get_md_attr(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    key = _require_str(params, "key")
    md = ctrl.get_current_md()
    sentinel = object()
    value = md.get(key, sentinel)
    if value is sentinel:
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown md key: {key!r}")
    return {"key": key, "value": _json_safe(value)}


def _h_context_get_ml(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    del params
    ml = ctrl.get_current_ml()
    return {
        "modules": sorted(ml.modules.keys()),
        "waveforms": sorted(ml.waveforms.keys()),
    }


def _h_context_set_md_attr(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    key = _require_str(params, "key")
    value = require_json_safe(params, "value")
    try:
        ctrl.set_md_attr(key, value)
    except RuntimeError as exc:
        raise RemoteError(ErrorCode.PRECONDITION_FAILED, str(exc)) from exc
    return {}


def _h_context_del_md_attr(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    key = _require_str(params, "key")
    try:
        ctrl.del_md_attr(key)
    except (AttributeError, RuntimeError) as exc:
        raise RemoteError(ErrorCode.PRECONDITION_FAILED, str(exc)) from exc
    return {}


def _h_context_set_ml_module(
    ctrl, params: Mapping[str, object]
) -> Mapping[str, object]:
    name = _require_str(params, "name")
    raw = require_object(params, "raw")
    try:
        ctrl.set_ml_module_from_raw(name, dict(raw))
    except MlEntryValidationError as exc:
        raise RemoteError(ErrorCode.INVALID_PARAMS, str(exc)) from exc
    except RuntimeError as exc:
        raise RemoteError(ErrorCode.PRECONDITION_FAILED, str(exc)) from exc
    return {}


def _h_context_del_ml_module(
    ctrl, params: Mapping[str, object]
) -> Mapping[str, object]:
    name = _require_str(params, "name")
    try:
        ctrl.del_ml_module(name)
    except (KeyError, RuntimeError) as exc:
        raise RemoteError(ErrorCode.PRECONDITION_FAILED, str(exc)) from exc
    return {}


def _h_context_set_ml_waveform(
    ctrl, params: Mapping[str, object]
) -> Mapping[str, object]:
    name = _require_str(params, "name")
    raw = require_object(params, "raw")
    try:
        ctrl.set_ml_waveform_from_raw(name, dict(raw))
    except MlEntryValidationError as exc:
        raise RemoteError(ErrorCode.INVALID_PARAMS, str(exc)) from exc
    except RuntimeError as exc:
        raise RemoteError(ErrorCode.PRECONDITION_FAILED, str(exc)) from exc
    return {}


def _h_context_del_ml_waveform(
    ctrl, params: Mapping[str, object]
) -> Mapping[str, object]:
    name = _require_str(params, "name")
    try:
        ctrl.del_ml_waveform(name)
    except (KeyError, RuntimeError) as exc:
        raise RemoteError(ErrorCode.PRECONDITION_FAILED, str(exc)) from exc
    return {}


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
    try:
        ctrl.start_connect_device(req)
    except RuntimeError as exc:
        raise RemoteError(ErrorCode.PRECONDITION_FAILED, str(exc)) from exc
    return {}


def _h_device_disconnect(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    req = coerce_disconnect_device_request(params)
    try:
        ctrl.start_disconnect_device(req)
    except RuntimeError as exc:
        raise RemoteError(ErrorCode.PRECONDITION_FAILED, str(exc)) from exc
    return {}


def _h_device_reconnect(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    name = _require_str(params, "name")
    try:
        ctrl.start_reconnect_device(name)
    except RuntimeError as exc:
        raise RemoteError(ErrorCode.PRECONDITION_FAILED, str(exc)) from exc
    return {}


def _h_device_forget(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    name = _require_str(params, "name")
    try:
        ctrl.forget_device(name)
    except RuntimeError as exc:
        raise RemoteError(ErrorCode.PRECONDITION_FAILED, str(exc)) from exc
    return {}


def _h_device_set_value(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    req = coerce_set_device_value_request(params)
    try:
        ctrl.start_set_device_value(req)
    except RuntimeError as exc:
        raise RemoteError(ErrorCode.PRECONDITION_FAILED, str(exc)) from exc
    return {}


def _h_device_setup(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    name = _require_str(params, "name")
    updates = require_object(params, "updates")
    try:
        info = ctrl.get_device_info(name)
    except RuntimeError as exc:
        raise RemoteError(ErrorCode.PRECONDITION_FAILED, str(exc)) from exc
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
        ctrl.start_setup_device(SetupDeviceRequest(name=name, info=updated))
    except RuntimeError as exc:
        raise RemoteError(ErrorCode.PRECONDITION_FAILED, str(exc)) from exc
    return {}


def _h_device_cancel_operation(
    ctrl, params: Mapping[str, object]
) -> Mapping[str, object]:
    name = _require_str(params, "name")
    try:
        ctrl.cancel_device_operation(name)
    except RuntimeError as exc:
        raise RemoteError(ErrorCode.PRECONDITION_FAILED, str(exc)) from exc
    return {}


def _h_adapter_list(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    del params
    return {"adapters": list(ctrl.get_adapter_names())}


def _h_device_list(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    del params
    devices = [
        {
            "name": e.name,
            "type_name": e.type_name,
            "is_connected": bool(e.is_connected),
        }
        for e in ctrl.list_devices()
    ]
    return {"devices": devices}


def _h_device_snapshot(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    name = _require_str(params, "name")
    snap = ctrl.get_device_snapshot(name)
    if snap is None:
        return {"snapshot": None}
    # BaseDeviceInfo is intentionally omitted (not JSON-friendly).
    return {
        "snapshot": {
            "name": snap.name,
            "type_name": snap.type_name,
            "address": snap.address,
            "status": snap.status.value,
            "error": snap.error,
        }
    }


def _progress_wire(progress) -> list[dict[str, object]]:
    return [
        {
            "token": entry.token,
            "format": entry.format,
            "maximum": entry.maximum,
            "value": entry.value,
        }
        for entry in progress
    ]


def _h_device_active_setup(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    del params
    setup = ctrl.get_active_device_setup()
    if setup is None:
        return {"active_setup": None}
    return {
        "active_setup": {
            "device_name": setup.device_name,
            "progress": _progress_wire(setup.progress),
        }
    }


def _h_device_active_operation(
    ctrl, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    snap = ctrl.get_active_device_operation()
    if snap is None:
        return {"active_operation": None}
    return {
        "active_operation": {
            "name": snap.name,
            "type_name": snap.type_name,
            "address": snap.address,
            "status": snap.status.value,
            "error": snap.error,
            "progress": _progress_wire(snap.progress),
        }
    }


def _h_device_wait_setup(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    name = _require_str(params, "name")
    timeout_raw = params.get("timeout", 120.0)
    if not isinstance(timeout_raw, (int, float)):
        raise RemoteError(ErrorCode.INVALID_PARAMS, "'timeout' must be a number")
    timeout = float(timeout_raw)
    try:
        ctrl.wait_device_setup_done(name, timeout)
    except RuntimeError as exc:
        raise RemoteError(ErrorCode.PRECONDITION_FAILED, str(exc)) from exc
    return {"done": True}


# ---------------------------------------------------------------------------
# Dialog / view-query handlers (Phase 81a)
# ---------------------------------------------------------------------------


def _h_dialog_open(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    from .dialogs import DialogName, parse_dialog_name

    name_raw = _require_str(params, "name")
    try:
        name: DialogName = parse_dialog_name(name_raw)
    except ValueError as exc:
        raise RemoteError(ErrorCode.INVALID_PARAMS, str(exc)) from exc
    ctrl.open_dialog(name)
    return {"opened": name.value}


def _h_dialog_close(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    from .dialogs import DialogName, parse_dialog_name

    name_raw = _require_str(params, "name")
    try:
        name: DialogName = parse_dialog_name(name_raw)
    except ValueError as exc:
        raise RemoteError(ErrorCode.INVALID_PARAMS, str(exc)) from exc
    ctrl.close_dialog(name)
    return {"closed": name.value}


def _h_dialog_list_open(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    del params
    open_names = [n.value for n in ctrl.list_open_dialogs()]
    return {"open": open_names}


def _h_view_snapshot(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    del params
    snap = ctrl.get_view_snapshot()
    if not isinstance(snap, dict):
        raise RemoteError(
            ErrorCode.INTERNAL,
            f"view snapshot returned non-dict {type(snap).__name__}",
        )
    return snap


def _h_dialog_screenshot(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    import base64

    from .dialogs import parse_dialog_name

    name_str = _require_str(params, "dialog_name")
    try:
        dialog_name = parse_dialog_name(name_str)
        png = ctrl.take_dialog_screenshot(dialog_name)
    except (ValueError, RuntimeError) as exc:
        raise RemoteError(ErrorCode.PRECONDITION_FAILED, str(exc)) from exc
    if not isinstance(png, (bytes, bytearray)):
        raise RemoteError(
            ErrorCode.INTERNAL,
            f"screenshot returned non-bytes {type(png).__name__}",
        )
    payload = base64.b64encode(bytes(png)).decode("ascii")
    return {"png_b64": payload, "bytes": len(png)}


def _h_view_screenshot(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    import base64

    tab_id: Optional[str] = None
    if "tab_id" in params and params["tab_id"] is not None:
        if not isinstance(params["tab_id"], str):
            raise RemoteError(
                ErrorCode.INVALID_PARAMS, "'tab_id' must be a string if present"
            )
        tab_id = params["tab_id"]
    try:
        png = ctrl.take_screenshot(tab_id)
    except RuntimeError as exc:
        raise RemoteError(ErrorCode.PRECONDITION_FAILED, str(exc)) from exc
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
    ctrl, params: Mapping[str, object]
) -> Mapping[str, object]:
    tab_id = _require_str(params, "tab_id")
    if not ctrl.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    result = ctrl.get_tab_analyze_result(tab_id)
    if result is None:
        return {"summary": None}
    if not hasattr(result, "to_summary_dict"):
        raise RemoteError(
            ErrorCode.INTERNAL,
            "analyze result does not implement to_summary_dict()",
        )
    return {"summary": result.to_summary_dict()}


def _h_tab_get_analyze_params(
    ctrl, params: Mapping[str, object]
) -> Mapping[str, object]:
    import dataclasses

    tab_id = _require_str(params, "tab_id")
    if not ctrl.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    snap = ctrl.get_tab_snapshot(tab_id)
    if snap.analyze_params is None:
        return {"analyze_params": None}
    ap = snap.analyze_params
    if not dataclasses.is_dataclass(ap) or isinstance(ap, type):
        return {"analyze_params": {}}
    return {"analyze_params": dataclasses.asdict(ap)}


def _h_analyze_start(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    import dataclasses

    tab_id = _require_str(params, "tab_id")
    if not ctrl.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    snap = ctrl.get_tab_snapshot(tab_id)
    if snap.analyze_params is None:
        raise RemoteError(ErrorCode.PRECONDITION_FAILED, "no analyze params available")
    raw_updates = params.get("updates", {})
    if not isinstance(raw_updates, dict):
        raise RemoteError(ErrorCode.INVALID_PARAMS, "'updates' must be a dict")
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
        ctrl.analyze(tab_id, updated)
    except RuntimeError as exc:
        raise RemoteError(ErrorCode.PRECONDITION_FAILED, str(exc)) from exc
    return {}


def _strip_cfg_tags(raw: object) -> object:
    if isinstance(raw, dict):
        kind = raw.get("__kind")
        if kind == "direct":
            return raw.get("value") if not raw.get("is_unset") else None
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


def _h_tab_get_cfg_summary(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    tab_id = _require_str(params, "tab_id")
    if not ctrl.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    schema = ctrl.get_tab_cfg_schema(tab_id)
    raw = _SCHEMA_CODEC.schema_to_raw(schema, ml=None)
    return {"summary": _strip_cfg_tags(raw)}


# ---------------------------------------------------------------------------
# Run progress handler
# ---------------------------------------------------------------------------


def _h_run_progress(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    del params
    bars = ctrl.get_run_progress()
    if not bars:
        return {"active": False, "bars": []}
    return {
        "active": True,
        "bars": [
            {
                "token": s.token,
                "desc": s.desc,
                "n": s.n,
                "total": s.total,
                "elapsed": s.elapsed,
                "remaining": s.remaining,
                "format": s.format,
            }
            for s in bars
        ],
    }


# ---------------------------------------------------------------------------
# Tab figure screenshot handler
# ---------------------------------------------------------------------------


def _h_tab_figure_screenshot(
    ctrl, params: Mapping[str, object]
) -> Mapping[str, object]:
    import base64
    from pathlib import Path

    tab_id = _require_str(params, "tab_id")
    out_path: Optional[str] = None
    if "out_path" in params and params["out_path"] is not None:
        if not isinstance(params["out_path"], str):
            raise RemoteError(
                ErrorCode.INVALID_PARAMS, "'out_path' must be a string if present"
            )
        out_path = params["out_path"]
    try:
        png = ctrl.take_figure_screenshot(tab_id)
    except RuntimeError as exc:
        raise RemoteError(ErrorCode.PRECONDITION_FAILED, str(exc)) from exc
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


def _h_predictor_load(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    from zcu_tools.gui.services.connection import (
        LoadPredictorRequest,
        PredictorLoadError,
    )

    path = _require_str(params, "path")
    flux_bias_raw = params.get("flux_bias", 0.0)
    if not isinstance(flux_bias_raw, (int, float)):
        raise RemoteError(ErrorCode.INVALID_PARAMS, "'flux_bias' must be a number")
    try:
        ctrl.load_predictor(
            LoadPredictorRequest(path=path, flux_bias=float(flux_bias_raw))
        )
    except PredictorLoadError as exc:
        raise RemoteError(ErrorCode.PRECONDITION_FAILED, str(exc)) from exc
    return {}


def _h_predictor_clear(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    del params
    ctrl.clear_predictor()
    return {}


def _h_predictor_predict(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    from zcu_tools.gui.services.connection import PredictFreqRequest, PredictorNotLoaded

    value_raw = params.get("value")
    if not isinstance(value_raw, (int, float)):
        raise RemoteError(ErrorCode.INVALID_PARAMS, "'value' is required (number)")
    from_lvl_raw = params.get("from_lvl", 0)
    to_lvl_raw = params.get("to_lvl", 1)
    if not isinstance(from_lvl_raw, int) or not isinstance(to_lvl_raw, int):
        raise RemoteError(
            ErrorCode.INVALID_PARAMS, "'from_lvl' and 'to_lvl' must be integers"
        )
    try:
        freq = ctrl.predict_freq(
            PredictFreqRequest(
                value=float(value_raw),
                transition=(int(from_lvl_raw), int(to_lvl_raw)),
            )
        )
    except PredictorNotLoaded as exc:
        raise RemoteError(ErrorCode.PRECONDITION_FAILED, str(exc)) from exc
    return {"freq_mhz": freq}


def _h_predictor_info(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    del params
    return {"info": ctrl.get_predictor_info()}


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
    "cfg.set_field": MethodSpec(
        _h_cfg_set_field, 5.0, "Set a single cfg field by dotted path"
    ),
    "run.start": MethodSpec(_h_run_start, 5.0, "Start a run (fire-and-forget)"),
    "run.cancel": MethodSpec(_h_run_cancel, 5.0, "Cancel current run"),
    "run.running_tab": MethodSpec(_h_run_running_tab, 5.0, "Current running tab"),
    "save.data": MethodSpec(_h_save_data, 30.0, "Save data file"),
    "save.image": MethodSpec(_h_save_image, 30.0, "Save image file"),
    "save.both": MethodSpec(_h_save_both, 30.0, "Save data and image"),
    "save.set_paths": MethodSpec(_h_save_set_paths, 5.0, "Set tab save path overrides"),
    "context.use": MethodSpec(_h_context_use, 5.0, "Switch context"),
    "context.new": MethodSpec(_h_context_new, 10.0, "Create new context"),
    "context.labels": MethodSpec(_h_context_labels, 5.0, "List context labels"),
    "context.active": MethodSpec(_h_context_active, 5.0, "Active context label"),
    "context.get_md": MethodSpec(_h_context_get_md, 5.0, "List MetaDict keys"),
    "context.get_md_attr": MethodSpec(
        _h_context_get_md_attr, 5.0, "Read one MetaDict attribute"
    ),
    "context.get_ml": MethodSpec(
        _h_context_get_ml, 5.0, "List ModuleLibrary module/waveform names"
    ),
    "context.set_md_attr": MethodSpec(
        _h_context_set_md_attr, 5.0, "Set one MetaDict attribute"
    ),
    "context.del_md_attr": MethodSpec(
        _h_context_del_md_attr, 5.0, "Delete one MetaDict attribute"
    ),
    "context.set_ml_module": MethodSpec(
        _h_context_set_ml_module, 10.0, "Set one ModuleLibrary module from raw dict"
    ),
    "context.del_ml_module": MethodSpec(
        _h_context_del_ml_module, 5.0, "Delete one ModuleLibrary module"
    ),
    "context.set_ml_waveform": MethodSpec(
        _h_context_set_ml_waveform,
        10.0,
        "Set one ModuleLibrary waveform from raw dict",
    ),
    "context.del_ml_waveform": MethodSpec(
        _h_context_del_ml_waveform, 5.0, "Delete one ModuleLibrary waveform"
    ),
    "state.has_project": MethodSpec(_h_state_has_project, 5.0, ""),
    "state.has_context": MethodSpec(_h_state_has_context, 5.0, ""),
    "state.has_active_context": MethodSpec(_h_state_has_active_context, 5.0, ""),
    "state.has_soc": MethodSpec(_h_state_has_soc, 5.0, ""),
    "session.persist": MethodSpec(_h_session_persist, 10.0, "Persist tab session"),
    "session.restore": MethodSpec(_h_session_restore, 10.0, "Restore tab session"),
    "connect.start": MethodSpec(_h_connect_start, 30.0, "Connect to SoC"),
    "startup.apply": MethodSpec(_h_startup_apply, 30.0, "Apply startup project"),
    "device.connect": MethodSpec(_h_device_connect, 30.0, "Connect device"),
    "device.disconnect": MethodSpec(_h_device_disconnect, 30.0, "Disconnect device"),
    "device.reconnect": MethodSpec(_h_device_reconnect, 30.0, "Reconnect device"),
    "device.forget": MethodSpec(_h_device_forget, 5.0, "Forget memory-only device"),
    "device.set_value": MethodSpec(_h_device_set_value, 30.0, "Set device value"),
    "device.setup": MethodSpec(_h_device_setup, 30.0, "Setup device"),
    "device.cancel_operation": MethodSpec(
        _h_device_cancel_operation, 5.0, "Cancel active device setup"
    ),
    "device.active_setup": MethodSpec(
        _h_device_active_setup, 5.0, "Read active device setup progress"
    ),
    "device.active_operation": MethodSpec(
        _h_device_active_operation, 5.0, "Read active device operation"
    ),
    "device.wait_setup": MethodSpec(
        _h_device_wait_setup, 130.0, "Block until device setup completes"
    ),
    "device.list": MethodSpec(_h_device_list, 5.0, "List registered devices"),
    "device.snapshot": MethodSpec(
        _h_device_snapshot, 5.0, "Read one device cached snapshot"
    ),
    "adapter.list": MethodSpec(_h_adapter_list, 5.0, "List available adapters"),
    "dialog.open": MethodSpec(_h_dialog_open, 10.0, "Open a named dialog"),
    "dialog.close": MethodSpec(_h_dialog_close, 5.0, "Close a named dialog"),
    "dialog.list_open": MethodSpec(_h_dialog_list_open, 5.0, "List open dialogs"),
    "dialog.screenshot": MethodSpec(
        _h_dialog_screenshot, 10.0, "Capture a named dialog as base64 PNG"
    ),
    "view.snapshot": MethodSpec(_h_view_snapshot, 5.0, "Capture view state summary"),
    "view.screenshot": MethodSpec(
        _h_view_screenshot, 10.0, "Capture window or tab as base64 PNG"
    ),
    "run.progress": MethodSpec(
        _h_run_progress, 5.0, "Read current run progress bar snapshots"
    ),
    "tab.figure_screenshot": MethodSpec(
        _h_tab_figure_screenshot, 10.0, "Capture tab figure area as PNG"
    ),
    "predictor.load": MethodSpec(_h_predictor_load, 30.0, "Load FluxoniumPredictor"),
    "predictor.clear": MethodSpec(_h_predictor_clear, 5.0, "Clear predictor"),
    "predictor.predict": MethodSpec(
        _h_predictor_predict, 10.0, "Predict transition frequency"
    ),
    "predictor.info": MethodSpec(_h_predictor_info, 5.0, "Get predictor info"),
    "tab.get_analyze_result": MethodSpec(
        _h_tab_get_analyze_result, 5.0, "Read tab analyze result scalar summary"
    ),
    "tab.get_analyze_params": MethodSpec(
        _h_tab_get_analyze_params, 5.0, "Read current analyze params"
    ),
    "analyze.start": MethodSpec(
        _h_analyze_start, 30.0, "Start analyze (fire-and-forget)"
    ),
    "tab.get_cfg_summary": MethodSpec(
        _h_tab_get_cfg_summary, 5.0, "Read tab cfg as clean scalar dict"
    ),
}


# Silence "unused" hint for helpers re-exported by Phase 81 dispatch additions.
_ = _require_int  # noqa: PLW0123
