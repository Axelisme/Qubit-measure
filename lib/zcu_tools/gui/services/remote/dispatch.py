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

from zcu_tools.gui.adapter import (
    CfgSchema,
    MetaDictWriteback,
    ModuleWriteback,
    WaveformWriteback,
)
from zcu_tools.gui.services.context import MlEntryValidationError
from zcu_tools.gui.services.device import SetupDeviceRequest
from zcu_tools.gui.services.session_persistence import SessionPersistenceService

from .errors import ErrorCode, RemoteError
from .method_specs import METHOD_SPECS, MethodSpec
from .param_spec import ParamSpec
from .wire import (
    coerce_connect_device_request,
    coerce_connect_request,
    coerce_disconnect_device_request,
    coerce_set_device_value_request,
)

logger = logging.getLogger(__name__)

Handler = Callable[["object", Mapping[str, object]], Mapping[str, object]]

# Shared serializer instance: we only use its pure schema_to_raw / raw_to_schema
# methods, never its persistence side effects.
_SCHEMA_CODEC = SessionPersistenceService()


# ---------------------------------------------------------------------------
# Runtime registry entry — binds a synchronous handler to a Qt-free MethodSpec.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoundMethod:
    handler: Handler
    spec: MethodSpec

    @property
    def timeout_seconds(self) -> float:
        return self.spec.timeout_seconds

    @property
    def params(self) -> tuple[ParamSpec, ...]:
        return self.spec.params

    @property
    def off_main_thread(self) -> bool:
        return self.spec.off_main_thread


# ---------------------------------------------------------------------------
# Tab handlers
# ---------------------------------------------------------------------------


def _h_tab_new(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    name = str(params["adapter_name"])
    if name not in ctrl.get_adapter_names():
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown adapter: {name!r}")
    tab_id = ctrl.new_tab(name)
    return {"tab_id": tab_id}


def _h_tab_close(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    tab_id = str(params["tab_id"])
    if not ctrl.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    ctrl.close_tab(tab_id)
    return {}


def _h_tab_set_active(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    tab_id = str(params["tab_id"])
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
        # Shared cfg-editor session id for this tab (None until the tab's form
        # is populated). Address it with the editor.* methods to edit cfg with
        # the GUI reflecting every change. (A tab uses its tab_id as owner key.)
        "editor_id": ctrl.editor_id_for_owner(tab_id),
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
    tab_id = str(tab_id_raw)
    if not ctrl.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    return _tab_snapshot_wire(ctrl, tab_id)


def _save_paths_wire(paths) -> Optional[dict[str, str]]:
    if paths is None:
        return None
    return {"data_path": paths.data_path, "image_path": paths.image_path}


def _h_tab_get_cfg(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    tab_id = str(params["tab_id"])
    if not ctrl.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    schema = ctrl.get_tab_cfg_schema(tab_id)
    raw = _SCHEMA_CODEC.schema_to_raw(schema, ml=None)
    return {"raw": raw}


def _h_tab_list_paths(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    from .path_resolver import list_settable_paths

    tab_id = str(params["tab_id"])
    if not ctrl.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    # Walk the same live LiveModel that cfg.set_field mutates, so every listed
    # path is guaranteed settable. Requires the tab's form to be populated.
    try:
        root = ctrl.get_tab_live_model_root(tab_id)
    except (KeyError, RuntimeError) as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason=getattr(exc, "reason_code", ""),
        ) from exc
    return {"paths": list_settable_paths(root)}


def _h_tab_update_cfg(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    tab_id = str(params["tab_id"])
    if not ctrl.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    if ctrl.get_running_tab_id() == tab_id:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            f"tab {tab_id!r} is currently running; cancel the run before editing cfg",
        )
    raw = params["raw"]
    assert isinstance(raw, dict)
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
    tab_id = str(params["tab_id"])
    if not ctrl.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    if ctrl.get_running_tab_id() == tab_id:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            f"tab {tab_id!r} is currently running; cancel the run before editing cfg",
        )
    path = str(params["path"])
    value = params["value"]
    try:
        ctrl.set_tab_field(tab_id, path, value)
    except (KeyError, RuntimeError) as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason=getattr(exc, "reason_code", ""),
        ) from exc
    return {}


# ---------------------------------------------------------------------------
# Run / Save handlers
# ---------------------------------------------------------------------------


def _h_run_start(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    tab_id = str(params["tab_id"])
    if not ctrl.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    try:
        operation_id = ctrl.start_run(tab_id)
    except RuntimeError as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason=getattr(exc, "reason_code", ""),
        ) from exc
    return {"operation_id": operation_id}


def _h_run_cancel(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    del params
    ctrl.cancel_run()
    return {}


def _h_run_running_tab(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    del params
    return {"tab_id": ctrl.get_running_tab_id()}


def _h_save_data(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    tab_id = str(params["tab_id"])
    data_path = params["data_path"]
    comment = str(params["comment"])
    try:
        ctrl.save_data(
            tab_id, str(data_path) if data_path is not None else None, comment=comment
        )
    except RuntimeError as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason=getattr(exc, "reason_code", ""),
        ) from exc
    return {}


def _h_save_image(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    tab_id = str(params["tab_id"])
    image_path = params["image_path"]
    try:
        ctrl.save_image(tab_id, str(image_path) if image_path is not None else None)
    except RuntimeError as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason=getattr(exc, "reason_code", ""),
        ) from exc
    return {}


def _h_save_both(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    tab_id = str(params["tab_id"])
    data_path = params["data_path"]
    image_path = params["image_path"]
    comment = str(params["comment"])
    try:
        ctrl.save_both(
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
    return {}


def _h_save_set_paths(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    tab_id = str(params["tab_id"])
    if not ctrl.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    data_path = str(params["data_path"])
    image_path = str(params["image_path"])
    ctrl.update_tab_save_paths(tab_id, data_path, image_path)
    return {}


# ---------------------------------------------------------------------------
# Context / state / session handlers
# ---------------------------------------------------------------------------


def _h_context_use(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    ctrl.use_context(str(params["label"]))
    return {}


def _h_context_new(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    value = params["value"]
    unit = str(params["unit"])
    clone = bool(params["clone_from_current"])
    ctrl.new_context(
        value=float(value) if value is not None else None,  # type: ignore[arg-type]
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
    key = str(params["key"])
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


def _h_ml_list_roles(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    """List the experiment-role templates available for create_from_role."""
    del params
    try:
        catalog = ctrl.get_role_catalog()
    except RuntimeError as exc:
        raise RemoteError(ErrorCode.PRECONDITION_FAILED, str(exc)) from exc
    return {"roles": list(catalog.list_meta())}


def _h_ml_create_from_role(
    ctrl, params: Mapping[str, object]
) -> Mapping[str, object]:
    """Create a blank ml module/waveform from a named role and register it.

    One-shot: seeds md-linked defaults (lowered against the live md), writes ml.
    Edit afterwards via editor.open(from_name=...).
    """
    item_kind = str(params["item_kind"])
    role_id = str(params["role_id"])
    name = str(params["name"])
    try:
        ctrl.create_from_role(item_kind, role_id, name)
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


def _h_context_set_md_attr(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    key = str(params["key"])
    value = params["value"]
    try:
        ctrl.set_md_attr(key, value)
    except RuntimeError as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason=getattr(exc, "reason_code", ""),
        ) from exc
    return {}


def _h_context_del_md_attr(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    key = str(params["key"])
    try:
        ctrl.del_md_attr(key)
    except (AttributeError, RuntimeError) as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason=getattr(exc, "reason_code", ""),
        ) from exc
    return {}


def _h_context_set_ml_module(
    ctrl, params: Mapping[str, object]
) -> Mapping[str, object]:
    name = str(params["name"])
    raw = params["raw"]
    assert isinstance(raw, dict)
    try:
        ctrl.set_ml_module_from_raw(name, dict(raw))
    except MlEntryValidationError as exc:
        raise RemoteError(ErrorCode.INVALID_PARAMS, str(exc)) from exc
    except RuntimeError as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason=getattr(exc, "reason_code", ""),
        ) from exc
    return {}


def _h_context_del_ml_module(
    ctrl, params: Mapping[str, object]
) -> Mapping[str, object]:
    name = str(params["name"])
    try:
        ctrl.del_ml_module(name)
    except (KeyError, RuntimeError) as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason=getattr(exc, "reason_code", ""),
        ) from exc
    return {}


def _h_context_set_ml_waveform(
    ctrl, params: Mapping[str, object]
) -> Mapping[str, object]:
    name = str(params["name"])
    raw = params["raw"]
    assert isinstance(raw, dict)
    try:
        ctrl.set_ml_waveform_from_raw(name, dict(raw))
    except MlEntryValidationError as exc:
        raise RemoteError(ErrorCode.INVALID_PARAMS, str(exc)) from exc
    except RuntimeError as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason=getattr(exc, "reason_code", ""),
        ) from exc
    return {}


def _h_context_del_ml_waveform(
    ctrl, params: Mapping[str, object]
) -> Mapping[str, object]:
    name = str(params["name"])
    try:
        ctrl.del_ml_waveform(name)
    except (KeyError, RuntimeError) as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason=getattr(exc, "reason_code", ""),
        ) from exc
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


def _h_resources_versions(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    del params
    return {"versions": ctrl.resources_versions()}


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
    operation_id = ctrl.start_connect(req)
    return {"operation_id": operation_id}


def _h_startup_apply(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    # Params are ParamSpec-validated; result_dir/database_path are optional and
    # default to "" (empty result_dir leaves the context in DRAFT — editable but
    # not runnable until a result_dir is set).
    from zcu_tools.gui.services.startup import StartupProjectRequest

    req = StartupProjectRequest(
        chip_name=str(params["chip_name"]),
        qub_name=str(params["qub_name"]),
        res_name=str(params["res_name"]),
        result_dir=str(params["result_dir"] or ""),
        database_path=str(params["database_path"] or ""),
    )
    ok = ctrl.apply_startup_project(req)
    return {"ok": bool(ok)}


def _h_device_connect(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    req = coerce_connect_device_request(params)
    try:
        ctrl.start_connect_device(req)
    except RuntimeError as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason=getattr(exc, "reason_code", ""),
        ) from exc
    return {}


def _h_device_disconnect(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    req = coerce_disconnect_device_request(params)
    try:
        ctrl.start_disconnect_device(req)
    except RuntimeError as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason=getattr(exc, "reason_code", ""),
        ) from exc
    return {}


def _h_device_reconnect(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    name = str(params["name"])
    try:
        ctrl.start_reconnect_device(name)
    except RuntimeError as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason=getattr(exc, "reason_code", ""),
        ) from exc
    return {}


def _h_device_forget(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    name = str(params["name"])
    try:
        ctrl.forget_device(name)
    except RuntimeError as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason=getattr(exc, "reason_code", ""),
        ) from exc
    return {}


def _h_device_set_value(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    req = coerce_set_device_value_request(params)
    try:
        ctrl.start_set_device_value(req)
    except RuntimeError as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason=getattr(exc, "reason_code", ""),
        ) from exc
    return {}


def _h_device_setup(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    name = str(params["name"])
    updates = params["updates"]
    assert isinstance(updates, dict)
    try:
        info = ctrl.get_device_info(name)
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
        operation_id = ctrl.start_setup_device(
            SetupDeviceRequest(name=name, info=updated)
        )
    except RuntimeError as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason=getattr(exc, "reason_code", ""),
        ) from exc
    return {"operation_id": operation_id}


def _h_device_cancel_operation(
    ctrl, params: Mapping[str, object]
) -> Mapping[str, object]:
    name = str(params["name"])
    try:
        ctrl.cancel_device_operation(name)
    except RuntimeError as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason=getattr(exc, "reason_code", ""),
        ) from exc
    return {}


def _h_adapter_list(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    del params
    return {"adapters": list(ctrl.get_adapter_names())}


def _h_adapter_cfg_spec(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    from .path_resolver import list_spec_paths

    name = str(params["adapter_name"])
    if name not in ctrl.get_adapter_names():
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown adapter: {name!r}")
    spec = ctrl.get_adapter_cfg_spec(name)
    return {"paths": list_spec_paths(spec)}


def _h_adapter_analyze_spec(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    name = str(params["adapter_name"])
    if name not in ctrl.get_adapter_names():
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown adapter: {name!r}")
    return {"params": ctrl.get_adapter_analyze_params(name)}


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
    name = str(params["name"])
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


def _h_operation_await(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    # off_main_thread handler: blocks the IO worker thread on the gate's
    # thread-safe registry (never touches main-thread-owned state). Returns the
    # terminal outcome; failed/cancelled become a PRECONDITION_FAILED so the
    # caller's await raises (mirrors the old wait_setup "failed -> raise").
    operation_id = int(params["operation_id"])  # type: ignore[arg-type]
    timeout = float(params["timeout"])  # type: ignore[arg-type]
    outcome = ctrl.await_operation(operation_id, timeout)
    if outcome is None:
        raise RemoteError(
            ErrorCode.TIMEOUT,
            f"operation {operation_id} did not complete within {timeout}s",
        )
    if outcome.status in ("failed", "cancelled"):
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            outcome.error or f"operation {outcome.status}",
            reason=outcome.status,
        )
    return {"status": outcome.status}


# ---------------------------------------------------------------------------
# Dialog / view-query handlers (Phase 81a)
# ---------------------------------------------------------------------------


def _h_dialog_open(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    from .dialogs import DialogName, parse_dialog_name

    name_raw = str(params["name"])
    try:
        name: DialogName = parse_dialog_name(name_raw)
    except ValueError as exc:
        raise RemoteError(ErrorCode.INVALID_PARAMS, str(exc)) from exc
    ctrl.open_dialog(name)
    return {"opened": name.value}


def _h_dialog_close(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    from .dialogs import DialogName, parse_dialog_name

    name_raw = str(params["name"])
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

    name_str = str(params["dialog_name"])
    try:
        dialog_name = parse_dialog_name(name_str)
        png = ctrl.take_dialog_screenshot(dialog_name)
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


def _h_view_screenshot(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    import base64

    tab_id_raw = params.get("tab_id")
    tab_id: Optional[str] = str(tab_id_raw) if tab_id_raw is not None else None
    try:
        png = ctrl.take_screenshot(tab_id)
    except RuntimeError as exc:
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
    ctrl, params: Mapping[str, object]
) -> Mapping[str, object]:
    tab_id = str(params["tab_id"])
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

    tab_id = str(params["tab_id"])
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

    tab_id = str(params["tab_id"])
    if not ctrl.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    snap = ctrl.get_tab_snapshot(tab_id)
    if snap.analyze_params is None:
        raise RemoteError(ErrorCode.PRECONDITION_FAILED, "no analyze params available")
    raw_updates = params["updates"]
    assert isinstance(raw_updates, dict)
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
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason=getattr(exc, "reason_code", ""),
        ) from exc
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
    tab_id = str(params["tab_id"])
    if not ctrl.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    schema = ctrl.get_tab_cfg_schema(tab_id)
    raw = _SCHEMA_CODEC.schema_to_raw(schema, ml=None)
    return {"summary": _strip_cfg_tags(raw)}


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


def _h_writeback_preview(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    tab_id = str(params["tab_id"])
    if not ctrl.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    items = ctrl.get_tab_writeback_items(tab_id)
    return {"items": [_writeback_item_wire(it) for it in items]}


def _h_writeback_set(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    """Edit a persistent writeback item by id (selected / target_name /
    metadict proposed_value). Module/waveform cfg edits go through editor.* on
    the item's editor_id, not here."""
    tab_id = str(params["tab_id"])
    if not ctrl.has_tab(tab_id):
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
        changes["proposed_value"] = params["proposed_value"]
    try:
        ctrl.set_writeback_item(tab_id, session_id, **changes)
    except RuntimeError as exc:
        raise RemoteError(ErrorCode.INVALID_PARAMS, str(exc)) from exc
    return {}


def _h_writeback_apply(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    """Apply the tab's persistent writeback draft as-is (edit it first via
    writeback.set / editor.*)."""
    tab_id = str(params["tab_id"])
    if not ctrl.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    try:
        applied = ctrl.apply_writeback(tab_id)
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


def _h_editor_open(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    from zcu_tools.gui.services.cfg_editor import CfgEditorError

    item_kind = str(params["item_kind"])
    from_name = str(params["from_name"])
    # editor.open is modify-only: it edits an existing ml entry. Creating a blank
    # entry goes through ml.create_from_role (role_id='<disc>:blank').
    try:
        editor_id, paths = ctrl.open_cfg_editor(
            item_kind, discriminator=None, from_name=from_name
        )
    except CfgEditorError as exc:
        raise RemoteError(ErrorCode.INVALID_PARAMS, str(exc)) from exc
    return {"editor_id": editor_id, "paths": paths}


def _h_editor_set_field(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    from zcu_tools.gui.services.cfg_editor import CfgEditorError

    editor_id = str(params["editor_id"])
    path = str(params["path"])
    value = params["value"]
    try:
        return ctrl.cfg_editor_set_field(editor_id, path, value)
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


def _h_editor_get(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    from zcu_tools.gui.services.cfg_editor import CfgEditorError

    editor_id = str(params["editor_id"])
    try:
        return {"paths": ctrl.cfg_editor_get(editor_id)}
    except CfgEditorError as exc:
        raise RemoteError(ErrorCode.INVALID_PARAMS, str(exc)) from exc


def _h_editor_commit(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    from zcu_tools.gui.services.cfg_editor import CfgEditorError

    editor_id = str(params["editor_id"])
    name = str(params["name"])
    try:
        ctrl.commit_cfg_editor(editor_id, name)
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


def _h_editor_discard(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    from zcu_tools.gui.services.cfg_editor import CfgEditorError

    editor_id = str(params["editor_id"])
    try:
        ctrl.discard_cfg_editor(editor_id)
    except CfgEditorError as exc:
        raise RemoteError(ErrorCode.INVALID_PARAMS, str(exc)) from exc
    return {}


# ---------------------------------------------------------------------------
# Run progress handler
# ---------------------------------------------------------------------------


def _h_run_progress(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    del params
    bars = ctrl.get_run_progress()
    if not bars:
        return {"active": False, "bars": []}
    # bars are ProgressEntrySnapshot(token, format, maximum, value). `format` is
    # the human-readable bar string (e.g. "Rounds 23/100 [0:25<1:15]"); maximum
    # is the Qt-scaled total (0 when unknown), value the current scaled position.
    # `percent` is the convenience 0-100 derivation (None when total unknown).
    return {
        "active": True,
        "bars": [
            {
                "token": s.token,
                "format": s.format,
                "maximum": s.maximum,
                "value": s.value,
                "percent": (
                    None if s.maximum == 0 else round(s.value / s.maximum * 100, 1)
                ),
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

    tab_id = str(params["tab_id"])
    out_path_raw = params.get("out_path")
    out_path: Optional[str] = str(out_path_raw) if out_path_raw is not None else None
    try:
        png = ctrl.take_figure_screenshot(tab_id)
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


def _h_predictor_load(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    from zcu_tools.gui.services.connection import (
        LoadPredictorRequest,
        PredictorLoadError,
    )

    path = str(params["path"])
    flux_bias = float(params["flux_bias"])  # type: ignore[arg-type]
    try:
        ctrl.load_predictor(LoadPredictorRequest(path=path, flux_bias=flux_bias))
    except PredictorLoadError as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason=getattr(exc, "reason_code", ""),
        ) from exc
    return {}


def _h_predictor_clear(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    del params
    ctrl.clear_predictor()
    return {}


def _h_predictor_predict(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    from zcu_tools.gui.services.connection import PredictFreqRequest, PredictorNotLoaded

    value = float(params["value"])  # type: ignore[arg-type]
    from_lvl = int(params["from_lvl"])  # type: ignore[arg-type]
    to_lvl = int(params["to_lvl"])  # type: ignore[arg-type]
    try:
        freq = ctrl.predict_freq(
            PredictFreqRequest(value=value, transition=(from_lvl, to_lvl))
        )
    except PredictorNotLoaded as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason=getattr(exc, "reason_code", ""),
        ) from exc
    return {"freq_mhz": freq}


def _h_predictor_info(ctrl, params: Mapping[str, object]) -> Mapping[str, object]:
    del params
    return {"info": ctrl.get_predictor_info()}


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
    "cfg.set_field": _h_cfg_set_field,
    "run.start": _h_run_start,
    "run.cancel": _h_run_cancel,
    "run.running_tab": _h_run_running_tab,
    "run.progress": _h_run_progress,
    "save.data": _h_save_data,
    "save.image": _h_save_image,
    "save.both": _h_save_both,
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
    "context.set_ml_module": _h_context_set_ml_module,
    "context.del_ml_module": _h_context_del_ml_module,
    "context.set_ml_waveform": _h_context_set_ml_waveform,
    "context.del_ml_waveform": _h_context_del_ml_waveform,
    "ml.list_roles": _h_ml_list_roles,
    "ml.create_from_role": _h_ml_create_from_role,
    "state.has_project": _h_state_has_project,
    "state.has_context": _h_state_has_context,
    "state.has_active_context": _h_state_has_active_context,
    "state.has_soc": _h_state_has_soc,
    "resources.versions": _h_resources_versions,
    "session.persist": _h_session_persist,
    "session.restore": _h_session_restore,
    "connect.start": _h_connect_start,
    "startup.apply": _h_startup_apply,
    "device.connect": _h_device_connect,
    "device.disconnect": _h_device_disconnect,
    "device.reconnect": _h_device_reconnect,
    "device.forget": _h_device_forget,
    "device.set_value": _h_device_set_value,
    "device.setup": _h_device_setup,
    "device.cancel_operation": _h_device_cancel_operation,
    "device.active_setup": _h_device_active_setup,
    "device.active_operation": _h_device_active_operation,
    "operation.await": _h_operation_await,
    "device.list": _h_device_list,
    "device.snapshot": _h_device_snapshot,
    "adapter.list": _h_adapter_list,
    "adapter.cfg_spec": _h_adapter_cfg_spec,
    "adapter.analyze_spec": _h_adapter_analyze_spec,
    "dialog.open": _h_dialog_open,
    "dialog.close": _h_dialog_close,
    "dialog.list_open": _h_dialog_list_open,
    "dialog.screenshot": _h_dialog_screenshot,
    "view.snapshot": _h_view_snapshot,
    "view.screenshot": _h_view_screenshot,
    "tab.figure_screenshot": _h_tab_figure_screenshot,
    "predictor.load": _h_predictor_load,
    "predictor.clear": _h_predictor_clear,
    "predictor.predict": _h_predictor_predict,
    "predictor.info": _h_predictor_info,
    "tab.get_analyze_result": _h_tab_get_analyze_result,
    "tab.get_analyze_params": _h_tab_get_analyze_params,
    "analyze.start": _h_analyze_start,
    "tab.get_cfg_summary": _h_tab_get_cfg_summary,
    "writeback.preview": _h_writeback_preview,
    "writeback.set": _h_writeback_set,
    "writeback.apply": _h_writeback_apply,
    "editor.open": _h_editor_open,
    "editor.set_field": _h_editor_set_field,
    "editor.get": _h_editor_get,
    "editor.commit": _h_editor_commit,
    "editor.discard": _h_editor_discard,
}

# Every spec must have a handler and vice versa — fail fast on drift.
if set(_HANDLERS) != set(METHOD_SPECS):
    missing_spec = sorted(set(_HANDLERS) - set(METHOD_SPECS))
    missing_handler = sorted(set(METHOD_SPECS) - set(_HANDLERS))
    raise RuntimeError(
        "dispatch/method_specs drift — "
        f"handlers without spec: {missing_spec}; specs without handler: {missing_handler}"
    )

# `auth` is a sentinel handled by the service before the registry — left out here.
METHOD_REGISTRY: dict[str, BoundMethod] = {
    method: BoundMethod(handler=_HANDLERS[method], spec=METHOD_SPECS[method])
    for method in METHOD_SPECS
}
