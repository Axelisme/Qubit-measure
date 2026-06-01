"""Wire types for RemoteControlAdapter.

Frozen dataclasses for request / response envelopes plus strict JSON ↔ dataclass
coercion helpers. All raw-dict validation happens here; no ``Any`` from the wire
ever flows into ``Controller`` or domain services.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

from zcu_tools.gui.services.connection import (
    ConnectMockRequest,
    ConnectRemoteRequest,
    ConnectRequest,
)
from zcu_tools.gui.services.device import (
    ConnectDeviceRequest,
    DisconnectDeviceRequest,
)

from .errors import ErrorCode, ErrorEnvelope, RemoteError

# Two independent hand-maintained versions reported by the no-auth
# ``wire.version`` handshake (which also surfaces the MCP server's own
# MCP_VERSION). Only WIRE_VERSION is *compared*; the code revisions are
# *reported* so an agent can eyeball whether a reload took effect:
#
#   WIRE_VERSION — the mcp<->RPC *interface contract* (RPC method set, their
#     params, event/serialization shape). The MCP server pins it; a mismatch
#     means the two sides speak different protocols → hard MISMATCH. Bump ONLY
#     on a contract change.
#   GUI_VERSION  — this GUI code's *revision*. Reported, never compared (the
#     MCP server does not pin it — a GUI revision is the GUI process's own
#     property). Bump on any meaningful GUI change you want to be able to spot a
#     reload of, INCLUDING pure-internal logic changes that DON'T touch the wire
#     (the whole point of the split: an internal change bumps GUI_VERSION, not
#     WIRE_VERSION, so the contract version stays put).
#
# (Replaces the old single-version scheme that conflated "is the contract
# compatible" with "is this process running the latest code".)
# v2: added ml.list_roles / ml.create_from_role (role catalog) and
#     context.rename_ml_module / context.rename_ml_waveform; editor.open dropped
#     its discriminator param (from_name-only).
# v3: removed device.set_value (set values via device.setup updates={"value":..});
#     device.connect / device.disconnect now return operation_id (operation handle
#     parity with device.setup); device.snapshot includes the device info payload.
# v4: removed context.set_ml_module / context.set_ml_waveform (raw-dict RPC); ml
#     entries are built/edited via the editor session (create_from_role + editor.*)
#     — ADR-0011, the single ml/md write authority is ContextService.
# v5: added device.setup_spec (discover the fields settable via device.setup's
#     updates — name/type/choices/current/settable — from the live info model).
# v6 (ADR-0013): (a) removed cfg.set_field — a tab's cfg is edited through its
#     CfgEditorService session (editor.set_field on the tab's editor_id), the
#     same draft the form attaches to, so agent + user share one model (F11);
#     tab.list_paths now reads that session too (wire shape unchanged). (b) new
#     unsolicited ``diagnostic`` event push ({severity: error|info, title,
#     message}) — the Controller fans diagnostics to the adapter (a diagnostic
#     View) out-of-band of the event subscription set; agents receive it via the
#     normal events poll without subscribing.
# v7: run_lock_changed split into run_started{tab_id} + run_finished{tab_id,
#     outcome, error_message} — one event name per real transition instead of a
#     single event whose meaning depended on which fields were present.
# v8: cfg path grammar drops the ModuleRef 'value' wrapper segment
#     (modules.qub_pulse.value.waveform.value.length -> ...qub_pulse.waveform.length;
#     a stale 'value' path is rejected); editor.set_field result adds
#     'removed'/'added' (paths a ref switch dropped/created).
# v9: run.progress bars add raw 'n'/'total' (alongside scaled maximum/value) —
#     progress derivation moved to the main-thread ProgressBarModel (SSOT), so
#     format/percent/timing are computed live at read.
# v10: device setup ↔ run alignment. device_setup_changed split into
#      device_setup_started{name} + device_setup_finished{name, outcome,
#      error_message} (mirrors run_started/run_finished); new device.setup_progress
#      (same shape as run.progress); device.active_setup now only {device_name}
#      and device.active_operation drops progress (live progress via
#      device.setup_progress).
# v11: added soc.info — read the connected SoC's QICK soccfg (human-readable
#      description + structured cfg: DAC/ADC channels, sample rates, freq ranges).
# v12: added adapter.guide — read an adapter's human-facing orientation guide
#      (behavior / expects_md / expects_ml / typical_writeback / recommended)
#      before running it. New method = contract change.
WIRE_VERSION = 12

# GUI code revision (see header). Bump on any meaningful GUI change you want a
# stale-process check to flag; independent of WIRE_VERSION.
# v2: tab_id is now '<adapter-slug>-<hash>' and an owner-keyed editor_id is
#     '<owner>-ed' (readability; ids stay opaque string keys — no wire change).
# v3: run_lock_changed split into run_started / run_finished (also a wire change
#     — WIRE_VERSION 7; bumped here too since it's a GUI code change).
# v4: cfg path grammar drops 'value' wrapper + set_field returns removed/added
#     (WIRE_VERSION 8).
# v5: progress refactor — device_progress.py -> pbar_host.py (beside plot_host),
#     mutable ProgressBarModel SSOT (worker forwards raw + throttles, main thread
#     computes format/timing live), run.progress adds raw n/total (WIRE 9).
# v6: device setup ↔ run alignment (WIRE 10) — split setup event, device.setup_progress.
# v7: progress big refactor — Qt-free ProgressService + ProgressContainer (owns
#     dict[operation_id, container]) behind a ProgressTransport port whose Qt
#     marshal (QtProgressTransport) is a driven adapter; run/device no longer
#     rebuild a ProgressModel; Views attach by owner_id. Wire shape unchanged.
# v8: soc.info RPC (WIRE 11) — expose the connected SoC's soccfg to the agent;
#     mcp folds the description into connect replies.
# v9: adapter.guide RPC (WIRE 12) — adapters carry a static AdapterGuide; GUI adds
#     a read-only "Guide" tab beside Config/Analysis.
GUI_VERSION = 9

# ---------------------------------------------------------------------------
# Wire envelopes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Request:
    id: str
    method: str
    params: Mapping[str, object]


@dataclass(frozen=True)
class Response:
    id: str
    ok: bool
    result: Optional[Mapping[str, object]] = None
    error: Optional[ErrorEnvelope] = None

    def to_wire(self) -> dict[str, object]:
        if self.ok:
            wire: dict[str, object] = {
                "id": self.id,
                "ok": True,
                "result": self.result or {},
            }
        else:
            assert self.error is not None
            wire = {"id": self.id, "ok": False, "error": self.error.to_wire()}
        return wire


def parse_request(raw: Mapping[str, object]) -> Request:
    rid = _require_str(raw, "id")
    method = _require_str(raw, "method")
    params = raw.get("params", {})
    if not isinstance(params, dict):
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            f"'params' must be an object, got {type(params).__name__}",
        )
    return Request(id=rid, method=method, params=params)


# ---------------------------------------------------------------------------
# Field helpers
# ---------------------------------------------------------------------------


def _require_str(params: Mapping[str, object], key: str) -> str:
    val = params.get(key)
    if val is None:
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"missing '{key}'")
    if not isinstance(val, str):
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            f"'{key}' must be a string, got {type(val).__name__}",
        )
    if not val:
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"'{key}' must be non-empty")
    return val


def _require_int(params: Mapping[str, object], key: str) -> int:
    val = params.get(key)
    if val is None:
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"missing '{key}'")
    # bool is a subclass of int in Python; reject it explicitly
    if isinstance(val, bool) or not isinstance(val, int):
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            f"'{key}' must be an integer, got {type(val).__name__}",
        )
    return val


def _require_bool(params: Mapping[str, object], key: str) -> bool:
    val = params.get(key)
    if val is None:
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"missing '{key}'")
    if not isinstance(val, bool):
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            f"'{key}' must be a boolean, got {type(val).__name__}",
        )
    return val


def _optional_str(params: Mapping[str, object], key: str) -> Optional[str]:
    val = params.get(key)
    if val is None:
        return None
    if not isinstance(val, str):
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            f"'{key}' must be a string if present, got {type(val).__name__}",
        )
    return val


def _optional_bool(params: Mapping[str, object], key: str, default: bool) -> bool:
    val = params.get(key)
    if val is None:
        return default
    if not isinstance(val, bool):
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            f"'{key}' must be a boolean if present, got {type(val).__name__}",
        )
    return val


def _optional_float(params: Mapping[str, object], key: str) -> Optional[float]:
    val = params.get(key)
    if val is None:
        return None
    if isinstance(val, bool):
        raise RemoteError(
            ErrorCode.INVALID_PARAMS, f"'{key}' must be a number if present"
        )
    if not isinstance(val, (int, float)):
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            f"'{key}' must be a number if present, got {type(val).__name__}",
        )
    return float(val)


def require_object(params: Mapping[str, object], key: str) -> Mapping[str, object]:
    val = params.get(key)
    if val is None:
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"missing '{key}'")
    if not isinstance(val, dict):
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            f"'{key}' must be an object, got {type(val).__name__}",
        )
    return val


def require_json_safe(params: Mapping[str, object], key: str) -> object:
    import json

    val = params.get(key)
    if key not in params:
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"missing '{key}'")
    try:
        json.dumps(val)
    except (TypeError, ValueError) as exc:
        raise RemoteError(
            ErrorCode.INVALID_PARAMS, f"'{key}' must be JSON-serializable"
        ) from exc
    return val


# ---------------------------------------------------------------------------
# Typed-request coercion
# ---------------------------------------------------------------------------


def coerce_connect_request(params: Mapping[str, object]) -> ConnectRequest:
    """Coerce ``{kind: 'mock'}`` or ``{kind: 'remote', ip, port}``."""
    kind = _require_str(params, "kind")
    if kind == "mock":
        return ConnectMockRequest()
    if kind == "remote":
        return ConnectRemoteRequest(
            ip=_require_str(params, "ip"),
            port=_require_int(params, "port"),
        )
    raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown connect kind: {kind!r}")


def coerce_connect_device_request(
    params: Mapping[str, object],
) -> ConnectDeviceRequest:
    return ConnectDeviceRequest(
        type_name=_require_str(params, "type_name"),
        name=_require_str(params, "name"),
        address=_require_str(params, "address"),
        remember=_optional_bool(params, "remember", True),
    )


def coerce_disconnect_device_request(
    params: Mapping[str, object],
) -> DisconnectDeviceRequest:
    return DisconnectDeviceRequest(
        name=_require_str(params, "name"),
        remember=_optional_bool(params, "remember", True),
    )
