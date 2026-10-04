"""Explicit setup workflows over an already connected, captured GUI.

Mutations are sent once. Timeout never cancels them; uncertain receipts and
partial snapshots stay visible. No stage attaches, launches or reconnects a GUI.
Native wire views retain opaque payloads and extra keys without domain conversion.
"""

import math
from collections.abc import Callable
from functools import partial
from typing import Any, Literal, NotRequired, TypedDict

from zcu_tools.mcp.core.reply import ToolReply
from zcu_tools.mcp.core.stdio_server import ToolTable
from zcu_tools.mcp.measure.session import GuiRpcError
from zcu_tools.mcp.measure.tool_context import MeasureToolContext
from zcu_tools.mcp.measure.tools_operation import wait, wait_timeout

type _JsonValue = (
    None | bool | int | float | str | list[_JsonValue] | dict[str, _JsonValue]
)
type _JsonObject = dict[str, _JsonValue]
type _Stage = Literal["pre_read", "start", "wait", "post_read"]
type _Phase = Literal["before", "after"]
type _StepStatus = Literal["not_started", "completed", "running", "failed", "unknown"]
type _Outcome = Literal[
    "pending", "finished", "failed", "cancelled", "running", "unknown"
]


class SetupError(TypedDict, total=False):
    """Boundary failure retaining the existing session error classification.

    code is the native error code or null. reason is the native/session reason
    code or null; no new taxonomy is introduced. message is diagnostic text.
    request_rejected is true only when admission explicitly rejected the request;
    false does not prove the mutation started or did not start.
    """

    code: str | None
    reason: str | None
    message: str
    request_rejected: bool


class SetupStep(TypedDict):
    """One workflow stage, independent of the native operation outcome.

    status is not_started, completed, running, failed or unknown. completed means
    only that this stage succeeded; unknown means its outcome was not confirmed.
    error is present when a stage failed or became uncertain and retains the
    session's native diagnostic/classification fields.
    """

    status: _StepStatus
    error: NotRequired[SetupError]


class _Steps(TypedDict):
    pre_read: SetupStep
    start: SetupStep
    wait: SetupStep
    post_read: SetupStep


class _SocSnapshot(TypedDict, total=False):
    is_mock: bool
    cfg: _JsonValue


class _DeviceField(TypedDict, total=False):
    name: str
    settable: bool


class _DeviceInfo(TypedDict, total=False):
    value: _JsonValue


class _DeviceSnapshot(TypedDict, total=False):
    name: str
    type_name: str
    address: str
    status: str
    unit: str
    error: _JsonValue
    info: _DeviceInfo | None
    fields: list[_DeviceField]


class _DeviceListEntry(TypedDict):
    name: str


class _DeviceListReply(TypedDict):
    devices: list[_DeviceListEntry]


class _PredictorSnapshot(TypedDict, total=False):
    loaded: bool


class _SimulationFacts(TypedDict):
    soc: _SocSnapshot | None
    devices: list[_DeviceSnapshot] | None
    predictor: _PredictorSnapshot | None


class _DeviceFacts(TypedDict):
    device: _DeviceSnapshot | None


class _SimulationRequest(TypedDict):
    pass


class _DeviceRequest(TypedDict):
    name: str
    value: int | float
    unit: str


class _SimulationVerification(TypedDict):
    ready: bool


class _DeviceVerification(TypedDict):
    actual: _JsonValue
    exact_match: bool


class _NativeOperationError(TypedDict, total=False):
    code: str | None
    reason: str | None
    message: str
    request_rejected: bool


class _Operation(TypedDict):
    """Native wait reply, not a second operation-completion policy."""

    status: _Outcome
    elapsed_s: float
    error: NotRequired[_NativeOperationError | None]
    feedback: NotRequired[_JsonValue]
    progress: NotRequired[list[_JsonObject]]
    eta_s: NotRequired[float]


class _Progress(TypedDict):
    op: int | None
    status: _Outcome
    steps: _Steps
    operation: _Operation | None
    error: SetupError | None


class _SimulationResult(_Progress):
    tool: Literal["simulation_initialize"]
    before: _SimulationFacts
    after: _SimulationFacts
    requested: _SimulationRequest
    verification: _SimulationVerification | None


class _DeviceResult(_Progress):
    tool: Literal["device_set_value"]
    before: _DeviceFacts
    after: _DeviceFacts
    requested: _DeviceRequest
    verification: _DeviceVerification | None


type SetupResult = _SimulationResult | _DeviceResult


class _RecipeGuideResult(TypedDict):
    recipe: str
    adapter: str
    guide: _JsonObject


def simulation_initialize(
    ctx: MeasureToolContext, arguments: dict[str, Any]
) -> ToolReply:
    """Initialize simulation once on ctx's existing GUI connection.

    arguments may contain timeout (default 60 seconds, finite non-bool number in
    0..300). The GUI coordinator may reuse simulation or disconnect real devices
    during switching, so the caller needs authorization for that effect.
    This tool does not connect hardware, apply a project or attach a GUI.

    Returns a ToolReply with op, native operation, four steps, before/after
    SoC/device/predictor facts and verification.ready. Missing facts are null.
    Invalid arguments raise ValueError before GUI access. Connection, query and
    mutation failures return an error reply with the known progress. Timeout
    does not cancel; running is not ready, and failed reads do not erase op.
    """
    timeout = wait_timeout(arguments)
    result: _SimulationResult = {
        **_initial_progress(),
        "tool": "simulation_initialize",
        "before": {"soc": None, "devices": None, "predictor": None},
        "after": {"soc": None, "devices": None, "predictor": None},
        "requested": {},
        "verification": None,
    }
    return _run_setup(ctx, result, timeout, _read_simulation, _verify_simulation)


def device_set_value(ctx: MeasureToolContext, arguments: dict[str, Any]) -> ToolReply:
    """Set one device value once on ctx's existing GUI connection.

    arguments requires nonempty name and unit strings and a finite non-bool
    int/float value. Optional timeout defaults to 60 seconds and accepts finite
    non-bool numbers in 0..300. unit must match the native snapshot; FakeDevice
    unit=none requires native. No conversion, tolerance, or output/mode/rampstep
    change is performed. The caller needs authorization to change that device.

    Returns a ToolReply with requested value, op, native operation, four steps,
    before/after device facts and verification.actual/exact_match. Invalid input
    raises ValueError before GUI access. Disconnected/unsettable devices, unit
    mismatch and native failures return an error reply retaining known facts.
    Timeout never cancels, reconnects, retries or proves absence of side effects.
    """
    name = arguments.get("name")
    value = arguments.get("value")
    unit = arguments.get("unit")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name must be a nonempty string")
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError("value must be a finite number")
    if not isinstance(unit, str) or not unit.strip():
        raise ValueError("unit must be a nonempty string")
    timeout = wait_timeout(arguments)
    result: _DeviceResult = {
        **_initial_progress(),
        "tool": "device_set_value",
        "before": {"device": None},
        "after": {"device": None},
        "requested": {"name": name, "value": value, "unit": unit},
        "verification": None,
    }
    return _run_setup(ctx, result, timeout, _read_device, _verify_device)


def recipe_guide(
    ctx: MeasureToolContext, arguments: dict[str, Any]
) -> _RecipeGuideResult:
    """Read the guide for arguments['recipe'] on ctx's existing GUI connection.

    recipe must name an entry in the authoritative recipe registry (for example
    t1). Returns recipe, its declared adapter name and the native guide object.
    No mutation, operation, attach or reconnect occurs. A missing/unknown recipe
    raises ValueError before GUI access; native query/connection errors raise
    GuiRpcError. Guide contents and adapter mapping are not guessed or rewritten.
    """
    recipe = arguments.get("recipe")
    if not isinstance(recipe, str):
        raise ValueError("recipe must be a string")
    definition = next(
        (entry for entry in ctx.session.recipes.definitions if entry.name == recipe),
        None,
    )
    if definition is None:
        raise ValueError(f"unknown recipe: {recipe!r}")
    ctx = ctx.bound(require_connected=True)
    reply = ctx.send_gui_rpc("adapter.guide", {"adapter_name": definition.adapter_name})
    return {
        "recipe": recipe,
        "adapter": definition.adapter_name,
        "guide": reply["guide"],
    }


def _initial_progress() -> _Progress:
    return {
        "op": None,
        "status": "failed",
        "steps": {
            "pre_read": {"status": "not_started"},
            "start": {"status": "not_started"},
            "wait": {"status": "not_started"},
            "post_read": {"status": "not_started"},
        },
        "operation": None,
        "error": None,
    }


def _record_error(
    result: SetupResult,
    step: _Stage,
    exc: GuiRpcError,
    status: Literal["failed", "unknown"],
) -> None:
    error: SetupError = {
        "code": exc.code,
        "reason": exc.reason,
        "message": str(exc),
        "request_rejected": exc.request_rejected,
    }
    result["steps"][step] = {"status": status, "error": error}
    # A later read failure must not erase the uncertain mutation/wait cause.
    if result["error"] is None:
        result["error"] = error


def _run_setup[ResultT: SetupResult](
    ctx: MeasureToolContext,
    result: ResultT,
    timeout: float,
    read_facts: Callable[[MeasureToolContext, ResultT, _Phase], None],
    verify: Callable[[ResultT], None],
) -> ToolReply:
    try:
        ctx = ctx.bound(require_connected=True)
        read_facts(ctx, result, "before")
    except GuiRpcError as exc:
        _record_error(result, "pre_read", exc, "failed")
        return ToolReply(data=dict(result), is_error=True)
    result["steps"]["pre_read"] = {"status": "completed"}

    if not _start_setup(ctx, result):
        return ToolReply(data=dict(result), is_error=True)
    if result["op"] is not None:
        _wait_setup(ctx, result, timeout)

    try:
        read_facts(ctx, result, "after")
    except GuiRpcError as exc:
        _record_error(result, "post_read", exc, "failed")
        if result["status"] == "finished":
            result["status"] = "failed"
    else:
        result["steps"]["post_read"] = {"status": "completed"}
        try:
            verify(result)
        except GuiRpcError as exc:
            _record_error(result, "post_read", exc, "failed")
            result["status"] = "failed"
    return ToolReply(
        data=dict(result),
        is_error=result["error"] is not None
        or result["status"] not in {"finished", "running"},
    )


def _start_setup(ctx: MeasureToolContext, result: SetupResult) -> bool:
    params: _JsonObject
    if result["tool"] == "simulation_initialize":
        method, params = "simulation.initialize", {}
    else:
        requested = result["requested"]
        method = "device.setup"
        params = {"name": requested["name"], "updates": {"value": requested["value"]}}
    try:
        receipt = ctx.send_gui_rpc(method, params)
        handle = receipt.get("handle")
        if isinstance(handle, bool) or not isinstance(handle, int):
            raise GuiRpcError("missing operation handle", reason="incompatible_wire")
        result["op"] = handle
    except GuiRpcError as exc:
        status = "failed" if exc.request_rejected else "unknown"
        result["status"] = status
        _record_error(result, "start", exc, status)
        return not exc.request_rejected
    result["steps"]["start"] = {"status": "completed"}
    return True


def _wait_setup(ctx: MeasureToolContext, result: SetupResult, timeout: float) -> None:
    try:
        operation = _Operation(**wait(ctx, {"op": result["op"], "timeout": timeout}))
    except GuiRpcError as exc:
        result["status"] = "unknown"
        _record_error(result, "wait", exc, "unknown")
        return
    result["operation"] = operation
    result["status"] = operation["status"]
    result["steps"]["wait"] = {
        "status": "running" if operation["status"] == "running" else "completed"
    }
    native_error = operation.get("error")
    if native_error is not None:
        error: SetupError = {
            "code": native_error.get("code"),
            "reason": native_error.get("reason"),
            "message": native_error.get("message", "operation failed"),
            "request_rejected": native_error.get("request_rejected", False),
        }
        result["error"] = error


def _read_simulation(
    ctx: MeasureToolContext, result: _SimulationResult, phase: _Phase
) -> None:
    facts = result[phase]
    if ctx.gui.read_internal("state.has_soc", {})["value"]:
        facts["soc"] = _SocSnapshot(
            **ctx.send_gui_rpc("soc.info", {"include_cfg": True})
        )
    listing = _DeviceListReply(**ctx.send_gui_rpc("device.list", {}))
    facts["devices"] = []
    for device in listing["devices"]:
        snapshot = _DeviceSnapshot(
            **ctx.send_gui_rpc("device.snapshot", {"name": device["name"]})["snapshot"]
        )
        facts["devices"].append(snapshot)
    facts["predictor"] = _PredictorSnapshot(**ctx.send_gui_rpc("predictor.info", {}))


def _verify_simulation(result: _SimulationResult) -> None:
    facts = result["after"]
    soc = facts["soc"]
    devices = facts["devices"]
    predictor = facts["predictor"]
    ready = (
        result["status"] == "finished"
        and soc is not None
        and soc.get("is_mock") is True
        and devices is not None
        and any(
            device.get("name") == "fake_flux"
            and device.get("type_name") == "FakeDevice"
            and device.get("status") == "connected"
            for device in devices
        )
        and predictor is not None
        and predictor.get("loaded") is True
    )
    result["verification"] = {"ready": ready}
    if result["status"] == "finished" and not ready:
        raise GuiRpcError("native facts do not confirm simulation readiness")


def _read_device(ctx: MeasureToolContext, result: _DeviceResult, phase: _Phase) -> None:
    requested = result["requested"]
    snapshot = _DeviceSnapshot(
        **ctx.send_gui_rpc("device.snapshot", {"name": requested["name"]})["snapshot"]
    )
    result[phase]["device"] = snapshot
    if phase == "before":
        _validate_device(snapshot, requested)


def _validate_device(snapshot: _DeviceSnapshot, requested: _DeviceRequest) -> None:
    if (
        snapshot.get("name") != requested["name"]
        or snapshot.get("status") != "connected"
    ):
        raise GuiRpcError("requested device is not connected")
    if not any(
        field.get("name") == "value" and field.get("settable") is True
        for field in snapshot.get("fields", [])
    ):
        raise GuiRpcError("device does not support setting value")
    if snapshot.get("unit") == "none":
        valid_unit = (
            snapshot.get("type_name") == "FakeDevice" and requested["unit"] == "native"
        )
    else:
        valid_unit = snapshot.get("unit") == requested["unit"]
    if not valid_unit:
        raise GuiRpcError("unit does not match the device's native coordinate")


def _verify_device(result: _DeviceResult) -> None:
    before = result["before"]["device"]
    after = result["after"]["device"]
    if before is None or after is None:
        raise GuiRpcError("native device facts were not acquired")
    info = after.get("info")
    actual = info.get("value") if info is not None else None
    result["verification"] = {
        "actual": actual,
        "exact_match": actual == result["requested"]["value"],
    }
    if result["status"] != "finished":
        return
    if (
        after.get("status") != "connected"
        or any(
            key not in before or key not in after or before.get(key) != after.get(key)
            for key in ("name", "type_name", "address", "unit")
        )
        or info is None
        or "value" not in info
    ):
        raise GuiRpcError(
            "native facts do not confirm the connected device identity/value"
        )


def build_setup_tools(ctx: MeasureToolContext) -> ToolTable:
    """Build three fixed setup tools over ctx, without connecting or performing setup.

    ctx supplies the application's configured MCP session. Returned handlers
    accept the simulation_initialize, device_set_value and recipe_guide argument
    objects documented above; callers must explicitly connect before invocation.
    """
    timeout_schema = {"type": "number", "minimum": 0, "maximum": 300, "default": 60.0}
    return {
        "simulation_initialize": {
            "handler": partial(simulation_initialize, ctx),
            "description": "Initialize the simulated environment through the GUI coordinator. "
            "May disconnect real devices; requires caller authorization. Reuses valid simulation. "
            "Does not apply a project, launch a GUI or connect real hardware. "
            "Wait timeout does not cancel. Inspect steps, operation and snapshots before retrying.",
            "inputSchema": {
                "type": "object",
                "properties": {"timeout": timeout_schema},
                "additionalProperties": False,
            },
        },
        "device_set_value": {
            "handler": partial(device_set_value, ctx),
            "description": "Set only a connected device working value. Unit must match its GUI "
            "snapshot; FakeDevice unit=none requires explicit native. No unit conversion, "
            "output/mode/rampstep change, retry or rollback. Reports requested and actual values "
            "without hardware tolerance. Wait timeout does not cancel; inspect partial progress.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "value": {"type": "number"},
                    "unit": {"type": "string", "minLength": 1},
                    "timeout": timeout_schema,
                },
                "required": ["name", "value", "unit"],
                "additionalProperties": False,
            },
        },
        "recipe_guide": {
            "handler": partial(recipe_guide, ctx),
            "description": "Read the native guide of a registered recipe adapter. "
            "Uses the authoritative recipe registry; does not run a recipe or mutate the GUI.",
            "inputSchema": {
                "type": "object",
                "properties": {"recipe": {"type": "string", "minLength": 1}},
                "required": ["recipe"],
                "additionalProperties": False,
            },
        },
    }
