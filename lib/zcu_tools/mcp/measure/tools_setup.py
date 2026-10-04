"""Explicit setup workflows over one captured GUI connection.

Mutations are sent once. Timeout never cancels them; uncertain receipts and
partial snapshots stay visible. They never launch a GUI or reconnect between steps.
"""

import math
from collections.abc import Callable
from functools import partial
from typing import Any, Literal, NotRequired, TypedDict

from recipes import RECIPES
from zcu_tools.mcp.core.reply import ToolReply
from zcu_tools.mcp.core.stdio_server import ToolTable
from zcu_tools.mcp.measure.session import GuiRpcError
from zcu_tools.mcp.measure.tool_context import MeasureToolContext
from zcu_tools.mcp.measure.tools_operation import wait, wait_timeout


class SetupError(TypedDict, total=False):
    code: str | None
    reason: str | None
    message: str
    request_rejected: bool


class SetupStep(TypedDict):
    status: Literal["not_started", "completed", "running", "failed", "unknown"]
    error: NotRequired[SetupError]


class SetupResult(TypedDict):
    tool: str
    op: int | None
    status: str
    steps: dict[str, SetupStep]
    before: dict[str, Any]
    after: dict[str, Any]
    requested: dict[str, Any]
    operation: dict[str, Any] | None
    verification: dict[str, Any] | None
    error: SetupError | None


def simulation_initialize(
    ctx: MeasureToolContext, arguments: dict[str, Any]
) -> ToolReply:
    """Initialize once, wait, then verify mock SoC, fake_flux and predictor facts."""
    timeout = wait_timeout(arguments)
    result = _setup_result("simulation_initialize", {}, ("soc", "devices", "predictor"))
    return _run_setup(
        ctx,
        result,
        timeout,
        _read_simulation,
        _verify_simulation,
    )


def device_set_value(ctx: MeasureToolContext, arguments: dict[str, Any]) -> ToolReply:
    """Set only value in the declared native unit and retain actual snapshots."""
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
    result = _setup_result(
        "device_set_value", {"name": name, "value": value, "unit": unit}, ("device",)
    )
    return _run_setup(
        ctx,
        result,
        timeout,
        _read_device,
        _verify_device,
    )


def recipe_guide(ctx: MeasureToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    """Query the registry-declared adapter guide without interpreting recipe names."""
    recipe = arguments.get("recipe")
    if not isinstance(recipe, str):
        raise ValueError("recipe must be a string")
    definition = next((entry for entry in RECIPES if entry.name == recipe), None)
    if definition is None:
        raise ValueError(f"unknown recipe: {recipe!r}")
    ctx = ctx.bound()
    reply = ctx.send_gui_rpc("adapter.guide", {"adapter_name": definition.adapter_name})
    return {
        "recipe": recipe,
        "adapter": definition.adapter_name,
        "guide": reply["guide"],
    }


def _setup_result(
    tool: str, requested: dict[str, Any], fact_names: tuple[str, ...]
) -> SetupResult:
    return {
        "tool": tool,
        "op": None,
        "status": "failed",
        "steps": {
            step: {"status": "not_started"}
            for step in ("pre_read", "start", "wait", "post_read")
        },
        "before": dict.fromkeys(fact_names),
        "after": dict.fromkeys(fact_names),
        "requested": requested,
        "operation": None,
        "verification": None,
        "error": None,
    }


def _record_error(
    result: SetupResult,
    step: str,
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


def _run_setup(
    ctx: MeasureToolContext,
    result: SetupResult,
    timeout: float,
    read_facts: Callable[
        [MeasureToolContext, SetupResult, Literal["before", "after"]], None
    ],
    verify: Callable[[SetupResult], None],
) -> ToolReply:
    try:
        ctx = ctx.bound()
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
        data=dict(result), is_error=result["status"] not in {"finished", "running"}
    )


def _start_setup(ctx: MeasureToolContext, result: SetupResult) -> bool:
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
        operation = wait(ctx, {"op": result["op"], "timeout": timeout})
    except GuiRpcError as exc:
        result["status"] = "unknown"
        _record_error(result, "wait", exc, "unknown")
        return
    result["operation"] = operation
    result["status"] = operation["status"]
    result["steps"]["wait"] = {
        "status": "running" if operation["status"] == "running" else "completed"
    }
    if operation.get("error") is not None:
        native_error = operation["error"]
        result["error"] = {
            "code": native_error.get("code"),
            "reason": native_error.get("reason"),
            "message": native_error.get("message", "operation failed"),
            "request_rejected": native_error.get("request_rejected", False),
        }


def _read_simulation(
    ctx: MeasureToolContext, result: SetupResult, phase: Literal["before", "after"]
) -> None:
    facts = result[phase]
    if ctx.gui.read_internal("state.has_soc", {})["value"]:
        facts["soc"] = ctx.send_gui_rpc("soc.info", {"include_cfg": True})
    devices = ctx.send_gui_rpc("device.list", {})["devices"]
    facts["devices"] = []
    for device in devices:
        snapshot = ctx.send_gui_rpc("device.snapshot", {"name": device["name"]})[
            "snapshot"
        ]
        facts["devices"].append(snapshot)
    facts["predictor"] = ctx.send_gui_rpc("predictor.info", {})


def _verify_simulation(result: SetupResult) -> None:
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


def _read_device(
    ctx: MeasureToolContext, result: SetupResult, phase: Literal["before", "after"]
) -> None:
    requested = result["requested"]
    snapshot = ctx.send_gui_rpc("device.snapshot", {"name": requested["name"]})[
        "snapshot"
    ]
    result[phase]["device"] = snapshot
    if phase == "before":
        _validate_device(snapshot, requested)


def _validate_device(snapshot: dict[str, Any], requested: dict[str, Any]) -> None:
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


def _verify_device(result: SetupResult) -> None:
    before = result["before"]["device"]
    after = result["after"]["device"]
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
            key not in before or key not in after or before[key] != after[key]
            for key in ("name", "type_name", "address", "unit")
        )
        or info is None
        or "value" not in info
    ):
        raise GuiRpcError(
            "native facts do not confirm the connected device identity/value"
        )


def build_setup_tools(ctx: MeasureToolContext) -> ToolTable:
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
