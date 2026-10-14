"""Explicit setup workflows over an already connected, captured GUI.

Mutations are sent once. Timeout never cancels them; uncertain receipts and
partial snapshots stay visible. No stage attaches, launches or reconnects a GUI.
Native wire views retain opaque payloads and extra keys without domain conversion.
"""

from collections.abc import Callable
from copy import deepcopy
from functools import partial
from typing import Any, Literal, NotRequired, TypedDict

from zcu_tools.gui.app.measure.remote.guide_types import AdapterGuideView
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


class SetupSteps(TypedDict):
    """Four workflow stages; completed is not an operation-success claim."""

    pre_read: SetupStep
    """Before acquisition and candidate preparation; failure prevents submission."""
    start: SetupStep
    """Submission; completed means a confirmed handle, not driver completion."""
    wait: SetupStep
    """Bounded native wait; running retains the handle without cancellation."""
    post_read: SetupStep
    """After acquisition; completed does not prove the requested value was reached."""


class _SocSnapshot(TypedDict, total=False):
    is_mock: bool
    cfg: _JsonValue


class DeviceFieldWire(TypedDict, total=False):
    """Native cached field projection; missing fields are not filled in."""

    name: str
    """Info field name."""
    type: str
    """Native scalar/enum description."""
    current: _JsonValue
    """Cached value, including native null; not a fresh hardware read."""
    settable: bool
    """Owner's editing flag, not an additional full-cfg validator."""
    choices: list[_JsonValue]
    """Optional native enum choices, in their original order."""


type DeviceSetupCfg = _JsonObject
"""Info field names mapped to recursive JSON values; validity belongs to Info."""


class DeviceSnapshotWire(TypedDict, total=False):
    """Native cached snapshot, retaining opaque extra keys without conversion."""

    name: str
    """Native registry name, never filled from requested.name."""
    type_name: str
    """Native driver name; not a local model registry key."""
    address: str
    """Native snapshot connection address."""
    status: str
    """Cached device lifecycle status, not an operation outcome."""
    unit: str
    """Native unit string; no conversion or none compatibility rule."""
    error: _JsonValue
    """Opaque native error or null."""
    info: DeviceSetupCfg | None
    """Complete cached Info object or null; fields never substitute for Info."""
    fields: list[DeviceFieldWire]
    """Native field projections, preserving order and optional keys."""


class _DeviceListEntry(TypedDict):
    name: str


class _DeviceListReply(TypedDict):
    devices: list[_DeviceListEntry]


class _SimulationFacts(TypedDict):
    soc: _SocSnapshot | None
    devices: list[DeviceSnapshotWire] | None


class DeviceSetupFacts(TypedDict):
    """Facts acquired in one setup phase, not evidence of absent side effects."""

    device: DeviceSnapshotWire | None
    """Native snapshot or null when this phase has not acquired it."""


class _SimulationRequest(TypedDict):
    pass


class DeviceSetupRequest(TypedDict):
    """Requested target and full candidate, not a claim of model admission."""

    name: str
    """Nonempty caller-specified registry name."""
    cfg: DeviceSetupCfg | None
    """Null until before-read and merge finish; then the complete candidate."""


class _SimulationVerification(TypedDict):
    ready: bool


class NativeSetupOperationError(TypedDict, total=False):
    """Native operation diagnostic, retaining missing versus null fields."""

    code: str | None
    """Native code or null, when supplied."""
    reason: str | None
    """Native reason or null, when supplied."""
    message: str
    """Native diagnostic text."""
    request_rejected: bool
    """Native admission fact; false does not prove no side effects."""


class NativeSetupOperation(TypedDict):
    """Native wait reply, not a second operation-completion policy."""

    status: _Outcome
    """Native pending/finished/failed/cancelled/running/unknown outcome."""
    elapsed_s: float
    """Nonnegative time measured by wait, not guaranteed hardware duration."""
    error: NotRequired[NativeSetupOperationError | None]
    """Optional native error or null; missing is not filled in."""
    feedback: NotRequired[_JsonValue]
    """Opaque native feedback, not a physical-success interpretation."""
    progress: NotRequired[list[_JsonObject]]
    """Optional native progress objects, retained without conversion."""
    eta_s: NotRequired[float]
    """Optional native estimated remaining seconds."""


class _Progress(TypedDict):
    """Shared workflow receipt fields, inherited by both setup results."""

    op: int | None
    """Opaque MCP handle, preserved once known; null does not prove no mutation."""
    status: _Outcome
    """Workflow outcome, distinct from native operation.status."""
    steps: SetupSteps
    """Four stage receipts; each retains its own failure or uncertainty."""
    operation: NativeSetupOperation | None
    """Native wait reply, or null until one is acquired."""
    error: SetupError | None
    """First workflow error, not erased by a later read failure."""


class _SimulationResult(_Progress):
    tool: Literal["simulation_initialize"]
    before: _SimulationFacts
    after: _SimulationFacts
    requested: _SimulationRequest
    verification: _SimulationVerification | None


class DeviceSetupResult(_Progress):
    """Setup workflow receipt; native operation and cached facts stay separate."""

    tool: Literal["device_setup"]
    """Fixed tool name."""
    before: DeviceSetupFacts
    """Detached cached facts acquired before submission."""
    after: DeviceSetupFacts
    """Detached post-read facts, never substituted with before facts."""
    requested: DeviceSetupRequest
    """Full candidate after merge; failure does not clear it."""


type SetupResult = _SimulationResult | DeviceSetupResult


class _RecipeGuideResult(TypedDict):
    """Recipe registry identity and its adapter's unmodified wire guide."""

    recipe: str
    """Registered recipe name requested by the caller."""
    adapter: str
    """Adapter name declared by that recipe's definition."""
    guide: AdapterGuideView
    """Complete Markdown and absolute resource directory string or null."""


def simulation_initialize(
    ctx: MeasureToolContext, arguments: dict[str, Any]
) -> ToolReply:
    """Initialize simulation once on ctx's existing GUI connection.

    arguments may contain timeout (default 60 seconds, finite non-bool number in
    0..300). The GUI coordinator may reuse simulation or disconnect real devices
    during switching, so the caller needs authorization for that effect.
    This tool does not connect hardware, apply a project or attach a GUI.

    Returns a ToolReply with op, native operation, four steps, before/after
    SoC/device facts and verification.ready. Missing facts are null.
    Invalid arguments raise ValueError before GUI access. Connection, query and
    mutation failures return an error reply with the known progress. Timeout
    does not cancel; running is not ready, and failed reads do not erase op.
    """
    timeout = wait_timeout(arguments)
    result: _SimulationResult = {
        **_initial_progress(),
        "tool": "simulation_initialize",
        "before": {"soc": None, "devices": None},
        "after": {"soc": None, "devices": None},
        "requested": {},
        "verification": None,
    }
    return _run_setup(ctx, result, timeout, _read_simulation, _verify_simulation)


def device_setup(ctx: MeasureToolContext, arguments: dict[str, Any]) -> ToolReply:
    """Apply a partial Info cfg once to name on ctx's already connected GUI.

    arguments requires a nonempty name and cfg object, with no timeout argument.
    Before Info is read and merged with cfg. A YOKO patch containing value must
    explicitly contain value_unit; a before unit never supplies caller intent.
    All model-required, extra, mode/unit and limit validation belongs to the RPC
    Info model. Connected admission belongs to the GUI device service.

    Returns DeviceSetupResult in ToolReply.data: four stages, requested full
    candidate (null before merge), known MCP op, native operation and detached
    before/after cached snapshots. Wait is bounded to 300 seconds. Timeout
    returns running with the handle, never cancels; use wait(op) to continue.
    Native rejection/driver errors and unknown receipts retain their facts.
    Invalid argument shape raises ValueError before GUI access. No conversion,
    verification, retry, reconnect or rollback is performed.
    """
    name = arguments.get("name")
    patch = arguments.get("cfg")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name must be a nonempty string")
    if not isinstance(patch, dict):
        raise ValueError("cfg must be an object")
    if set(arguments) - {"name", "cfg"}:
        raise ValueError("device_setup accepts only name and cfg")
    result: DeviceSetupResult = {
        **_initial_progress(),
        "tool": "device_setup",
        "before": {"device": None},
        "after": {"device": None},
        "requested": {"name": name, "cfg": None},
    }
    return _run_setup(ctx, result, 300.0, partial(_read_device, patch=patch))


def recipe_guide(
    ctx: MeasureToolContext, arguments: dict[str, Any]
) -> _RecipeGuideResult:
    """Read the guide for arguments['recipe'] on ctx's existing GUI connection.

    recipe must name an entry in the authoritative recipe registry (for example
    t1). Returns recipe, its declared adapter name and guide containing complete
    Markdown and an absolute resource directory string or null. Images use
    relative paths; the agent can read them from resource_dir when needed.
    This tool does not read or attach images. No mutation, operation, attach or
    reconnect occurs. A missing/unknown recipe
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
    verify: Callable[[ResultT], None] | None = None,
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
            if verify is not None:
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
        params = {"name": requested["name"], "cfg": requested["cfg"]}
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
        operation = NativeSetupOperation(
            **wait(ctx, {"op": result["op"], "timeout": timeout})
        )
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
        snapshot = DeviceSnapshotWire(
            **ctx.send_gui_rpc("device.snapshot", {"name": device["name"]})["snapshot"]
        )
        facts["devices"].append(snapshot)


def _verify_simulation(result: _SimulationResult) -> None:
    facts = result["after"]
    soc = facts["soc"]
    devices = facts["devices"]
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
    )
    result["verification"] = {"ready": ready}
    if result["status"] == "finished" and not ready:
        raise GuiRpcError("native facts do not confirm simulation readiness")


def _read_device(
    ctx: MeasureToolContext,
    result: DeviceSetupResult,
    phase: _Phase,
    *,
    patch: DeviceSetupCfg,
) -> None:
    requested = result["requested"]
    snapshot = DeviceSnapshotWire(
        **deepcopy(
            ctx.send_gui_rpc("device.snapshot", {"name": requested["name"]})["snapshot"]
        )
    )
    result[phase]["device"] = snapshot
    if phase != "before":
        return
    info = snapshot.get("info")
    if info is None:
        raise GuiRpcError("before snapshot has no Info to merge")
    if (
        snapshot.get("type_name") == "YOKOGS200"
        and "value" in patch
        and "value_unit" not in patch
    ):
        raise GuiRpcError("YOKO value patch requires explicit value_unit")
    requested["cfg"] = {**deepcopy(info), **deepcopy(patch)}


def build_setup_tools(ctx: MeasureToolContext) -> ToolTable:
    """Build three fixed setup tools over ctx, without connecting or performing setup.

    ctx supplies the application's configured MCP session. Returned handlers
    accept the simulation_initialize, device_setup and recipe_guide argument
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
        "device_setup": {
            "handler": partial(device_setup, ctx),
            "description": "Merge partial cfg with before Info and submit complete cfg once. "
            "YOKO value patches require explicit value_unit. RPC Info owns model validation; "
            "GUI service owns connected admission. Wait up to 300 seconds, without cancellation, "
            "conversion, retry or rollback. Inspect stages, native operation and cached snapshots.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "cfg": {"type": "object"},
                },
                "required": ["name", "cfg"],
                "additionalProperties": False,
            },
        },
        "recipe_guide": {
            "handler": partial(recipe_guide, ctx),
            "description": "Read a registered recipe adapter's complete Markdown guide "
            "and absolute resource_dir string (null when no guide is declared). "
            "Images use relative paths; read them yourself from resource_dir when needed. "
            "This tool does not read or attach images. Uses the authoritative recipe "
            "registry; does not run a recipe or mutate the GUI.",
            "inputSchema": {
                "type": "object",
                "properties": {"recipe": {"type": "string", "minLength": 1}},
                "required": ["recipe"],
                "additionalProperties": False,
            },
        },
    }
