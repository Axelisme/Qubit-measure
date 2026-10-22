#!/usr/bin/env python
"""Measure MCP stdio server, attached to a live GUI control socket."""

from __future__ import annotations

import logging
import runpy
from collections.abc import Sequence
from pathlib import Path
from tempfile import gettempdir
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zcu_tools.mcp.measure.recipe_api import RecipeDeclaration

logger = logging.getLogger(__name__)

# Validate GUI dependencies before assembling the injected server.
_BOOTSTRAP = runpy.run_path(str(Path(__file__).resolve().parents[1] / "_standalone.py"))
_BOOTSTRAP["bootstrap_standalone_server"](
    __file__,
    required_modules=(
        (
            "qtpy",
            "measure-gui MCP server requires the 'gui' extra (qtpy + PyQt6); "
            "'{module}' is missing. Rebuild the environment with:\n"
            "    uv sync --extra gui\n",
        ),
        (
            "PyQt6",
            "measure-gui MCP server requires the 'gui' extra (qtpy + PyQt6); "
            "'{module}' is missing. Rebuild the environment with:\n"
            "    uv sync --extra gui\n",
        ),
    ),
)

# tolerated: Standalone lib-path setup and GUI dependency preflight must precede GUI imports.
from zcu_tools.gui.app.measure.remote.wire_version import (  # noqa: E402
    WIRE_VERSION as MCP_WIRE_VERSION,
)

# tolerated: Standalone lib-path setup and GUI dependency preflight must precede bridge imports.
from zcu_tools.mcp.core.bridge import (  # noqa: E402
    McpBridge,
    MCPBridgeConfig,
    port_is_open,
    resolve_connect_port,
)

# tolerated: Standalone lib-path setup and GUI dependency preflight must precede tool assembly.
from zcu_tools.mcp.measure.assembly import build_measure_tools  # noqa: E402

# tolerated: Standalone lib-path setup and GUI dependency preflight must precede session imports.
from zcu_tools.mcp.measure.session import MeasureMcpSession  # noqa: E402

# tolerated: Standalone lib-path setup and GUI dependency preflight must precede tool context imports.
from zcu_tools.mcp.measure.tool_context import MeasureToolContext  # noqa: E402

# v89: forward GUI-owned guard requests once and expose observed operation state.
# v90: fixed cfg/library tools use shared drafts and report partial library edits.
# v91: run and analyze tools share GUI operations, bounded waits and complete results.
# v92: writeback forwards preview and explicit-item batch writes to the GUI owner.
# v93: artifact save, guarded tab close and graceful shutdown tools.
# v94: fixed tab_interact tool over the shared GUI plugin session.
# v95: finalized shared-state workflow guidance and interactive concurrency instructions.
# v96: the project tool applies through the project.apply GUI method.
# v98: tab_run requires and forwards the caller's explicit cfg ref once.
# v99: generic RPC accepts all public methods, including tool-backed commands.
# v100: accept writes all current Primary and Post candidates with partial progress.
# v103: Lookback recipe execution and shared cancel/finish-early control.
# v104: Recipe interaction delivery and post-Run result provenance checks.
# v105: Onetone recipes with GUI-owned ranges and source-bound Run preview.
# v106: Two-tone spectrum and Rabi recipes with explicit frequency source precedence.
# v107: recipe-first fixed tools, shared analysis/control and complete public RPC.
# v108: analysis results retain invalid paths from the GUI's wire projection.
# v109: flux unit assertions and explicit native coordinate opt-in for FakeDevice.
# v111: injected generator recipes, captured questions and explicit answer/writeback.
# v112: status projects the opened EntryInfo rather than chip/qubit identity.
# v113: bounded partial device_setup over complete Info device.setup RPC.
# v114: integration revision above the base's device_setup release.
# v115: read-only predictor inputs and readiness without an installed predictor.
# v116: explicit working-point selection and saved writeback destinations.
# v117: holder/ref-free cfg integrated above the working-point and UI releases.
# v122: predictor RPC uses entry-local filenames or independent inline params.
# v123: actionable launch collisions and explicit raw-save analysis observations.
# v124: Async declarations, request-scoped answers, full owner evidence and scoped cancel.
MCP_VERSION = 125

_SERVER_INSTRUCTIONS = """\
Attach to the live qubit-measure GUI with connect. This does not connect hardware.
Use launch='if_missing'/'new' only when authorized to start a GUI. Pass token if
required, keep it secret, and inspect connect.status before acting. The user may
also edit the GUI. Reconnection reloads its live catalog; incompatible wire
versions fail before forwarding an action.

For routine measurement, prefer the recipe matching the experimental goal.
Read its tool schema and adapter guide through recipe_guide(adapter).
For analysis of existing data, use tab_analyze and tab_interact rather than
starting another measurement. If tools are deferred by the client, discover
the recipe or shared analysis/control tool first. Routine work and diagnosis
are guidance, not modes or permissions. For detailed setup, cfg, save,
writeback or diagnosis, use rpc_list(domain?), rpc_describe(method), then
rpc_call(method, params). Every listed public method remains callable even
when a recipe or shared tool covers it. Raw RPC does not aggregate tool
results, decode PNG replies, or run the analysis artifact save pipeline.


For an authorized launch, connect with launch='new' and script set to a trusted
Python file defining startup(ctx). It replaces the default startup script;
clean=True skips all scripts and cannot be combined with script. Scripts may
connect hardware, so review the selected script and obtain authorization first.
For offline FTF use zcu_lab/simulation_envs/FTF/startup.py after preparing its YAML.
Add project_root to isolate result/Database data; it must be an absolute path to
an existing directory. Omit it to use the repo root. Code, catalogs and script
paths keep their existing resolution. Explicit script or project_root options
are refused when attaching; the existing GUI stays unchanged and no startup is
rerun. Close it before selecting another script or data root. Entry names remain
single path segments.
device_setup(name, cfg) accepts a partial Info patch, reads before, merges full
cfg, submits once and waits at most 300 seconds. It has no timeout argument.
When before Info stores value_unit, value patches require caller-supplied
value_unit; output/rampstep-only patches do not. RPC Info validators own
field/limit/mode-unit validation.
For asynchronous setup or a shorter wait, read device.snapshot, build complete
Info cfg, call device.setup(name, cfg), wait(op), then read device.snapshot.
Raw RPC requires every Info model-required field and never merges cached values.
YOKO mode is required; raw RPC may omit value_unit, generated by Info from mode.
Inspect stages, native operation, requested full candidate and before/after
cached facts. Requested.cfg is null until before-read and merge complete; a
candidate is not a validation receipt. Timeout returns running with a known op,
without cancelling. Unknown receipts do not prove no side effects. No unit
conversion, automatic reconnect, retry or rollback; use wait(op) to continue.

A recipe call waits up to 300 seconds before returning a still-running execution;
missing parameters, failures and interactive handoffs return earlier. Configure
the client deadline above 300 seconds with room for transport and reply overhead.
The stdio server is synchronous: another request on that connection is not
guaranteed service during the first wait. A client timeout is not cancellation.
Never automatically rerun a recipe or mutation after timeout, disconnect or busy.
Inspect current state and confirmed files before deciding.

Use global status for GUI operations, non-terminal executions and terminal count.
Keep known execution IDs for later queries; global status does not embed history.
status(execution) reads progress and the live scope path without reconnecting.
Completed, failed and cancelled recipe wait summaries contain only status,
authored report and runs. Status adds active_path and structured live_paths, including at terminal.
A failure's diagnostic is available in status(execution, detail="full").
Nonterminal summaries include execution. A paused question supplies
pending_question with request_id, evidence and answer_schema. Interactive
analysis supplies pending={tab,stage,op}; complete that same GUI operation with
tab_interact. Missing parameters return needs_parameters, missing paths and
outcome=null, not a failed RecipeOutcome.
Full recipe detail retains per-Run/tab/operation conditions from the owner's
fixed Form observation, native analysis estimates and invalid paths, and explicit
writeback candidates. Reading detail never reruns analysis, writes or saves.
Runs retain acquisition run_id and every explicit Save attempt, even failed
outputs. Output status/error, not run_id, determines save success.
Post result reads use the owner's start-receipt budget: 5 seconds plus 5 per
measured tone, capped at 60 seconds; an absent budget uses 5 seconds.
A result_read failure retains recovery with rpc, params, original op and stage.
First read tab.snapshot and compare post_analysis_state.source_operation_id with
recovery.params.operation_id. If different, report superseded and do not deliver
the replacement as the old result. Otherwise rpc_call the supplied source-bound
route; replacement races return result_superseded. Do not rerun analysis to
recover a result read. GUI Post read handlers have a 60-second ceiling.
Standalone analysis retains its native summary and Save reports. Artifacts
preserve producer sections, names, members, paths, formats, statuses and errors. Synchronous tab.save_data and
tab.save_artifacts return the complete current report, not a Save handle. No Save
await or snapshot reread is needed. Reserved destinations are not saved files.
A resave replaces the selected artifacts with that attempt, not a saved prefix.
Status never attaches images; wait(op) observes only the GUI operation;
wait(execution) includes downstream reads, saves and current image delivery.
Failed outcomes are data. A wait timeout stops waiting,
not the operation or its continuation. Operation handles belong to one GUI
connection generation; execution IDs belong to this MCP server session and
are not durable recovery tokens. After GUI reconnect, discover current handles
through status; do not reuse an old op.

cancel(execution) requests root recipe stop; native workflow roots use scope="workflow".
The next await raises the shared ordinary Exception ScriptCancelled. An in-flight
Run drains and retains run_id/partial/completion facts before raising. The author
continues synchronous finally cleanup; no new await operation starts after root stop.
Synchronous edits and explicit saves still run. The terminal status is cancelled
with report=null and retained runs. cancel accepts exactly one of execution,
run_id (native workflow), or op. Narrow scopes preserve their parents:
- acquire stops one acquisition; the next acquire in the same Schedule is clean.
- schedule stops this Schedule and its later acquires; the next operation is clean.
- iteration skips the current point (跳過目前這一點). Releasing it lets the next
  iteration start clean. Authors continue the main loop or return from a point helper;
  a synchronous finally commit preserves the point without awaiting during cleanup.
- run stops the current operation. Recipe admits only an active tab.run; analysis
  and questions continue. Workflow admits any operation. operation is its label.
Without a recipe Run, run/schedule/acquire return cancelled=false with reason
"no run in progress". Status includes active_path. For example,
cancel(execution="recipe-...", scope="acquire", expected=3) accepts only acquire#3
within the current GUI Run root. Sequences may restart in the next Run. For exact
cross-Run targeting use cancel(op=17, scope="acquire", expected=3) with the observed
Run handle; stale fine-scope op targets return "scope not active", never cancel
the next Run. Generic cancel(op=17) retains script Stop behavior.
Status active_path is the single root-to-first-fork chain. live_paths is a list
of every live leaf full path, not a comma-joined string. Select by kind, kind#N,
operation:<name>, operation:<name>#N, or an exact full path from live_paths.
Multiple matches return cancelled=false, reason="ambiguous", and full-path candidates.
Use #N for same-parent same-name operations; cross-parent kind#N may remain ambiguous,
so use the exact full path. No latest-child or callable-reflection selection occurs.
expected and N are positive integers, not bool, and must agree if both supplied.
Malformed syntax is invalid_params; inactive selectors and stale numbers return
scope not active or sequence mismatch. Roots have no sequence.
Accepted requests return cancelled=true, reason="scope cancellation requested".
This is not cleanup completion; use wait/status. Recipe schedule/acquire use the
exact GUI Run handle; its active_path appends the GUI tail to local scopes.
live_paths and acknowledgement candidates use the same projection. It is not atomic
across processes. Expected analysis fitting failures return status="failed", result=null
and reason for the author to decide; arbitrary programming or recording errors still raise.
ScriptCancelled is not recorded as an error; genuine cleanup/owner failures are retained.
Standalone analysis cancellation
stops its continuation. Unregistered cancel(op) keeps the direct GUI operation hook.
Terminal execution results are not rewritten.

Library creation uses rpc_call(method="context.ml_create_from_role", params=...).
List templates with context.ml_list_roles. Provide role_id and a new name such as
"R1.readout.mux". Optional edits use the same shape as context.ml_edit:
[{"path": ["tones"], "value": [{"freq": 100.0, "gain": 0.2, "ro_ch": 0}]}].
Roles with required fields, such as readout/mux:blank's tones, need these edits
before creation. Expressions use {"expr": "<text>"}. The GUI applies edits to
template input before validation and writes only a valid Spec. Invalid paths,
values or Form results return invalid_params with data.errors: [{path, message}]
and leave the library unchanged. Success returns {created: name}.

Tab get_cfg, edit_cfg, reset_cfg and use_inline return Form input/values/errors,
without schema. Use tab.describe_cfg(path=[]) for root fields, then request a
child's full model-field path for its metadata and direct children. Options
contain label/value and commands, not fields; the owner reports selected.
Select by writing the option's whole value once. A single ordinary wrapper may
be flattened in children, but paths still include it. Ref options have no children.
Describe replies include current local input_present/input/value and Form preview
errors from one owner turn. For lowered cfg errors use get_cfg or Run admission.
Recipe tab.read makes one atomic read_recipe_field RPC per call from fixed
sources. Its locked flag includes selection_fixed, fixed and readonly.
Ordinary strings are literal; expressions use {expr: text}, complex uses the
reversible __complex__ tag, and library keys are actual keys. Read description
options in order; do not infer choices from unions. Invalid input remains editable.
Generated tag leaves are not recipe logical paths.
Inline leaf paths include module. use_inline runs the owner's guarded to_inline
command against fixed library capture, not a private copy path.
Edit unique model-field leaf paths;
write range start/stop/n separately. Invalid input remains editable with errors.
Open and explicit Reset capture sources once. Recipe preparation always Reset
before edits, including new tabs. Reset(aliases=...) is the single-use source
override; it changes neither global aliases nor saved input. Parameter reads,
cfg, edits and Run use that fixed capture until the next Reset. parameter_read
and get_cfg take no aliases. run_start rejects nonempty aliases before focus or
operation effects; None and {} use displayed values. Run accepts only current
valid Form values. Load restores results and analysis sources, not Form input;
inspect the result's native cfg separately. Load replies report result/analysis
facts, not cfg backfill. apply_writeback(tab, items) selects current
Primary/Post candidates by stable target_name. Omit items for all; an empty array
writes none. It ignores checkboxes and rejects unknown or ambiguous names before
any write, using current IDs rather than proposal IDs. It does not answer recipes.
apply_writeback stops on the first error and reports confirmed
progress without retry or rollback. Inspect proposals and the destination first.

When a recipe is paused, answer(execution, request_id, answer) supplies named
fields described by pending_question.answer_schema. request_id is required.
Stale or absent requests return accepted=false, reason and pending_request_id
(null without a pending request). Rejection does not consume the question.
Answer is not a write or permission; only explicit tab.accept_writeback writes.
Only one nonterminal recipe is admitted.

tab_interact without payload reads committed state and available commands.
Send payload={command,args} for one action. Later owner-loop commits win.
Reads preserve focus; commands follow Analysis.
done joins the original completion. For a recipe-owned analysis it waits for
that recipe's next result or question; standalone analysis retains its execution.
preview_active is a local interactive preview, not committed state. Current tool
replies may attach in-memory image content without session PNG files. To see a
figure later, use tab.get_figure for the GUI's current image; it is not guaranteed
to be the earlier image. Persistent outputs are recorded only in artifacts.
Recipes do not automatically close tabs. Use tab_close explicitly; busy cannot
be bypassed with discard_unsaved. app.shutdown via RPC requests graceful exit;
its reply is not proof that the responding process has exited.

The server disconnects and joins its workers on exit. It does not close the GUI
or promise to stop hardware. There are no subscribed MCP events: read snapshots or wait instead. Follow run-measure-gui
for hardware safety, task policy and measurement workflow.
"""

_CONFIG = MCPBridgeConfig(
    app_name="gui",
    app_slug="measure",
    tool_prefix="",
    default_port=8765,
    mcp_version=MCP_VERSION,
    wire_version=MCP_WIRE_VERSION,
    server_display_name="qubit-measure-control",
    server_instructions=_SERVER_INSTRUCTIONS,
    pid_file=Path(gettempdir()) / "zcu_tools_gui.pid",
    log_file=Path(gettempdir()) / "zcu_tools_gui_debug.log",
    run_script_name="run_measure_gui.py",
)


def _setup_logging() -> None:
    from zcu_tools.gui.logging_setup import setup_gui_logging

    setup_gui_logging(
        app_name="measure",
        log_root=Path(__file__).resolve().parents[4],
        group="mcp",
        extra_namespaces=("zcu_tools.mcp",),
    )


def main(*, recipes: Sequence[RecipeDeclaration]) -> None:
    """Run the stdio server with explicitly supplied async declarations.

    recipes is the composition root's immutable declaration sequence. Invalid
    definitions fail during session assembly, before GUI operations. On EOF/error,
    disconnect and drain session workers/PNGs; do not close the GUI or hardware.
    """
    from zcu_tools.mcp.core.stdio_server import StdioLoopHooks, run_stdio_loop

    session = MeasureMcpSession(
        _CONFIG,
        recipes=recipes,
        resolve_connect_port=resolve_connect_port,
        port_is_open=port_is_open,
    )
    bridge = McpBridge(_CONFIG)
    session.attach_bridge(bridge)
    context = MeasureToolContext(
        config=_CONFIG,
        session=session,
        resolve_connect_port=resolve_connect_port,
    )

    try:
        run_stdio_loop(
            _CONFIG,
            build_measure_tools(context, recipes=recipes),
            hooks=StdioLoopHooks(
                on_start=_setup_logging,
                on_error=logger.exception,
            ),
            server_version="1.1.1",
        )
    finally:
        session.close()
