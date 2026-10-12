# Fluxdep MCP control

**Last updated:** 2026-10-13, named alignment decisions and guarded polling

Fluxdep MCP composes native GUI RPCs into handwritten agent decision tools over
one shared bridge.
The GUI owns control authority, interactive identities and search operations.
Business calls require holder unless explicitly readonly. Reads return current
projections, not per-connection mutation baselines. MCP does not maintain another
authority or operation registry. Domain validity, busy and owner-lifetime checks
remain independent. See [ADR-0070](../../../../docs/adr/0070-control-identity.md).

`assembly.py` combines explicit pipeline/control tools with Fluxdep lifecycle
initialization. `fluxdep_connect(label="agent", mode="owner")` initializes fixed
socket identity after authentication and wire checks. Owner auto-acquires unlocked
Local; a GUI lock or another online holder still permits connection. The reply
contains native `control` (`holder`, `locked`, `online`, `me`), project paths,
published spectrum stages/drafts and fit freshness. An optional project supplies
only chip/qubit names; GUI computes `result_dir` and `data_root`. Wire mismatch
still rejects the connection, but versions are not part of the agent receipt. `readonly` never acquires, even when its
label matches the holder. Omitted label/mode retain the prior choices on reconnect;
launch supports the same choices, including a later attach after auto_connect=false.

`fluxdep_control_status`, `fluxdep_control_acquire` and `fluxdep_control_release`
forward to the GUI authority. `not_holder`, `control_locked`, `control_held` and
`readonly_connection` become MCP isError replies with JSON `error` text preserving
native code/message/reason/data. MCP never reclassifies these refusals or retries.
Dispersive's shared readonly lifecycle does not initialize a control identity.

`spectrum_tools.py` composes list with Filter/fit reads, and raw load with snapshot
and render. `projection.py` converts complete RPC point arrays into count/ranges,
rounds numerical spectrum summaries to three decimals and preserves GUI
alignment provenance. Load and snapshot attach the native evidence PNG as image
content, not base64 text or an image file. PNG integrity uses the same core
producer as Measure. MCP retains no spectrum, draft, provenance or fit state.

`align_tools.py` implements J2 as `fluxdep_align(name, command, params?, image?)`.
It opens/reuses the named line draft, sends commands and summarizes half/integer
lines, period and changed fields as before/after pairs. Every move_line first
sets conjugate, defaulting to false, so the command does not inherit a hidden GUI
mode. auto_align polls interactive.read once per second with the starting
context_id until the GUI is not busy. A retired draft becomes an interrupted tool
error; MCP does not compare identities or keep a registry. finish accepts manual
or load-inherited lines without auto_align and attaches spectrum.render evidence.
image=true requests interactive.render for unfinished drafts. GUI open receipts
supply discarded stage facts; downstream invalidation belongs to app publication,
not MCP inference.

The stdio entry supplies connection settings and instructions. Server exit only
disconnects; it does not kill or shut down the GUI. Tools preserve scalar JSON
types for GUI validation. Their descriptions identify composed reads; they do not retry, reconnect or rollback. Timeout/image failure may follow an accepted GUI
mutation. Failed search outcomes remain data, distinct from invocation errors.

Public tool/stdio behavior is tested through recording Transport in
`tests/mcp/fluxdep/`. Native GUI/domain behavior belongs to the app remote suite.
The agent workflow is `.agents/skills/run-fluxdep-gui/SKILL.md`.
