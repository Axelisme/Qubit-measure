# Fluxdep MCP control

**Last updated:** 2026-10-11, holder admission and structured control errors

Fluxdep MCP forwards the GUI's declared analysis methods over one shared bridge.
The GUI owns control authority, interactive identities and search operations.
Business calls require holder unless explicitly readonly. Reads return current
projections, not per-connection mutation baselines. MCP does not maintain another
authority or operation registry. Domain validity, busy and owner-lifetime checks
remain independent. See [ADR-0070](../../../../docs/adr/0070-control-identity.md).

`assembly.py` combines generated pipeline/control tools with Fluxdep lifecycle
initialization. `fluxdep_connect(label="agent", mode="owner")` initializes fixed
socket identity after authentication and wire checks. Owner auto-acquires unlocked
Local; a GUI lock or another online holder still permits connection. The reply
contains native `control` (`holder`, `locked`, `online`, `me`) and versions. GUI
revision is reported, never compared. `readonly` never acquires, even when its
label matches the holder. Omitted label/mode retain the prior choices on reconnect;
launch supports the same choices, including a later attach after auto_connect=false.

`fluxdep_control_status`, `fluxdep_control_acquire` and `fluxdep_control_release`
forward to the GUI authority. `not_holder`, `control_locked`, `control_held` and
`readonly_connection` become MCP isError replies with JSON `error` text preserving
native code/message/reason/data. MCP never reclassifies these refusals or retries.
Dispersive's shared readonly lifecycle does not initialize a control identity.

Interactive replies retain the captured context/effect and deliver their PNG as
MCP image content. Text carries MIME/byte metadata instead of base64; no image
files are created. PNG integrity is checked by the same core producer used by
Measure's existing consumers.

The stdio entry supplies connection settings and instructions. Server exit only
disconnects; it does not kill or shut down the GUI. Tools preserve scalar JSON
types for GUI validation. They do not hide reads, retry, reconnect or rollback. Timeout/image failure may follow an accepted GUI
mutation. Failed search outcomes remain data, distinct from invocation errors.

Public tool/stdio behavior is tested through recording Transport in
`tests/mcp/fluxdep/`. Native GUI/domain behavior belongs to the app remote suite.
The agent workflow is `.agents/skills/run-fluxdep-gui/SKILL.md`.
