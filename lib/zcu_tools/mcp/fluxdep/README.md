# Fluxdep MCP control

**Last updated:** 2026-10-06, analysis pipeline tools and native image delivery

Fluxdep MCP forwards the GUI's declared analysis methods over one shared bridge.
The GUI owns published-resource observations, interactive identities and search
operations. MCP does not maintain another guard or operation registry.

`assembly.py` combines generated pipeline tools with shared lifecycle tools.
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
