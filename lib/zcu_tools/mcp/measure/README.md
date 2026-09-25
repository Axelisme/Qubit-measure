**Last updated:** 2026-09-26 — GUI-owned operation status and wait

# `zcu_tools/mcp/measure/`

This package is the measure-gui MCP server. The shared `zcu_tools.mcp.core`
bridge owns socket and stdio transport; this package owns measure-specific
connection, guarded requests and fixed tool handlers. GUI core does not import
this package.

## Boundaries

- `server.py` creates one bridge, session and tool table per stdio session.
  Exiting MCP closes only its socket. Starting and stopping the GUI are
  separate, explicit operations.
- `assembly.py` registers handwritten tools. Wire methods never generate a
  dynamic MCP tool inventory. The connection ticket registers `connect` and
  `rpc_list` / `rpc_describe` / `rpc_call`; later tickets add the remaining
  specialized tools.
- `session.py` validates `wire.version`, loads `rpc.catalog` from the live GUI
  on each connection and caches that connection's policy, observed resource
  versions. Reconnecting discards the old observations.
  A failed or timed-out mutation is not automatically sent again.
- The GUI's `services.remote.method_entries` own each method's agent exposure,
  version guard dependencies, revealed resources and operation key. The MCP
  does not keep a separate method or guard table. `session_policy.py` only
  expands resource patterns and describes stale keys.
- `tool_context.py` binds fixed handlers to their session. Ordinary RPC
  timeouts come from the live catalog; wait methods supply their own deadline.
  GUI handlers validate parameter values and return stable error reasons.
- `tools_operation.py` reads current orientation and GUI-owned live operation
  handles for `status` and the `connect` reply. `wait` and `cancel` address known
  GUI handles, including GUI-started operations. Unknown or evicted handles
  fail; failed work is a returned outcome. Internal methods are not available
  through `rpc_call`.

There is no measure MCP event subscription, diagnostic queue or reply
piggyback. The GUI's own EventBus and wire event transport remain available
to other consumers. Operation state is read through request/reply methods.

## Tests

`tests/mcp/measure/` uses in-process recording transports. GUI control-socket
contract tests live in `tests/gui/app/main/services/remote/`; headless Qt
runs use `QT_QPA_PLATFORM=offscreen`, `QT_QPA_PLATFORMTHEME=` and
`MPLBACKEND=Agg`. Tool inventory and module boundaries are reviewed directly.
