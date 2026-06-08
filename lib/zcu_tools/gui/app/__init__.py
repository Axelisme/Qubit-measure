"""GUI app composition roots.

One subpackage per standalone GUI app: ``main`` (the measure-gui), ``fluxdep``
(flux-dependence analysis), ``dispersive`` (dispersive-shift fitting), and
``autofluxdep`` (automated multi-task flux sweeps).

The extracted shared layers (Stage E) carry the *mechanism*, never the domain
(decision D2 — domain stays per-app: each keeps its own state, handlers, method
specs, serializers):

- ``gui/remote`` — the shared RPC wire layer: wire primitives (``framing`` /
  ``errors`` / ``wire`` / ``param_spec`` / ``method_spec``) and the
  ``NdjsonRpcEndpoint`` GUI-side server (``rpc_endpoint``). The client-side
  ``McpBridge`` + the launchable MCP servers live under ``zcu_tools.mcp`` (a
  consumer of these primitives).
- ``gui/event_bus`` — the ``BaseEventBus`` / ``BasePayload`` publish/subscribe
  mechanism (payload-type keyed); each app supplies its own event enum +
  payloads. (main keeps a separate enum-keyed scheme; it does not build on this.)
- ``gui/plotting`` — the embedded matplotlib backend + plot host.

Each app's ``RemoteControlAdapter`` is a thin router over the shared endpoint;
its launchable MCP server (config + overrides over the shared bridge) lives under
``zcu_tools.mcp.<app>``. App-specific policy (measure-gui's version guard /
operation tracking / diagnostic channel) lives only in that app, never in the
shared layer.
"""
