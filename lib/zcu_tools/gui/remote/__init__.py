"""Shared transport layer for the GUI apps' remote control + MCP bridges.

App-agnostic (no app import); all four apps (``main`` / ``fluxdep`` /
``dispersive`` / ``autofluxdep``) build on these. This package is not eager —
each module is imported on demand, so the import-clean primitives stay free of Qt
even though the transport mechanisms pull it in. Two tiers:

- **Import-clean wire primitives** (no Qt, no matplotlib): ``framing`` (NDJSON),
  ``errors`` (envelopes), ``wire`` (Request/Response + field coercion),
  ``param_spec`` (the ParamSpec schema engine), ``method_spec``
  (MethodSpec/BoundMethod + ``build_method_registry``).
- **Transport mechanism**: ``rpc_endpoint`` — the ``NdjsonRpcEndpoint`` GUI-side
  server (socket + framing + handshake + push fan-out + the
  ``MainThreadDispatcher`` Qt marshal primitive; Qt-aware). The client-side
  ``McpBridge`` + ``run_stdio_loop`` the standalone MCP servers run on now lives in
  ``zcu_tools.mcp.core.bridge`` (a consumer of the wire primitives above).

Each app keeps its own domain: dispatch tables, method specs, the
``RemoteControlAdapter`` router (its ``route`` seam + event serializers); the
launchable MCP server bridge (config + overrides + any guard/operation/diagnostic
policy) lives under ``zcu_tools.mcp.<app>``.
"""
