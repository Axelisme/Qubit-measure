"""Shared transport layer for the GUI apps' remote control + MCP bridges.

App-agnostic (no app import); all four apps (``main`` / ``fluxdep`` /
``dispersive`` / ``autofluxdep``) build on these. This package is not eager —
each module is imported on demand, so the import-clean primitives stay free of Qt
even though the transport mechanisms pull it in. Two tiers:

- **Import-clean wire primitives** (no Qt, no matplotlib): ``framing`` (NDJSON),
  ``errors`` (envelopes), ``wire`` (Request/Response + field coercion),
  ``param_spec`` (the ParamSpec schema engine), ``method_spec``
  (MethodSpec/BoundMethod + ``build_method_registry``).
- **Transport mechanisms**: ``rpc_endpoint`` — the ``NdjsonRpcEndpoint`` GUI-side
  server (socket + framing + handshake + push fan-out + the
  ``MainThreadDispatcher`` Qt marshal primitive; Qt-aware); ``mcp_bridge`` — the
  ``McpBridge`` client-side bridge (socket + RID routing + lifecycle +
  ``run_stdio_loop``) the standalone MCP servers run on.

Each app keeps only its own domain: dispatch tables, method specs, the
``RemoteControlAdapter`` router (its ``route`` seam + event serializers) and the
``mcp_server`` config + overrides + any guard/operation/diagnostic policy.
"""
