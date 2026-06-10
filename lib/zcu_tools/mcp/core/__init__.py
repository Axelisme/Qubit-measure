"""Shared MCP transport for the zcu_tools MCP servers.

``bridge`` holds the stdio JSON-RPC protocol loop, the ``McpBridge`` client (socket
+ RID routing + GUI-subprocess lifecycle), and the tool-generation helpers — the
plumbing every server under ``zcu_tools.mcp.*`` runs on. It consumes the
app-agnostic wire primitives (``param_spec`` / ``method_spec``) from
``zcu_tools.gui.remote``, the shared remote layer the GUI's own RPC endpoint uses
too; ``mcp`` is a consumer of that layer, not a leaf.
"""
