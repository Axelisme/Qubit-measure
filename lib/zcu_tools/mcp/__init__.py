"""MCP servers for zcu_tools — one sub-package per stdio MCP server.

Each server is launchable standalone (registered in ``.mcp.json``). ``agent_memory``
is a self-contained, file-backed lab notebook with no GUI side. The four GUI bridges
(``measure`` / ``fluxdep`` / ``dispersive`` / ``autofluxdep``) and the shared
transport plumbing (``core``) live here; the bridges build on the import-clean
transport primitives under ``zcu_tools.gui.remote``.
"""
