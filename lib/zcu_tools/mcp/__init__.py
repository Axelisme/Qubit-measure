"""MCP servers for zcu_tools — one sub-package per stdio MCP server.

Each server is launchable standalone (registered in ``.mcp.json``). The three GUI bridges
(``measure`` / ``fluxdep`` / ``dispersive``) and the shared
transport plumbing (``core``) live here; the bridges build on the import-clean
transport primitives under ``zcu_tools.gui.remote``.
"""
