"""MCP servers for zcu_tools — one sub-package per stdio MCP server.

Each server is launchable standalone (registered in ``.mcp.json``). ``agent_memory``
is a self-contained, file-backed lab notebook with no GUI side. The GUI bridges
(measure / fluxdep / dispersive) and the shared transport plumbing (``core``) move
into this package in a later migration phase; until then ``agent_memory`` reuses the
import-clean helpers under ``zcu_tools.gui.remote``.
"""
