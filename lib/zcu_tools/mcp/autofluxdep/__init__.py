"""MCP server bridge package for the autofluxdep-gui ``RemoteControlAdapter``.

A standalone stdio MCP server (``server.py``) that forwards read-only tool calls
to a live autofluxdep-gui over a TCP control socket. Mirrors the fluxdep /
dispersive bridges; see ``server.py`` for the config + lifecycle tools.
"""

from __future__ import annotations
