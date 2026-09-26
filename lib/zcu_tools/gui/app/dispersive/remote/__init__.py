"""Remote control + RPC transport for dispersive-fit-gui (read-only surface).

NDJSON-over-TCP RPC so an automation agent (and an MCP server) can *observe* the
GUI the same way fluxdep-gui's remote does. The pure-mechanism transport is the
shared ``zcu_tools.gui.remote`` layer + a copied ``RemoteControlAdapter``; the
method set (method_specs / dispatch) is dispersive-specific and read-only — the
user drives load / preprocess / tune / fit / export in the GUI.
"""
