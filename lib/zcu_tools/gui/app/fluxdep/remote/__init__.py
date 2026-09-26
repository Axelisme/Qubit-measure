"""Remote control + RPC transport for fluxdep-gui.

NDJSON-over-TCP RPC so an automation agent (and an MCP server) can drive the GUI
the same way a user does. The pure-mechanism transport (framing / errors /
param_spec / the wire envelope) is copied from measure-gui; the method set
(method_specs / dispatch) is fluxdep-specific (the pipeline actions on the
Controller).
"""
