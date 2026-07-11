"""Per-app wire / GUI code version constants for the dispersive-fit-gui.

These are the two hand-maintained version integers reported by the no-auth
``wire.version`` handshake. They are intentionally NOT in the shared
``zcu_tools.gui.remote.wire`` module — each GUI app evolves its own wire contract
independently, so the constants live beside the app's own RemoteControlAdapter.

  WIRE_VERSION — the mcp<->RPC *interface contract* (RPC method set, params,
    event shape). The MCP server pins it; a mismatch is a hard MISMATCH. Bump ONLY
    on a contract change.
  GUI_VERSION  — this GUI code's *revision*. Reported, never compared. Bump on any
    meaningful GUI change you want a stale-process check to flag.
"""

from __future__ import annotations

WIRE_VERSION = 4  # v4: event envelopes carry seq/origin

GUI_VERSION = 6  # owner-thread guard and scheduler injection
