"""Measure-gui versions reported by the no-auth ``wire.version`` handshake.

Each app owns its wire contract and GUI code revision. MCP pins and compares
WIRE_VERSION; it reports GUI_VERSION without comparing it.
"""

from __future__ import annotations

# Workflow commands share the measure session and headless tab owners.
WIRE_VERSION = 112

# Bump for meaningful GUI code changes that need a reload signal.
# Wire-contract changes bump both versions.
GUI_VERSION = 145
