"""Measure-gui versions reported by the no-auth ``wire.version`` handshake.

Each app owns its wire contract and GUI code revision. MCP pins and compares
WIRE_VERSION; it reports GUI_VERSION without comparing it.
"""

from __future__ import annotations

# The combined lazy-description and flat Readout contract differs from both v104 branches.
WIRE_VERSION = 105

# Bump for meaningful GUI code changes that need a reload signal.
# Wire-contract changes bump both versions.
GUI_VERSION = 138
