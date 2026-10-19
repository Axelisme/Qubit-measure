"""Measure-gui versions reported by the no-auth ``wire.version`` handshake.

Each app owns its wire contract and GUI code revision. MCP pins and compares
WIRE_VERSION; it reports GUI_VERSION without comparing it.
"""

from __future__ import annotations

# Predictor sources use entry-local JSON filenames instead of qubit names.
WIRE_VERSION = 111

# Bump for meaningful GUI code changes that need a reload signal.
# Wire-contract changes bump both versions.
GUI_VERSION = 144
