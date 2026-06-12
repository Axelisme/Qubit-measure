"""Read-only remote-control bridge for autofluxdep-gui.

The RPC face onto the autofluxdep ``Controller`` — a second View peer to the Qt
MainWindow (ADR-0013). autofluxdep is READ-ONLY over the wire: the agent observes
a workflow the user drives in the GUI. Re-exports the adapter + its options so the
entry script and the MCP server import from one place.
"""

from __future__ import annotations

from .service import ControlOptions, RemoteControlAdapter

__all__ = ["ControlOptions", "RemoteControlAdapter"]
