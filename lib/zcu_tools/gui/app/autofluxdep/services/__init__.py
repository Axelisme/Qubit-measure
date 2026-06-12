"""App-local service sub-packages for autofluxdep-gui.

Currently holds only the read-only ``remote`` bridge (the RPC/MCP face onto the
Controller). The session services (connection / context / device / startup) are
the shared ones composed in the Controller, not app-local; this package is the
home for autofluxdep's own service code, of which the remote bridge is the first.
"""

from __future__ import annotations
