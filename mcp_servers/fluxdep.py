#!/usr/bin/env python
"""Supply the fluxdep GUI module to the framework MCP stdio server."""

from zcu_tools.mcp.fluxdep.server import main as run_fluxdep_mcp


def main() -> None:
    """Run stdio tools; GUI launch requires an explicit lifecycle tool call."""
    run_fluxdep_mcp(gui_module="apps.fluxdep")


if __name__ == "__main__":
    main()
