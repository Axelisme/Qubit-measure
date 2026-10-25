#!/usr/bin/env python
"""Supply the dispersive GUI module to the framework MCP stdio server."""

from zcu_tools.mcp.dispersive.server import main as run_dispersive_mcp


def main() -> None:
    """Run stdio tools; GUI launch requires an explicit lifecycle tool call."""
    run_dispersive_mcp(gui_module="apps.dispersive")


if __name__ == "__main__":
    main()
