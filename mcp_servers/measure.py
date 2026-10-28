#!/usr/bin/env python
"""Inject the lab description into the measure MCP stdio server."""

from zcu_tools.mcp.measure.server import main as run_measure_mcp

from zcu_lab.assembly import lab_definitions


def main() -> None:
    """Delegate the lab description to the measure stdio server.

    Tool calls explicitly connect or launch the GUI. On EOF the framework drains
    its session and disconnects, without closing the GUI or hardware.
    """
    run_measure_mcp(lab_definitions(), gui_module="apps.measure")


if __name__ == "__main__":
    main()
