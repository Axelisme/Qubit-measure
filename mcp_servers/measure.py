#!/usr/bin/env python
"""Compose user recipe definitions into the measure MCP stdio server."""

from zcu_tools.mcp.measure.server import main as run_measure_mcp
from zcu_tools.resources.entry.registry import component_registry

from zcu_lab.components import register_all
from zcu_lab.recipes import RECIPES


def main() -> None:
    """Bootstrap lab components and delegate recipes to the measure stdio server.

    Tool calls explicitly connect or launch the GUI. On EOF the framework drains
    its session and disconnects, without closing the GUI or hardware.
    """
    register_all(component_registry)
    run_measure_mcp(recipes=RECIPES)


if __name__ == "__main__":
    main()
