#!/usr/bin/env python
"""Compose user recipe definitions into the measure MCP stdio server."""

from zcu_tools.mcp.measure.server import main
from zcu_tools.resources.entry.registry import component_registry

from zcu_lab.components import register_all
from zcu_lab.recipes import RECIPES

if __name__ == "__main__":
    register_all(component_registry)
    main(recipes=RECIPES)
