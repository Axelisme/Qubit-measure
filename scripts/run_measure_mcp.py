#!/usr/bin/env python
"""Compose user recipe definitions into the measure MCP stdio server."""

from zcu_tools.mcp.measure.server import main

from zcu_lab.recipes import RECIPES

if __name__ == "__main__":
    main(recipes=RECIPES)
