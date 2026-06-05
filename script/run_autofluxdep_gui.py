"""Launcher for the autofluxdep workflow GUI (Phase C prototype).

Opens the node-list / node-detail window. Prototype: Setup builds fake resources
and Run drives the orchestrator on fake data — no hardware. Use it to exercise
the interaction (add/remove/reorder Nodes, Setup→Run enable, edit↔run lock,
auto-follow, Run/Stop toggle, global flux progress).

Example:
    uv run python script/run_autofluxdep_gui.py
"""

from __future__ import annotations

import logging
import sys


def _setup_logging() -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.WARNING)
    handler.setFormatter(logging.Formatter("[%(levelname)-7s] %(name)s: %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)


if __name__ == "__main__":
    _setup_logging()

    # Configure the embedded matplotlib backend BEFORE importing pyplot-using
    # code (the "configure backend before pyplot" invariant), so Phase C's real
    # liveplots route into the GUI rather than detached windows. Import-clean.
    from zcu_tools.gui.plotting.setup import configure_matplotlib_backend

    configure_matplotlib_backend()

    from zcu_tools.gui.app.autofluxdep.app import run_app

    run_app()
