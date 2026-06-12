"""Launcher for the autofluxdep workflow GUI (Phase C prototype).

Opens the node-list / node-detail window. Prototype: Setup builds fake resources
and Run drives the orchestrator on synthetic data — no hardware. Use it to
exercise the interaction (add/remove/reorder/rename Nodes, Setup→Run enable,
edit↔run lock, auto-follow, Run/Stop toggle, global flux progress).

DEBUG output is written to a per-session file under ``logs/gui/autofluxdep/`` at
the repo root (the whole run loop: each flux point, each Node entered / skipped /
fitted). Disable with ``--no-log`` or redirect with ``--log-file``.

Example:
    uv run python script/run_autofluxdep_gui.py
    uv run python script/run_autofluxdep_gui.py --no-log
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from zcu_tools.gui.logging_setup import setup_gui_logging

# Repo root: this script lives in script/, so its parent is the root.
PROJECT_ROOT = Path(__file__).parent.parent


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_autofluxdep_gui",
        description="Launch the autofluxdep workflow GUI",
    )
    parser.add_argument("--no-log", action="store_true", help="Disable file logging")
    parser.add_argument(
        "--control-port",
        type=int,
        default=None,
        help="Start the read-only remote-control TCP server on this port (for agents/MCP)",
    )
    parser.add_argument(
        "--control-token",
        type=str,
        default=None,
        help="Shared auth token required by remote-control clients",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Override the DEBUG log file path",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])
    setup_gui_logging(
        app_name="autofluxdep",
        log_root=PROJECT_ROOT,
        to_file=not args.no_log,
        log_file=Path(args.log_file) if args.log_file else None,
    )

    # Force a NON-interactive backend BEFORE any pyplot import. Unlike
    # measure-gui / fluxdep, autofluxdep does NOT use the custom
    # ``module://zcu_tools.gui.plotting.backend`` (that backend is for the
    # worker-draws-then-marshals model — ADR-0017). autofluxdep's worker never
    # touches matplotlib; the main thread embeds each Node's figure with a bare
    # ``FigureCanvasQTAgg`` (ADR-0017). With ``Agg`` selected, pyplot can never
    # pop a detached window, while the directly-constructed embed canvases work
    # as normal. (Selecting the custom pyplot backend here would route stray
    # pyplot figures to detached windows — the bug this avoids.)
    import matplotlib

    matplotlib.use("Agg")

    from zcu_tools.gui.app.autofluxdep.app import run_app
    from zcu_tools.gui.app.autofluxdep.services.remote.service import ControlOptions

    control = (
        ControlOptions(port=args.control_port, token=args.control_token)
        if args.control_port is not None
        else None
    )

    run_app(control=control)
