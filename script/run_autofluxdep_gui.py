"""Launcher for the autofluxdep workflow GUI (Phase C prototype).

Opens the node-list / node-detail window. Prototype: Setup builds fake resources
and Run drives the orchestrator on synthetic data — no hardware. Use it to
exercise the interaction (add/remove/reorder/rename Nodes, Setup→Run enable,
edit↔run lock, auto-follow, Run/Stop toggle, global flux progress).

DEBUG output is written to a per-session file under ``logs/gui/autofluxdep/`` at
the repo root (the whole run loop: each flux point, each Node entered / skipped /
fitted). Disable with ``--no-log`` or redirect with ``--log-file``.

Example:
    uv run python script/run_autofluxdep_gui.py             # opens control socket on port 8768
    uv run python script/run_autofluxdep_gui.py --no-control # disable the remote-control socket
    uv run python script/run_autofluxdep_gui.py --no-log
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from zcu_tools.gui.launcher import add_runtime_cli_options, runtime_options_from_args

# Repo root: this script lives in script/, so its parent is the root.
PROJECT_ROOT = Path(__file__).parent.parent


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_autofluxdep_gui",
        description="Launch the autofluxdep workflow GUI",
    )
    add_runtime_cli_options(
        parser,
        control_port_help=(
            "Start the read-only remote-control TCP server on this port. Omit to "
            "use the agreed-upon port 8768 (auto-falls back to an ephemeral port "
            "if 8768 is taken, advertised via session discovery); pass an explicit "
            "port to pin it (fast-fails if taken). 0 = OS-assigned ephemeral port. "
            "Use --no-control to disable the socket entirely."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    project_root = str(PROJECT_ROOT)

    from zcu_tools.gui.app.autofluxdep.app import AutoFluxDepGuiBehavior
    from zcu_tools.gui.runtime import launch_gui_runtime

    return launch_gui_runtime(
        AutoFluxDepGuiBehavior,
        runtime_options_from_args(args, log_root=PROJECT_ROOT),
        project_root=project_root,
    )


if __name__ == "__main__":
    sys.exit(main())
