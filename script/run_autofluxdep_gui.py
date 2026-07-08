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

# Repo root: this script lives in script/, so its parent is the root.
PROJECT_ROOT = Path(__file__).parent.parent


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_autofluxdep_gui",
        description="Launch the autofluxdep workflow GUI",
    )
    parser.add_argument("--no-log", action="store_true", help="Disable file logging")
    parser.add_argument(
        "--no-control",
        action="store_true",
        help="Disable the remote-control TCP socket entirely (overrides --control-port).",
    )
    parser.add_argument(
        "--control-port",
        type=int,
        default=None,
        help=(
            "Start the read-only remote-control TCP server on this port. Omit to "
            "use the agreed-upon port 8768 (auto-falls back to an ephemeral port "
            "if 8768 is taken, advertised via session discovery); pass an explicit "
            "port to pin it (fast-fails if taken). 0 = OS-assigned ephemeral port. "
            "Use --no-control to disable the socket entirely."
        ),
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


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    project_root = str(PROJECT_ROOT)

    from zcu_tools.gui.app.autofluxdep.app import AutoFluxDepGuiBehavior
    from zcu_tools.gui.runtime import GuiLaunchOptions, launch_gui_runtime

    return launch_gui_runtime(
        AutoFluxDepGuiBehavior,
        GuiLaunchOptions(
            log_root=PROJECT_ROOT,
            to_file=not args.no_log,
            log_file=Path(args.log_file) if args.log_file else None,
            control_port=args.control_port,
            control_token=args.control_token,
            no_control=args.no_control,
        ),
        project_root=project_root,
    )


if __name__ == "__main__":
    sys.exit(main())
