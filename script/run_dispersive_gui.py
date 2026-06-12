"""Launcher for the dispersive-shift analysis GUI.

Examples:
    uv run python run_dispersive_gui.py
    uv run python run_dispersive_gui.py --chip Q12_2D --qub Q1 --result-dir ../result/Q12_2D/Q1
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
        prog="run_dispersive_gui",
        description="Launch the fluxonium dispersive-shift fitting GUI",
    )
    parser.add_argument("--no-log", action="store_true", help="Disable file logging")
    parser.add_argument("--chip", type=str, default="", help="Chip name")
    parser.add_argument("--qub", type=str, default="", help="Qubit name")
    parser.add_argument("--result-dir", type=str, default="", help="Result directory")
    parser.add_argument(
        "--database-path", type=str, default="", help="Raw one-tone root"
    )
    parser.add_argument(
        "--control-port",
        type=int,
        default=None,
        help="Start the remote-control TCP server on this port (for agents/MCP)",
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
        app_name="dispersive",
        log_root=PROJECT_ROOT,
        to_file=not args.no_log,
        log_file=Path(args.log_file) if args.log_file else None,
    )

    # Select the embedded matplotlib backend BEFORE importing anything that uses
    # matplotlib (the "configure backend before pyplot" invariant) — preprocessing
    # / auto-fit may touch pyplot via notebook helpers. The setup module is
    # import-clean (pulls in no matplotlib).
    from zcu_tools.gui.plotting.setup import configure_matplotlib_backend

    configure_matplotlib_backend()

    from zcu_tools.gui.app.dispersive.app import run_app
    from zcu_tools.gui.app.dispersive.state import ProjectInfo

    control = None
    if args.control_port is not None:
        from zcu_tools.gui.app.dispersive.services.remote.service import ControlOptions

        control = ControlOptions(port=args.control_port, token=args.control_token)

    # Anchor default result/database paths at the repo root (this script lives in
    # script/, so its parent is the repo root) rather than cwd — a .bat launcher
    # does `cd /d "%~dp0"` into script/, which would otherwise scope defaults
    # under script/.
    project_root = str(PROJECT_ROOT)

    # Only override a ProjectInfo field when the arg was actually given, so an
    # omitted --chip / --qub keeps the unknown_* defaults instead of becoming "".
    # root_dir seeds the derived-default anchoring inside ProjectInfo.__post_init__.
    project_kwargs = {
        k: v
        for k, v in (
            ("chip_name", args.chip),
            ("qub_name", args.qub),
            ("result_dir", args.result_dir),
            ("database_path", args.database_path),
        )
        if v
    }
    run_app(
        ProjectInfo(root_dir=project_root, **project_kwargs),
        control=control,
        project_root=project_root,
    )
