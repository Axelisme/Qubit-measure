"""Launcher for the fluxdep analysis GUI.

Examples:
    uv run python run_fluxdep_gui.py
    uv run python run_fluxdep_gui.py --chip Q5_2D --qub Q1 --result-dir ../result/Q5_2D/Q1
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

LOG_FILE = Path(__file__).parent.parent / "fluxdep_gui_debug.log"
LOG_FORMAT = "%(asctime)s.%(msecs)03d [%(levelname)-7s] %(name)s: %(message)s"
LOG_DATE = "%H:%M:%S"


def _setup_logging(to_file: bool = True, log_file: Path = LOG_FILE) -> None:
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE))
    root.addHandler(stderr_handler)

    if to_file:
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE))
        log = logging.getLogger("zcu_tools.gui.app.fluxdep")
        log.addHandler(file_handler)
        log.setLevel(logging.DEBUG)
        print(f"[run_fluxdep_gui] Logging DEBUG output to: {log_file}", file=sys.stderr)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_fluxdep_gui",
        description="Launch the fluxonium flux-dependence fitting GUI",
    )
    parser.add_argument("--no-log", action="store_true", help="Disable file logging")
    parser.add_argument("--chip", type=str, default="", help="Chip name")
    parser.add_argument("--qub", type=str, default="", help="Qubit name")
    parser.add_argument("--result-dir", type=str, default="", help="Result directory")
    parser.add_argument("--database-path", type=str, default="", help="Raw data root")
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
    log_file = Path(args.log_file) if args.log_file else LOG_FILE
    _setup_logging(to_file=not args.no_log, log_file=log_file)

    # Select the embedded matplotlib backend BEFORE importing anything that uses
    # matplotlib (the "configure backend before pyplot" invariant) — this routes
    # search_in_database(plot=True)'s pyplot figure into the GUI instead of a
    # detached window. The setup module is import-clean (pulls in no matplotlib).
    from zcu_tools.gui.plotting.setup import configure_matplotlib_backend

    configure_matplotlib_backend()

    from zcu_tools.gui.app.fluxdep.app import run_app
    from zcu_tools.gui.app.fluxdep.services.remote.service import ControlOptions
    from zcu_tools.gui.app.fluxdep.state import ProjectInfo

    control = (
        ControlOptions(port=args.control_port, token=args.control_token)
        if args.control_port is not None
        else None
    )

    # Anchor default result/database paths at the repo root (this script lives in
    # script/, so its parent is the repo root) rather than cwd — a .bat launcher
    # does `cd /d "%~dp0"` into script/, which would otherwise scope defaults
    # under script/.
    project_root = str(Path(__file__).parent.parent)

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
