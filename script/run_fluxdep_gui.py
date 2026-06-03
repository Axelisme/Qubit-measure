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


def _setup_logging(to_file: bool = True) -> None:
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE))
    root.addHandler(stderr_handler)

    if to_file:
        file_handler = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE))
        log = logging.getLogger("zcu_tools.fluxdep_gui")
        log.addHandler(file_handler)
        log.setLevel(logging.DEBUG)
        print(f"[run_fluxdep_gui] Logging DEBUG output to: {LOG_FILE}", file=sys.stderr)


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
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])
    _setup_logging(to_file=not args.no_log)

    from zcu_tools.fluxdep_gui.app import run_app
    from zcu_tools.fluxdep_gui.state import ProjectInfo

    run_app(
        ProjectInfo(
            chip_name=args.chip,
            qub_name=args.qub,
            result_dir=args.result_dir,
            database_path=args.database_path,
        )
    )
