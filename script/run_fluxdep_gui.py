"""Launcher for the fluxdep analysis GUI.

Examples:
    uv run python run_fluxdep_gui.py                          # opens control socket on port 8766
    uv run python run_fluxdep_gui.py --no-control             # disable the remote-control socket
    uv run python run_fluxdep_gui.py --chip Q5_2D --qub Q1 --result-dir ../result/Q5_2D/Q1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from zcu_tools.gui.launcher import (
    add_analysis_project_cli_options,
    add_runtime_cli_options,
    project_info_from_args,
    runtime_options_from_args,
)

# Repo root: this script lives in script/, so its parent is the root.
PROJECT_ROOT = Path(__file__).parent.parent


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_fluxdep_gui",
        description="Launch the fluxonium flux-dependence fitting GUI",
    )
    add_runtime_cli_options(
        parser,
        control_port_help=(
            "Start the remote-control TCP server on this port. Omit to use the "
            "agreed-upon port 8766 (auto-falls back to an ephemeral port if 8766 "
            "is taken, advertised via session discovery); pass an explicit port to "
            "pin it (fast-fails if taken). 0 = OS-assigned ephemeral port. "
            "Use --no-control to disable the socket entirely."
        ),
    )
    add_analysis_project_cli_options(parser, database_path_help="Raw data root")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    # Anchor default result/database paths at the repo root (this script lives in
    # script/, so its parent is the repo root) rather than cwd — a .bat launcher
    # does `cd /d "%~dp0"` into script/, which would otherwise scope defaults
    # under script/.
    project_root = str(PROJECT_ROOT)

    from zcu_tools.gui.app.fluxdep.app import FluxDepGuiBehavior
    from zcu_tools.gui.runtime import launch_gui_runtime

    return launch_gui_runtime(
        FluxDepGuiBehavior,
        runtime_options_from_args(args, log_root=PROJECT_ROOT),
        project=project_info_from_args(args, project_root=project_root),
        project_root=project_root,
    )


if __name__ == "__main__":
    sys.exit(main())
