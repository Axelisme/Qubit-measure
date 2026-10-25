"""Launcher for the dispersive-shift analysis GUI.

Examples:
    uv run python scripts/app/run_dispersive_gui.py                       # opens control socket on port 8767
    uv run python scripts/app/run_dispersive_gui.py --no-control          # disable the remote-control socket
    uv run python scripts/app/run_dispersive_gui.py --project-root .. --entry demo --component Q1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from zcu_tools.gui.app.analysis_project.launcher import (
    add_analysis_entry_cli_options,
    analysis_target_from_args,
)
from zcu_tools.gui.launcher import (
    add_runtime_cli_options,
    runtime_options_from_args,
)

# Repo root: this script lives two directories below it, in scripts/app/.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_dispersive_gui",
        description="Launch the fluxonium dispersive-shift fitting GUI",
    )
    add_runtime_cli_options(
        parser,
        control_port_help=(
            "Start the remote-control TCP server on this port. Omit to use the "
            "agreed-upon port 8767 (auto-falls back to an ephemeral port if 8767 "
            "is taken, advertised via session discovery); pass an explicit port to "
            "pin it (fast-fails if taken). 0 = OS-assigned ephemeral port. "
            "Use --no-control to disable the socket entirely."
        ),
    )
    add_analysis_entry_cli_options(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    # Anchor default result/database paths at the repo root, not cwd.
    project_root = str(Path(args.project_root or PROJECT_ROOT).resolve())

    from zcu_tools.gui.app.dispersive.app import DispersiveGuiBehavior
    from zcu_tools.resources.entry import component_registry

    from zcu_lab.components import register_all

    register_all(component_registry)
    from zcu_tools.gui.runtime import launch_gui_runtime

    return launch_gui_runtime(
        DispersiveGuiBehavior,
        runtime_options_from_args(args, log_root=PROJECT_ROOT),
        project=analysis_target_from_args(args, project_root=project_root),
        project_root=project_root,
    )


if __name__ == "__main__":
    sys.exit(main())
