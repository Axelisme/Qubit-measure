"""Launcher for the v2 GUI.

Examples:
    uv run python run_measure_gui.py                          # default file logging; opens control socket on port 8765
    uv run python run_measure_gui.py --no-log                 # no file log
    uv run python run_measure_gui.py --clean                  # don't restore the previous persisted session
    uv run python run_measure_gui.py --no-control             # disable the remote-control socket entirely
    uv run python run_measure_gui.py --control-port 0         # start remote control on an ephemeral loopback port
    uv run python run_measure_gui.py --control-port 8765 --control-token <hex>
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
        prog="run_measure_gui",
        description="Launch the v2 GUI for ZCU qubit-measure",
    )
    add_runtime_cli_options(
        parser,
        no_log_help="Disable file logging (stderr WARNING+ only)",
        log_file_help=(
            "Override the DEBUG log file path (default: a per-session file under "
            "logs/gui/measure/)."
        ),
        control_port_help=(
            "Start RemoteControlService on this TCP port. Omit to use the "
            "agreed-upon port 8765 (auto-falls back to an OS-assigned ephemeral "
            "port if 8765 is taken, advertised via session discovery); pass an "
            "explicit port to pin it (fast-fails if that port is taken). "
            "0 = OS-assigned ephemeral port. Bound to 127.0.0.1 unless "
            "--control-allow-external is set. Use --no-control to disable entirely."
        ),
        control_token_help=(
            "Shared token required from clients via the 'auth' RPC. Optional on loopback."
        ),
        allow_external=True,
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help=(
            "Start without restoring the previous persisted session "
            "(gui_state_v1.json is left untouched at startup; a normal close "
            "still flushes over it)."
        ),
    )
    return parser.parse_args(argv)


def _build_measure_catalogs():
    """Build measure-gui's experiment catalogs after runtime pre-Qt setup."""
    from zcu_tools.experiment.v2_gui.registry import register_all, register_all_roles

    # Composition root: wire the experiment-adapter layer (experiment.v2_gui)
    # into the GUI framework. The behavior receives a factory, so these imports
    # happen after runtime logging and matplotlib policy setup.
    from zcu_tools.gui.app.main.registry import Registry
    from zcu_tools.gui.app.main.role_catalog import RoleCatalog

    registry = Registry()
    register_all(registry)

    role_catalog = RoleCatalog()
    register_all_roles(role_catalog)
    return registry, role_catalog


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    # Anchor default result/database paths at the repo root (this script lives in
    # script/, so its parent is the repo root) rather than cwd — a .bat launcher
    # does `cd /d "%~dp0"` into script/, which would otherwise scope defaults
    # under script/.
    project_root = str(PROJECT_ROOT)

    from zcu_tools.gui.app.main.app import MeasureGuiBehavior
    from zcu_tools.gui.runtime import launch_gui_runtime

    return launch_gui_runtime(
        MeasureGuiBehavior,
        runtime_options_from_args(args, log_root=PROJECT_ROOT),
        registry_factory=_build_measure_catalogs,
        clean=args.clean,
        project_root=project_root,
    )


if __name__ == "__main__":
    sys.exit(main())
