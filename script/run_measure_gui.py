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

# Repo root: this script lives in script/, so its parent is the root.
PROJECT_ROOT = Path(__file__).parent.parent


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_measure_gui",
        description="Launch the v2 GUI for ZCU qubit-measure",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Disable file logging (stderr WARNING+ only)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Override the DEBUG log file path (default: a per-session file under logs/gui/measure/).",
    )
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
            "Start RemoteControlService on this TCP port. Omit to use the "
            "agreed-upon port 8765 (auto-falls back to an OS-assigned ephemeral "
            "port if 8765 is taken, advertised via session discovery); pass an "
            "explicit port to pin it (fast-fails if that port is taken). "
            "0 = OS-assigned ephemeral port. Bound to 127.0.0.1 unless "
            "--control-allow-external is set. Use --no-control to disable entirely."
        ),
    )
    parser.add_argument(
        "--control-token",
        type=str,
        default=None,
        help="Shared token required from clients via the 'auth' RPC. Optional on loopback.",
    )
    parser.add_argument(
        "--control-allow-external",
        action="store_true",
        help="Bind the control socket to 0.0.0.0 (requires --control-token).",
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
    from zcu_tools.gui.runtime import GuiLaunchOptions, launch_gui_runtime

    return launch_gui_runtime(
        MeasureGuiBehavior,
        GuiLaunchOptions(
            log_root=PROJECT_ROOT,
            to_file=not args.no_log,
            log_file=Path(args.log_file) if args.log_file else None,
            control_port=args.control_port,
            control_token=args.control_token,
            control_allow_external=args.control_allow_external,
            no_control=args.no_control,
        ),
        registry_factory=_build_measure_catalogs,
        clean=args.clean,
        project_root=project_root,
    )


if __name__ == "__main__":
    sys.exit(main())
