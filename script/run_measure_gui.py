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

from zcu_tools.gui.logging_setup import setup_gui_logging

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
        default=8765,
        help=(
            "Start RemoteControlService on this TCP port (default: 8765, the "
            "agreed-upon port for agent attach; 0 = OS-assigned ephemeral port). "
            "Bound to 127.0.0.1 unless --control-allow-external is set. "
            "Use --no-control to disable the socket entirely."
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


if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])
    setup_gui_logging(
        app_name="measure",
        log_root=PROJECT_ROOT,
        to_file=not args.no_log,
        log_file=Path(args.log_file) if args.log_file else None,
        extra_namespaces=("zcu_tools.experiment.v2_gui",),
    )

    # Configure the matplotlib backend before importing anything that uses
    # matplotlib. ``zcu_tools.gui.plotting.setup`` is import-clean (it does not
    # pull in matplotlib), so this import cannot load pyplot too early.
    from zcu_tools.gui.plotting.setup import configure_matplotlib_backend

    configure_matplotlib_backend()

    from zcu_tools.experiment.v2_gui.registry import register_all, register_all_roles
    from zcu_tools.gui.app.main.app import run_app

    # Composition root: wire the experiment-adapter layer (experiment.v2_gui)
    # into the GUI framework. run_app receives the populated registry/catalog,
    # so the GUI framework itself never imports the experiment layer.
    from zcu_tools.gui.app.main.registry import Registry
    from zcu_tools.gui.app.main.role_catalog import RoleCatalog

    registry = Registry()
    register_all(registry)

    role_catalog = RoleCatalog()
    register_all_roles(role_catalog)

    control_opts = None
    if not args.no_control:
        from zcu_tools.gui.app.main.services.remote import ControlOptions

        control_opts = ControlOptions(
            port=args.control_port,
            token=args.control_token,
            allow_external=args.control_allow_external,
        )

    # Anchor default result/database paths at the repo root (this script lives in
    # script/, so its parent is the repo root) rather than cwd — a .bat launcher
    # does `cd /d "%~dp0"` into script/, which would otherwise scope defaults
    # under script/.
    project_root = str(PROJECT_ROOT)

    run_app(
        registry,
        role_catalog,
        control_opts=control_opts,
        clean=args.clean,
        project_root=project_root,
    )
