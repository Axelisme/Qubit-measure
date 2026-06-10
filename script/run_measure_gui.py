"""Launcher for the v2 GUI.

Examples:
    uv run python run_measure_gui.py                          # default file logging
    uv run python run_measure_gui.py --no-log                 # no file log
    uv run python run_measure_gui.py --clean                  # don't restore the previous persisted session
    uv run python run_measure_gui.py --control-port 0         # start remote control on an ephemeral loopback port
    uv run python run_measure_gui.py --control-port 8765 --control-token <hex>
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

LOG_FILE = Path(__file__).parent.parent / "gui_debug.log"
LOG_FORMAT = "%(asctime)s.%(msecs)03d [%(levelname)-7s] %(name)s: %(message)s"
LOG_DATE = "%H:%M:%S"


def _setup_logging(to_file: bool = True, log_file: Path | None = None) -> None:
    """Configure root logger: DEBUG to file, WARNING to stderr.

    ``log_file`` overrides the default ``LOG_FILE`` location (e.g. an automated
    launcher pointing it at the OS temp dir).
    """
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE))
    root.addHandler(stderr_handler)

    if to_file:
        target = log_file or LOG_FILE
        file_handler = logging.FileHandler(target, mode="w", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE))
        for name in ("zcu_tools.gui.app.main", "zcu_tools.experiment.v2_gui"):
            log = logging.getLogger(name)
            log.addHandler(file_handler)
            log.setLevel(logging.DEBUG)

        print(f"[run_measure_gui] Logging DEBUG output to: {target}", file=sys.stderr)


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
        help="Override the DEBUG log file path (default: gui_debug.log beside run_measure_gui.py).",
    )
    parser.add_argument(
        "--control-port",
        type=int,
        default=None,
        help=(
            "Start RemoteControlService on this TCP port (0 = pick an ephemeral "
            "free port). Bound to 127.0.0.1 unless --control-allow-external is set."
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
    _setup_logging(
        to_file=not args.no_log,
        log_file=Path(args.log_file) if args.log_file else None,
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
    if args.control_port is not None:
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
    project_root = str(Path(__file__).parent.parent)

    run_app(
        registry,
        role_catalog,
        control_opts=control_opts,
        clean=args.clean,
        project_root=project_root,
    )
