"""Launcher for the v2 GUI.

Examples:
    uv run python run_gui.py                          # default file logging
    uv run python run_gui.py --no-log                 # no file log
    uv run python run_gui.py --control-port 0         # start remote control on an ephemeral loopback port
    uv run python run_gui.py --control-port 8765 --control-token <hex>
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make sure lib/ is on the path when running from the repo root
sys.path.insert(0, str(Path(__file__).parent / "lib"))

LOG_FILE = Path(__file__).parent / "gui_debug.log"
LOG_FORMAT = "%(asctime)s.%(msecs)03d [%(levelname)-7s] %(name)s: %(message)s"
LOG_DATE = "%H:%M:%S"


def _setup_logging(to_file: bool = True) -> None:
    """Configure root logger: DEBUG to file, WARNING to stderr."""
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
        for name in ("zcu_tools.gui", "zcu_tools.experiment.v2_gui"):
            log = logging.getLogger(name)
            log.addHandler(file_handler)
            log.setLevel(logging.DEBUG)

        print(f"[run_gui] Logging DEBUG output to: {LOG_FILE}", file=sys.stderr)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_gui",
        description="Launch the v2 GUI for ZCU qubit-measure",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Disable file logging (stderr WARNING+ only)",
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
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])
    _setup_logging(to_file=not args.no_log)

    from zcu_tools.gui.mpl_backend_setup import configure_gui_matplotlib_backend

    configure_gui_matplotlib_backend()

    from zcu_tools.experiment.v2_gui.app import run_app

    control_opts = None
    if args.control_port is not None:
        from zcu_tools.gui.services.remote import ControlOptions

        control_opts = ControlOptions(
            port=args.control_port,
            token=args.control_token,
            allow_external=args.control_allow_external,
        )

    run_app(control_opts=control_opts)
