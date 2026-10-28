"""Launcher for the v2 GUI.

Examples:
    uv run python -m apps.measure                          # default file logging; opens control socket on port 8765
    uv run python -m apps.measure --no-log                 # no file log
    uv run python -m apps.measure --clean                  # skip all startup scripts
    uv run python -m apps.measure --script startup.py      # run only this trusted startup(ctx)
    uv run python -m apps.measure --no-control             # disable the remote-control socket entirely
    uv run python -m apps.measure --control-port 0         # start remote control on an ephemeral loopback port
    uv run python -m apps.measure --control-port 8765 --control-token <hex>
"""

from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

from zcu_tools.gui.app.measure import default_startup
from zcu_tools.gui.launcher import (
    add_runtime_cli_options,
    runtime_options_from_args,
    select_startup_script,
)

# Logs and the default data project stay anchored at the repo, not cwd.
REPO_ROOT = Path(__file__).resolve().parents[1]


def _existing_project_root(value: str) -> Path:
    root = Path(value)
    if not root.is_absolute():
        raise argparse.ArgumentTypeError("project root must be an absolute path")
    try:
        is_directory = root.is_dir()
    except (OSError, ValueError):
        is_directory = False
    if not is_directory:
        raise argparse.ArgumentTypeError("project root must be an existing directory")
    return root


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
    startup = parser.add_mutually_exclusive_group()
    startup.add_argument(
        "--clean",
        action="store_true",
        help="Skip all startup scripts, including the default script.",
    )
    startup.add_argument(
        "--script",
        type=Path,
        help="Run only this trusted Python file defining startup(ctx).",
    )
    parser.add_argument(
        "--project-root",
        type=_existing_project_root,
        default=REPO_ROOT,
        help=(
            "Data project root: an absolute path to an existing directory "
            "(default: repo root). Entry owners manage result/ and Database/; "
            "code, catalogs and startup-script paths keep their existing resolution."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Launch the GUI; invalid project roots exit before GUI or lab assembly."""
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    from zcu_tools.gui.app.measure.app import MeasureGuiBehavior
    from zcu_tools.gui.runtime import launch_gui_runtime

    from zcu_lab.assembly import lab_definitions

    lab = lab_definitions()
    return launch_gui_runtime(
        MeasureGuiBehavior.spec,
        runtime_options_from_args(args, log_root=REPO_ROOT),
        logging_namespaces=lab.logging_namespaces,
        behavior_factory=partial(
            MeasureGuiBehavior,
            lab,
            clean=args.clean,
            startup_script=select_startup_script(
                clean=args.clean,
                script=args.script,
                default_script=Path(default_startup.__file__),
            ),
            project_root=str(args.project_root),
        ),
    )


if __name__ == "__main__":
    sys.exit(main())
