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

# Code, catalogs and logs stay anchored at the repo, not the data project.
REPO_ROOT = Path(__file__).resolve().parent.parent


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
    parser.add_argument(
        "--clean",
        action="store_true",
        help=(
            "Start without restoring the previous persisted session "
            "(gui_state_v1.json is left untouched at startup; a normal close "
            "still flushes over it)."
        ),
    )
    parser.add_argument(
        "--simulate",
        type=Path,
        help=(
            "Trusted Python environment defining build_environment(); fixed for this "
            "process. Example: --simulate zcu_lab/simulation_envs/FTF/environment.py"
        ),
    )
    parser.add_argument(
        "--project-root",
        type=_existing_project_root,
        default=REPO_ROOT,
        help=(
            "Data project root: an absolute path to an existing directory "
            "(default: repo root). Entry owners manage result/ and Database/; "
            "code, catalogs and simulation paths keep their existing resolution."
        ),
    )
    return parser.parse_args(argv)


def _build_measure_catalogs():
    """Build measure-gui's experiment catalogs after runtime pre-Qt setup."""
    from zcu_tools.gui.app.measure.catalog_loader import SourceExperimentCatalogLoader

    # Composition root: wire the user-owned experiment attachments
    # into the GUI framework. The behavior receives a factory, so these imports
    # happen after runtime logging and matplotlib policy setup.
    from zcu_tools.gui.app.measure.registry import Registry
    from zcu_tools.gui.catalog_loader import SourcePackage
    from zcu_tools.resources.entry.registry import component_registry

    from zcu_lab.definitions import register_all
    from zcu_lab.templates import TEMPLATES

    registry = Registry(templates=TEMPLATES)
    register_all(registry, components=component_registry)
    loader = SourceExperimentCatalogLoader(
        sources=(
            SourcePackage("zcu_tools", REPO_ROOT / "lib" / "zcu_tools"),
            SourcePackage("zcu_lab", REPO_ROOT / "zcu_lab"),
        ),
        reload_modules=("zcu_lab.v2", "zcu_lab.definitions"),
        preserved_modules=(
            "zcu_lab.templates",
            "zcu_lab.v2._support.measure",
            "zcu_lab.v2._support.singleshot",
        ),
        catalog_module="zcu_lab.definitions",
        current_registry=registry,
    )
    return registry, loader


def main(argv: list[str] | None = None) -> int:
    """Launch the GUI; invalid project roots exit before GUI or lab assembly."""
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    from zcu_tools.gui.app.measure.app import MeasureGuiBehavior
    from zcu_tools.gui.runtime import launch_gui_runtime

    from zcu_lab import load_alias_presets
    from zcu_lab.components import params_for

    return launch_gui_runtime(
        MeasureGuiBehavior,
        runtime_options_from_args(args, log_root=REPO_ROOT),
        extra_logging_namespaces=("zcu_lab",),
        registry_factory=_build_measure_catalogs,
        params_for=params_for,
        alias_presets=load_alias_presets(),
        clean=args.clean,
        simulate=args.simulate,
        project_root=str(args.project_root),
    )


if __name__ == "__main__":
    sys.exit(main())
