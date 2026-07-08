"""Import-light helpers for standalone GUI launcher scripts.

This module owns CLI edge repetition only: shared runtime flags and the
analysis-GUI project flags. It deliberately avoids importing Qt or matplotlib at
module import time, so launcher scripts can import these helpers before the
runtime configures process-level plot policy.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zcu_tools.gui.project import ProjectInfo
    from zcu_tools.gui.runtime import GuiLaunchOptions

DEFAULT_LOG_FILE_HELP = "Override the DEBUG log file path"
DEFAULT_NO_LOG_HELP = "Disable file logging"
DEFAULT_NO_CONTROL_HELP = (
    "Disable the remote-control TCP socket entirely (overrides --control-port)."
)
DEFAULT_CONTROL_TOKEN_HELP = "Shared auth token required by remote-control clients"


def add_runtime_cli_options(
    parser: argparse.ArgumentParser,
    *,
    control_port_help: str,
    no_log_help: str = DEFAULT_NO_LOG_HELP,
    log_file_help: str = DEFAULT_LOG_FILE_HELP,
    control_token_help: str = DEFAULT_CONTROL_TOKEN_HELP,
    allow_external: bool = False,
) -> None:
    """Add common process-runtime flags to a GUI launcher parser."""
    parser.add_argument("--no-log", action="store_true", help=no_log_help)
    parser.add_argument(
        "--no-control", action="store_true", help=DEFAULT_NO_CONTROL_HELP
    )
    parser.add_argument(
        "--control-port",
        type=int,
        default=None,
        help=control_port_help,
    )
    parser.add_argument(
        "--control-token",
        type=str,
        default=None,
        help=control_token_help,
    )
    if allow_external:
        parser.add_argument(
            "--control-allow-external",
            action="store_true",
            help="Bind the control socket to 0.0.0.0 (requires --control-token).",
        )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help=log_file_help,
    )


def runtime_options_from_args(
    args: argparse.Namespace,
    *,
    log_root: Path,
) -> GuiLaunchOptions:
    """Build runtime launch options from parser output."""
    from zcu_tools.gui.runtime import GuiLaunchOptions

    log_file = getattr(args, "log_file", None)
    return GuiLaunchOptions(
        log_root=log_root,
        to_file=not bool(getattr(args, "no_log")),
        log_file=Path(log_file) if log_file else None,
        control_port=getattr(args, "control_port", None),
        control_token=getattr(args, "control_token", None),
        control_allow_external=bool(getattr(args, "control_allow_external", False)),
        no_control=bool(getattr(args, "no_control")),
    )


def add_analysis_project_cli_options(
    parser: argparse.ArgumentParser,
    *,
    database_path_help: str,
) -> None:
    """Add common chip/qubit/path flags used by analysis GUIs."""
    parser.add_argument("--chip", type=str, default="", help="Chip name")
    parser.add_argument("--qub", type=str, default="", help="Qubit name")
    parser.add_argument("--result-dir", type=str, default="", help="Result directory")
    parser.add_argument(
        "--database-path", type=str, default="", help=database_path_help
    )


def project_info_from_args(
    args: argparse.Namespace,
    *,
    project_root: str,
) -> ProjectInfo:
    """Build a ProjectInfo while preserving omitted-field defaults."""
    from zcu_tools.gui.project import ProjectInfo

    project_kwargs = {
        key: value
        for key, value in (
            ("chip_name", getattr(args, "chip", "")),
            ("qub_name", getattr(args, "qub", "")),
            ("result_dir", getattr(args, "result_dir", "")),
            ("database_path", getattr(args, "database_path", "")),
        )
        if value
    }
    return ProjectInfo(root_dir=project_root, **project_kwargs)


__all__ = [
    "add_analysis_project_cli_options",
    "add_runtime_cli_options",
    "project_info_from_args",
    "runtime_options_from_args",
]
