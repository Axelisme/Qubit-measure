"""Import-light helpers for standalone GUI launcher scripts.

This module owns CLI edge repetition for shared runtime flags only. It avoids
importing Qt or matplotlib at module import time, so launcher scripts can import
these helpers before the
runtime configures process-level plot policy.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zcu_tools.gui.runtime import GuiLaunchOptions


def select_startup_script(
    *, clean: bool, script: Path | None, default_script: Path
) -> Path | None:
    """Select exactly one trusted script, or None for clean startup.

    script is the explicit CLI path; default_script is supplied by composition,
    never inferred from a user package in lib. clean skips both scripts and
    conflicts with script, raising ValueError before GUI assembly. No file is
    loaded or validated here; the background runner reports path/load failures.
    This pure selection function has no thread or resource ownership requirement.
    """
    if clean and script is not None:
        raise ValueError("--clean and --script are mutually exclusive")
    if clean:
        return None
    return default_script if script is None else script


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
        to_file=not bool(getattr(args, "no_log", False)),
        log_file=Path(log_file) if log_file else None,
        control_port=getattr(args, "control_port", None),
        control_token=getattr(args, "control_token", None),
        control_allow_external=bool(getattr(args, "control_allow_external", False)),
        no_control=bool(getattr(args, "no_control", False)),
    )


__all__ = [
    "add_runtime_cli_options",
    "runtime_options_from_args",
    "select_startup_script",
]
