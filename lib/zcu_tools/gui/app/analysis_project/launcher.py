"""Import-light CLI project helpers for the two analysis apps."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zcu_tools.gui.app.analysis_project.entry import AnalysisTarget
    from zcu_tools.gui.app.analysis_project.project import ProjectInfo


def add_analysis_project_cli_options(
    parser: argparse.ArgumentParser,
    *,
    data_root_help: str,
) -> None:
    """Add common chip/qubit/path flags used by analysis GUIs."""
    parser.add_argument("--chip", type=str, default="", help="Chip name")
    parser.add_argument("--qub", type=str, default="", help="Qubit name")
    parser.add_argument("--result-dir", type=str, default="", help="Result directory")
    parser.add_argument("--data-root", type=str, default="", help=data_root_help)


def project_info_from_args(
    args: argparse.Namespace,
    *,
    project_root: str,
) -> ProjectInfo:
    """Build a ProjectInfo while preserving omitted-field defaults."""
    from zcu_tools.gui.app.analysis_project.project import ProjectInfo

    project_kwargs = {
        key: value
        for key, value in (
            ("chip_name", getattr(args, "chip", "")),
            ("qub_name", getattr(args, "qub", "")),
            ("result_dir", getattr(args, "result_dir", "")),
            ("data_root", getattr(args, "data_root", "")),
        )
        if value
    }
    return ProjectInfo(root_dir=project_root, **project_kwargs)


def add_analysis_entry_cli_options(parser: argparse.ArgumentParser) -> None:
    """Add project root, existing entry name and optional concrete component."""
    parser.add_argument(
        "--project-root",
        type=str,
        default="",
        help="Root containing result/ and Database/",
    )
    parser.add_argument(
        "--entry", type=str, default="", help="Existing result entry name"
    )
    parser.add_argument(
        "--component",
        type=str,
        default=None,
        help="Fluxonium component; omission uses setup qubit alias",
    )


def analysis_target_from_args(
    args: argparse.Namespace, *, project_root: str
) -> AnalysisTarget | None:
    """Open the requested target, or leave an unconfigured GUI when entry is omitted.

    --component without --entry raises ValueError. Entry and kind errors propagate.
    No entry, component or Database output is created.
    """
    from zcu_tools.gui.app.analysis_project.entry import AnalysisTarget

    if not args.entry:
        if args.component is not None:
            raise ValueError("--component requires --entry")
        return None
    return AnalysisTarget.open(
        args.project_root or project_root, args.entry, args.component
    )
