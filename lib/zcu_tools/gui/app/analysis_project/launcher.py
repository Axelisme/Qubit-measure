"""Import-light CLI project helpers for the two analysis apps."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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
