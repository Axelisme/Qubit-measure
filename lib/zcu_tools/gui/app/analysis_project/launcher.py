"""Import-light CLI project helpers for the two analysis apps."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zcu_tools.gui.app.analysis_project.entry import AnalysisTarget


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
