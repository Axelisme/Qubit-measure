"""Import-light CLI project helpers for the two analysis apps."""

from __future__ import annotations

import argparse

from .selection import AnalysisSelection


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


def analysis_selection_from_args(
    args: argparse.Namespace, *, project_root: str
) -> AnalysisSelection:
    """Describe CLI choices without accessing files or component definitions.

    project_root is the default when --project-root is omitted. An empty entry
    leaves the GUI unopened. A component without an entry raises ValueError.
    """
    if not args.entry and args.component is not None:
        raise ValueError("--component requires --entry")
    return AnalysisSelection(
        args.project_root or project_root, args.entry, args.component
    )
