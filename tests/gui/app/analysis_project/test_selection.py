"""Analysis CLI choices stay unresolved until the lab is registered."""

import argparse
from pathlib import Path

import pytest
from zcu_tools.gui.app.analysis_project.entry import AnalysisTarget
from zcu_tools.gui.app.analysis_project.launcher import analysis_selection_from_args
from zcu_tools.gui.app.analysis_project.selection import AnalysisSelection


def test_cli_selection_does_not_open_the_requested_entry(tmp_path: Path) -> None:
    parser = argparse.Namespace(project_root="", entry="not-created", component="Q1")
    selection = analysis_selection_from_args(parser, project_root=str(tmp_path))
    assert selection == AnalysisSelection(str(tmp_path), "not-created", "Q1")


def test_cli_selection_preserves_empty_entry_and_rejects_component_alone(
    tmp_path: Path,
) -> None:
    args = argparse.Namespace(project_root="", entry="", component=None)
    assert analysis_selection_from_args(
        args, project_root=str(tmp_path)
    ) == AnalysisSelection(str(tmp_path))
    args.component = "Q1"
    with pytest.raises(ValueError, match="--component requires --entry"):
        analysis_selection_from_args(args, project_root=str(tmp_path))


def test_target_open_accepts_an_unresolved_selection(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not-created"):
        AnalysisTarget.open(AnalysisSelection(tmp_path, "not-created", "Q1"))
