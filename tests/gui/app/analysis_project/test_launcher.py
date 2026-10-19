from __future__ import annotations

import argparse

from zcu_tools.gui.app.analysis_project.launcher import (
    add_analysis_project_cli_options,
    project_info_from_args,
)


def test_project_info_from_args_preserves_omitted_defaults() -> None:
    parser = argparse.ArgumentParser()
    add_analysis_project_cli_options(parser, data_root_help="database help")

    default_project = project_info_from_args(
        parser.parse_args([]),
        project_root="/repo",
    )
    explicit_project = project_info_from_args(
        parser.parse_args(
            [
                "--chip",
                "Q_TEST",
                "--qub",
                "Q1",
                "--result-dir",
                "result/Q_TEST/Q1",
                "--data-root",
                "Database/Q_TEST/Q1",
            ]
        ),
        project_root="/repo",
    )

    assert default_project.chip_name == "unknown_chip"
    assert default_project.qub_name == "unknown_qubit"
    assert default_project.result_dir == "/repo/result/unknown_chip/unknown_qubit"
    assert default_project.data_root == "/repo/Database/unknown_chip/unknown_qubit"

    assert explicit_project.chip_name == "Q_TEST"
    assert explicit_project.qub_name == "Q1"
    assert explicit_project.result_dir == "result/Q_TEST/Q1"
    assert explicit_project.data_root == "Database/Q_TEST/Q1"
