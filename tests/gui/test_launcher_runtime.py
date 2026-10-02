from __future__ import annotations

import argparse
from pathlib import Path

from zcu_tools.gui.launcher import (
    add_analysis_project_cli_options,
    add_runtime_cli_options,
    project_info_from_args,
    runtime_options_from_args,
)
from zcu_tools.gui.runtime import GuiLaunchOptions


def test_runtime_options_from_args_preserves_common_cli_contract() -> None:
    parser = argparse.ArgumentParser()
    add_runtime_cli_options(
        parser,
        control_port_help="control help",
        allow_external=True,
    )

    args = parser.parse_args(
        [
            "--no-log",
            "--log-file",
            "custom.log",
            "--no-control",
            "--control-port",
            "9001",
            "--control-token",
            "secret",
            "--control-allow-external",
        ]
    )

    options = runtime_options_from_args(args, log_root=Path("/tmp/gui-root"))

    assert options == GuiLaunchOptions(
        log_root=Path("/tmp/gui-root"),
        to_file=False,
        log_file=Path("custom.log"),
        control_port=9001,
        control_token="secret",
        control_allow_external=True,
        no_control=True,
    )


def test_runtime_options_from_args_has_safe_defaults() -> None:
    options = runtime_options_from_args(
        argparse.Namespace(), log_root=Path("/tmp/root")
    )

    assert options == GuiLaunchOptions(log_root=Path("/tmp/root"))


def test_project_info_from_args_preserves_omitted_defaults() -> None:
    parser = argparse.ArgumentParser()
    add_analysis_project_cli_options(parser, database_path_help="database help")

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
                "--database-path",
                "Database/Q_TEST/Q1",
            ]
        ),
        project_root="/repo",
    )

    assert default_project.chip_name == "unknown_chip"
    assert default_project.qub_name == "unknown_qubit"
    assert default_project.result_dir == "/repo/result/unknown_chip/unknown_qubit"
    assert default_project.database_path == "/repo/Database/unknown_chip/unknown_qubit"

    assert explicit_project.chip_name == "Q_TEST"
    assert explicit_project.qub_name == "Q1"
    assert explicit_project.result_dir == "result/Q_TEST/Q1"
    assert explicit_project.database_path == "Database/Q_TEST/Q1"
