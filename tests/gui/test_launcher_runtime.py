from __future__ import annotations

import argparse
from pathlib import Path

from zcu_tools.gui.launcher import (
    add_runtime_cli_options,
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
