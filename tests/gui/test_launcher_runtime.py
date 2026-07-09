from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest
from zcu_tools.gui import runtime
from zcu_tools.gui.launcher import (
    add_analysis_project_cli_options,
    add_runtime_cli_options,
    project_info_from_args,
    runtime_options_from_args,
)
from zcu_tools.gui.runtime import GuiLaunchOptions, GuiRuntimeBehavior


def _load_launcher(module_name: str) -> ModuleType:
    return importlib.import_module(module_name)


def test_launcher_module_is_import_light() -> None:
    code = "\n".join(
        [
            "import importlib",
            "import json",
            "import sys",
            "importlib.import_module('zcu_tools.gui.launcher')",
            "watched = [",
            "    'zcu_tools.gui.runtime',",
            "    'qtpy',",
            "    'PySide6',",
            "    'PyQt5',",
            "    'matplotlib',",
            "    'zcu_tools.gui.app.main.app',",
            "    'zcu_tools.gui.app.autofluxdep.app',",
            "    'zcu_tools.gui.app.fluxdep.app',",
            "    'zcu_tools.gui.app.dispersive.app',",
            "]",
            "print(json.dumps([name for name in watched if name in sys.modules]))",
        ]
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == []


@pytest.mark.parametrize(
    ("module_name", "behavior_name"),
    [
        ("zcu_tools.gui.app.main.app", "MeasureGuiBehavior"),
        ("zcu_tools.gui.app.autofluxdep.app", "AutoFluxDepGuiBehavior"),
        ("zcu_tools.gui.app.fluxdep.app", "FluxDepGuiBehavior"),
        ("zcu_tools.gui.app.dispersive.app", "DispersiveGuiBehavior"),
    ],
)
def test_app_modules_expose_runtime_behavior_not_legacy_run_app(
    module_name: str,
    behavior_name: str,
) -> None:
    module = importlib.import_module(module_name)

    assert getattr(module, behavior_name).__name__ == behavior_name
    assert not hasattr(module, "run_app")
    assert "run_app" not in getattr(module, "__all__", ())


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


@pytest.mark.parametrize(
    ("module_name", "behavior_name", "app_slug", "default_port", "return_code"),
    [
        ("script.run_fluxdep_gui", "FluxDepGuiBehavior", "fluxdep", 8766, 41),
        (
            "script.run_dispersive_gui",
            "DispersiveGuiBehavior",
            "dispersive",
            8767,
            42,
        ),
    ],
)
def test_launcher_main_delegates_to_gui_runtime(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    behavior_name: str,
    app_slug: str,
    default_port: int,
    return_code: int,
) -> None:
    calls: list[
        tuple[
            type[GuiRuntimeBehavior],
            GuiLaunchOptions,
            tuple[object, ...],
            dict[str, object],
        ]
    ] = []

    def fake_launch_gui_runtime(
        behavior_cls: type[GuiRuntimeBehavior],
        options: GuiLaunchOptions,
        *args: object,
        **kwargs: object,
    ) -> int:
        calls.append((behavior_cls, options, args, kwargs))
        return return_code

    monkeypatch.setattr(runtime, "launch_gui_runtime", fake_launch_gui_runtime)

    launcher = _load_launcher(module_name)
    main = cast(Callable[[list[str] | None], int], getattr(launcher, "main"))
    project_root = cast(Path, getattr(launcher, "PROJECT_ROOT"))
    log_file = project_root / "launcher-test.log"

    code = main(
        [
            "--no-log",
            "--chip",
            "Q_TEST",
            "--qub",
            "Q1",
            "--result-dir",
            "result/Q_TEST/Q1",
            "--database-path",
            "database",
            "--control-port",
            "9001",
            "--control-token",
            "secret",
            "--log-file",
            str(log_file),
        ]
    )

    assert code == return_code
    assert len(calls) == 1

    behavior_cls, options, args, kwargs = calls[0]
    assert args == ()
    assert behavior_cls.__name__ == behavior_name
    assert behavior_cls.spec.app_slug == app_slug
    assert behavior_cls.spec.default_control_port == default_port

    assert options.log_root == project_root
    assert options.to_file is False
    assert options.log_file == log_file
    assert options.control_port == 9001
    assert options.control_token == "secret"
    assert options.no_control is False

    assert kwargs["project_root"] == str(project_root)
    project = kwargs["project"]
    assert getattr(project, "root_dir") == str(project_root)
    assert getattr(project, "chip_name") == "Q_TEST"
    assert getattr(project, "qub_name") == "Q1"
    assert getattr(project, "result_dir") == "result/Q_TEST/Q1"
    assert getattr(project, "database_path") == "database"


@pytest.mark.parametrize(
    "module_name",
    [
        "script.run_fluxdep_gui",
        "script.run_dispersive_gui",
        "script.run_measure_gui",
        "script.run_autofluxdep_gui",
    ],
)
def test_launcher_main_preserves_no_control_override(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
) -> None:
    calls: list[GuiLaunchOptions] = []

    def fake_launch_gui_runtime(
        behavior_cls: type[GuiRuntimeBehavior],
        options: GuiLaunchOptions,
        *args: object,
        **kwargs: object,
    ) -> int:
        del behavior_cls, args, kwargs
        calls.append(options)
        return 0

    monkeypatch.setattr(runtime, "launch_gui_runtime", fake_launch_gui_runtime)

    launcher = _load_launcher(module_name)
    main = cast(Callable[[list[str] | None], int], getattr(launcher, "main"))

    assert main(["--no-control", "--control-port", "9001"]) == 0

    assert len(calls) == 1
    assert calls[0].no_control is True
    assert calls[0].control_port == 9001


def test_measure_launcher_main_delegates_to_gui_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[
        tuple[
            type[GuiRuntimeBehavior],
            GuiLaunchOptions,
            tuple[object, ...],
            dict[str, object],
        ]
    ] = []

    def fake_launch_gui_runtime(
        behavior_cls: type[GuiRuntimeBehavior],
        options: GuiLaunchOptions,
        *args: object,
        **kwargs: object,
    ) -> int:
        calls.append((behavior_cls, options, args, kwargs))
        return 43

    monkeypatch.setattr(runtime, "launch_gui_runtime", fake_launch_gui_runtime)

    launcher = _load_launcher("script.run_measure_gui")
    main = cast(Callable[[list[str] | None], int], getattr(launcher, "main"))
    project_root = cast(Path, getattr(launcher, "PROJECT_ROOT"))
    log_file = project_root / "measure-launcher-test.log"

    code = main(
        [
            "--no-log",
            "--clean",
            "--control-port",
            "9002",
            "--control-token",
            "secret",
            "--control-allow-external",
            "--log-file",
            str(log_file),
        ]
    )

    assert code == 43
    assert len(calls) == 1

    behavior_cls, options, args, kwargs = calls[0]
    assert args == ()
    assert behavior_cls.__name__ == "MeasureGuiBehavior"
    assert behavior_cls.spec.app_slug == "measure"
    assert behavior_cls.spec.default_control_port == 8765
    assert behavior_cls.spec.logging_extra_namespaces == (
        "zcu_tools.experiment.v2_gui",
    )

    assert options.log_root == project_root
    assert options.to_file is False
    assert options.log_file == log_file
    assert options.control_port == 9002
    assert options.control_token == "secret"
    assert options.control_allow_external is True
    assert options.no_control is False

    assert callable(kwargs["registry_factory"])
    assert kwargs["clean"] is True
    assert kwargs["project_root"] == str(project_root)


def test_autofluxdep_launcher_main_delegates_to_gui_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[
        tuple[
            type[GuiRuntimeBehavior],
            GuiLaunchOptions,
            tuple[object, ...],
            dict[str, object],
        ]
    ] = []

    def fake_launch_gui_runtime(
        behavior_cls: type[GuiRuntimeBehavior],
        options: GuiLaunchOptions,
        *args: object,
        **kwargs: object,
    ) -> int:
        calls.append((behavior_cls, options, args, kwargs))
        return 44

    monkeypatch.setattr(runtime, "launch_gui_runtime", fake_launch_gui_runtime)

    launcher = _load_launcher("script.run_autofluxdep_gui")
    main = cast(Callable[[list[str] | None], int], getattr(launcher, "main"))
    project_root = cast(Path, getattr(launcher, "PROJECT_ROOT"))
    log_file = project_root / "autofluxdep-launcher-test.log"

    code = main(
        [
            "--no-log",
            "--control-port",
            "9003",
            "--control-token",
            "secret",
            "--log-file",
            str(log_file),
        ]
    )

    assert code == 44
    assert len(calls) == 1

    behavior_cls, options, args, kwargs = calls[0]
    assert args == ()
    assert behavior_cls.__name__ == "AutoFluxDepGuiBehavior"
    assert behavior_cls.spec.app_slug == "autofluxdep"
    assert behavior_cls.spec.default_control_port == 8768
    assert behavior_cls.spec.logging_extra_namespaces == ("zcu_tools.program.v2",)

    assert options.log_root == project_root
    assert options.to_file is False
    assert options.log_file == log_file
    assert options.control_port == 9003
    assert options.control_token == "secret"
    assert options.no_control is False

    assert kwargs["project_root"] == str(project_root)
