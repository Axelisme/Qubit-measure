from __future__ import annotations

from pathlib import Path, PurePosixPath

import check_test_path_correspondence as checker
import pytest


def _build(root: Path, *, test_dirs: tuple[str, ...], source_dirs: tuple[str, ...]):
    for relative in source_dirs:
        (root / relative).mkdir(parents=True, exist_ok=True)
    for relative in test_dirs:
        directory = root / "tests" / relative
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "test_sample.py").write_text("", encoding="utf-8")


def test_reserved_segment_exempts_itself_and_everything_below_it(tmp_path: Path):
    _build(
        tmp_path,
        test_dirs=("mcp/contract", "gui/parity/acceptance"),
        source_dirs=("lib/zcu_tools/mcp", "lib/zcu_tools/gui"),
    )

    assert checker.violations(tmp_path) == ()


def test_reserved_segment_does_not_exempt_an_unmatched_prefix(tmp_path: Path):
    _build(
        tmp_path,
        test_dirs=("nonexistent/contract",),
        source_dirs=("lib/zcu_tools/mcp",),
    )

    found = checker.violations(tmp_path)

    assert len(found) == 1
    assert found[0].test_dir == PurePosixPath("tests/nonexistent/contract")
    assert found[0].expected_source_dir == PurePosixPath("lib/zcu_tools/nonexistent")


def test_directory_naming_no_module_is_a_violation_that_names_the_missing_path(
    tmp_path: Path,
):
    _build(
        tmp_path,
        test_dirs=("autofluxdep_gui",),
        source_dirs=("lib/zcu_tools/gui/app/autofluxdep",),
    )

    found = checker.violations(tmp_path)

    assert len(found) == 1
    assert found[0].expected_source_dir == PurePosixPath(
        "lib/zcu_tools/autofluxdep_gui"
    )


def test_directory_mirroring_a_module_path_passes(tmp_path: Path):
    _build(
        tmp_path,
        test_dirs=("gui/app/autofluxdep",),
        source_dirs=("lib/zcu_tools/gui/app/autofluxdep",),
    )

    assert checker.violations(tmp_path) == ()


@pytest.mark.parametrize("root", ["script", "tools"])
def test_declared_source_roots_resolve_outside_the_library(root: str, tmp_path: Path):
    """script/ and tools/ sit at the repository root, not under lib/zcu_tools/."""
    _build(tmp_path, test_dirs=(root,), source_dirs=(root,))

    assert checker.violations(tmp_path) == ()


@pytest.mark.parametrize("root", ["script", "tools"])
def test_a_declared_source_root_is_a_mapping_not_an_exemption(
    root: str, tmp_path: Path
):
    """The mapped target must exist; declaring the root does not waive the rule."""
    _build(tmp_path, test_dirs=(f"{root}/missing",), source_dirs=(root,))

    found = checker.violations(tmp_path)

    assert len(found) == 1
    assert found[0].expected_source_dir == PurePosixPath(f"{root}/missing")


def test_file_names_are_unconstrained_because_the_rule_is_module_level(tmp_path: Path):
    _build(tmp_path, test_dirs=("device",), source_dirs=("lib/zcu_tools/device",))
    (tmp_path / "tests" / "device" / "test_no_such_source_file.py").write_text(
        "", encoding="utf-8"
    )

    assert checker.violations(tmp_path) == ()


def test_directories_without_test_modules_are_not_checked(tmp_path: Path):
    (tmp_path / "tests" / "helpers").mkdir(parents=True)
    (tmp_path / "tests" / "helpers" / "support.py").write_text("", encoding="utf-8")

    assert checker.test_directories(tmp_path) == ()
    assert checker.violations(tmp_path) == ()
