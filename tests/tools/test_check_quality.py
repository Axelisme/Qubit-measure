"""Quality-checker contracts, using synthetic source trees in-process."""

from __future__ import annotations

import ast
import json
from pathlib import Path, PurePosixPath

import check_quality as checker
import pytest


@pytest.mark.parametrize(
    "command",
    [
        None,
        "file-size",
        "test-paths",
        "test-structure",
        "test-capabilities",
        "suppressions",
        "all",
    ],
)
def test_help_describes_the_selected_command_and_exits_zero(
    command: str | None, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as caught:
        checker.main([command, "-h"] if command else ["-h"])
    assert caught.value.code == 0
    output = capsys.readouterr()
    assert "usage:" in output.out
    assert output.err == ""
    if command:
        assert f" {command} " in output.out
    else:
        assert all(
            name in output.out
            for name in (
                "file-size",
                "test-paths",
                "test-structure",
                "test-capabilities",
                "suppressions",
                "all",
            )
        )


@pytest.fixture
def quality_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for name in ("tools", "tests", "lib"):
        (tmp_path / name).mkdir()
    # Only test-structure has --root; the other CLIs locate their own checkout.
    monkeypatch.setattr(checker, "__file__", str(tmp_path / "tools/check_quality.py"))
    return tmp_path


@pytest.mark.parametrize(
    ("scenario", "expected_code"),
    [
        ("clean", 0),
        ("oversize", 1),
        ("paths", 1),
        ("structure", 1),
        ("capability", 1),
        ("suppression", 0),
        ("invalid-root", 2),
        ("oversize-and-invalid-root", 2),
    ],
)
def test_all_combines_receipts_diagnostics_and_exit_codes(
    quality_root: Path,
    capsys: pytest.CaptureFixture[str],
    scenario: str,
    expected_code: int,
) -> None:
    if "oversize" in scenario:
        _write_lines(quality_root, "lib/big.py", 1001)
    if scenario == "paths":
        _write_test_source(quality_root, "missing/test_sample.py", "")
    if scenario == "structure":
        _write_test_source(
            quality_root, "test_sample.py", "def test_a(): pass\ndef test_a(): pass\n"
        )
    if scenario == "capability":
        _write_test_source(
            quality_root, "test_sample.py", "import time\ntime.sleep(1)\n"
        )
    if scenario == "suppression":
        (quality_root / "lib/sample.py").write_text("x = 1  # noqa: F841\n")
    if "invalid-root" in scenario:
        (quality_root / "tests").rmdir()
    names = [
        "file-size",
        "test-paths",
        "test-structure",
        "test-capabilities",
        "suppressions",
    ]
    receipts = {}
    diagnostics = []
    for name in names:
        checker.main([name])
        output = capsys.readouterr()
        receipts[name] = json.loads(output.out) if output.out else {}
        diagnostics.append(output.err)

    assert checker.main(["all"]) == expected_code
    output = capsys.readouterr()
    combined = json.loads(output.out)
    assert set(combined) == set(names)
    assert combined == receipts
    assert output.err == "".join(diagnostics)


@pytest.mark.parametrize("scenario", ["clean", "blocking", "missing", "syntax"])
def test_structure_main_reports_receipts_and_read_failures(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    scenario: str,
) -> None:
    if scenario != "missing":
        (tmp_path / "tests").mkdir()
    if scenario == "blocking":
        _write_test_source(
            tmp_path, "test_sample.py", "def test_a(): pass\ndef test_a(): pass\n"
        )
    if scenario == "syntax":
        _write_test_source(tmp_path, "test_sample.py", "def (:\n")
    code = checker.main(["test-structure", "--root", str(tmp_path)])
    output = capsys.readouterr()
    if scenario in ("missing", "syntax"):
        assert code == 2
        assert output.out == ""
        assert "test structure check failed:" in output.err
    else:
        expected = int(scenario == "blocking")
        assert code == expected
        receipt = json.loads(output.out)
        assert receipt["status"] == ("FAIL" if expected else "PASS")
        assert receipt["violation_count"] == expected
        assert receipt["advisory_count"] == 0
        assert len(receipt["findings"]) == expected
        if expected:
            assert receipt["findings"][0]["rule"] == "duplicate-test-definition"
            assert "blocking duplicate-test-definition" in output.err
        else:
            assert output.err == ""


def test_help_exits_successfully_without_scanning_file_size() -> None:
    with pytest.raises(SystemExit) as caught:
        checker.main(["file-size", "-h"])
    assert caught.value.code == 0


def _write_lines(root: Path, relative: str, lines: int) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x = 1\n" * lines, encoding="utf-8")
    return path


def test_a_file_at_the_limit_passes_and_one_line_over_fails_file_size(tmp_path: Path):
    _write_lines(tmp_path, "lib/exactly_at_limit.py", 10)
    _write_lines(tmp_path, "lib/one_over.py", 11)

    found = checker.oversize_files(tmp_path, limit=10)

    assert [item.path for item in found] == [PurePosixPath("lib/one_over.py")]


def test_violations_are_ordered_longest_first_file_size(tmp_path: Path):
    _write_lines(tmp_path, "lib/small.py", 12)
    _write_lines(tmp_path, "lib/huge.py", 40)
    _write_lines(tmp_path, "lib/medium.py", 20)

    found = checker.oversize_files(tmp_path, limit=10)

    assert [item.lines for item in found] == [40, 20, 12]


def test_every_checked_root_is_covered_not_only_the_library_file_size(tmp_path: Path):
    for root in ("lib", "tests", "scripts", "tools"):
        _write_lines(tmp_path, f"{root}/big.py", 40)

    found = checker.oversize_files(tmp_path, limit=10)

    assert {item.path.parts[0] for item in found} == {
        "lib",
        "tests",
        "scripts",
        "tools",
    }


def test_generated_and_virtual_environment_trees_are_skipped_file_size(tmp_path: Path):
    _write_lines(tmp_path, "lib/__pycache__/big.py", 40)
    _write_lines(tmp_path, "lib/.venv/big.py", 40)
    _write_lines(tmp_path, "lib/real.py", 40)

    found = checker.oversize_files(tmp_path, limit=10)

    assert [item.path for item in found] == [PurePosixPath("lib/real.py")]


def test_a_final_line_without_a_newline_still_counts_file_size(tmp_path: Path):
    path = _write_lines(tmp_path, "lib/unterminated.py", 3)
    path.write_text("a = 1\nb = 2\nc = 3", encoding="utf-8")

    assert checker.line_count(path) == 3


def test_an_empty_file_counts_as_no_lines_file_size(tmp_path: Path):
    path = _write_lines(tmp_path, "lib/empty.py", 0)

    assert checker.line_count(path) == 0


def test_named_paths_narrow_the_walk_file_size(tmp_path: Path):
    """The ratchet scans only what changed; a file grows only by being edited."""
    _write_lines(tmp_path, "lib/edited.py", 40)
    _write_lines(tmp_path, "lib/untouched.py", 40)

    found = checker.oversize_files(tmp_path, limit=10, paths=["lib/edited.py"])

    assert [item.path for item in found] == [PurePosixPath("lib/edited.py")]


def test_a_named_path_that_no_longer_exists_is_skipped_file_size(tmp_path: Path):
    """A file deleted by the candidate is absent from the base tree, and vice versa."""
    assert checker.oversize_files(tmp_path, limit=10, paths=["lib/gone.py"]) == ()


def test_help_exits_successfully_without_scanning_suppressions() -> None:
    with pytest.raises(SystemExit) as caught:
        checker.main(["suppressions", "-h"])
    assert caught.value.code == 0


def test_each_comment_silencer_is_counted_suppressions() -> None:
    source = (
        "a = 1  # noqa: C901\n"
        "b = 2  # type: ignore[arg-type]\n"
        "c = 3  # pyright: ignore[reportAny]\n"
        "# pyright: reportUnusedFunction=false\n"
    )

    assert checker.comment_suppressions(source) == {
        "noqa": 1,
        "type-ignore": 1,
        "pyright-ignore": 1,
        "pyright-file-rule": 1,
    }


def test_directives_after_another_comment_are_counted_suppressions() -> None:
    source = (
        "a = 1  # (shape) # type: ignore[index]\n"
        "b = 2  # explanation # pyright: ignore[reportX]\n"
        "c = 3  # explanation # noqa: E501\n"
    )

    assert checker.comment_suppressions(source) == {
        "type-ignore": 1,
        "pyright-ignore": 1,
        "noqa": 1,
    }


def test_mid_comment_file_rules_are_not_counted_suppressions() -> None:
    source = (
        "# explanation # pyright: reportAssignmentType=false\n"
        "# explanation `# pyright: reportAssignmentType=none`\n"
    )

    assert checker.comment_suppressions(source) == {}


def test_a_silencer_inside_a_string_is_not_a_silencer_suppressions() -> None:
    """Matching the file text would count documentation and test data."""
    source = 'message = "add # noqa: C901 to silence it"\n'

    assert checker.comment_suppressions(source) == {}


def test_prose_that_mentions_a_directive_is_not_a_use_of_it_suppressions() -> None:
    """Words without a directive at a hash-delimited segment do not silence."""
    source = (
        "# avoid type: ignore here\n"
        "# mention pyright: ignore here\n"
        "# mention noqa: E501 here\n"
        "# mention pyright: reportUnusedFunction=false here\n"
    )

    assert checker.comment_suppressions(source) == {}


def test_repeated_silencers_in_one_file_are_all_counted_suppressions() -> None:
    source = "a = 1  # noqa\nb = 2  # noqa\nc = 3  # noqa\n"

    assert checker.comment_suppressions(source) == {"noqa": 3}


def test_a_pyright_ignore_is_not_also_counted_as_a_file_rule_suppressions() -> None:
    assert checker.comment_suppressions("a = 1  # pyright: ignore\n") == {
        "pyright-ignore": 1
    }


def test_cast_is_counted_through_its_import_suppressions() -> None:
    tree = ast.parse("from typing import cast\nx = cast(int, 1)\n")

    assert checker.call_suppressions(tree) == {"cast": 1}


def test_an_aliased_cast_is_still_counted_suppressions() -> None:
    tree = ast.parse("import typing as t\nx = t.cast(int, 1)\n")

    assert checker.call_suppressions(tree) == {"cast": 1}


def test_a_local_function_named_cast_is_not_typing_cast_suppressions() -> None:
    tree = ast.parse("def cast(kind, value):\n    return value\n\nx = cast(int, 1)\n")

    assert checker.call_suppressions(tree) == {}


def test_a_file_reports_one_entry_per_kind_suppressions(tmp_path: Path):
    source = "from typing import cast\nx = cast(int, 1)  # noqa: F821\n"
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib" / "module.py").write_text(source, encoding="utf-8")

    found = checker.suppressions(tmp_path)

    assert {(str(item.path), item.kind, item.count) for item in found} == {
        ("lib/module.py", "cast", 1),
        ("lib/module.py", "noqa", 1),
    }


def test_a_file_that_silences_nothing_reports_nothing_suppressions(tmp_path: Path):
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib" / "clean.py").write_text("x = 1\n", encoding="utf-8")

    assert checker.suppressions(tmp_path) == ()


def test_generated_trees_are_skipped_suppressions(tmp_path: Path):
    (tmp_path / "lib" / "__pycache__").mkdir(parents=True)
    (tmp_path / "lib" / "__pycache__" / "m.py").write_text(
        "x = 1  # noqa\n", encoding="utf-8"
    )

    assert checker.suppressions(tmp_path) == ()


def test_an_unparsable_file_still_reports_its_comments_suppressions(tmp_path: Path):
    """A syntax error must not hide the silencers already in the file."""
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib" / "broken.py").write_text(
        "x = 1  # noqa\ndef (:\n", encoding="utf-8"
    )

    found = checker.suppressions(tmp_path)

    assert [(item.path, item.kind) for item in found] == [
        (PurePosixPath("lib/broken.py"), "noqa")
    ]


def _pyproject(root: Path, body: str) -> None:
    (root / "pyproject.toml").write_text(body, encoding="utf-8")


def test_a_per_file_ignore_is_counted_per_pattern_suppressions(tmp_path: Path):
    _pyproject(
        tmp_path,
        "[tool.ruff.lint.per-file-ignores]\n"
        '"__init__.py" = ["F401"]\n'
        '"tests/**" = ["PLR0913", "ARG001"]\n',
    )

    found = checker.config_suppressions(tmp_path)

    assert {(str(item.path), item.count) for item in found} == {
        ("pyproject.toml[per-file-ignores:__init__.py]", 1),
        ("pyproject.toml[per-file-ignores:tests/**]", 2),
    }


def test_global_ignores_are_counted_by_size_suppressions(tmp_path: Path):
    _pyproject(tmp_path, '[tool.ruff.lint]\nignore = ["E402", "E501"]\n')

    found = checker.config_suppressions(tmp_path)

    assert [(item.kind, item.count) for item in found] == [("ruff-global-ignore", 2)]


def test_a_pyright_rule_set_to_none_is_counted_suppressions(tmp_path: Path):
    _pyproject(
        tmp_path,
        "[tool.pyright]\n"
        'typeCheckingMode = "strict"\n'
        'reportUnknownMemberType = "none"\n'
        'reportPrivateUsage = "error"\n',
    )

    found = checker.config_suppressions(tmp_path)

    assert [(item.kind, item.count) for item in found] == [("pyright-global-rule", 1)]


def test_a_project_that_silences_nothing_by_configuration_suppressions(tmp_path: Path):
    _pyproject(tmp_path, '[project]\nname = "x"\n')

    assert checker.config_suppressions(tmp_path) == ()


def test_configuration_entries_join_the_file_entries_suppressions(tmp_path: Path):
    _pyproject(tmp_path, '[tool.ruff.lint]\nignore = ["E402"]\n')
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib" / "m.py").write_text("x = 1  # noqa\n", encoding="utf-8")

    kinds = {item.kind for item in checker.suppressions(tmp_path)}

    assert kinds == {"noqa", "ruff-global-ignore"}


def test_named_paths_narrow_the_walk_suppressions(tmp_path: Path):
    (tmp_path / "lib").mkdir()
    for name in ("edited.py", "untouched.py"):
        (tmp_path / "lib" / name).write_text("x = 1  # noqa\n", encoding="utf-8")

    found = checker.suppressions(tmp_path, paths=["lib/edited.py"])

    assert [str(item.path) for item in found] == ["lib/edited.py"]


def test_configuration_entries_survive_a_narrowed_walk_suppressions(tmp_path: Path):
    """They are not keyed by a source file, so no file list can exclude them."""
    _pyproject(tmp_path, '[tool.ruff.lint]\nignore = ["E402"]\n')

    found = checker.suppressions(tmp_path, paths=[])

    assert [item.kind for item in found] == ["ruff-global-ignore"]


def test_help_exits_successfully_without_scanning_test_capabilities() -> None:
    with pytest.raises(SystemExit) as caught:
        checker.main(["test-capabilities", "-h"])
    assert caught.value.code == 0


def _write_test_source(root: Path, relative: str, source: str) -> None:
    path = root / "tests" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def test_each_capability_is_recognised_at_its_call_site_test_capabilities() -> None:
    tree = ast.parse(
        "import asyncio, socket, subprocess, time\n"
        "time.sleep(1)\n"
        "asyncio.sleep(1)\n"
        "socket.socket()\n"
        "subprocess.run(['true'])\n"
    )

    assert checker.capability_uses(tree) == {
        "wall-clock": 2,
        "loopback": 1,
        "subprocess": 1,
    }


def test_a_name_imported_directly_is_still_a_use_test_capabilities() -> None:
    """`from time import sleep` is ordinary Python, not evasion, and must count."""
    tree = ast.parse("from time import sleep\nsleep(1)\n")

    assert checker.capability_uses(tree) == {"wall-clock": 1}


def test_an_aliased_module_is_still_a_use_test_capabilities() -> None:
    tree = ast.parse("import time as t\nt.sleep(1)\n")

    assert checker.capability_uses(tree) == {"wall-clock": 1}


def test_an_aliased_name_is_still_a_use_test_capabilities() -> None:
    tree = ast.parse("from time import sleep as nap\nnap(1)\n")

    assert checker.capability_uses(tree) == {"wall-clock": 1}


def test_the_indirect_spellings_of_a_capability_count_test_capabilities() -> None:
    tree = ast.parse(
        "import os\nimport socket\nos.system('true')\nsocket.socketpair()\n"
    )

    assert checker.capability_uses(tree) == {"subprocess": 1, "loopback": 1}


def test_a_name_that_was_never_imported_is_not_resolved_test_capabilities() -> None:
    """A local helper called sleep() is not time.sleep and must not be claimed."""
    tree = ast.parse("def sleep(seconds):\n    return seconds\n\nsleep(1)\n")

    assert checker.capability_uses(tree) == {}


def test_a_relative_import_binds_nothing_test_capabilities() -> None:
    tree = ast.parse("from .helpers import sleep\nsleep(1)\n")

    assert checker.capability_uses(tree) == {}


def test_a_bare_reference_is_not_a_use_test_capabilities() -> None:
    tree = ast.parse("import time\nhandler = time.sleep\n")

    assert checker.capability_uses(tree) == {}


def test_a_declared_module_is_not_reported_test_capabilities(tmp_path: Path):
    _write_test_source(
        tmp_path,
        "test_declared.py",
        "import time\n\nimport pytest\n\n"
        "pytestmark = pytest.mark.uses_wall_clock\n\n"
        "def test_waits():\n    time.sleep(0)\n",
    )

    assert checker.undeclared_capabilities(tmp_path) == ()


def test_a_marker_on_a_single_test_declares_it_for_the_module_test_capabilities(
    tmp_path: Path,
):
    _write_test_source(
        tmp_path,
        "test_decorated.py",
        "import time\n\nimport pytest\n\n"
        "@pytest.mark.uses_wall_clock\n"
        "def test_waits():\n    time.sleep(0)\n",
    )

    assert checker.undeclared_capabilities(tmp_path) == ()


def test_an_undeclared_module_names_the_marker_it_is_missing_test_capabilities(
    tmp_path: Path,
):
    _write_test_source(
        tmp_path,
        "test_undeclared.py",
        "import time\n\ndef test_waits():\n    time.sleep(0)\n    time.sleep(1)\n",
    )

    found = checker.undeclared_capabilities(tmp_path)

    assert len(found) == 1
    assert found[0].module == PurePosixPath("tests/test_undeclared.py")
    assert found[0].marker == "uses_wall_clock"
    assert found[0].uses == 2


def test_the_wrong_marker_does_not_declare_a_capability_test_capabilities(
    tmp_path: Path,
):
    _write_test_source(
        tmp_path,
        "test_mismarked.py",
        "import subprocess\n\nimport pytest\n\n"
        "pytestmark = pytest.mark.uses_wall_clock\n\n"
        "def test_spawns():\n    subprocess.run(['true'])\n",
    )

    found = checker.undeclared_capabilities(tmp_path)

    assert [item.capability for item in found] == ["subprocess"]


def test_a_module_that_uses_nothing_special_is_silent_test_capabilities(tmp_path: Path):
    _write_test_source(
        tmp_path, "test_plain.py", "def test_adds():\n    assert 1 + 1 == 2\n"
    )

    assert checker.undeclared_capabilities(tmp_path) == ()


def test_named_paths_narrow_the_walk_test_capabilities(tmp_path: Path):
    source = "import time\n\ndef test_waits():\n    time.sleep(0)\n"
    _write_test_source(tmp_path, "test_edited.py", source)
    _write_test_source(tmp_path, "test_untouched.py", source)

    found = checker.undeclared_capabilities(tmp_path, paths=["tests/test_edited.py"])

    assert [item.module for item in found] == [PurePosixPath("tests/test_edited.py")]


def test_named_paths_outside_tests_are_ignored_test_capabilities(tmp_path: Path):
    _write_test_source(tmp_path, "test_edited.py", "import time\ntime.sleep(0)\n")
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib" / "test_lookalike.py").write_text(
        "import time\ntime.sleep(0)\n", encoding="utf-8"
    )

    found = checker.undeclared_capabilities(tmp_path, paths=["lib/test_lookalike.py"])

    assert found == ()


def test_help_exits_successfully_without_scanning_test_path_correspondence() -> None:
    with pytest.raises(SystemExit) as caught:
        checker.main(["test-paths", "-h"])
    assert caught.value.code == 0


def _build(root: Path, *, test_dirs: tuple[str, ...], source_dirs: tuple[str, ...]):
    for relative in source_dirs:
        (root / relative).mkdir(parents=True, exist_ok=True)
    for relative in test_dirs:
        directory = root / "tests" / relative
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "test_sample.py").write_text("", encoding="utf-8")


def test_reserved_segment_exempts_itself_and_everything_below_it_test_path_correspondence(
    tmp_path: Path,
):
    _build(
        tmp_path,
        test_dirs=("mcp/contract", "gui/parity/acceptance"),
        source_dirs=("lib/zcu_tools/mcp", "lib/zcu_tools/gui"),
    )

    assert checker.path_violations(tmp_path) == ()


def test_reserved_segment_does_not_exempt_an_unmatched_prefix_test_path_correspondence(
    tmp_path: Path,
):
    _build(
        tmp_path,
        test_dirs=("nonexistent/contract",),
        source_dirs=("lib/zcu_tools/mcp",),
    )

    found = checker.path_violations(tmp_path)

    assert len(found) == 1
    assert found[0].test_dir == PurePosixPath("tests/nonexistent/contract")
    assert found[0].expected_source_dir == PurePosixPath("lib/zcu_tools/nonexistent")


def test_directory_naming_no_module_is_a_violation_that_names_the_missing_path_test_path_correspondence(
    tmp_path: Path,
):
    _build(
        tmp_path,
        test_dirs=("autofluxdep_gui",),
        source_dirs=("lib/zcu_tools/gui/app/autofluxdep",),
    )

    found = checker.path_violations(tmp_path)

    assert len(found) == 1
    assert found[0].expected_source_dir == PurePosixPath(
        "lib/zcu_tools/autofluxdep_gui"
    )


def test_directory_mirroring_a_module_path_passes_test_path_correspondence(
    tmp_path: Path,
):
    _build(
        tmp_path,
        test_dirs=("gui/app/autofluxdep",),
        source_dirs=("lib/zcu_tools/gui/app/autofluxdep",),
    )

    assert checker.path_violations(tmp_path) == ()


@pytest.mark.parametrize("root", ["scripts", "tools"])
def test_declared_source_roots_resolve_outside_the_library_test_path_correspondence(
    root: str, tmp_path: Path
):
    """scripts/ and tools/ sit at the repository root, not under lib/zcu_tools/."""
    _build(tmp_path, test_dirs=(root,), source_dirs=(root,))

    assert checker.path_violations(tmp_path) == ()


@pytest.mark.parametrize("root", ["scripts", "tools"])
def test_a_declared_source_root_is_a_mapping_not_an_exemption_test_path_correspondence(
    root: str, tmp_path: Path
):
    """The mapped target must exist; declaring the root does not waive the rule."""
    _build(tmp_path, test_dirs=(f"{root}/missing",), source_dirs=(root,))

    found = checker.path_violations(tmp_path)

    assert len(found) == 1
    assert found[0].expected_source_dir == PurePosixPath(f"{root}/missing")


def test_file_names_are_unconstrained_because_the_rule_is_module_level_test_path_correspondence(
    tmp_path: Path,
):
    _build(tmp_path, test_dirs=("device",), source_dirs=("lib/zcu_tools/device",))
    (tmp_path / "tests" / "device" / "test_no_such_source_file.py").write_text(
        "", encoding="utf-8"
    )

    assert checker.path_violations(tmp_path) == ()


def test_directories_without_test_modules_are_not_checked_test_path_correspondence(
    tmp_path: Path,
):
    (tmp_path / "tests" / "helpers").mkdir(parents=True)
    (tmp_path / "tests" / "helpers" / "support.py").write_text("", encoding="utf-8")

    assert checker.test_directories(tmp_path) == ()
    assert checker.path_violations(tmp_path) == ()
