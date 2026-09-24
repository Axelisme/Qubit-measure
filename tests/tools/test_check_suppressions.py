from __future__ import annotations

import ast
from pathlib import Path, PurePosixPath

import check_suppressions as checker


def test_each_comment_silencer_is_counted() -> None:
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


def test_a_silencer_inside_a_string_is_not_a_silencer() -> None:
    """Matching the file text would count documentation and test data."""
    source = 'message = "add # noqa: C901 to silence it"\n'

    assert checker.comment_suppressions(source) == {}


def test_prose_that_mentions_a_directive_is_not_a_use_of_it() -> None:
    """A comment explaining `# noqa` is documentation, not a silenced finding."""
    source = "# the `# noqa` comment below silences it\n# see also # type: ignore\n"

    assert checker.comment_suppressions(source) == {}


def test_repeated_silencers_in_one_file_are_all_counted() -> None:
    source = "a = 1  # noqa\nb = 2  # noqa\nc = 3  # noqa\n"

    assert checker.comment_suppressions(source) == {"noqa": 3}


def test_a_pyright_ignore_is_not_also_counted_as_a_file_rule() -> None:
    assert checker.comment_suppressions("a = 1  # pyright: ignore\n") == {
        "pyright-ignore": 1
    }


def test_cast_is_counted_through_its_import() -> None:
    tree = ast.parse("from typing import cast\nx = cast(int, 1)\n")

    assert checker.call_suppressions(tree) == {"cast": 1}


def test_an_aliased_cast_is_still_counted() -> None:
    tree = ast.parse("import typing as t\nx = t.cast(int, 1)\n")

    assert checker.call_suppressions(tree) == {"cast": 1}


def test_a_local_function_named_cast_is_not_typing_cast() -> None:
    tree = ast.parse("def cast(kind, value):\n    return value\n\nx = cast(int, 1)\n")

    assert checker.call_suppressions(tree) == {}


def test_a_file_reports_one_entry_per_kind(tmp_path: Path):
    source = "from typing import cast\nx = cast(int, 1)  # noqa: F821\n"
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib" / "module.py").write_text(source, encoding="utf-8")

    found = checker.suppressions(tmp_path)

    assert {(str(item.path), item.kind, item.count) for item in found} == {
        ("lib/module.py", "cast", 1),
        ("lib/module.py", "noqa", 1),
    }


def test_a_file_that_silences_nothing_reports_nothing(tmp_path: Path):
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib" / "clean.py").write_text("x = 1\n", encoding="utf-8")

    assert checker.suppressions(tmp_path) == ()


def test_generated_trees_are_skipped(tmp_path: Path):
    (tmp_path / "lib" / "__pycache__").mkdir(parents=True)
    (tmp_path / "lib" / "__pycache__" / "m.py").write_text(
        "x = 1  # noqa\n", encoding="utf-8"
    )

    assert checker.suppressions(tmp_path) == ()


def test_an_unparsable_file_still_reports_its_comments(tmp_path: Path):
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


def test_a_per_file_ignore_is_counted_per_pattern(tmp_path: Path):
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


def test_global_ignores_are_counted_by_size(tmp_path: Path):
    _pyproject(tmp_path, '[tool.ruff.lint]\nignore = ["E402", "E501"]\n')

    found = checker.config_suppressions(tmp_path)

    assert [(item.kind, item.count) for item in found] == [("ruff-global-ignore", 2)]


def test_a_pyright_rule_set_to_none_is_counted(tmp_path: Path):
    _pyproject(
        tmp_path,
        "[tool.pyright]\n"
        'typeCheckingMode = "strict"\n'
        'reportUnknownMemberType = "none"\n'
        'reportPrivateUsage = "error"\n',
    )

    found = checker.config_suppressions(tmp_path)

    assert [(item.kind, item.count) for item in found] == [("pyright-global-rule", 1)]


def test_a_project_that_silences_nothing_by_configuration(tmp_path: Path):
    _pyproject(tmp_path, '[project]\nname = "x"\n')

    assert checker.config_suppressions(tmp_path) == ()


def test_configuration_entries_join_the_file_entries(tmp_path: Path):
    _pyproject(tmp_path, '[tool.ruff.lint]\nignore = ["E402"]\n')
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib" / "m.py").write_text("x = 1  # noqa\n", encoding="utf-8")

    kinds = {item.kind for item in checker.suppressions(tmp_path)}

    assert kinds == {"noqa", "ruff-global-ignore"}


def test_named_paths_narrow_the_walk(tmp_path: Path):
    (tmp_path / "lib").mkdir()
    for name in ("edited.py", "untouched.py"):
        (tmp_path / "lib" / name).write_text("x = 1  # noqa\n", encoding="utf-8")

    found = checker.suppressions(tmp_path, paths=["lib/edited.py"])

    assert [str(item.path) for item in found] == ["lib/edited.py"]


def test_configuration_entries_survive_a_narrowed_walk(tmp_path: Path):
    """They are not keyed by a source file, so no file list can exclude them."""
    _pyproject(tmp_path, '[tool.ruff.lint]\nignore = ["E402"]\n')

    found = checker.suppressions(tmp_path, paths=[])

    assert [item.kind for item in found] == ["ruff-global-ignore"]
