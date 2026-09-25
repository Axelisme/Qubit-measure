from __future__ import annotations

from pathlib import Path, PurePosixPath

import check_file_size as checker


def _write(root: Path, relative: str, lines: int) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x = 1\n" * lines, encoding="utf-8")
    return path


def test_a_file_at_the_limit_passes_and_one_line_over_fails(tmp_path: Path):
    _write(tmp_path, "lib/exactly_at_limit.py", 10)
    _write(tmp_path, "lib/one_over.py", 11)

    found = checker.oversize_files(tmp_path, limit=10)

    assert [item.path for item in found] == [PurePosixPath("lib/one_over.py")]


def test_violations_are_ordered_longest_first(tmp_path: Path):
    _write(tmp_path, "lib/small.py", 12)
    _write(tmp_path, "lib/huge.py", 40)
    _write(tmp_path, "lib/medium.py", 20)

    found = checker.oversize_files(tmp_path, limit=10)

    assert [item.lines for item in found] == [40, 20, 12]


def test_every_checked_root_is_covered_not_only_the_library(tmp_path: Path):
    for root in ("lib", "tests", "script", "tools"):
        _write(tmp_path, f"{root}/big.py", 40)

    found = checker.oversize_files(tmp_path, limit=10)

    assert {item.path.parts[0] for item in found} == {"lib", "tests", "script", "tools"}


def test_generated_and_virtual_environment_trees_are_skipped(tmp_path: Path):
    _write(tmp_path, "lib/__pycache__/big.py", 40)
    _write(tmp_path, "lib/.venv/big.py", 40)
    _write(tmp_path, "lib/real.py", 40)

    found = checker.oversize_files(tmp_path, limit=10)

    assert [item.path for item in found] == [PurePosixPath("lib/real.py")]


def test_a_final_line_without_a_newline_still_counts(tmp_path: Path):
    path = _write(tmp_path, "lib/unterminated.py", 3)
    path.write_text("a = 1\nb = 2\nc = 3", encoding="utf-8")

    assert checker.line_count(path) == 3


def test_an_empty_file_counts_as_no_lines(tmp_path: Path):
    path = _write(tmp_path, "lib/empty.py", 0)

    assert checker.line_count(path) == 0


def test_named_paths_narrow_the_walk(tmp_path: Path):
    """The ratchet scans only what changed; a file grows only by being edited."""
    _write(tmp_path, "lib/edited.py", 40)
    _write(tmp_path, "lib/untouched.py", 40)

    found = checker.oversize_files(tmp_path, limit=10, paths=["lib/edited.py"])

    assert [item.path for item in found] == [PurePosixPath("lib/edited.py")]


def test_a_named_path_that_no_longer_exists_is_skipped(tmp_path: Path):
    """A file deleted by the candidate is absent from the base tree, and vice versa."""
    assert checker.oversize_files(tmp_path, limit=10, paths=["lib/gone.py"]) == ()
