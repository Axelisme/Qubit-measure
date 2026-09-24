from __future__ import annotations

import ast
from pathlib import Path, PurePosixPath

import check_test_capabilities as checker


def _write(root: Path, relative: str, source: str) -> None:
    path = root / "tests" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def test_each_capability_is_recognised_at_its_call_site() -> None:
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


def test_a_name_imported_directly_is_still_a_use() -> None:
    """`from time import sleep` is ordinary Python, not evasion, and must count."""
    tree = ast.parse("from time import sleep\nsleep(1)\n")

    assert checker.capability_uses(tree) == {"wall-clock": 1}


def test_an_aliased_module_is_still_a_use() -> None:
    tree = ast.parse("import time as t\nt.sleep(1)\n")

    assert checker.capability_uses(tree) == {"wall-clock": 1}


def test_an_aliased_name_is_still_a_use() -> None:
    tree = ast.parse("from time import sleep as nap\nnap(1)\n")

    assert checker.capability_uses(tree) == {"wall-clock": 1}


def test_the_indirect_spellings_of_a_capability_count() -> None:
    tree = ast.parse(
        "import os\nimport socket\nos.system('true')\nsocket.socketpair()\n"
    )

    assert checker.capability_uses(tree) == {"subprocess": 1, "loopback": 1}


def test_a_name_that_was_never_imported_is_not_resolved() -> None:
    """A local helper called sleep() is not time.sleep and must not be claimed."""
    tree = ast.parse("def sleep(seconds):\n    return seconds\n\nsleep(1)\n")

    assert checker.capability_uses(tree) == {}


def test_a_relative_import_binds_nothing() -> None:
    tree = ast.parse("from .helpers import sleep\nsleep(1)\n")

    assert checker.capability_uses(tree) == {}


def test_a_bare_reference_is_not_a_use() -> None:
    tree = ast.parse("import time\nhandler = time.sleep\n")

    assert checker.capability_uses(tree) == {}


def test_a_declared_module_is_not_reported(tmp_path: Path):
    _write(
        tmp_path,
        "test_declared.py",
        "import time\n\nimport pytest\n\n"
        "pytestmark = pytest.mark.uses_wall_clock\n\n"
        "def test_waits():\n    time.sleep(0)\n",
    )

    assert checker.undeclared_capabilities(tmp_path) == ()


def test_a_marker_on_a_single_test_declares_it_for_the_module(tmp_path: Path):
    _write(
        tmp_path,
        "test_decorated.py",
        "import time\n\nimport pytest\n\n"
        "@pytest.mark.uses_wall_clock\n"
        "def test_waits():\n    time.sleep(0)\n",
    )

    assert checker.undeclared_capabilities(tmp_path) == ()


def test_an_undeclared_module_names_the_marker_it_is_missing(tmp_path: Path):
    _write(
        tmp_path,
        "test_undeclared.py",
        "import time\n\ndef test_waits():\n    time.sleep(0)\n    time.sleep(1)\n",
    )

    found = checker.undeclared_capabilities(tmp_path)

    assert len(found) == 1
    assert found[0].module == PurePosixPath("tests/test_undeclared.py")
    assert found[0].marker == "uses_wall_clock"
    assert found[0].uses == 2


def test_the_wrong_marker_does_not_declare_a_capability(tmp_path: Path):
    _write(
        tmp_path,
        "test_mismarked.py",
        "import subprocess\n\nimport pytest\n\n"
        "pytestmark = pytest.mark.uses_wall_clock\n\n"
        "def test_spawns():\n    subprocess.run(['true'])\n",
    )

    found = checker.undeclared_capabilities(tmp_path)

    assert [item.capability for item in found] == ["subprocess"]


def test_a_module_that_uses_nothing_special_is_silent(tmp_path: Path):
    _write(tmp_path, "test_plain.py", "def test_adds():\n    assert 1 + 1 == 2\n")

    assert checker.undeclared_capabilities(tmp_path) == ()


def test_named_paths_narrow_the_walk(tmp_path: Path):
    source = "import time\n\ndef test_waits():\n    time.sleep(0)\n"
    _write(tmp_path, "test_edited.py", source)
    _write(tmp_path, "test_untouched.py", source)

    found = checker.undeclared_capabilities(tmp_path, paths=["tests/test_edited.py"])

    assert [item.module for item in found] == [PurePosixPath("tests/test_edited.py")]


def test_named_paths_outside_tests_are_ignored(tmp_path: Path):
    _write(tmp_path, "test_edited.py", "import time\ntime.sleep(0)\n")
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib" / "test_lookalike.py").write_text(
        "import time\ntime.sleep(0)\n", encoding="utf-8"
    )

    found = checker.undeclared_capabilities(tmp_path, paths=["lib/test_lookalike.py"])

    assert found == ()
