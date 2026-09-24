from __future__ import annotations

import ast
from pathlib import Path

import _support


def _call(source: str, index: int = 0) -> ast.Call:
    """Return the call expression at `index`, checked rather than assumed."""
    statement = ast.parse(source).body[index]
    assert isinstance(statement, ast.Expr)
    assert isinstance(statement.value, ast.Call)
    return statement.value


def test_an_attribute_chain_reads_as_a_dotted_name() -> None:
    assert _support.dotted_name(_call("a.b.c()").func) == "a.b.c"


def test_a_chain_not_rooted_in_a_name_has_no_dotted_form() -> None:
    """`f().g` names nothing resolvable; a caller must not treat it as `?.g`."""
    assert _support.dotted_name(_call("f().g()").func) is None


def test_each_import_form_binds_its_local_name() -> None:
    tree = ast.parse(
        "import time\n"
        "import typing as t\n"
        "from os import path\n"
        "from time import sleep as nap\n"
        "import a.b.c\n"
    )

    assert _support.import_bindings(tree) == {
        "time": "time",
        "t": "typing",
        "path": "os.path",
        "nap": "time.sleep",
        "a": "a",
    }


def test_a_relative_import_binds_nothing() -> None:
    assert _support.import_bindings(ast.parse("from .local import helper\n")) == {}


def test_a_call_resolves_through_its_binding() -> None:
    source = "import typing as t\nt.cast(int, 1)\n"
    bindings = _support.import_bindings(ast.parse(source))

    assert _support.resolve_call(_call(source, 1).func, bindings) == "typing.cast"


def test_an_unimported_name_resolves_to_nothing() -> None:
    """A local function named cast is not typing.cast."""
    source = "def cast(k, v):\n    return v\n\ncast(int, 1)\n"
    bindings = _support.import_bindings(ast.parse(source))

    assert _support.resolve_call(_call(source, 1).func, bindings) is None


def test_the_walk_covers_every_checked_root(tmp_path: Path):
    for root in _support.CHECKED_ROOTS:
        (tmp_path / root).mkdir()
        (tmp_path / root / "m.py").write_text("x = 1\n", encoding="utf-8")

    found = _support.python_files(tmp_path)

    assert {path.parent.name for path in found} == set(_support.CHECKED_ROOTS)


def test_generated_trees_are_skipped(tmp_path: Path):
    for skipped in _support.SKIPPED_DIRS:
        (tmp_path / "lib" / skipped).mkdir(parents=True)
        (tmp_path / "lib" / skipped / "m.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "lib" / "real.py").write_text("x = 1\n", encoding="utf-8")

    found = _support.python_files(tmp_path)

    assert [path.name for path in found] == ["real.py"]


def test_named_paths_narrow_the_walk_and_tolerate_absence(tmp_path: Path):
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib" / "here.py").write_text("x = 1\n", encoding="utf-8")

    found = _support.python_files(tmp_path, paths=["lib/here.py", "lib/gone.py"])

    assert [path.name for path in found] == ["here.py"]


def test_a_sibling_check_loads_by_name() -> None:
    module = _support.load_tool("check_file_size")

    assert module.LINE_LIMIT == 1000
