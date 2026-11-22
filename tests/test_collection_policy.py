"""Exercise subtree collection and simulation marking through pytest sessions."""

import shutil
from pathlib import Path

import pytest

pytest_plugins = ["pytester"]


@pytest.fixture
def checkout(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Load the suite's real hooks in a small checkout with sibling subtrees."""
    repo = Path(__file__).resolve().parents[1]
    # Give copied conftests unique importlib names, rather than reusing the
    # outer session's tests.conftest module and its original __file__ path.
    (pytester.path / "__init__.py").write_text("", encoding="utf-8")
    tests = pytester.path / "tests"
    tests.mkdir()
    for name in (
        "__init__.py",
        "conftest.py",
        "_cache_environment.py",
        "_failure_log.py",
        "_temp_directory_guard.py",
        "simulate/__init__.py",
        "simulate/conftest.py",
        "simulate/evolution/contract/accuracy.py",
        "simulators/ftf/physics/conftest.py",
    ):
        target = tests / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo / "tests" / name, target)
    shutil.copyfile(repo / "pyproject.toml", pytester.path / "pyproject.toml")
    monkeypatch.delenv("PYTEST_ADDOPTS", raising=False)
    for subtree in (tests, tests / "simulate", tests / "simulators/ftf/physics"):
        (subtree / "test_example.py").write_text(
            "def test_example():\n    pass\n", encoding="utf-8"
        )
    return tests


class CollectionRecorder:
    """Observe imported items before command-line marker deselection."""

    def __init__(self) -> None:
        self.nodeids: list[str] = []

    def pytest_itemcollected(self, item: pytest.Item) -> None:
        self.nodeids.append(item.nodeid)


@pytest.mark.parametrize(
    "arguments",
    [
        (),
        ("-m", "not sim"),
        ("-m", "not (slow or sim)"),
        ("-m", "not slow and not sim", "tests"),
    ],
)
def test_default_selection_does_not_collect_simulate(
    pytester: pytest.Pytester, checkout: Path, arguments: tuple[str, ...]
) -> None:
    recorder = CollectionRecorder()
    result = pytester.runpytest_inprocess(
        "-q", "-n", "0", *arguments, plugins=[recorder]
    )

    result.assert_outcomes(passed=1, deselected=1)
    assert sorted(recorder.nodeids) == [
        "tests/simulators/ftf/physics/test_example.py::test_example",
        "tests/test_example.py::test_example",
    ]


@pytest.mark.parametrize(
    ("expression", "passed", "deselected"),
    [
        ("sim", 2, 1),
        ("sim or requires_loopback", 2, 1),
        ("not sim or requires_loopback", 1, 2),
        ("", 3, 0),
        ("not sim.with.dot", 3, 0),
        ("not sim-test", 3, 0),
    ],
)
def test_marker_selection_keeps_subtree_ownership(
    pytester: pytest.Pytester,
    checkout: Path,
    expression: str,
    passed: int,
    deselected: int,
) -> None:
    recorder = CollectionRecorder()
    result = pytester.runpytest_inprocess(
        "-q", "-n", "0", "-m", expression, plugins=[recorder]
    )

    result.assert_outcomes(passed=passed, deselected=deselected)
    assert sorted(recorder.nodeids) == [
        "tests/simulate/test_example.py::test_example",
        "tests/simulators/ftf/physics/test_example.py::test_example",
        "tests/test_example.py::test_example",
    ]


@pytest.mark.parametrize(
    "target",
    [
        "tests/simulate",
        "tests/simulate/test_example.py",
        "tests/simulate/test_example.py::test_example",
    ],
)
@pytest.mark.parametrize("absolute", [False, True])
def test_explicit_simulate_paths_keep_normal_collection_and_deselection(
    pytester: pytest.Pytester, checkout: Path, target: str, absolute: bool
) -> None:
    if absolute:
        target = str(pytester.path / target)
    recorder = CollectionRecorder()
    result = pytester.runpytest_inprocess(
        "-q", "-n", "0", "tests", target, plugins=[recorder]
    )

    result.assert_outcomes(passed=1, deselected=2)
    assert "tests/simulate/test_example.py::test_example" in recorder.nodeids
