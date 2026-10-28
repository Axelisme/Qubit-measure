"""Session-level filesystem effects, exercised through public pytest hooks."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import pytest

from tests._temp_directory_guard import ContextDirectoryGuard


class SyntheticItem(pytest.Item):
    def runtest(self) -> None:
        pass


class RecordingTerminal:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def section(self, title: str, **kwargs) -> None:
        self.lines.append(title)

    def write_line(self, line: str, **kwargs) -> None:
        self.lines.append(line)


def test_default_temp_directories_are_owned_by_the_pytest_session(tmp_path_factory):
    with TemporaryDirectory(prefix="ordinary-") as directory:
        assert Path(directory).is_relative_to(tmp_path_factory.getbasetemp())


def test_worker_crash_without_output_does_not_report_a_directory_leak():
    guard = ContextDirectoryGuard()
    terminal = RecordingTerminal()

    guard.pytest_testnodedown(SimpleNamespace(), "worker crashed")
    guard.pytest_terminal_summary(terminal)

    assert terminal.lines == []


def test_worker_directory_leaks_are_merged_and_reported_with_their_owners():
    config = pytest.Config.fromdictargs({}, [])
    session = pytest.Session.from_config(config)
    session.exitstatus = pytest.ExitCode.OK
    guard = ContextDirectoryGuard()
    terminal = RecordingTerminal()
    try:
        for path, owner in (
            ("/worker-0/context-first", "test_first.py::test_polluter"),
            ("/worker-1/context-second", "test_second.py::test_polluter"),
        ):
            node = SimpleNamespace(
                workeroutput={"context_directory_leaks": {path: owner}}
            )
            guard.pytest_testnodedown(node, None)
        guard.pytest_sessionfinish(session)
        guard.pytest_terminal_summary(terminal)

        assert session.exitstatus == pytest.ExitCode.TESTS_FAILED
        report = "\n".join(terminal.lines)
        assert (
            "/worker-0/context-first — observed during test_first.py::test_polluter"
            in report
        )
        assert (
            "/worker-1/context-second — observed during test_second.py::test_polluter"
            in report
        )
    finally:
        config.hook.pytest_unconfigure(config=config)


@pytest.mark.parametrize("leak", [False, True])
def test_session_rejects_only_new_context_directories(tmp_path, leak):
    root = tmp_path / "tmp"
    root.mkdir()
    existing = root / "context-existing"
    existing.mkdir()
    config = pytest.Config.fromdictargs({}, [])
    session = pytest.Session.from_config(config)
    session.exitstatus = pytest.ExitCode.OK
    guard = ContextDirectoryGuard(root)
    terminal = RecordingTerminal()
    module = pytest.Module.from_parent(session, path=tmp_path / "test_sample.py")
    item = SyntheticItem.from_parent(module, name="test_polluter")
    try:
        guard.pytest_sessionstart(session)
        protocol = guard.pytest_runtest_protocol(item)
        next(protocol)
        (root / "other-directory").mkdir()
        new = root / "context-new"
        new.mkdir()
        if not leak:
            new.rmdir()
        with pytest.raises(StopIteration):
            next(protocol)
        guard.pytest_sessionfinish(session)
        guard.pytest_terminal_summary(terminal)

        if leak:
            assert session.exitstatus == pytest.ExitCode.TESTS_FAILED
            report = "\n".join(terminal.lines)
            assert str(new) in report
            assert item.nodeid in report
            assert str(existing) not in report
        else:
            assert session.exitstatus == pytest.ExitCode.OK
            assert terminal.lines == []
    finally:
        config.hook.pytest_unconfigure(config=config)
