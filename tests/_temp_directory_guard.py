"""Pytest session guard for unowned context directories in a private temp root."""

from pathlib import Path

import pytest


class ContextDirectoryGuard:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root
        self.before: set[Path] = set()
        self.observed: set[Path] = set()
        self.owners: dict[Path, str] = {}
        self.leaks: dict[str, str] = {}

    def start(self, root: Path) -> None:
        self.root = root
        self.before = {path for path in root.glob("context-*") if path.is_dir()}
        self.observed = self.before.copy()

    def pytest_sessionstart(self, session: pytest.Session) -> None:
        if self.root is not None:
            self.start(self.root)

    @pytest.hookimpl(wrapper=True)
    def pytest_runtest_protocol(self, item: pytest.Item):
        try:
            return (yield)
        finally:
            if self.root is not None:
                current = {
                    path for path in self.root.glob("context-*") if path.is_dir()
                }
                for path in current - self.observed:
                    self.owners[path] = item.nodeid
                self.observed = current

    @pytest.hookimpl(optionalhook=True)
    def pytest_testnodedown(self, node, error) -> None:
        self.leaks.update(node.workeroutput.get("context_directory_leaks", {}))

    def pytest_sessionfinish(self, session: pytest.Session) -> None:
        if self.root is not None:
            current = {path for path in self.root.glob("context-*") if path.is_dir()}
            for path in current - self.before:
                self.leaks.setdefault(
                    str(path), self.owners.get(path, "session setup/teardown")
                )
        # A worker has no terminal reporter; forward its observations to the
        # controller so xdist reports the nodeid as well as the new directory.
        worker_output = getattr(session.config, "workeroutput", None)
        if worker_output is not None:
            worker_output["context_directory_leaks"] = self.leaks
        if self.leaks and session.exitstatus == pytest.ExitCode.OK:
            session.exitstatus = pytest.ExitCode.TESTS_FAILED

    def pytest_terminal_summary(self, terminalreporter) -> None:
        if self.leaks:
            terminalreporter.section("Leaked session context-* directories", red=True)
            terminalreporter.write_line(
                f"{len(self.leaks)} new directories in private tempfile roots."
            )
            for path, owner in sorted(self.leaks.items()):
                terminalreporter.write_line(f"{path} — observed during {owner}")
