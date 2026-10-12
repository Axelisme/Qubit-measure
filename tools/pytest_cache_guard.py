"""Pytest GUI cache isolation and pollution diagnostics."""

from __future__ import annotations

import hashlib
import os
import stat
import sys
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from platformdirs import user_cache_dir

_REAL_CACHE_ENV = "ZCU_TOOLS_PYTEST_REAL_GUI_CACHE"


@dataclass(frozen=True)
class _CacheEntry:
    mode: int
    size: int
    mtime_ns: int
    ctime_ns: int
    inode: int
    content: str


def _snapshot(root: Path) -> dict[str, _CacheEntry]:
    """Read tree metadata and content without following directory symlinks."""
    # Test fixtures may replace pathlib.exists for their own I/O error cases.
    # Read the real filesystem here so that protection does not use that fake.
    try:
        os.lstat(root)
    except FileNotFoundError:
        return {}
    paths = [root, *sorted(root.rglob("*"))]
    entries: dict[str, _CacheEntry] = {}
    for path in paths:
        info = os.lstat(path)
        if stat.S_ISLNK(info.st_mode):
            content = str(path.readlink())
        elif stat.S_ISREG(info.st_mode):
            content = hashlib.sha256(path.read_bytes()).hexdigest()
        else:
            content = ""
        entries[str(path.relative_to(root))] = _CacheEntry(
            info.st_mode,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
            info.st_ino,
            content,
        )
    return entries


class GuiCacheGuard:
    """Isolate default GUI persistence and report writes to a protected cache.

    ``real_cache`` is the GUI directory to protect; omitted means the current
    platformdirs GUI cache (or the protected path inherited from an xdist
    controller via ZCU_TOOLS_PYTEST_REAL_GUI_CACHE). ``cache_root`` is the
    temporary XDG cache root; omitted means an owned temporary directory inside
    this checkout. Construct before importing test modules, register with
    pytest, and let unconfigure restore both environment variables. Each worker
    gets its own temporary cache, but all protect the original user directory.

    Changes in existence, metadata or file contents fail collection/test
    reports with the responsible node and phase. Changes outside those reports
    fail startup or session finish with that lifecycle phase. Only Linux
    supports XDG configuration; other platforms raise pytest.UsageError before
    changing the environment. The guard reports pollution, never restores or
    deletes user cache data. Filesystem read errors propagate rather than
    silently disabling protection.
    """

    def __init__(
        self, *, real_cache: Path | None = None, cache_root: Path | None = None
    ) -> None:
        if sys.platform != "linux":
            raise pytest.UsageError(
                "GUI cache isolation only supports Linux (XDG_CACHE_HOME)"
            )
        inherited_real_cache = os.environ.get(_REAL_CACHE_ENV)
        if real_cache is not None:
            self._real_cache = real_cache
        elif inherited_real_cache is not None:
            self._real_cache = Path(inherited_real_cache)
        else:
            self._real_cache = Path(user_cache_dir("zcu_tools", "zcu_tools")) / "gui"
        self._last_snapshot = _snapshot(self._real_cache)
        self._pollutions: list[str] = []
        self._previous_cache_home = os.environ.get("XDG_CACHE_HOME")
        self._previous_real_cache = inherited_real_cache
        self._temporary_cache: TemporaryDirectory[str] | None = None
        self._closed = False
        if cache_root is None:
            self._temporary_cache = TemporaryDirectory(
                prefix=".pytest-tmp-cache-", dir=Path(__file__).resolve().parents[1]
            )
            cache_root = Path(self._temporary_cache.name)
        os.environ["XDG_CACHE_HOME"] = str(cache_root.resolve())
        os.environ[_REAL_CACHE_ENV] = str(self._real_cache.resolve())

    def _check(self, owner: str) -> str | None:
        current = _snapshot(self._real_cache)
        changed = sorted(
            name
            for name in self._last_snapshot.keys() | current.keys()
            if self._last_snapshot.get(name) != current.get(name)
        )
        self._last_snapshot = current
        if not changed:
            return None
        message = (
            f"GUI cache pollution by {owner}: {self._real_cache}; "
            f"changed paths: {', '.join(changed)}"
        )
        self._pollutions.append(message)
        return message

    def pytest_configure(self) -> None:
        """Reject writes made while initial conftests were imported."""
        message = self._check("startup (initial conftest imports)")
        if message is not None:
            raise pytest.UsageError(message)

    @pytest.hookimpl(wrapper=True)
    def pytest_make_collect_report(
        self, collector: pytest.Collector
    ) -> Generator[None, pytest.CollectReport, pytest.CollectReport]:
        """Attribute module-import writes to their collection node."""
        report = yield
        message = self._check(f"{collector.nodeid or '<session>'} (collection)")
        if message is not None:
            report.outcome = "failed"
            report.longrepr = str(report.longrepr or "") + "\n" + message
        return report

    @pytest.hookimpl(wrapper=True)
    def pytest_runtest_makereport(
        self, item: pytest.Item
    ) -> Generator[None, pytest.TestReport, pytest.TestReport]:
        """Fail the setup, call or teardown report that polluted the cache."""
        report = yield
        message = self._check(f"{item.nodeid} ({report.when})")
        if message is not None:
            report.outcome = "failed"
            report.longrepr = str(report.longrepr or "") + "\n" + message
        return report

    @pytest.hookimpl(wrapper=True, tryfirst=True)
    def pytest_sessionfinish(
        self, session: pytest.Session
    ) -> Generator[None, None, None]:
        """Fail on remaining lifecycle pollution, including collection-only runs."""
        yield
        self._check("session finish (outside test reports)")
        if self._pollutions:
            session.exitstatus = pytest.ExitCode.TESTS_FAILED
            for message in self._pollutions:
                session.config.get_terminal_writer().line(message)

    def pytest_unconfigure(self) -> None:
        """Restore the environment and remove owned temporary cache files."""
        if self._closed:
            return
        self._closed = True
        if self._previous_cache_home is None:
            os.environ.pop("XDG_CACHE_HOME", None)
        else:
            os.environ["XDG_CACHE_HOME"] = self._previous_cache_home
        if self._previous_real_cache is None:
            os.environ.pop(_REAL_CACHE_ENV, None)
        else:
            os.environ[_REAL_CACHE_ENV] = self._previous_real_cache
        if self._temporary_cache is not None:
            self._temporary_cache.cleanup()
