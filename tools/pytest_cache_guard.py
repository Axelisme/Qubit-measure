"""Pytest GUI cache isolation and pollution diagnostics."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from platformdirs import user_cache_dir


class GuiCacheGuard:
    """Isolate default GUI persistence and report writes to a protected cache.

    ``real_cache`` is the GUI directory to protect; omitted means the current
    platformdirs GUI cache. ``cache_root`` is the temporary XDG cache root;
    omitted means an owned temporary directory inside this checkout. Construct
    before importing test modules, register with pytest, and let unconfigure
    restore the prior environment. Only Linux supports the XDG configuration;
    other platforms raise pytest.UsageError before changing the environment.
    """

    def __init__(
        self, *, real_cache: Path | None = None, cache_root: Path | None = None
    ) -> None:
        if sys.platform != "linux":
            raise pytest.UsageError(
                "GUI cache isolation only supports Linux (XDG_CACHE_HOME)"
            )
        self._real_cache = (
            real_cache
            if real_cache is not None
            else Path(user_cache_dir("zcu_tools", "zcu_tools")) / "gui"
        )
        self._previous_cache_home = os.environ.get("XDG_CACHE_HOME")
        self._temporary_cache: TemporaryDirectory[str] | None = None
        self._closed = False
        if cache_root is None:
            self._temporary_cache = TemporaryDirectory(
                prefix=".pytest-tmp-cache-", dir=Path(__file__).resolve().parents[1]
            )
            cache_root = Path(self._temporary_cache.name)
        os.environ["XDG_CACHE_HOME"] = str(cache_root.resolve())

    def pytest_unconfigure(self) -> None:
        """Restore the environment and remove owned temporary cache files."""
        if self._closed:
            return
        self._closed = True
        if self._previous_cache_home is None:
            os.environ.pop("XDG_CACHE_HOME", None)
        else:
            os.environ["XDG_CACHE_HOME"] = self._previous_cache_home
        if self._temporary_cache is not None:
            self._temporary_cache.cleanup()
