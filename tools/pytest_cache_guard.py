"""Pytest GUI cache isolation and pollution diagnostics."""

from __future__ import annotations

from pathlib import Path

import pytest


class GuiCacheGuard:
    """Isolate default GUI persistence and report writes to a protected cache.

    ``real_cache`` is the GUI directory to protect; omitted means the current
    platformdirs GUI cache. ``cache_root`` is the temporary XDG cache root;
    omitted means an owned temporary directory inside this checkout. Construct
    before importing test modules, register with pytest, and let unconfigure
    restore the prior environment. Only Linux supports the XDG configuration.
    """

    def __init__(
        self, *, real_cache: Path | None = None, cache_root: Path | None = None
    ) -> None:
        # Inert harness for the first assertion-red persistence contract.
        self.real_cache = real_cache
        self.cache_root = cache_root

    def pytest_unconfigure(self, config: pytest.Config) -> None:
        """Restore the environment and remove owned temporary cache files."""
