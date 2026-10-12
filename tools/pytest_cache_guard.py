"""Inert public cache-guard interface for the test-first contract commit."""

from pathlib import Path


class GuiCacheGuard:
    """Expose the plugin API without implementing isolation or diagnostics yet."""

    def __init__(
        self, *, real_cache: Path | None = None, cache_root: Path | None = None
    ) -> None:
        """Accept protected GUI and isolated XDG paths; the red harness does nothing."""

    def pytest_unconfigure(self) -> None:
        """Provide the cleanup hook without changing the process environment."""
