"""PySide6 mock-first GUI toolkit for experiment/v2 workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def launch_mock_gui(
    project_root: Optional[str] = None,
    backend: str = "mock",
) -> None:
    """Launch the mock GUI entrypoint.

    PySide6 is imported lazily so that notebook-only users are unaffected.
    Fake-only 階段僅支援 backend='mock'.
    """
    if backend != "mock":
        raise ValueError("v2_gui fake-only mode currently supports backend='mock' only")

    from .ui import run_app

    run_app(Path(project_root) if project_root else None, backend=backend)


__all__ = ["launch_mock_gui"]
