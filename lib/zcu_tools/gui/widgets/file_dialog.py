"""Neutral filesystem helpers for GUI file-dialog starting directories."""

from __future__ import annotations

import os


def nearest_existing(path: str) -> str:
    """The deepest existing ancestor of ``path`` (``path`` itself if it exists).

    Lets a dialog open near a not-yet-created target instead of at the cwd/root.
    Returns "" if nothing in the chain exists (then Qt opens the cwd).
    """
    path = os.path.abspath(path) if path else ""
    while path and not os.path.isdir(path):
        parent = os.path.dirname(path)
        if parent == path:  # reached the filesystem root
            return ""
        path = parent
    return path
