"""Shared path helpers for autofluxdep run artifacts."""

from __future__ import annotations

import re
from pathlib import Path

_SAFE_PATH_COMPONENT_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def safe_artifact_slug(value: str, *, fallback: str = "unnamed") -> str:
    """Return a filesystem-safe artifact path component."""
    slug = _SAFE_PATH_COMPONENT_RE.sub("-", value.strip()).strip("-")
    return slug or fallback


def relative_to_artifact(root: str | Path, path: str | Path) -> str:
    """Return ``path`` relative to an artifact root as a manifest string."""
    return str(Path(path).relative_to(Path(root)))


__all__ = ["relative_to_artifact", "safe_artifact_slug"]
