"""Canonical persistence keys for inline reference variants."""

from __future__ import annotations

_CUSTOM_PREFIX = "<Custom:"


def make_custom_reference_key(label: str) -> str:
    """Return the canonical persistence key for a custom reference label."""
    if not isinstance(label, str) or not label:
        raise ValueError("Custom reference label must be a non-empty string")
    if ">" in label:
        raise ValueError("Custom reference label must not contain '>'")
    return f"{_CUSTOM_PREFIX}{label}>"


def parse_custom_reference_key(key: str) -> str | None:
    """Parse a custom key, returning ``None`` for an ordinary library key.

    A key that starts like a custom key is never treated as a library key: its
    complete syntax is validated so corrupt persisted state fails immediately.
    """
    if not isinstance(key, str):
        raise ValueError("Reference key must be a string")
    if not key.startswith(_CUSTOM_PREFIX):
        return None
    if not key.endswith(">"):
        raise ValueError(f"Invalid custom reference key: {key!r}")
    label = key[len(_CUSTOM_PREFIX) : -1]
    if not label or ">" in label:
        raise ValueError(f"Invalid custom reference key: {key!r}")
    return label


def is_custom_reference_key(key: str) -> bool:
    """Return whether ``key`` is a valid custom reference key."""
    return parse_custom_reference_key(key) is not None


__all__ = [
    "is_custom_reference_key",
    "make_custom_reference_key",
    "parse_custom_reference_key",
]
