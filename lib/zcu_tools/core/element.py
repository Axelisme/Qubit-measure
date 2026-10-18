"""Shared original-position identity for captured element facts."""

from typing import TypedDict


class ElementIdentity(TypedDict):
    """Display identity captured together with a value, never a write destination.

    index is the nonnegative position in the complete original model list,
    not the element's position in participants or output columns. name is that
    same element's declared literal name at capture time. Neither field grants
    write permission or resolves a target, alias or library entry.
    """

    index: int
    name: str
