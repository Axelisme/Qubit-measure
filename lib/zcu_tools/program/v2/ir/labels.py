from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .instructions import Instruction

PSEUDO_LABELS = frozenset({"PREV", "HERE", "NEXT", "SKIP"})


class Label:
    """A first-class logical label identity."""

    label_set: set[str] = set()

    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @classmethod
    def make_new(cls, base_name: str) -> "Label":
        """Create a new logical label, ensuring the name is unique."""
        name = base_name
        counter = 0
        while name in cls.label_set:
            name = f"{base_name}_{counter}"
            counter += 1
        cls.label_set.add(name)
        return cls(name)

    def clone_new(self) -> "Label":
        """Create a new label derived from this one's name."""
        return Label.make_new(self._name)

    def __deepcopy__(self, memo: dict[int, Any]) -> "Label":
        # Ensure deepcopy preserves shared references within the cloned subtree
        # while creating a fresh unique label identity overall.
        new_label = self.clone_new()
        memo[id(self)] = new_label
        return new_label

    @classmethod
    def reset(cls) -> None:
        """Clear the allocated label set."""
        cls.label_set.clear()

    def __str__(self) -> str:
        return self._name

    def __repr__(self) -> str:
        return f"Label({self._name})"

    def __hash__(self):
        return hash(id(self))

    def __eq__(self, other):
        return self is other


def iter_label_references(inst: "Instruction") -> Iterable["Label"]:
    label = inst.need_label
    if label:
        return (label,)
    return ()
