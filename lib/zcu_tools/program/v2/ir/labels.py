from __future__ import annotations

from typing_extensions import Any

PSEUDO_LABELS = {"PREV", "HERE", "NEXT", "SKIP"}


class Label:
    """A first-class logical label identity."""

    # Global allocator: tracks all allocated label names to guarantee uniqueness.
    # INVARIANT: Label.reset() must be called before each top-level build pass
    # (i.e., before IRBuilder.build()).
    _instances: dict[str, Label] = {}

    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @classmethod
    def make_new(cls, base_name: str) -> Label:
        """Create a new logical label, ensuring the name is unique."""
        if base_name in PSEUDO_LABELS:
            return cls(base_name)

        name = base_name
        counter = 0
        while name in cls._instances:
            name = f"{base_name}_{counter}"
            counter += 1
        inst = cls(name)
        cls._instances[name] = inst
        return inst

    @classmethod
    def use_existing(cls, name: str) -> Label:
        """Use an existing label by name, without guaranteeing uniqueness."""
        if name in PSEUDO_LABELS:
            return cls(name)
        if name not in cls._instances:
            raise ValueError(f"Label name '{name}' has not been allocated yet.")
        return cls._instances[name]

    @property
    def label_set(self) -> set[str]:
        # Keep as property for backward compatibility if needed by other modules
        return set(self._instances.keys())

    def is_pseudo_name(self) -> bool:
        return self._name in PSEUDO_LABELS

    def clone_new(self) -> Label:
        """Create a new label derived from this one's name."""
        if self.is_pseudo_name():
            return self
        return Label.make_new(self._name)

    def __deepcopy__(self, memo: dict[int, Any]) -> "Label":
        if self.is_pseudo_name():
            memo[id(self)] = self
            return self

        # Ensure deepcopy preserves shared references within the cloned subtree
        # while creating a fresh unique label identity overall.
        new_label = self.clone_new()
        memo[id(self)] = new_label
        return new_label

    @classmethod
    def reset(cls) -> None:
        """Clear the allocated label set. Must be called before each top-level build."""
        cls._instances.clear()

    def __str__(self) -> str:
        if self.is_pseudo_name():
            return self._name
        return f"&{self._name}"

    def __repr__(self) -> str:
        return f"Label({self._name})"
