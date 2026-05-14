from __future__ import annotations

from typing import TYPE_CHECKING

from typing_extensions import Any

if TYPE_CHECKING:
    from .pipeline import ChunkList

PSEUDO_LABELS = {"PREV", "HERE", "NEXT", "SKIP"}


class Label:
    """A first-class logical label identity."""

    # Global allocator: tracks all allocated label names to guarantee uniqueness.
    # INVARIANT: Label.reset() must be called before each top-level build pass
    # (i.e., before IRBuilder.build()).
    _allocated: dict[str, Label] = {}

    def __new__(cls, name: str):
        if name in PSEUDO_LABELS:
            return super().__new__(cls)
        if name not in cls._allocated:
            raise ValueError(f"Label name '{name}' has not been allocated yet.")
        return cls._allocated[name]

    def __init__(self, name: str):
        self.name = name

    @classmethod
    def make_new(cls, base_name: str) -> Label:
        """Create a new logical label, ensuring the name is unique."""
        if base_name in PSEUDO_LABELS:
            return cls(base_name)

        name = base_name
        counter = 0
        while name in cls._allocated:
            name = f"{base_name}_{counter}"
            counter += 1

        inst = super().__new__(cls)
        inst.__init__(name)
        cls._allocated[name] = inst
        return inst

    @property
    def label_set(self) -> set[str]:
        # Keep as property for backward compatibility if needed by other modules
        return set(self._allocated.keys())

    def is_pseudo_name(self) -> bool:
        return self.name in PSEUDO_LABELS

    def clone_new(self) -> Label:
        """Create a new label derived from this one's name."""
        if self.is_pseudo_name():
            return self
        return Label.make_new(self.name)

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
        cls._allocated.clear()

    def __str__(self) -> str:
        if self.is_pseudo_name():
            return self.name
        return f"&{self.name}"

    def __repr__(self) -> str:
        return f"Label({self.name})"


def collect_referenced_labels(chunks: ChunkList) -> set[Label]:
    """Collect all Labels referenced by need_label across a chunk list."""
    from .instructions import BaseInst
    from .node import BasicBlockNode

    refs: set[Label] = set()
    for chunk in chunks:
        if not isinstance(chunk, BasicBlockNode):
            continue
        for inst in (
            *chunk.labels,
            *chunk.insts,
            *([chunk.branch] if chunk.branch else []),
        ):
            if isinstance(inst, BaseInst) and inst.need_label is not None:
                refs.add(inst.need_label)
    return refs
