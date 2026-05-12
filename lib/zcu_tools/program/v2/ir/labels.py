from __future__ import annotations

from collections.abc import Iterable

from typing_extensions import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .instructions import BaseInst

PSEUDO_LABELS = frozenset({"PREV", "HERE", "NEXT", "SKIP"})


def is_pseudo_label_name(name: str) -> bool:
    return name in PSEUDO_LABELS


def is_register_addr(name: str) -> bool:
    core_name = name[1:] if name.startswith("&") else name
    # Using a simple check to keep labels.py free of complex regex if possible.
    # Actually, we can just use the standard checks.
    if core_name.startswith("r_wave") or core_name.startswith("s_"):
        return True
    return bool(core_name) and core_name[0] in "rswp" and core_name[1:].isdigit()


def is_system_reg_name(name: str) -> bool:
    """Returns True if the register name is a system register (s0-s15, w0-w5, or aliases)."""
    if name.startswith("s_") or name.startswith("w_") or name == "r_wave":
        return True
    if name.startswith("s") or name.startswith("w"):
        return name[1:].isdigit()
    return False


def is_volatile_reg_name(name: str) -> bool:
    """Returns True if writes to this register have hardware/external side effects (s0-s14)."""
    if not is_system_reg_name(name):
        return False
    if name.startswith("w") or name == "r_wave":
        return False
    # s15 is the only system register whose side effects are purely local to program flow.
    return name != "s15" and name != "s_addr"


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
        if is_pseudo_label_name(base_name):
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
        if is_pseudo_label_name(name):
            return cls(name)
        if name not in cls._instances:
            raise ValueError(f"Label name '{name}' has not been allocated yet.")
        return cls._instances[name]

    @property
    def label_set(self) -> set[str]:
        # Keep as property for backward compatibility if needed by other modules
        return set(self._instances.keys())

    def is_pseudo_name(self) -> bool:
        return is_pseudo_label_name(self._name)

    def clone_new(self) -> Label:
        """Create a new label derived from this one's name."""
        if is_pseudo_label_name(self._name):
            return self
        return Label.make_new(self._name)

    def __deepcopy__(self, memo: dict[int, Any]) -> "Label":
        if is_pseudo_label_name(self._name):
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
        if is_pseudo_label_name(self._name):
            return self._name
        return f"&{self._name}"

    def __repr__(self) -> str:
        return f"Label({self._name})"


def iter_label_references(inst: BaseInst) -> Iterable[Label]:
    label = inst.need_label
    if label:
        return (label,)
    return ()
