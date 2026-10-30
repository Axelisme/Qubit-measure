"""Immutable table identity and selected lookup provenance."""

from dataclasses import dataclass


@dataclass(frozen=True, eq=False)
class DeclaredTable:
    """Identity of one declared integer table, in original selector order.

    values are unpacked scalars or raw uint32 words, before signed dmem encoding.
    The consumer decides whether a value represents time or a hardware word.
    Equal contents do not make two
    separately declared resources interchangeable for selected playback proof.
    """

    values: tuple[int, ...]


@dataclass(frozen=True)
class TableSelection:
    """One table's actual destination and captured selection register.

    table identifies the declared resource. register is its loaded destination.
    selector identifies the index at the lookup position. Ordinary lookups use
    the author's register; selected playback uses a loader-owned capture so later
    author writes cannot change that selection. Exact lookup writers are
    certified separately by the producer owner.
    """

    table: DeclaredTable
    register: str
    selector: str
