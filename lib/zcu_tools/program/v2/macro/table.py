"""Immutable table identity and selected lookup provenance."""

from dataclasses import dataclass


@dataclass(frozen=True, eq=False)
class DeclaredTable:
    """Identity of one declared scalar table, in original selector order.

    values are unpacked scalar integers. Equal contents do not make two
    separately declared resources interchangeable for selected playback proof.
    """

    values: tuple[int, ...]


@dataclass(frozen=True)
class TableSelection:
    """One table's actual destination and captured selection register.

    table identifies the declared resource. register is its loaded destination.
    selector is the loader-owned captured index, not the mutable author input.
    Exact lookup writers are certified separately by the producer owner.
    """

    table: DeclaredTable
    register: str
    selector: str
