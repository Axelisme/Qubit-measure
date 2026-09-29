"""Operation-owned named figures, independent of presentation and save policy."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from threading import RLock
from weakref import ReferenceType, WeakKeyDictionary, ref

from matplotlib.figure import Figure

# Neither side roots an operation, even if a figure callback refers to its owner.
_owners: WeakKeyDictionary[Figure, ReferenceType[FigureCollection]] = (
    WeakKeyDictionary()
)
_registration_lock = RLock()


class FigureCollection(Mapping[str, Figure]):
    """Own one name per native Figure until this collection is released.

    Registration is synchronized across collections, but artist mutation remains
    the caller's responsibility. Sealing fixes membership, not figure content;
    it neither closes a canvas nor declares an operation successful. Keeping a
    collection keeps its figures and ownership, including after sealing.
    """

    def __init__(self) -> None:
        self._figures: dict[str, Figure] = {}
        self._sealed = False

    def __getitem__(self, name: str) -> Figure:
        with _registration_lock:
            return self._figures[name]

    def __iter__(self) -> Iterator[str]:
        with _registration_lock:
            return iter(tuple(self._figures))

    def __len__(self) -> int:
        with _registration_lock:
            return len(self._figures)

    def adopt(self, name: str, figure: Figure) -> None:
        """Register a figure without moving its canvas or changing its contents.

        Repeating the same name/figure is idempotent, even after sealing. Name,
        alias and ownership conflicts raise ValueError without changing either
        collection. Adding a new entry after sealing raises RuntimeError.
        """
        if not name:
            raise ValueError("Figure name must not be empty")
        with _registration_lock:
            if name in self._figures:
                if self._figures[name] is figure:
                    return
                raise ValueError(
                    f"Figure name {name!r} already belongs to another figure"
                )
            if any(existing is figure for existing in self._figures.values()):
                raise ValueError("Figure already has a name in this collection")
            owner_ref = _owners.get(figure)
            owner = None if owner_ref is None else owner_ref()
            if owner is not None and owner is not self:
                raise ValueError("Figure is owned by another collection")
            if self._sealed:
                raise RuntimeError("Cannot add a figure to a sealed collection")
            self._figures[name] = figure
            _owners[figure] = ref(self)

    def seal(self) -> None:
        """Stop accepting new figures while retaining native figure references."""
        with _registration_lock:
            self._sealed = True
