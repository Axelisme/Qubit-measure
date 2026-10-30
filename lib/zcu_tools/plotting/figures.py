"""Operation-owned named figures, independent of presentation and save policy."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from threading import RLock
from typing import Any, Literal, overload
from weakref import ReferenceType, WeakKeyDictionary, ref

import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from numpy.typing import NDArray

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
        self._figure_number = 0

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

    @overload
    def subplots(
        self,
        nrows: Literal[1] = 1,
        ncols: Literal[1] = 1,
        *,
        num: str | None = None,
        sharex: bool | Literal["none", "all", "row", "col"] = False,
        sharey: bool | Literal["none", "all", "row", "col"] = False,
        squeeze: Literal[True] = True,
        subplot_kw: dict[str, Any] | None = None,
        gridspec_kw: dict[str, Any] | None = None,
        **figure_kwargs: Any,
    ) -> tuple[Figure, Axes]: ...

    @overload
    def subplots(
        self,
        nrows: int = 1,
        ncols: int = 1,
        *,
        num: str | None = None,
        sharex: bool | Literal["none", "all", "row", "col"] = False,
        sharey: bool | Literal["none", "all", "row", "col"] = False,
        squeeze: bool = True,
        subplot_kw: dict[str, Any] | None = None,
        gridspec_kw: dict[str, Any] | None = None,
        **figure_kwargs: Any,
    ) -> tuple[Figure, Axes | NDArray[np.object_]]: ...

    @overload
    def subplots(
        self,
        name: str,
        *,
        nrows: Literal[1] = 1,
        ncols: Literal[1] = 1,
        num: str | None = None,
        sharex: bool | Literal["none", "all", "row", "col"] = False,
        sharey: bool | Literal["none", "all", "row", "col"] = False,
        squeeze: Literal[True] = True,
        subplot_kw: dict[str, Any] | None = None,
        gridspec_kw: dict[str, Any] | None = None,
        **figure_kwargs: Any,
    ) -> tuple[Figure, Axes]: ...

    @overload
    def subplots(
        self,
        name: str,
        *,
        nrows: int = 1,
        ncols: int = 1,
        num: str | None = None,
        sharex: bool | Literal["none", "all", "row", "col"] = False,
        sharey: bool | Literal["none", "all", "row", "col"] = False,
        squeeze: bool = True,
        subplot_kw: dict[str, Any] | None = None,
        gridspec_kw: dict[str, Any] | None = None,
        **figure_kwargs: Any,
    ) -> tuple[Figure, Axes | NDArray[np.object_]]: ...

    # tolerated: Keep native Matplotlib shape, sharing and layout options explicit.
    def subplots(  # noqa: PLR0913
        self,
        *args: int | str,
        nrows: int = 1,
        ncols: int = 1,
        num: str | None = None,
        sharex: bool | Literal["none", "all", "row", "col"] = False,
        sharey: bool | Literal["none", "all", "row", "col"] = False,
        squeeze: bool = True,
        subplot_kw: dict[str, Any] | None = None,
        gridspec_kw: dict[str, Any] | None = None,
        **figure_kwargs: Any,
    ) -> tuple[Figure, Axes | NDArray[np.object_]]:
        """Create native subplots without presenting or registering with pyplot.

        Accept Matplotlib's positional nrows/ncols and keyword sharing, squeeze,
        subplot, gridspec and Figure options. num is a nonempty unique string;
        omitted names use the next unused figure_N in creation order. Conflicts
        raise ValueError. A leading string remains a legacy name and cannot be
        combined with num. Adding figures after seal raises RuntimeError.
        Return a Figure and Matplotlib's scalar, 1-D or 2-D axes shape.
        """
        if args and isinstance(args[0], str):
            # Migration compatibility: remove the leading string name at contraction.
            if len(args) != 1 or num is not None:
                raise TypeError(
                    "A legacy name cannot be combined with num or dimensions"
                )
            num = args[0]
        else:
            if len(args) > 2 or any(not isinstance(arg, int) for arg in args):
                raise TypeError("Expected at most two integer dimensions")
            if args:
                nrows = int(args[0])
            if len(args) == 2:
                ncols = int(args[1])
        if num is None:
            self._figure_number += 1
            while f"figure_{self._figure_number}" in self:
                self._figure_number += 1
            num = f"figure_{self._figure_number}"
        if not num:
            raise ValueError("Figure name must be a nonempty string")
        figure = Figure(**figure_kwargs)
        self._create_canvas(figure)
        axes = figure.subplots(
            nrows=nrows,
            ncols=ncols,
            sharex=sharex,
            sharey=sharey,
            squeeze=squeeze,
            subplot_kw=subplot_kw,
            gridspec_kw=gridspec_kw,
        )
        self.adopt(num, figure)
        return figure, axes

    def _create_canvas(self, figure: Figure) -> None:
        FigureCanvasAgg(figure)

    def seal(self) -> None:
        """Stop accepting new figures while retaining native figure references."""
        with _registration_lock:
            self._sealed = True


class NamedFigures(Mapping[str, Figure]):
    """Read-only completed figures, retaining native artists and their owner.

    Construction seals membership. Keeping this mapping preserves ownership,
    even when the caller releases presentation or drops the operation handle.
    Figure contents remain mutable through the native Matplotlib interface.
    """

    def __init__(self, owner: FigureCollection) -> None:
        owner.seal()
        self._owner = owner

    def __getitem__(self, name: str) -> Figure:
        return self._owner[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._owner)

    def __len__(self) -> int:
        return len(self._owner)
