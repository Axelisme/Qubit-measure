"""Operation-owned named figures, independent of presentation and save policy."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from threading import RLock
from typing import Any, Literal, cast, overload
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
        name: str,
        *,
        nrows: Literal[1] = 1,
        ncols: Literal[1] = 1,
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
        sharex: bool | Literal["none", "all", "row", "col"] = False,
        sharey: bool | Literal["none", "all", "row", "col"] = False,
        squeeze: bool = True,
        subplot_kw: dict[str, Any] | None = None,
        gridspec_kw: dict[str, Any] | None = None,
        **figure_kwargs: Any,
    ) -> tuple[Figure, Axes | NDArray[np.object_]]: ...

    def subplots(  # noqa: PLR0913 - preserve native Matplotlib subplot options
        self,
        name: str,
        *,
        nrows: int = 1,
        ncols: int = 1,
        sharex: bool | Literal["none", "all", "row", "col"] = False,
        sharey: bool | Literal["none", "all", "row", "col"] = False,
        squeeze: bool = True,
        subplot_kw: dict[str, Any] | None = None,
        gridspec_kw: dict[str, Any] | None = None,
        **figure_kwargs: Any,
    ) -> tuple[Figure, Axes | NDArray[np.object_]]:
        """Build and register native subplots without opening a presentation.

        Shape, shared axes and styling follow Figure.subplots/Figure. The Agg
        canvas makes the figure saveable without choosing a process-wide backend
        or registering a pyplot manager. An adapter can attach its own canvas
        after the producing operation completes.
        """
        figure = Figure(**figure_kwargs)
        FigureCanvasAgg(figure)
        axes = cast(
            "Axes | NDArray[np.object_]",
            figure.subplots(
                nrows=nrows,
                ncols=ncols,
                sharex=sharex,
                sharey=sharey,
                squeeze=squeeze,
                subplot_kw=subplot_kw,
                gridspec_kw=gridspec_kw,
            ),
        )
        self.adopt(name, figure)
        return figure, axes

    def seal(self) -> None:
        """Stop accepting new figures while retaining native figure references."""
        with _registration_lock:
            self._sealed = True
