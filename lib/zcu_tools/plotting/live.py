"""Owner-managed updates for caller-defined live figures."""

from collections.abc import Callable
from copy import deepcopy
from typing import Generic, TypeVar

from matplotlib.figure import Figure

from .plots import PlotHost

_T = TypeVar("_T")


class LiveUpdate(Generic[_T]):
    """Snapshot typed data and apply a caller's drawing callback on one owner.

    Construct inside a ``Plots.create_live`` initializer, passing its ``host``,
    ``figure`` and ``ensure_active`` arguments unchanged. ``callback`` accepts
    one payload of type T and mutates only that live figure's artists. Retain
    this handle in the producer, not the active artists. The collection owns
    the figure and its finish/release lifetime; this handle does not finish it.

    ``update`` deep-copies the payload before synchronous owner dispatch.
    Payloads must support ``copy.deepcopy`` (including nested arrays and
    containers); custom deepcopy implementations must preserve isolation.
    Callbacks run on the host owner, may retain the copied payload, and must
    not mutate producer-owned data. No data shape or plotting format is imposed.
    Copy, callback and host failures propagate without rollback. A failed
    callback does not refresh. Stop producers before finishing their collection.
    """

    def __init__(
        self,
        host: PlotHost,
        figure: Figure,
        callback: Callable[[_T], None],
        ensure_active: Callable[[], None],
    ) -> None:
        """Bind the initializer's owner, figure, drawing callback and lifetime check."""
        self._host = host
        self._figure = figure
        self._callback = callback
        self._ensure_active = ensure_active

    def ensure_active(self) -> None:
        """Check lifetime before caller-side validation or data preparation.

        Raise RuntimeError after finish/release without dispatching to the host.
        update and refresh also check lifetime themselves, including on the owner.
        """
        self._ensure_active()

    def update(self, data: _T, *, refresh: bool = True) -> None:
        """Copy data, run callback on the owner, then refresh unless disabled.

        Block until completion. ``refresh=False`` permits batching followed by
        ``refresh()`` or ``Plots.refresh(name)``. Raise RuntimeError after finish
        or release, checking both before copying and on the owner before mutation.
        Other copy/callback/host errors propagate; partial artist changes remain.
        """
        self._ensure_active()
        snapshot = deepcopy(data)

        def apply() -> None:
            self._ensure_active()
            self._callback(snapshot)
            if refresh:
                self._host.refresh(self._figure)

        self._host.call(apply)

    def refresh(self) -> None:
        """Synchronously refresh the bound figure on its owner without new data.

        Raise RuntimeError after finish/release, including when finish occurs
        before owner delivery. Host errors propagate without changing lifetime.
        """
        self._ensure_active()

        def apply() -> None:
            self._ensure_active()
            self._host.refresh(self._figure)

        self._host.call(apply)
