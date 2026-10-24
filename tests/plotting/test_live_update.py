from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from zcu_tools.plotting.live import LiveUpdate
from zcu_tools.plotting.plots import NonPresentingHost, PlotHost, Plots

_T = TypeVar("_T")


class OwnerHost(NonPresentingHost):
    def __init__(self) -> None:
        self.in_owner = False
        self.before_call: Callable[[], None] | None = None
        self.refreshed: list[tuple[Figure, bool]] = []

    def call(self, callback: Callable[[], _T]) -> _T:
        if self.before_call is not None:
            self.before_call()
        self.in_owner = True
        try:
            return callback()
        finally:
            self.in_owner = False

    def refresh(self, figure: Figure, *, final: bool = False) -> None:
        assert self.in_owner
        self.refreshed.append((figure, final))


@dataclass
class Frame:
    values: NDArray[np.float64]
    labels: list[str]


def test_update_snapshots_nested_data_before_owner_callback_and_refresh() -> None:
    host = OwnerHost()
    plots = Plots(host)
    observed: list[Frame] = []

    def initialize(
        owner: PlotHost,
        figure: Figure,
        axes: tuple[Axes, ...],
        ensure_active: Callable[[], None],
    ) -> LiveUpdate[Frame]:
        assert host.in_owner

        def apply(frame: Frame) -> None:
            assert host.in_owner
            assert host.refreshed == []
            axes[0].plot(frame.values)
            observed.append(frame)

        return LiveUpdate(owner, figure, apply, ensure_active)

    handle = plots.create_live("custom", initialize)
    source = Frame(np.array([2.0, 3.0]), ["original"])

    def mutate_source() -> None:
        source.values[:] = 99
        source.labels.append("later")

    host.before_call = mutate_source
    handle.update(source)
    host.before_call = None
    assert len(observed) == 1
    np.testing.assert_array_equal(observed[0].values, [2, 3])
    assert observed[0].labels == ["original"]
    assert host.refreshed == [(plots["custom"], False)]
    plots.finish()
    plots.release()


@pytest.mark.parametrize("release", [False, True])
def test_finished_handle_rejects_updates_and_refresh(release: bool) -> None:
    host = OwnerHost()
    plots = Plots(host)
    observed: list[int] = []
    handle = plots.create_live(
        "custom",
        lambda owner, figure, axes, active: LiveUpdate(
            owner, figure, observed.append, active
        ),
    )
    handle.update(1, refresh=False)
    assert observed == [1]
    assert host.refreshed == []
    handle.refresh()
    assert host.refreshed == [(plots["custom"], False)]
    plots.finish()
    if release:
        plots.release()
    for command in (lambda: handle.update(2), handle.refresh):
        with pytest.raises(RuntimeError, match="finished"):
            command()
    assert observed == [1]
    assert host.refreshed == [(plots["custom"], False), (plots["custom"], True)]
    plots.release()


def test_owner_rechecks_lifetime_before_applying_snapshot() -> None:
    host = OwnerHost()
    plots = Plots(host)
    observed: list[int] = []
    handle = plots.create_live(
        "custom",
        lambda owner, figure, axes, active: LiveUpdate(
            owner, figure, observed.append, active
        ),
    )

    def finish_before_delivery() -> None:
        host.before_call = None
        plots.finish()

    host.before_call = finish_before_delivery
    with pytest.raises(RuntimeError, match="finished"):
        handle.update(1)
    assert observed == []
    assert host.refreshed == [(plots["custom"], True)]
    plots.release()
