from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import get_ident
from typing import TypeVar

import numpy as np
import pytest
from matplotlib.artist import Artist
from matplotlib.figure import Figure
from numpy.typing import NDArray
from zcu_tools.plotting.plots import NonPresentingHost, Plots

_T = TypeVar("_T")


class OwnerHost(NonPresentingHost):
    def __init__(self) -> None:
        self.in_owner = False
        self.before_call: Callable[[], None] | None = None
        self.refreshed: list[tuple[Figure, bool]] = []
        self.presented: list[Figure] = []

    def call(self, callback: Callable[[], _T]) -> _T:
        if self.before_call is not None:
            self.before_call()
        self.in_owner = True
        try:
            return callback()
        finally:
            self.in_owner = False

    def present(self, figure: Figure) -> None:
        self.presented.append(figure)

    def refresh(self, figure: Figure, *, final: bool = False) -> None:
        assert self.in_owner
        self.refreshed.append((figure, final))


@dataclass
class Frame:
    values: NDArray[np.float64]
    labels: list[str]


def test_decorator_copies_before_owner_and_refreshes_distinct_root_figures() -> None:
    host = OwnerHost()
    plots = Plots(host)
    first, axes = plots.subplots(num="first")
    second, other = plots.subplots(num="second")
    (line,) = axes.plot([], [])
    subfigure = second.add_subfigure(second.add_gridspec(1, 1)[0])
    (nested_line,) = subfigure.subplots().plot([], [])
    first.show()
    second.show()
    observed: list[Frame] = []

    @plots.on_plot_thread
    def push(frame: Frame) -> Iterable[Artist]:
        assert host.in_owner
        assert host.refreshed == []
        line.set_data([0, 1], frame.values)
        observed.append(frame)
        return (line, axes, first, nested_line, subfigure, other)

    source = Frame(np.array([2.0, 3.0]), ["original"])

    def mutate_source() -> None:
        source.values[:] = 99
        source.labels.append("later")

    host.before_call = mutate_source
    assert push(source) is None
    host.before_call = None
    np.testing.assert_array_equal(observed[0].values, [2, 3])
    assert observed[0].labels == ["original"]
    assert host.refreshed == [(first, False), (second, False)]
    assert host.presented == [first, second]
    plots.finish()
    plots.release()


@pytest.mark.parametrize("worker", [False, True])
def test_nonpresenting_decorator_executes_synchronously_on_caller(worker: bool) -> None:
    plots = Plots(NonPresentingHost())
    figure, axes = plots.subplots()
    observed: list[int] = []

    @plots.on_plot_thread
    def push(title: str) -> Iterable[Artist]:
        observed.append(get_ident())
        return (axes.set_title(title),)

    def produce() -> int:
        push("Caller thread")
        return get_ident()

    try:
        figure.show()
        with ThreadPoolExecutor(max_workers=1) as pool:
            caller = pool.submit(produce).result() if worker else produce()
        assert observed == [caller]
        assert axes.get_title() == "Caller thread"
    finally:
        plots.finish()
        plots.release()


@pytest.fixture
def decorated_plots() -> Iterator[tuple[Plots, OwnerHost]]:
    host = OwnerHost()
    plots = Plots(host)
    plots.subplots(num="frame")
    try:
        yield plots, host
    finally:
        plots.finish(present=False)
        plots.release()


def test_decorator_before_show_and_mixed_figures_refresh_only_presented(
    decorated_plots: tuple[Plots, OwnerHost],
) -> None:
    plots, host = decorated_plots
    first = plots["frame"]
    second, _ = plots.subplots(num="later")

    @plots.on_plot_thread
    def update(value: int) -> Iterable[Artist]:
        first.axes[0].set_title(str(value))
        second.axes[0].set_title(str(value))
        return (first, second)

    update(1)
    assert host.refreshed == []
    assert host.presented == []
    first.show()
    assert first.axes[0].get_title() == "1"
    update(2)
    assert host.refreshed == [(first, False)]
    assert host.presented == [first]
    plots.finish()
    assert second.axes[0].get_title() == "2"
    assert host.presented == [first, second]


def test_decorator_empty_and_missing_return_do_not_refresh(
    decorated_plots: tuple[Plots, OwnerHost],
) -> None:
    plots, host = decorated_plots
    plots["frame"].show()

    @plots.on_plot_thread
    def empty(_: int) -> Iterable[Artist]:
        return ()

    @plots.on_plot_thread
    def missing(_: int) -> None:
        pass

    empty(0)
    with pytest.raises(TypeError, match="return.*iterable.*artists"):
        missing(0)
    assert host.refreshed == []
    assert host.presented == [plots["frame"]]


@pytest.mark.parametrize("presented", [False, True])
def test_decorator_validates_all_returned_figures_before_refresh(
    decorated_plots: tuple[Plots, OwnerHost], presented: bool
) -> None:
    plots, host = decorated_plots
    if presented:
        plots["frame"].show()
    foreign = Figure()
    (foreign_line,) = foreign.subplots().plot([], [])

    @plots.on_plot_thread
    def update(_: int) -> Iterable[Artist]:
        yield plots["frame"]
        yield foreign_line

    with pytest.raises(ValueError, match="belong to this Plots"):
        update(0)
    assert host.refreshed == []
    assert host.presented == ([plots["frame"]] if presented else [])


def test_decorator_body_exception_propagates_without_refresh(
    decorated_plots: tuple[Plots, OwnerHost],
) -> None:
    plots, host = decorated_plots
    plots["frame"].show()
    error = LookupError("cannot draw")

    @plots.on_plot_thread
    def update(_: int) -> Iterable[Artist]:
        plots["frame"].axes[0].set_title("Partial mutation")
        raise error

    with pytest.raises(LookupError, match="cannot draw") as caught:
        update(0)
    assert caught.value is error
    assert plots["frame"].axes[0].get_title() == "Partial mutation"
    assert host.refreshed == []


@pytest.mark.parametrize("release", [False, True])
def test_decorator_checks_finished_lifetime_before_copy(
    decorated_plots: tuple[Plots, OwnerHost], release: bool
) -> None:
    plots, host = decorated_plots

    class Uncopyable:
        def __deepcopy__(self, memo: dict[int, object]) -> "Uncopyable":
            raise AssertionError("must check lifetime before copying")

    @plots.on_plot_thread
    def update(_: Uncopyable) -> Iterable[Artist]:
        raise AssertionError("must not execute after finish")

    plots.finish(present=False)
    if release:
        plots.release()
    with pytest.raises(RuntimeError, match="finished"):
        update(Uncopyable())
    assert host.refreshed == []


def test_decorator_rechecks_lifetime_on_owner(
    decorated_plots: tuple[Plots, OwnerHost],
) -> None:
    plots, host = decorated_plots

    @plots.on_plot_thread
    def update(_: int) -> Iterable[Artist]:
        raise AssertionError("must not execute after finish")

    def finish_before_delivery() -> None:
        host.before_call = None
        plots.finish(present=False)

    host.before_call = finish_before_delivery
    with pytest.raises(RuntimeError, match="finished"):
        update(1)
    assert host.refreshed == []


def test_decorators_share_axes_for_repeated_and_one_off_updates(
    decorated_plots: tuple[Plots, OwnerHost],
) -> None:
    plots, host = decorated_plots
    figure = plots["frame"]
    axes = figure.axes[0]
    (line,) = axes.plot([], [])
    figure.show()

    @plots.on_plot_thread
    def update(value: int) -> Iterable[Artist]:
        line.set_data([0], [value])
        return (line,)

    @plots.on_plot_thread
    def once(value: str) -> Iterable[Artist]:
        return (axes.set_title(value),)

    update(1)
    once("One-off")
    assert axes.get_title() == "One-off"
    assert host.refreshed == [(figure, False), (figure, False)]
    assert host.presented == [figure]






