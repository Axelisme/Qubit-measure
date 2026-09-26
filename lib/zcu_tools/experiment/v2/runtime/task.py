from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Generic

from matplotlib.axes import Axes
from numpy.typing import NDArray
from typing_extensions import TypeVar

from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.v2.utils import Result
from zcu_tools.liveplot import AbsLivePlot

from .result_tree import ResultUpdateEvent
from .schedule import ScheduleStep

T_Cfg = TypeVar("T_Cfg", bound=ExpCfgModel)
T_Env = TypeVar("T_Env")
T_Result = TypeVar("T_Result", bound=Result)
T_PlotDict = TypeVar("T_PlotDict", bound=Mapping[str, AbsLivePlot])
T_SaveAxis = TypeVar("T_SaveAxis", bound=NDArray[Any])


class Acquirer(ABC, Generic[T_Cfg, T_Env, T_Result]):
    def init(self, dynamic_pbar: bool = False) -> None:
        pass

    @abstractmethod
    def run(self, state: ScheduleStep[T_Cfg, Any, T_Env]) -> None: ...

    @abstractmethod
    def get_default_result(self) -> T_Result: ...

    def cleanup(self) -> None:
        pass


class TaskPlotter(ABC, Generic[T_Env, T_Result, T_PlotDict]):
    @abstractmethod
    def num_axes(self) -> dict[str, int]: ...

    @abstractmethod
    def make_plotter(self, name: str, axs: dict[str, list[Axes]]) -> T_PlotDict: ...

    @abstractmethod
    def update_plotter(
        self,
        plotters: T_PlotDict,
        event: ResultUpdateEvent[T_Env, T_Result],
        result: T_Result,
        /,
    ) -> None: ...


class TaskPersister(ABC, Generic[T_Result, T_SaveAxis]):
    @abstractmethod
    def save(
        self,
        filepath: str,
        axis_values: T_SaveAxis,
        result: T_Result,
        comment: str | None,
        prefix_tag: str,
        /,
    ) -> None: ...


class MeasurementBundle(
    Acquirer[T_Cfg, T_Env, T_Result],
    TaskPlotter[T_Env, T_Result, T_PlotDict],
    TaskPersister[T_Result, T_SaveAxis],
    Generic[T_Cfg, T_Env, T_Result, T_PlotDict, T_SaveAxis],
):
    pass


class ComposedMeasurementBundle(
    MeasurementBundle[T_Cfg, T_Env, T_Result, T_PlotDict, T_SaveAxis],
    Generic[T_Cfg, T_Env, T_Result, T_PlotDict, T_SaveAxis],
):
    def __init__(
        self,
        *,
        acquirer: Acquirer[T_Cfg, T_Env, T_Result],
        plotter: TaskPlotter[T_Env, T_Result, T_PlotDict],
        persister: TaskPersister[T_Result, T_SaveAxis],
    ) -> None:
        self.acquirer = acquirer
        self.plotter = plotter
        self.persister = persister

    def init(self, dynamic_pbar: bool = False) -> None:
        self.acquirer.init(dynamic_pbar=dynamic_pbar)

    def run(self, state: ScheduleStep[T_Cfg, Any, T_Env]) -> None:
        self.acquirer.run(state)

    def cleanup(self) -> None:
        self.acquirer.cleanup()

    def get_default_result(self) -> T_Result:
        return self.acquirer.get_default_result()

    def num_axes(self) -> dict[str, int]:
        return self.plotter.num_axes()

    def make_plotter(self, name: str, axs: dict[str, list[Axes]]) -> T_PlotDict:
        return self.plotter.make_plotter(name, axs)

    def update_plotter(
        self,
        plotters: T_PlotDict,
        event: ResultUpdateEvent[T_Env, T_Result],
        result: T_Result,
        /,
    ) -> None:
        self.plotter.update_plotter(plotters, event, result)

    def save(
        self,
        filepath: str,
        axis_values: T_SaveAxis,
        result: T_Result,
        comment: str | None,
        prefix_tag: str,
        /,
    ) -> None:
        self.persister.save(filepath, axis_values, result, comment, prefix_tag)


class MeasurementTask(
    MeasurementBundle[T_Cfg, T_Env, T_Result, T_PlotDict, T_SaveAxis],
    Generic[T_Cfg, T_Env, T_Result, T_PlotDict, T_SaveAxis],
):
    """Single executor measurement contract used by autofluxdep and overnight."""
