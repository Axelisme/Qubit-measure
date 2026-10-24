"""Keyword-aware callbacks retain each definition's cfg/options/result pairing."""

from collections.abc import Callable
from typing import Protocol

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.interactive import PluginDefinition, Session
from zcu_tools.plotting.plots import Plots

from .types import AnalyzeRequest, AnalyzeResult, PostAnalyzeRequest


class RunHook[C: ConfigBase, R](Protocol):
    """Acquire raw data; the host retains accepted cfg and acquisition evidence."""

    def __call__(self, cfg: C, /, *, context: RunContext) -> R: ...


class AnalyzeHook[C: ConfigBase, R, O, A: AnalyzeResult | None](Protocol):
    """Analyze one retained source with the edited generated options."""

    def __call__(
        self, req: AnalyzeRequest[RunRecord[C, R], O], /, *, plots: Plots
    ) -> A: ...


class PostAnalyzeHook[
    C: ConfigBase,
    R,
    A: AnalyzeResult | None,
    O,
    P: AnalyzeResult | None,
](Protocol):
    """Analyze the committed primary result and its retained raw source."""

    def __call__(
        self, req: PostAnalyzeRequest[RunRecord[C, R], A, O], /, *, plots: Plots
    ) -> P: ...


class InteractiveFrontendEnv(Protocol):
    def run_background(
        self,
        compute: Callable[[], object],
        on_done: Callable[[object], None],
        on_error: Callable[[Exception], None],
    ) -> None: ...


class PluginHook[C: ConfigBase, R, O, S, A](Protocol):
    def __call__(
        self, req: AnalyzeRequest[RunRecord[C, R], O], /, *, plots: Plots
    ) -> PluginDefinition[S, A]: ...


class FrontendHook[S, A](Protocol):
    def __call__(
        self,
        plugin: PluginDefinition[S, A],
        session: Session[S],
        env: InteractiveFrontendEnv,
        request_finish: Callable[[], bool],
        request_cancel: Callable[[], bool],
        /,
        *,
        plots: Plots,
    ) -> object: ...


class SessionFactory(Protocol):
    def __call__[S, A](self, plugin: PluginDefinition[S, A], /) -> Session[S]: ...


type FrontendMount = Callable[
    [InteractiveFrontendEnv, Callable[[], bool], Callable[[], bool]], object
]
