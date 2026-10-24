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

    def __call__(self, cfg: C, /, *, context: RunContext) -> R:
        """Acquire using admitted cfg and keyword-only operation context.

        Return raw R, not RunRecord. context supplies acquisition resources and
        cancellation/progress. Hardware/I/O and callback failures propagate to
        the operation host, which owns evidence, partial results and settlement.
        """
        ...


class AnalyzeHook[C: ConfigBase, R, O, A: AnalyzeResult | None](Protocol):
    """Analyze one retained source with the edited generated options."""

    def __call__(
        self, req: AnalyzeRequest[RunRecord[C, R], O], /, *, plots: Plots
    ) -> A:
        """Compute from req's retained source, edited options and fixed parameters.

        plots is the operation's figure host. Return an AnalyzeResult or None
        for figures-only output. Failures propagate to the caller; the host
        adopts successful output and preserves previous output on failure.
        """
        ...


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
    ) -> P:
        """Compute from req's retained source, primary and edited post options.

        req also carries fixed parameters; plots belongs to this operation.
        Return an AnalyzeResult or None for figures-only output. Callback
        failures propagate; adoption and previous-output retention are host-owned.
        """
        ...


class InteractiveFrontendEnv(Protocol):
    """Qt-free access to the host's background runner, borrowed by a frontend.

    The host owns admission, execution, owner-loop delivery and shutdown.
    A frontend does not acquire ownership of the runner or session through env.
    """

    def run_background(
        self,
        compute: Callable[[], object],
        on_done: Callable[[object], None],
        on_error: Callable[[Exception], None],
    ) -> None:
        """Submit compute off-owner and return None without waiting for its result.

        Call on the UI owner thread. Deliver compute's result, including None,
        to on_done on that thread; deliver compute exceptions to on_error there.
        Submission/admission failures raise synchronously. Completion callback
        failures belong to the runner's callback boundary, not the adapter;
        they are not compute failures and are not rerouted to on_error.
        The runner owns draining and callback delivery during shutdown.
        """
        ...


class PluginHook[C: ConfigBase, R, O, S, A](Protocol):
    """Build a captured-input plugin before the host opens an interactive session."""

    def __call__(
        self, req: AnalyzeRequest[RunRecord[C, R], O], /, *, plots: Plots
    ) -> PluginDefinition[S, A]:
        """Return a plugin for req's retained source/options/parameters and plots.

        The definition pairs state S with terminal result A, which may be None
        for figures-only output. Do not open a session here; SessionFactory owns
        that step. Construction failures propagate before session creation.
        """
        ...


class FrontendHook[S, A](Protocol):
    """Build presentation for one already-paired plugin/session on the UI thread.

    The contract imports no Qt type. The UI checks the returned object against
    its InteractiveFrontend contract and owns widget mounting and cleanup.
    """

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
    ) -> object:
        """Create a frontend using plugin, its matching session, env and plots.

        env submits background work. request_finish asks the host to finish from
        committed state; request_cancel asks it to cancel. Both return False
        when no interactive operation is active, True when handled. Finish
        validation may raise while leaving input open; a handled finish is not
        necessarily a successful numeric result. The host owns settlement.

        Return the UI frontend object, not None or a result payload. Construction
        and synchronous callback failures propagate to their caller. The adapter
        adds no cleanup boundary; the host must dispose the session on mounting
        failure and terminal paths. The frontend must not dispose that session.
        """
        ...


class SessionFactory(Protocol):
    """Host-supplied generic session opener, preserving each plugin's state type.

    The host retains session lifetime, operation admission/settlement and cleanup.
    Interactive only captures the returned pair for a later frontend call.
    """

    def __call__[S, A](self, plugin: PluginDefinition[S, A], /) -> Session[S]:
        """Open/adopt one plugin session and return it, not an operation result.

        Supply the host's OwnerScheduler to plugin.open and retain the matching
        Session[S]. Opening/admission failures propagate; the factory must clean
        up any session it opened before failing. After return, the host remains
        responsible for disposal on finish, cancel and frontend-mount failure.
        This callback does not create, validate or mount a UI widget.
        """
        ...


type FrontendMount = Callable[
    [InteractiveFrontendEnv, Callable[[], bool], Callable[[], bool]], object
]
