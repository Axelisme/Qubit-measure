"""Qt-free plugin actions, commands and terminal result contract."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Generic, TypeVar

from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError
from zcu_tools.gui.remote.param_spec import ParamSpec
from zcu_tools.gui.session.ports import OwnerScheduler

from .session import Session

S = TypeVar("S")
P = TypeVar("P")
R = TypeVar("R")

BackgroundSubmitter = Callable[
    [Callable[[], object], Callable[[object], None], Callable[[Exception], None]], None
]


def _project_state(state: object) -> object:
    return (
        asdict(state) if is_dataclass(state) and not isinstance(state, type) else state
    )


@dataclass(frozen=True, slots=True)
class Action(Generic[S, P]):
    """A typed state transition shared by GUI and remote command callers.

    calculate receives a detached latest state and the domain payload P, and
    returns its full replacement. Validation errors propagate without commit.
    record_undo=True replaces history; False publishes while preserving it.
    """

    calculate: Callable[[S, P], S]
    record_undo: bool = True

    def execute(self, session: Session[S], params: P) -> S:
        """Apply params to the latest session state and return a detached result.

        Propagate calculate failures without publication. Session owner-loop and
        input gates apply; subscriber failures are isolated by the session.
        """
        return session.commit(
            lambda state: self.calculate(state, params), record_undo=self.record_undo
        )


@dataclass(frozen=True, slots=True)
class Command(Generic[S]):
    """A stable command id, its shared wire ParamSpec, and a typed action bridge.

    The GUI invokes the typed Action directly. Remote validates the request with
    gui.remote.param_spec.validate_params before calling the bridge. Both call
    the same Action; the domain policy never depends on a wire codec.

    name is a nonempty unique plugin-local id, excluding reserved 'done'.
    params declares wire parameter names, types and validation constraints.
    execute receives the session and already-validated named parameters, invokes
    the typed action, and returns its domain result. Names must be unique and
    execute callable; invalid declarations raise ValueError or TypeError.
    """

    name: str
    params: tuple[ParamSpec, ...]
    execute: Callable[[Session[S], Mapping[str, object]], object]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("interactive command name must be non-empty")
        if self.name == "done":
            raise ValueError("interactive command 'done' is reserved")
        names = [param.name for param in self.params]
        if len(names) != len(set(names)):
            raise ValueError(
                f"duplicate parameters for interactive command {self.name!r}"
            )
        if not callable(self.execute):
            raise TypeError("interactive command execute must be callable")


@dataclass(frozen=True, slots=True)
class PluginDefinition(Generic[S, R]):
    """Captured-input interactive definition, with independent sessions.

    plugin_id is a nonempty stable domain identifier. seed is the deepcopy-able
    initial state S. commands is a tuple of uniquely named command declarations.
    can_finish validates committed state, raising to keep input open on failure.
    build_result returns the terminal result R after input closes; failure leaves
    it closed. project_state converts S to a read-only presentation projection;
    the default projects dataclasses to field mappings and returns other values.
    Invalid names/callbacks raise ValueError or TypeError at construction.
    """

    plugin_id: str
    seed: S
    commands: tuple[Command[S], ...]
    can_finish: Callable[[S], None]
    build_result: Callable[[S], R]
    project_state: Callable[[S], object] = _project_state
    _background: BackgroundSubmitter | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if not self.plugin_id.strip():
            raise ValueError("interactive plugin id must be non-empty")
        names = [command.name for command in self.commands]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate interactive commands for {self.plugin_id!r}")
        if not callable(self.can_finish) or not callable(self.build_result):
            raise TypeError("interactive plugin terminal callbacks must be callable")
        if not callable(self.project_state):
            raise TypeError("interactive plugin project_state must be callable")

    def bind_background(self, submit: BackgroundSubmitter) -> None:
        """Bind the operation's owner-loop delivery runner before accepting commands."""
        if self._background is not None:
            raise RuntimeError("interactive background runner is already bound")
        object.__setattr__(self, "_background", submit)

    def run_background(
        self,
        compute: Callable[[], object],
        on_done: Callable[[object], None],
        on_error: Callable[[Exception], None],
    ) -> None:
        """Submit compute to the bound runner with owner-loop completion callbacks.

        compute returns the value supplied to on_done; on_error receives compute
        failures. Raise FailedPreconditionError before submission if unbound.
        Execution, cancellation and callback isolation belong to the runner.
        """
        if self._background is None:
            raise FailedPreconditionError(
                "interactive background runner is unavailable"
            )
        self._background(compute, on_done, on_error)

    def info(self) -> Mapping[str, object]:
        """Plugin-specific read-only operation status, separate from committed state."""
        return {}

    def open(self, owner: OwnerScheduler) -> Session[S]:
        """Open one independent framework-owned session for these captured inputs."""
        return Session(self.seed, owner)

    def execute_command(
        self, session: Session[S], name: str, params: Mapping[str, object]
    ) -> object:
        """Dispatch already-decoded params; the command invokes the shared action."""
        session.ensure_input_open()
        for command in self.commands:
            if command.name == name:
                return command.execute(session, params)
        raise InvalidInputError(f"unknown interactive command {name!r}")

    def finish(self, session: Session[S]) -> R:
        """Validate before closing input; result failure leaves the gate terminal."""
        session.ensure_input_open()
        committed = session.snapshot()
        self.can_finish(committed)
        session.close_input()
        return self.build_result(committed)
