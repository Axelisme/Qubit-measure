"""Qt-free plugin actions, commands and terminal result contract."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Generic, TypeVar

from matplotlib.figure import Figure

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
    calculate: Callable[[S, P], S]

    def execute(self, session: Session[S], params: P) -> S:
        """The GUI and command handler call this same validation/commit path."""
        return session.commit(lambda state: self.calculate(state, params))


@dataclass(frozen=True, slots=True)
class Command(Generic[S]):
    """A stable command id, its shared wire ParamSpec, and a typed action bridge.

    The GUI invokes the typed Action directly. Remote validates the request with
    gui.remote.param_spec.validate_params before calling the bridge. Both call
    the same Action; the domain policy never depends on a wire codec.
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
    plugin_id: str
    seed: S
    commands: tuple[Command[S], ...]
    can_finish: Callable[[S], None]
    build_result: Callable[[S], R]
    attach_figure: Callable[[R, Figure], R] | None = None
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
        if self.attach_figure is not None and not callable(self.attach_figure):
            raise TypeError("interactive plugin attach_figure must be callable")
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

    def finish(self, session: Session[S], figure: Figure | None = None) -> R:
        """Validate before closing input; result failure leaves the gate terminal."""
        session.ensure_input_open()
        committed = session.snapshot()
        self.can_finish(committed)
        session.close_input()
        result = self.build_result(committed)
        if figure is not None and self.attach_figure is not None:
            return self.attach_figure(result, figure)
        return result
